import glob
import math
import ntpath
import os
import random
import sys
from typing import List, Tuple

import numpy
import pandas
# limit memory usage..
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.layers import (Activation, AveragePooling3D, BatchNormalization,
                          Convolution2D, Convolution3D, Dense, Dropout,
                          Flatten, Input, LeakyReLU, MaxPooling2D,
                          MaxPooling3D, UpSampling2D, UpSampling3D,
                          ZeroPadding3D, merge)
from keras.metrics import (binary_accuracy, binary_crossentropy,
                           mean_absolute_error, mean_squared_error)
from keras.models import Model, load_model, model_from_json
from keras.optimizers import SGD, Adam
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import cv2
import helpers
import settings
import step2_train_nodule_detector

from step1_preprocess_ndsb import extract_dicom_images
from step1_preprocess_luna16 import process_image

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame,
                                       patient_id,
                                       view_size,
                                       luna16=False):
    src_dir = settings.LUNA_16_TRAIN_DIR2D2 if luna16 else settings.NDSB3_EXTRACTED_IMAGE_DIR
    patient_mask = helpers.load_patient_images(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y + view_size, start_x:
                          start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                nodule_in_mask = True

        if not nodule_in_mask:
            print("Nodule not in mask: ", (center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z,
                      " y_perc: ", y_perc)
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score

            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:",
                      center_z, " y_perc: ", y_perc)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id,
                      " center z:", center_z, " y_perc: ", y_perc)

    df_nodule_predictions.drop(
        df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions


def predict_cubes(model_path,
                  only_patient_id=None,
                  luna16=False,
                  magnification=1,
                  flip=False,
                  train_data=True,
                  holdout_no=-1,
                  ext_name="",
                  fold_count=2):

    dst_dir = "./"
    holdout_ext = ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    dst_dir += "predictions" + str(int(
        magnification * 10)) + holdout_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    sw = helpers.Stopwatch.start_new()
    model = step2_train_nodule_detector.get_net(
        input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),
        load_weight_path=model_path)

    patient_id = only_patient_id
    patient_index = 0

    print(patient_index, ": ", patient_id)
    csv_target_path = dst_dir + patient_id + ".csv"

    patient_img = helpers.load_patient_images(
        patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_i.png", [])
    if magnification != 1:
        patient_img = helpers.rescale_patient_images(
            patient_img, (1, 1, 1), magnification)

    patient_mask = helpers.load_patient_images(
        patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_m.png", [])
    if magnification != 1:
        patient_mask = helpers.rescale_patient_images(
            patient_mask, (1, 1, 1), magnification, is_mask_image=True)

    step = PREDICT_STEP
    CROP_SIZE = CUBE_SIZE
    # CROP_SIZE = 48

    predict_volume_shape_list = [0, 0, 0]
    for dim in range(3):
        dim_indent = 0
        while dim_indent + CROP_SIZE < patient_img.shape[dim]:
            predict_volume_shape_list[dim] += 1
            dim_indent += step

    predict_volume_shape = (predict_volume_shape_list[0],
                            predict_volume_shape_list[1],
                            predict_volume_shape_list[2])
    predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
    print("Predict volume shape: ", predict_volume.shape)
    done_count = 0
    skipped_count = 0
    batch_size = 128
    batch_list = []
    batch_list_coords = []
    patient_predictions_csv = []
    cube_img = None
    annotation_index = 0

    for z in range(0, predict_volume_shape[0]):
        print("z = {}/{}".format(z, predict_volume_shape[0]))
        for y in range(0, predict_volume_shape[1]):
            for x in range(0, predict_volume_shape[2]):
                cube_img = patient_img[z * step:z * step + CROP_SIZE,
                                       y * step:y * step + CROP_SIZE,
                                       x * step:x * step + CROP_SIZE]
                cube_mask = patient_mask[z * step:z * step + CROP_SIZE,
                                         y * step:y * step + CROP_SIZE,
                                         x * step:x * step + CROP_SIZE]

                if cube_mask.sum() < 2000:
                    skipped_count += 1
                else:
                    if flip:
                        cube_img = cube_img[:, :, ::-1]

                    if CROP_SIZE != CUBE_SIZE:
                        cube_img = helpers.rescale_patient_images2(
                            cube_img, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                        # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                        # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

                    img_prep = prepare_image_for_net3D(cube_img)
                    batch_list.append(img_prep)
                    batch_list_coords.append((z, y, x))
                    if len(batch_list) % batch_size == 0:
                        batch_data = numpy.vstack(batch_list)
                        p = model.predict(
                            batch_data, batch_size=batch_size)
                        for i in range(len(p[0])):
                            p_z = batch_list_coords[i][0]
                            p_y = batch_list_coords[i][1]
                            p_x = batch_list_coords[i][2]
                            nodule_chance = p[0][i][0]
                            predict_volume[p_z, p_y, p_x] = nodule_chance
                            if nodule_chance > P_TH:
                                p_z = p_z * step + CROP_SIZE / 2
                                p_y = p_y * step + CROP_SIZE / 2
                                p_x = p_x * step + CROP_SIZE / 2

                                p_z_perc = round(
                                    p_z / patient_img.shape[0], 4)
                                p_y_perc = round(
                                    p_y / patient_img.shape[1], 4)
                                p_x_perc = round(
                                    p_x / patient_img.shape[2], 4)
                                diameter_mm = round(p[1][i][0], 4)
                                print(p[1][i])
                                # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                diameter_perc = round(
                                    2 * step / patient_img.shape[2], 4)
                                diameter_perc = round(
                                    diameter_mm / patient_img.shape[2], 4)
                                nodule_chance = round(nodule_chance, 4)
                                patient_predictions_csv_line = [
                                    patient_id,
                                    annotation_index,
                                    p_x, p_y, p_z,
                                    p_x_perc, p_y_perc, p_z_perc,
                                    diameter_perc, nodule_chance,
                                    diameter_mm
                                ]
                                patient_predictions_csv.append(
                                    patient_predictions_csv_line)
                                # all_predictions_csv.append(
                                #     [patient_id] +
                                #     patient_predictions_csv_line)
                                annotation_index += 1

                        batch_list = []
                        batch_list_coords = []
                done_count += 1
                if done_count % 10000 == 0:
                    print("Done: ", done_count, " skipped:", skipped_count)

    df = pandas.DataFrame(
        patient_predictions_csv,
        columns=[
            "patient", "anno_index",
            "x_mm", "y_mm", "z_mm",
            "coord_x", "coord_y", "coord_z",
            "diameter", "nodule_chance", "diameter_mm"
        ])
    filter_patient_nodules_predictions(df, patient_id,
                                       CROP_SIZE * magnification)
    df.to_csv(csv_target_path, index=False)

    print(predict_volume.mean())
    print("Done in : ", sw.get_elapsed_seconds(), " seconds")


if __name__ == "__main__":

    if len(sys.argv) == 0:
        print("Please supply filename as an argument")
        exit()
    only_patient_id = sys.argv[1]

    # process_image("data/gumed/" + only_patient_id + ".mhd")
    extract_dicom_images(clean_targetdir_first=False, only_patient_id=only_patient_id)

    for magnification in [1, 1.5, 2]:
        predict_cubes(
            "models/model_luna16_full__fs_best.hd5",
            only_patient_id=only_patient_id,
            magnification=magnification,
            flip=False,
            train_data=True,
            holdout_no=None,
            ext_name="luna16_fs")
