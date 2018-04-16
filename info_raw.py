import os
import sys
import ntpath
import SimpleITK
import numpy

import helpers
import settings

def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    # print("Patient: ", patient_id)

    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    # print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    # print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    # print("Direction: ", direction)

    print("{};{};{};{};{};{};{}".format(patient_id, img_array.shape[0], img_array.shape[1], img_array.shape[2],
                                        origin[0], origin[1], origin[2]))


if __name__ == "__main__":

    if len(sys.argv) == 0:
        print("Please supply filename as an argument")
        exit()
    only_patient_id = sys.argv[1]

    process_image("data/gumed/" + only_patient_id + ".mhd")
