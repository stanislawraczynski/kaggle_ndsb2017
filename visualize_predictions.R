library(ggplot2)
library(dplyr)

foldername <- "predictions10_luna16_fs"
patientid <- "571900"

alldata <- do.call(rbind, lapply(paste(foldername, "/", list.files(foldername, paste0(patientid, "*")), sep = ""), read.csv))

p <- plot_ly(alldata, x = ~x_mm, y = ~y_mm, z = ~z_mm, color = ~patient) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'X'),
                      yaxis = list(title = 'Y'),
                      zaxis = list(title = 'Z')))
