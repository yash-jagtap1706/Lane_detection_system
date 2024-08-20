import sys
import os
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication,QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from image_page import Ui_image_page
from reference_files.detection import process_image
import cv2
import numpy as np

class imageWindow(QWidget):
    def __init__(self):
        super(imageWindow,self).__init__()
        self.imageGui = Ui_image_page()
        self.imageGui.setupUi(self)
        self.image_url = ''
        self.imageGui.browse_button.clicked.connect(self.browsefile)
        self.imageGui.clear_button.clicked.connect(self.clear_button_press)
        self.imageGui.detect_button.clicked.connect(self.press_detect_button)
        # self.movie = QMovie()

    def browsefile(self):
        fname = QFileDialog.getOpenFileName(self,'open file',"D:\Lane Detection")
        self.image_url = fname[0]
        if self.image_url:
            self.work_on_image()

        # print(self.image_url)

    def clear_button_press(self):
        self.imageGui.input_image.clear()
        self.imageGui.output_image.clear()

    def work_on_image(self):
        if self.image_url:
            img_name = os.path.basename(self.image_url)
            # img = process_image(self.image_url)
            pixmap = QPixmap(self.image_url)
            scaled_pixmap = pixmap.scaled(self.imageGui.input_image.size(), aspectRatioMode=QtCore.Qt.IgnoreAspectRatio)
            self.imageGui.input_image.setPixmap(scaled_pixmap)
            self.imageGui.input_image.setScaledContents(True)

    def press_detect_button(self):
        # self.start_gif()
        self.detect_road_lanes()

    def detect_road_lanes(self):
        img = cv2.imread(self.image_url)
        print(img)

        imag = process_image(img)
        # print(imag[0,0])
        imag_rgb = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        # print(imag_rgb[0,0])
        imag_rgb = np.ascontiguousarray(imag_rgb)
        height, width, channel = imag_rgb.shape
        bytes_per_line = channel * width
        q_image = QImage(imag_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.imageGui.input_image.size(), aspectRatioMode=QtCore.Qt.IgnoreAspectRatio)
        self.imageGui.output_image.setPixmap(scaled_pixmap)
        self.imageGui.output_image.setScaledContents(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = imageWindow()
    ui.show()
    sys.exit(app.exec_())
