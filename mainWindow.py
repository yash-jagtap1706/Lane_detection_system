import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
from home_main import Ui_Dialog
from image_page_control import imageWindow
from video_page_control import videoWindow
import subprocess

class mainFile(QWidget):
    def __init__(self):
        super(mainFile,self).__init__()
        # print("Main file")
        self.mainGui = Ui_Dialog()
        self.mainGui.setupUi(self)

        self.mainGui.quit_button.clicked.connect(self.close)
        self.mainGui.image_button.clicked.connect(self.open_image_window)
        self.mainGui.video_button.clicked.connect(self.open_video_window)
        self.setWindowIcon(QIcon(f"D:\\Lane Detection\\GUI\\road_icon.png"))

    def open_video_window(self):
        self.hide()
        self.video_window = videoWindow()
        self.video_window.show()
        self.video_window.videoGui.home_button_video.clicked.connect(self.show_main_window_after_video)

    def open_image_window(self):
        self.hide()
        self.image_window = imageWindow()
        self.image_window.show()
        self.image_window.imageGui.home_button.clicked.connect(self.show_main_window)
        # self.mainGui.close()

    def show_main_window_after_video(self):
        self.video_window.close()
        self.show()

    def show_main_window(self):
        self.image_window.close()
        self.show()


    # def open_image_window_2(self):
    #     subprocess.Popen(['python',"image_page_control.py"])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mainFile()
    ui.show()
    sys.exit(app.exec_())
