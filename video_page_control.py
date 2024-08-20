import sys
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from video_page_ui import Ui_video_page
import cv2
from reference_files.detection import process_image


class videoWindow(QWidget):
    def __init__(self):
        super(videoWindow, self).__init__()
        self.videoGui = Ui_video_page()
        self.videoGui.setupUi(self)
        self.video_url = ''
        self.videoGui.browse_button.clicked.connect(self.browsefile)
        self.videoGui.clear_button_video.clicked.connect(self.clear_button_press)

        self.cap = None  # Initialize with None

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # for input video
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                converted_qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.videoGui.input_video.setPixmap(QPixmap.fromImage(converted_qt_image))

                # for output video
                processed_frame = process_image(frame)
                h, w, ch = processed_frame.shape
                bytes_per_line = ch * w
                converted_output_qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.videoGui.output_video.setPixmap(QPixmap.fromImage(converted_output_qt_image))
            else:
                self.timer.stop()
                self.cap.release()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def clear_button_press(self):
        self.videoGui.input_video.clear()
        self.videoGui.output_video.clear()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()

    def browsefile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', "D:\\Lane Detection",
                                               "Video Files (*.mp4 *.avi *.mov)")
        if fname:
            self.video_url = fname
            # print(self.video_url)
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_url)
            if not self.cap.isOpened():
                print("Error: Cannot open video file.")
                return

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Start the timer for video playback


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = videoWindow()
    ui.show()
    sys.exit(app.exec_())
