import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML


global avgLeft, avgRight

avgLeft = (0, 0, 0, 0)
avgRight = (0, 0, 0, 0)
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def movingAverage(avg, new_sample, N=20):
    if (avg == 0):
        return new_sample
    avg -= avg / N
    avg += new_sample / N
    return avg

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    # Applies the Canny transform
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    # Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # a blank mask
    mask = np.zeros_like(img)

    # to handle 3 channel and 1 channel images
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # state variables to keep track of most dominant segment
    global avgLeft, avgRight
    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0, 0, 0, 0)
    largestRightLine = (0, 0, 0, 0)

    if lines is None:
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2 - y1) / (x2 - x1))
            # Filter slope based on incline and
            # find the most dominent segment based on length
            if (slope > 0.5):  # right
                if (size > largestRightLineSize):
                    largestRightLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.5):  # left
                if (size > largestLeftLineSize):
                    largestLeftLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Define an imaginary horizontal line in the center of the screen
    # and at the bottom of the image, to extrapolate determined segment
    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight / 3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight / 3))])
    downLinePoint1 = np.array([0, int(imgHeight)])
    downLinePoint2 = np.array([int(imgWidth), int(imgHeight)])

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    if (math.isnan(upLeftPoint[0]) or math.isnan(downLeftPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upLeftPoint[0]), int(upLeftPoint[1])), (int(downLeftPoint[0]), int(downLeftPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected left lane over multiple video frames and draw
    # global avgLeft
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (
    movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]), movingAverage(avgx2, downLeftPoint[0]),
    movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    if (math.isnan(upRightPoint[0]) or math.isnan(downRightPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upRightPoint[0]), int(upRightPoint[1])), (int(downRightPoint[0]), int(downRightPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected right lane over multiple video frames and draw
    # global avgRight
    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]),
                movingAverage(avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # grayscale conversion before processing causes more harm than good
    # because sometimes the lane and road have same amount of luminance
    # grayscaleImage = grayscale(image)

    # Blur to avoid edges from noise
    blurredImage = gaussian_blur(image, 11)

    # Detect edges using canny
    # high to low threshold factor of 3
    # it is necessary to keep a linient threshold at the lower end
    # to continue to detect faded lane markings
    edgesImage = canny(blurredImage, 40, 50)

    # mark out the trapezium region of interest
    # dont' be too agressive as the car may drift laterally
    # while driving, hence ample space is still left on both sides.
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([[
        [3 * width / 4, 3 * height / 5],
        [width / 4, 3 * height / 5],
        [40, height],
        [width - 40, height]
    ]], dtype=np.int32)

    # mask the canny output with trapezium region of interest
    regionInterestImage = region_of_interest(edgesImage, vertices)

    # parameters tuned using this method:
    # threshold 30 by modifying it and seeing where slightly curved
    # lane markings are barely detected
    # min line length 20 by modifying and seeing where broken short
    # lane markings are barely detected
    # max line gap as 100 to allow plenty of room for the algo to
    # connect spaced out lane markings
    lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi / 180, 40, 30, 200)

    # Test detected edges by uncommenting this
    # return cv2.cvtColor(regionInterestImage, cv2.COLOR_GRAY2RGB)

    # draw output on top of original
    return weighted_img(lineMarkedImage, image)


def process_video(input_video_path, output_video_path):
    # Load the video file
    clip = VideoFileClip(input_video_path)

    # Process each frame with the lane detection function
    processed_clip = clip.fl_image(process_image)

    # Save the processed video
    processed_clip.write_videofile(output_video_path, audio=False)

#
# # Paths for input and output videos
# input_video = "D:/Lane Detection/Examples/solidWhiteRight.mp4"
# # input_video = "D:/Lane Detection/Examples/solidWhiteRight.mp4"
# output_video = "D:/Lane Detection/output/output_solidRight.mp4"
# # output_video = "D:/Lane Detection/output/output_solidWhiteRight.mp4"

# Process and save the video
# process_video(input_video, output_video)

# image = mpimage.imread("D:/Lane Detection/Examples/solidWhiteCurve.jpg")
# img = process_image(image)
# cv2.imwrite(f"D:\Lane Detection\output\detected_road.jpg",img)
# plt.figure()
# plt.imshow(img)
# plt.show()

if __name__ == "__main__":
    print('hwllo')