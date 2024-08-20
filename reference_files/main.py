import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import cv2
import math


def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img,lines,color=[255,0,0],thickness=3):
    if lines is None:
        return
    img = np.copy(img)
    # blank image of same size as img
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # plt.figure()
    # plt.imshow(line_img)
    # plt.show()
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img
# image = mpimage.imread('D:\Lane Detection\Examples\solidWhiteCurve.jpg')
# print(image.shape)
# plt.imshow(image)  # (height, width, channel)
# plt.show()


def pipeline(image):

    # # image = mpimage.imread('D:\Lane Detection\Examples\solidWhiteCurve.jpg')
    # image = mpimage.imread('D:\Lane Detection\Examples\solidYellowCurve.jpg')
    height,width = image.shape[0],image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    canny_image = cv2.Canny(gray_image,100,200)


    cropped_image = region_of_interest(
        canny_image,
        np.array([region_of_interest_vertices], np.int32),
    )
    # plt.figure()
    # plt.imshow(cropped_image)
    # plt.show()


    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    # print(lines)

    # line_image = draw_lines(image, lines) # <---- Add this call.


    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = image.shape[0] * (3 / 5) # <-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    # Drawing lines
    line_image = draw_lines(
        image,
        [[
            [left_x_start, int(max_y), left_x_end, int(min_y)],
            [right_x_start, int(max_y), right_x_end, int(min_y)],
        ]],
        thickness=5,
    )

    plt.figure()
    plt.imshow(line_image)
    plt.show()


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print(f"Reading video from: {input_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    print(f"Writing video to: {output_video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = pipeline(frame)

        if processed_frame is None or processed_frame.size == 0:
            print("Error: Processed frame is invalid.")
            break

        out.write(processed_frame)
        cv2.imshow('Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_video_path = '/Examples/solidWhiteRight.mp4'  # Replace with your input video path
    output_video_path = 'D:\Lane Detection\output\output_video.mp4'  # Replace with your desired output path
    process_video(input_video_path, output_video_path)