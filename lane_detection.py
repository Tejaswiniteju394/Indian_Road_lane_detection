import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import warnings
from numpy.polynomial.polyutils import RankWarning
warnings.simplefilter('ignore', RankWarning)
def make_points(image, line):
    height, width, _ = image.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 0.6)
    if abs(slope) < 1e-6:
        slope = 1e-6 if slope >= 0 else -1e-6 

    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except Exception as e:
        print(f"Error in make_points calculation: {e}")
        return None
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    return (x1, y1, x2, y2)

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([
        [(int(width*0.1), height), (int(width*0.9), height), (int(width*0.55), int(height * 0.5)), (int(width*0.45), int(height * 0.5))]
    ])
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, polygons, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lanes(img, left_line, right_line, thickness=6):
    overlay = np.zeros_like(img)
    height = img.shape[0]

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness) 

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness) 

    if left_line is not None and right_line is not None:
        y1 = left_line[1] 
        y2 = left_line[3] 
        if y1 != y2:
            left_slope, left_intercept = np.polyfit((left_line[0], left_line[2]), (left_line[1], left_line[3]), 1)
            right_slope, right_intercept = np.polyfit((right_line[0], right_line[2]), (right_line[1], right_line[3]), 1)
            if abs(left_slope) < 1e-6: left_slope = 1e-6
            if abs(right_slope) < 1e-6: right_slope = 1e-6

            center_x1 = int(((y1 - left_intercept) / left_slope + (y1 - right_intercept) / right_slope) / 2)
            center_x2 = int(((y2 - left_intercept) / left_slope + (y2 - right_intercept) / right_slope) / 2)
            width = img.shape[1]
            center_x1 = max(0, min(width - 1, center_x1))
            center_x2 = max(0, min(width - 1, center_x2))
            y1 = max(0, min(img.shape[0] - 1, y1))
            y2 = max(0, min(img.shape[0] - 1, y2))


            cv2.line(overlay, (center_x1, y1), (center_x2, y2), (255, 255, 0), thickness) # Yellow for center

    return cv2.addWeighted(img, 0.8, overlay, 1, 1)

def detect_lanes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
   
    edges = cv2.Canny(blur, 50, 150) 

    roi_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi_edges, 2, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap=10) # Lower threshold and change line parameters

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0:
                continue
            try:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope, intercept = parameters
                if slope < -0.4 and slope > -2.0: 
                    left_lines.append((slope, intercept))
                elif slope > 0.4 and slope < 2.0: 
                    right_lines.append((slope, intercept))
            except Exception as e:
                print(f"Error in polyfit or slope filtering: {e}")
                continue


    left_line = right_line = None
    if len(left_lines) > 0:
        avg_left = np.average(left_lines, axis=0)
        left_line = make_points(frame, avg_left)
    if len(right_lines) > 0:
        avg_right = np.average(right_lines, axis=0)
        right_line = make_points(frame, avg_right)
    result = draw_lanes(frame, left_line, right_line)
    if left_line is None and right_line is None:
         return frame

    return result

print("Please upload an image file:")
uploaded = files.upload()
if uploaded:
    image_filename = list(uploaded.keys())[0]
    image = cv2.imread(image_filename)
    if image is None:
        print(f"Error: Could not load image from {image_filename}. Please check the filename and try again.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detect_lanes(image_rgb) 
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.title("Left, Right, and Center Lane Detection (on the road)")
        plt.axis('off')
        plt.show()
else:
    print("No file uploaded.")
