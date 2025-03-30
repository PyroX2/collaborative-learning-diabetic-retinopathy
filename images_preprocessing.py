import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


SCALE = 300

images_path = 'datasets/segmentation/original_images/train_set'
masks_path = 'datasets/segmentation/segmentation_ground_truths/train_set'


def pad_image(frame):
    # Get width and height of frame
    frame_height, frame_width = frame.shape[:2]

    # Assumes that width will be 640 and calculating new height so that image proportion stays the same
    new_height = 640 / frame_width * frame_height

    # Resize image to new shape
    frame = cv2.resize(frame, (640, int(new_height)))

    frame_height, frame_width = frame.shape[:2]
    pad_size = 640 - frame_height
    top = pad_size // 2
    bottom = pad_size - top 
    processed_frame = cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])

    return processed_frame, top, bottom

def cut_low_values(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(frame)
    mask = np.ones_like(y)*255
    mask[y < 10] = 0
    return mask

def circle_detection(frame):
    # Resize the image to reduce processing time
    processed_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Get height and width of the image for transforming circle coordinates to percentage
    frame_height, frame_width, _ = processed_frame.shape

    original_frame_height, original_frame_width, _ = frame.shape

    # Get the mask of fundus based on luminance
    mask = cut_low_values(processed_frame)

    # Blur for better circle detection
    mask_blurred = cv2.GaussianBlur(mask, (25, 25), 0) 
    
    # Detect circles in the image
    detected_circles = cv2.HoughCircles(mask_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, 
               param2 = 30, minRadius = 0, maxRadius = 0) 

    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles)) 
    
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            a = int(a / frame_width*original_frame_width)
            b = int(b / frame_height*original_frame_height)
            r = int(r / frame_width*original_frame_width)
    
            return frame, (a, b), r
    return frame, None, None

def fundus_fitting(frame, masks):
    processed_frame, center, radius = circle_detection(frame.copy())

    # Get left and right coordinates of the circle
    min_x = center[0] - radius
    max_x = center[0] + radius
    if min_x < 0:
        min_x = 0
    if max_x >= processed_frame.shape[1]:
        max_x = processed_frame.shape[1] - 1

    # Get only the fundus area
    processed_frame = processed_frame[:, min_x:max_x]

    for i, mask in enumerate(masks):
        mask = mask[:, min_x:max_x]
        mask, top_pad, bottom_pad = pad_image(mask)
        masks[i] = mask

    # Pad image to 640x640
    processed_frame, top_pad, bottom_pad = pad_image(processed_frame)

    return processed_frame, masks, top_pad, bottom_pad

def color_enhacement(frame):
    processed_frame = frame.copy()
    
    blurred = cv2.GaussianBlur(processed_frame, (0, 0), SCALE / 30)
    processed_frame = cv2.addWeighted(processed_frame, 4, blurred, -4, 128)

    return processed_frame

def mask_area_around_fundus(frame, top_pad, bottom_pad):
    processed_frame = frame.copy()

    mask = np.zeros_like(processed_frame)
    mask = cv2.circle(mask, (320, 320), 320, (255,255,255), -1)
    mask[0:top_pad] = 0
    mask[640 - bottom_pad:] = 0
    mask = mask.astype(np.uint8)
    processed_frame = processed_frame.astype(np.uint8)
    processed_frame = cv2.bitwise_and(processed_frame, mask)

    return processed_frame
    
images = sorted(os.listdir(images_path))

# Get dir for every type of mask
masks_dirs = sorted(os.listdir(masks_path))

# Create a list fo paths for every mask of every type
masks_list = [sorted(os.listdir(os.path.join(masks_path, masks_dir))) for masks_dir in masks_dirs]

for i in range(len(images)):
    image = cv2.imread(os.path.join(images_path, images[i]))

    # Read mask of every type for current image
    image_masks = []
    for mask_dir_index, dirname in enumerate(masks_dirs):
        # Read masks
        mask = np.array(Image.open(os.path.join(masks_path, masks_dirs[mask_dir_index], masks_list[mask_dir_index][i]))) 
        mask = mask*255 # Rescale from 1 to 255
        image_masks.append(mask)

    # Fit fundus so that it takes whole image
    processed_frame, processed_image_masks, top_pad, bottom_pad = fundus_fitting(image, image_masks)

    # Enhance features and colors
    processed_frame = color_enhacement(processed_frame)

    # Mask fundus surrounding to after color enhancement
    processed_frame = mask_area_around_fundus(processed_frame, top_pad, bottom_pad)
    
    if processed_frame is None:
        continue
    
    # Show processed image
    cv2.imshow('Processed images', processed_frame)
    
    # Show masks
    for mask_index, mask in enumerate(processed_image_masks):
        cv2.imshow(f"Processed mask {mask_index}", mask)

    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()


