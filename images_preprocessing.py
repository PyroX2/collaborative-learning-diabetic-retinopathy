import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import sys


SCALE = 300
USE_MASKS = False
DEBUG = False
IMAGE_SHAPE = 2848 # int because output image is expected to be square


def pad_image(frame):
    # Get width and height of frame
    frame_height, frame_width = frame.shape[:2]

    # Assumes that width will be 640 and calculating new height so that image proportion stays the same
    ratio = IMAGE_SHAPE / max(frame_width, frame_height)
    new_height = int(frame_height * ratio)
    new_width = int(frame_width * ratio)

    # Resize image to new shape
    frame = cv2.resize(frame, (new_width, new_height))


    frame_height, frame_width = frame.shape[:2]
    pad_size = IMAGE_SHAPE - min(new_width, new_height)
    if new_height < new_width:
        top = pad_size // 2
        bottom = pad_size - top 
        processed_frame = cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])
        width_pad = False
        return processed_frame, top, bottom, width_pad
    elif new_height > new_width:
        left = pad_size // 2
        right = pad_size - left
        processed_frame = cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])
        width_pad = True
        return processed_frame, left, right, width_pad
    else:
        return frame, 0, 0, True


def cut_low_values(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(frame)
    mask = np.ones_like(y)*255
    mask[y < 20] = 0
    return mask

def circle_detection(frame):
    print(f"Original frame shape: {frame.shape}")
    # Resize the image to reduce processing time
    processed_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Get height and width of the image for transforming circle coordinates to percentage
    frame_height, frame_width, _ = processed_frame.shape

    original_frame_height, original_frame_width, _ = frame.shape

    # Get the mask of fundus based on luminance
    mask = cut_low_values(processed_frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    print(f"Mask shape: {mask.shape}")

    # Blur for better circle detection
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0) 

    if DEBUG:
        cv2.imshow("Mask", mask_blurred)
        cv2.imshow("Original Frame", processed_frame)
        cv2.waitKey(0)

    # Detect circles in the image
    detected_circles = cv2.HoughCircles(mask_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, 
               param2 = 30, minRadius = 0, maxRadius = 0) 
    
    if DEBUG:
        # draw detected circles on the original frame
        if detected_circles is not None:
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                cv2.circle(frame, (int(a / frame_width * original_frame_width), int(b / frame_height * original_frame_height)), int(r / frame_width * original_frame_width), (0, 255, 0), 2)
                cv2.circle(frame, (int(a / frame_width * original_frame_width), int(b / frame_height * original_frame_height)), 2, (0, 0, 255), 3)
                cv2.imshow("Detected Circles", frame)
                cv2.waitKey(0)

    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles)) 
    
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            a = int(a / frame_width*original_frame_width)
            b = int(b / frame_height*original_frame_height)
            r = int(r / frame_width*original_frame_width)
    
            return frame, (a, b), r
        
    if DEBUG:
        print(f"Detected circles: {detected_circles}")
    return frame, None, None

def fundus_fitting(frame, masks=[]):
    processed_frame, center, radius = circle_detection(frame.copy())

    if center is None or radius is None:
        print("No circle detected, skipping image.")
        return None, None, 0, 0, False

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
        mask, top_pad, bottom_pad, width_pad = pad_image(mask)
        masks[i] = mask

    # Pad image to 640x640
    processed_frame, top_pad, bottom_pad, width_pad = pad_image(processed_frame)

    return processed_frame, masks, top_pad, bottom_pad, width_pad

def color_enhacement(frame):
    processed_frame = frame.copy()
    
    blurred = cv2.GaussianBlur(processed_frame, (0, 0), SCALE / 30)
    processed_frame = cv2.addWeighted(processed_frame, 4, blurred, -4, 128)
    
    return processed_frame

def mask_area_around_fundus(frame, top_pad, bottom_pad, width_pad):
    processed_frame = frame.copy()

    mask = np.zeros_like(processed_frame)
    mask = cv2.circle(mask, (int(IMAGE_SHAPE/2), int(IMAGE_SHAPE/2)), int(IMAGE_SHAPE/2), (255,255,255), -1)
    if not width_pad:
        mask[0:top_pad] = 0
        mask[IMAGE_SHAPE - bottom_pad:] = 0
    else:
        mask[:, 0:top_pad] = 0
        mask[:, IMAGE_SHAPE - bottom_pad:] = 0
    mask = mask.astype(np.uint8)
    processed_frame = processed_frame.astype(np.uint8)
    processed_frame = cv2.bitwise_and(processed_frame, mask)

    return processed_frame
    

def preprocess_images(images_path, output_images_path, masks_path=None, output_masks_path=None):
    images = sorted(os.listdir(images_path))

    # Get dir for every type of mask
    if USE_MASKS:
        masks_dirs = sorted(os.listdir(masks_path))
        masks_dirs = [masks_dir for masks_dir in masks_dirs if os.path.isdir(os.path.join(masks_path, masks_dir))]

        # Create a list fo paths for every mask of every type
        masks_list = [sorted(os.listdir(os.path.join(masks_path, masks_dir))) for masks_dir in masks_dirs]

    suffixes = {"microaneurysms": "_MA.tif", "haemorrhages":"_HE.tif", "hard_exudates":"_EX.tif", "soft_exudates":"_SE.tif", "optic_disc": "_OD.tif"}

    for i in range(len(images)):
        image = cv2.imread(os.path.join(images_path, images[i]))

        # Read mask of every type for current image
        image_masks = []

        if USE_MASKS:
            masks_filenames = []
            for mask_dir_index, dirname in enumerate(masks_dirs):
                # Read masks
                image_filename, extension = os.path.splitext(images[i])
                
                mask_filename = image_filename + suffixes[masks_dirs[mask_dir_index]]
                if mask_filename not in masks_list[mask_dir_index]:
                    mask = np.zeros_like(image[..., 0]).astype(np.uint8)
                    print(f"Mask {mask_filename} not found in {masks_dirs[mask_dir_index]}")
                else:
                    mask = np.array(Image.open(os.path.join(masks_path, masks_dirs[mask_dir_index], mask_filename))) 

                if mask.shape[-1] == 4:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
                    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                else:
                    mask *= 255

                masks_filenames.append(mask_filename)
                image_masks.append(mask)

        # Fit fundus so that it takes whole image
        if USE_MASKS:
            processed_frame, processed_image_masks, top_pad, bottom_pad, width_pad = fundus_fitting(image, image_masks)
        else:
            processed_frame, processed_image_masks, top_pad, bottom_pad, width_pad = fundus_fitting(image)

        if processed_frame is None:
            print(f"Skipping image {images[i]} due to no fundus detected.")
            continue

        # Enhance features and colors
        processed_frame = color_enhacement(processed_frame)

        # Mask fundus surrounding to after color enhancement
        processed_frame = mask_area_around_fundus(processed_frame, top_pad, bottom_pad, width_pad)
        
        if processed_frame is None:
            continue

        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        cv2.imwrite(os.path.join(output_images_path, images[i]), processed_frame)

        # Show masks
        if USE_MASKS:
            for mask_index, mask in enumerate(processed_image_masks):
                if not os.path.exists(os.path.join(output_masks_path, masks_dirs[mask_index])):
                    os.makedirs(os.path.join(output_masks_path, masks_dirs[mask_index]))

                mask_filename, extension = os.path.splitext(masks_filenames[mask_index])
                mask_filename = mask_filename + '.png'

                cv2.imwrite(os.path.join(output_masks_path, masks_dirs[mask_index], mask_filename), mask)

    cv2.destroyAllWindows()


def main():
    splits = ['train', 'test', 'val']
    classes = [0, 1, 2, 3, 4]

    for split in splits:
        for class_idx in classes:
            images_path = f'/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2/dr_unified_v2/{split}/{class_idx}'

            output_images_path = f'/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/eyepacs-aptos-messidor-diabetic-retinopathy-original-preprocessed-no-color-enhancement/{split}/{class_idx}'

            preprocess_images(images_path, output_images_path)


if __name__ == '__main__':
    main()