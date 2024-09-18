import re
import os
from pix2tex.cli import LatexOCR
import pytesseract
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

#hàm minh họa hình ảnh -> hiển thị kết quả cho việc debug
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

#hàm tăng độ dày chữ cho những trường hợp chữ quá mỏng (không sử dụng)
def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

#hàm giảm độ dày chữ cho những trường hợp chữ quá dày (không sử dụng)
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

#hàm thay đổi kích thước ảnh cho việc OCR
def resize(image):
    # Resize the image
    scale_percent = 150  # Percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

#hàm tăng độ tương phản (không sử dụng)
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

#hàm xóa background để tăng độ tương phản cho ảnh (chỉ sử dụng cho simple threshold)
def bg_removal(image):
    bg = cv2.dilate(image, np.ones((5,5), dtype=np.uint8))
    bg = cv2.GaussianBlur(bg, (5,5), 1)
    # subtract out background from source
    no_bg_image = 255 - cv2.absdiff(image, bg)
    return no_bg_image

#Simple threshold để đổi ảnh sang định dạng trắng đen 
def simple_threshold(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    thresh, im_bw = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)
    return im_bw

#otsu threshold để đổi ảnh sang định dạng trắng đen (ưu tiên hàm này hơn simple threshold vì hiệu năng tốt hơn)
def otsu_threshold(image):
    # Gaussian blur with adjustable kernel size
    blur_kernel_size = (3, 3)
    image = cv2.GaussianBlur(image, blur_kernel_size, 0)
    
    # Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

#Loại bỏ noise trong ảnh giúp OCR tốt hơn
def noise_removal(image):
    import numpy as np
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

#gọi các hàm trên vào một hàm để xử lí ảnh (dùng simple threshold)
def preprocess_STH_image(image):
    #image = resize(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("temp/gray.png", image)
    image = bg_removal(image)
    cv2.imwrite("temp/no_bg.png", image)
    image = simple_threshold(image)
    image = noise_removal(image)
    #image = deskew(image)
    #cv2.imwrite('temp/STH_image.png', image)
    return image

#gọi các hàm trên vào một hàm để xử lí ảnh (dùng otsu threshold)
def preprocess_otsuTH_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = bg_removal(image)
    image = otsu_threshold(image)
    image = noise_removal(image)
    image = thick_font(image)
    #image = deskew(image)
    cv2.imwrite('temp/otsuTH_image.png', image)
    return image

#chia ảnh thành từng dòng dùng findContours của cv2
def segment_lines(original_image, processed_image, padding=5, min_contour_area=100):
    """
    Segment lines from an image based on the preprocessed image.
    """
    # Invert the binary image if necessary
    binary_image = processed_image
    inv_image = cv2.bitwise_not(binary_image)

    kernel_dilate = np.ones((3, 85), np.uint8)  # Adjust kernel size for dilation

    dilated = cv2.dilate(inv_image, kernel_dilate, iterations=1)
    cv2.imwrite("temp/dilated.png", dilated)
    
    # Find contours from the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours = [ctr for ctr in contours if cv2.contourArea(ctr) > min_contour_area]

    if not contours:
        print("No contours found.")
        return []
    
    contours_image = original_image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 3)
    cv2.imwrite("temp/contours.png", contours_image)
    
    # Sort contours by vertical position (top to bottom)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Create images from contours' bounding boxes
    line_images = []
    image_with_boxes = original_image.copy()

    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)

        # Add padding to the bounding box
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(x + w + 2 * padding, original_image.shape[1]) - x
        h = min(y + h + 2 * padding, original_image.shape[0]) - y

        # Ensure valid dimensions
        if w <= 0 or h <= 0:
            continue

        # Draw bounding box on the original image for visualization
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (40, 100, 250), 2)

        # Extract the line image using the original image
        line_img = original_image[y:y + h, x:x + w]

        # Append the line image to the list
        line_images.append(line_img)

    cv2.imwrite("temp/bboxes.png", image_with_boxes)
    
    return line_images


#gọi hàm tiền xử lí ảnh và hàm chia ảnh thành từng dòng
def process_and_segment_image(image):
    # Read and resize image
    resized_image = image
    print(f"Image resized to: {resized_image.shape}")
    
    # Preprocess the resized image (e.g., thresholding, etc.)
    preprocessed_image = preprocess_STH_image(resized_image)
    cv2.imwrite("temp/STH_image.png", preprocessed_image)
    # Segment lines using the original image and the preprocessed image
    line_images = segment_lines(resized_image, preprocessed_image)

    # Display and save segmented lines
    for i, line_img in enumerate(line_images):
        if line_img.size == 0:
            print(f"Skipping empty line image at index {i}.")
            continue
        
        cv2.imwrite(f"temp/lines/line_{i+1}.png", line_img)

    return line_images
