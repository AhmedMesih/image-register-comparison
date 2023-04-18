import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as compare_ssim

global merged_image
global registered_second_image

def preview_merged_image():
    global merged_image
    if merged_image is None:
        print("No merged image available.")
        return
    merged_image_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(merged_image_rgb)
    image = ImageTk.PhotoImage(image)
    top = Toplevel()
    top.title("Merged Image Preview")
    top.columnconfigure(0, weight=1)
    top.rowconfigure(0, weight=1)
    canvas = Canvas(top, width=image.width(), height=image.height())
    canvas.grid(row=0, column=0)
    canvas.create_image(0, 0, anchor=NW, image=image)
    canvas.image = image

def open_first_image():
    global first_image
    file_path = filedialog.askopenfilename()
    first_image = cv2.imread(file_path)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    display_image(first_image, image_canvas1)
    
def open_second_image():
    global second_image, first_image, merged_image, registered_second_image
    file_path = filedialog.askopenfilename()
    second_image = cv2.imread(file_path)
    second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
    display_image(second_image, image_canvas2)
    if first_image is not None:
        registered_second_image = register_images(second_image, first_image)  # Swap the order of input images
        display_image(registered_second_image, image_canvas4)
    update_comparison()
    merged_image = merge_images(first_image, registered_second_image)
    display_image(merged_image, image_canvas5)

def update_comparison():
    global registered_second_image
    if first_image is not None and second_image is not None:
        threshold_value = threshold_slider.get()
        difference_image = compute_difference_image(first_image, registered_second_image, threshold_value)
        display_image(difference_image, image_canvas3)

def display_image(image, canvas):
    image = Image.fromarray(image)
    image.thumbnail((300, 300))
    image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=NW, image=image)
    canvas.image = image

def compute_difference_image(img1, img2, threshold_value):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(img1_gray, img2_gray)
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    _, diff_thresholded = cv2.threshold(blurred_diff, threshold_value, 255, cv2.THRESH_BINARY)
    diff_thresholded_color = cv2.cvtColor(diff_thresholded, cv2.COLOR_GRAY2RGB)

    return diff_thresholded_color

def compare_images():
    update_comparison()  # Replace the whole content with this line

def register_images(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width, _ = img2.shape
    registered_image = cv2.warpPerspective(img1, M, (width, height))

    return registered_image

def merge_images(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
    
    img1_green = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    img1_green[:, :, 0] = 0
    img1_green[:, :, 2] = 0
    
    img2_purple = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2RGB)
    img2_purple[:, :, 1] = 0
    
    merged_image = cv2.addWeighted(img1_green, 0.5, img2_purple, 0.5, 0)
    display_image(merged_image, image_canvas5)
    
    return merged_image



root = Tk()
root.title("Image Comparison")

first_image = None
second_image = None

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)

Label(root, text="First Image").grid(row=0, column=0, pady=5, padx=5, sticky="w")
Label(root, text="Second Image").grid(row=0, column=1, pady=5, padx=5, sticky="w")
Label(root, text="Registered Second Image").grid(row=0, column=2, pady=5, padx=5, sticky="w")
Label(root, text="Difference Image").grid(row=3, column=1, pady=5, padx=5, sticky="w")

Button(root, text="First Picture", command=open_first_image).grid(row=0, column=0, pady=5, padx=5, sticky="w")
image_canvas1 = Canvas(root, width=300, height=300)
image_canvas1.grid(row=1, column=0)

Button(root, text="Second Picture", command=open_second_image).grid(row=0, column=1, pady=5, padx=5, sticky="w")
image_canvas2 = Canvas(root, width=300, height=300)
image_canvas2.grid(row=1, column=1)

image_canvas4 = Canvas(root, width=300, height=300)
image_canvas4.grid(row=1, column=2)

threshold_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Threshold", command=lambda value: update_comparison())
threshold_slider.set(30)
threshold_slider.grid(row=2, column=0, columnspan=3, sticky="ew")

image_canvas3 = Canvas(root, width=300, height=300)
image_canvas3.grid(row=4, column=1)

Label(root, text="Merged Image").grid(row=3, column=2, pady=5, padx=5, sticky="w")
image_canvas5 = Canvas(root, width=300, height=300)
image_canvas5.grid(row=4, column=2)
Button(root, text="Preview", command=preview_merged_image).grid(row=5, column=2, pady=5, padx=5, sticky="w")


root.mainloop()
