import cv2
import numpy as np
import SimpleITK as sitk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

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
        if registration_method_var.get() == "SIFT":
            registered_second_image = register_images_sift(second_image, first_image)
        else:
            registered_second_image = register_images_simpleitk(first_image, second_image)
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
    update_comparison() 

def register_images_sift(img1, img2):
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


#def register_images_simpleitk(img1, img2):
#    img1_sitk = sitk.GetImageFromArray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
#    img2_sitk = sitk.GetImageFromArray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
#
#    registration_method = sitk.ImageRegistrationMethod()
#
#    registration_method.SetMetricAsMeanSquares()
#    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#    registration_method.SetOptimizerScalesFromPhysicalShift()
#
#    final_transform = sitk.Euler2DTransform()
#    registration_method.SetInitialTransform(final_transform)
#
#    registration_method.SetInterpolator(sitk.sitkLinear)
#
#    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
#    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
#    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
#
#    final_transform = registration_method.Execute(sitk.Cast(img1_sitk, sitk.sitkFloat32), sitk.Cast(img2_sitk, sitk.sitkFloat32))
#
#    resampler = sitk.ResampleImageFilter()
#    resampler.SetReferenceImage(img1_sitk)
#    resampler.SetInterpolator(sitk.sitkLinear)
#    resampler.SetTransform(final_transform)
#
#    registered_img2_sitk = resampler.Execute(img2_sitk)
#    registered_img2 = cv2.cvtColor(sitk.GetArrayFromImage(registered_img2_sitk), cv2.COLOR_GRAY2RGB)
#
#    return registered_img2

def register_images_simpleitk(img1, img2, transform_type="affine"):
   
    if transform_type == "affine":
        img1_sitk = sitk.GetImageFromArray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        img2_sitk = sitk.GetImageFromArray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
    
        registration_method = sitk.ImageRegistrationMethod()
    
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
    
        final_transform = sitk.AffineTransform(2)
        registration_method.SetInitialTransform(final_transform, inPlace=False)
    
        registration_method.SetInterpolator(sitk.sitkLinear)
    
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
        final_transform = registration_method.Execute(sitk.Cast(img1_sitk, sitk.sitkFloat32), sitk.Cast(img2_sitk, sitk.sitkFloat32))
    
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img1_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
    
        registered_img2_sitk = resampler.Execute(img2_sitk)
        registered_img2 = cv2.cvtColor(sitk.GetArrayFromImage(registered_img2_sitk), cv2.COLOR_GRAY2RGB)
    
        return registered_img2
    else:  # transform_type == "bspline"        
        img1_sitk = sitk.GetImageFromArray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        img2_sitk = sitk.GetImageFromArray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))

        registration_method = sitk.ImageRegistrationMethod()

        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        grid_physical_spacing = [50.0, 50.0]
        transform_domain_mesh_size = [4, 4]

        initial_transform = sitk.BSplineTransformInitializer(img1_sitk, transform_domain_mesh_size, order=3)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        final_transform = registration_method.Execute(sitk.Cast(img1_sitk, sitk.sitkFloat32), sitk.Cast(img2_sitk, sitk.sitkFloat32))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img1_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)

        registered_img2_sitk = resampler.Execute(img2_sitk)
        registered_img2 = cv2.cvtColor(sitk.GetArrayFromImage(registered_img2_sitk), cv2.COLOR_GRAY2RGB)

        return registered_img2


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

def register_images_wrapper(event=None):
    global registered_second_image
    if first_image is not None and second_image is not None:
        method = registration_method_var.get()
        if method == "SIFT":
            registered_second_image = register_images_sift(second_image, first_image)
        elif method == "Affine":
            registered_second_image = register_images_simpleitk(first_image, second_image, "affine")
        else:  # method == "B-spline"
            registered_second_image = register_images_simpleitk(first_image, second_image, "bspline")
        display_image(registered_second_image, image_canvas4)
        update_comparison()
        merged_image = merge_images(first_image, registered_second_image)
        display_image(merged_image, image_canvas5)


root = Tk()
root.title("Image Comparison")

first_image = None
second_image = None

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)

Label(root, text="Registration Method").grid(row=0, column=2, pady=5, padx=5, sticky="w")
registration_method_var = StringVar()
registration_method_var.set("SIFT")
registration_method_menu = ttk.Combobox(root, textvariable=registration_method_var, values=["SIFT", "Affine", "B-spline"], state='readonly', width=10)
registration_method_menu.grid(row=0, column=2, pady=5, padx=5, sticky="e")
registration_method_menu.bind("<<ComboboxSelected>>", register_images_wrapper)

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
