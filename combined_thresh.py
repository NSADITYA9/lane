import tempfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    img = np.uint8(img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        gray = img
    else:
        raise ValueError("Unsupported image format. Please provide a 3-channel RGB image or a grayscale image.")
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    else:
        raise ValueError("Invalid orientation. Use 'x' or 'y'.")
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def hls_thresh(img, thresh=(100, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def combined_thresh(img):
    img_array = np.array(img)
    if len(img_array.shape) < 3 or img_array.shape[2] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.uint8(img_array)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    abs_bin = abs_sobel_thresh(gray, orient='x', thresh_min=50, thresh_max=255)
    mag_bin = mag_thresh(gray, sobel_kernel=3, mag_thresh=(50, 255))
    dir_bin = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_bin = hls_thresh(img_array, thresh=(170, 255))
    mag_bin_resized = cv2.resize(mag_bin, (abs_bin.shape[1], abs_bin.shape[0]))
    dir_bin_resized = cv2.resize(dir_bin, (abs_bin.shape[1], abs_bin.shape[0]))
    hls_bin_resized = cv2.resize(hls_bin, (abs_bin.shape[1], abs_bin.shape[0]))
    combined = np.zeros_like(abs_bin)
    combined[np.logical_or(np.logical_or(abs_bin == 1, (mag_bin_resized == 1) & (dir_bin_resized == 1)), hls_bin_resized == 1)] = 1
    return combined, abs_bin, mag_bin_resized, dir_bin_resized, hls_bin_resized

if __name__ == '__main__':
    img_file = 'test_images/straight_lines1.jpg'
    img_file = 'test_images/test5.jpg'
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    combined, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
    plt.subplot(2, 3, 1)
    plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 2)
    plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 5)
    plt.imshow(img)
    plt.subplot(2, 3, 6)
    plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()

    video_file = 'path_to_your_video.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    video = VideoFileClip(temp_video_path)
    # Proceed with processing the video
