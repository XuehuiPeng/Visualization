import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def THRESH(rgb_img):
    # 阈值分割
    gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_otsu(gray_image)
    dst = (rgb_img <= thresh) * 0.991
    plt.figure(figsize=(16, 16), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(dst, plt.cm.gray)
    plt.savefig("./thresh.png")
    thresh_img = cv2.imread("thresh.png", 1)[:, :, ::-1]
    thresh_img = cv2.resize(thresh_img, dsize=(224, 224))
    thresh_img = np.float32(thresh_img) / 255
    mask = thresh_img[:, :, 0]
    # 闭操作
    kernel1 = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel1)  # 膨胀
    kernel2 = np.ones((11, 11), np.uint8)
    mask = cv2.erode(mask, kernel2)  # 腐蚀

    # 开操作
    kernel1 = np.ones((11, 11), np.uint8)
    mask = cv2.erode(mask, kernel1)  # 腐蚀
    kernel2 = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel2)  # 膨胀
    # 边缘平滑
    # grayscale_cam = cv2.blur(grayscale_cam, (2, 2))  #均值滤波
    # grayscale_cam = cv2.boxFilter(grayscale_cam,-1,(2,2))  #方框滤波
    # grayscale_cam = cv2.GaussianBlur(grayscale_cam,(3,3),0,0)  #高斯滤波
    # grayscale_cam = cv2.medianBlur(grayscale_cam,5)  #中值滤波
    # grayscale_cam = cv2.bilateralFilter(grayscale_cam,2,100,10)  #双边滤波

    # kernel = np.ones((9,9),np.float32)/81
    # grayscale_cam = cv2.filter2D(grayscale_cam , -1, kernel)  #2D
    return mask

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      grayscale_cam: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    #p = 0.3
    #cam = heatmap * p + img * (1 - p)
    # 像素点直接替换
    for i in range(0, 224):
        for j in range(0, 224):
            for k in range(0, 3):
                if mask[i, j] == 0:
                    cam[i, j, k] = img[i, j, k]
    #溢出置1
    #for i in range(0,224):
    #    for j in range(0,224):
    #        for k in range(0,3):
    #            if cam[i,j,k]>1:
    #                cam[i,j,k] = 1


    return np.uint8(255 * cam)