import numpy as np
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def read_image():
    image = []
    for filename in os.listdir('imagesE'):
        img = io.imread(os.path.join('imagesE', filename))
        if len(img.shape) < 3:
            image.append(img)

        elif len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            image.append(img_gray)

    return image


def hist_equal(images):

    list_equal = []
    for img in images:
        hist, bin = np.histogram(img.ravel(), 256, [0, 256])
        # print(hist/np.sum(hist))
        # value = [0]*256
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         value[img[i,j]] +=1
        # print('value',value)

        cdf = [0] * len(hist)  # cumulative distribution frequency
        cdf[0] = hist[0]
        for i in range(1, len(hist)):
            cdf[i] = cdf[i - 1] + hist[i]  # cumsum

        # mapping histogram (probability of each bin) and normalize cdf --> cdf/max(cdf)
        cdf_norm = [ah * 255 / max(cdf) for ah in cdf]

        # mapping original image and equalize image
        img_equalize = np.interp(img, range(0, 256), cdf_norm)
        img_equalize = img_equalize.astype(int)

        # img_equ = np.hstack((img, img_equalize))

        list_equal.append(img_equalize)

    plot_image(images,list_equal)


def plot_image(main_img,equal_img):
    fig = plt.figure(figsize=(10,10))  # size for imageE folder
    # fig = plt.figure(figsize=(16,30))    # size for imageT folder
    spec = gridspec.GridSpec(ncols=4, nrows=len(main_img), figure=fig )

    for i in range(len(main_img)):
        ax1 = fig.add_subplot(spec[i, 0] )
        ax2 = fig.add_subplot(spec[i, 1])
        ax3 = fig.add_subplot(spec[i, 2])
        ax4 = fig.add_subplot(spec[i, 3])

        ax1.set_title('Original image',fontsize=10)
        ax2.set_title('Equalize image',fontsize=10)
        ax3.set_title('Original histogram',fontsize=10)
        ax4.set_title('Equalize Histogram',fontsize=10)

        ax1.imshow(main_img[i],cmap='gray')
        ax2.imshow(equal_img[i] ,cmap='gray')
        ax3.hist(main_img[i].ravel(),256,[0,256],color='skyblue')
        ax4.hist(equal_img[i].ravel(),256,[0,256] ,color='red')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('histogram equalization.jpg')
    plt.show()


    # opencv
    equ =cv2.equalizeHist(main_img[0])
    plt.hist(main_img[0].ravel(),255,[0,255])
    plt.hist(equ.ravel(),255,[0,255])
    plt.figure(figsize=(15, 6))
    fig ,(ax1 ,ax2) = plt.subplots(1,2)
    ax1.set_title('Original image')
    ax1.imshow(main_img[0], cmap='gray')
    ax2.set_title('OpenCv image')
    ax2.imshow(equ, cmap='gray')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()



image = read_image()
hist_equal(image)