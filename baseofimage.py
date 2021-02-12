# author: ZHANG wentong
# date: 2017.05.08
# email: wentong.zhang@groupe-esigelec.org
# code for building the data base of our model
# we prepare 5 photos of everyone, change them and apply LBP and face detection

import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy
import numpy as np
from numpy import *
from random import randint

import argparse
import os
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
# import the photo of someone and change randomly for the base of images.
# here i take 5 photos for everyone, change them into 100 photos for each photo
# so totally 500 photos for everyone, totally 2000 photos as the input of CNN

facedir = './P2'

image_paths = []
if os.path.isdir(facedir):
    images = os.listdir(facedir)
    image_paths = [os.path.join(facedir,img) for img in images]



def resize(image, size):
    #size = check_size(size)
    image = imresize(image, size)
    return image

def horizontal_flip(image, rate=0.5):
    #if np.random.rand() < rate:
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    #if np.random.rand() < rate:
    image = image[::-1, :, :]
    return image

def random_rotation(image, angle_range=(0, 270)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image

def larger(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def smaller(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 100)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

'''def lighter(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = dst.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] <= 255 - num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] + num)
                else:
                    dst[xj, xi, i] = 255
    return dst

def darker(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] >= num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] - num)
                else:
                    dst[xj, xi, i] = 0
    return dst'''

def moveright(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def moveleft(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,-num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movetop(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,-num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movebot(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def turnright(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def turnleft(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def changeandsave(name,time,choice,img,i):
    # the new name of changed picture is "changed?.png",? means it's the ?-th picture changed
    name = './DIV2K_aug_HR/'  +str(time) + str(i) + '.jpg'
    # do different changes by the choice
    if choice == 1:
        newimg = horizontal_flip(img)
    elif choice == 5:
        newimg = larger(img)
    elif choice == 2:
        newimg = vertical_flip(img)
    elif choice == 4:
        newimg = random_rotation(img)
    elif choice == 3:
        newimg = horizontal_flip(vertical_flip(img)) 
    # save the new picture
    cv2.imwrite(name, newimg)


# take fu's 5 photos, change each photo into 100 photos, so totally 500
k=0
for j in image_paths:
    print(j)
    img = cv2.imread(j,1)
    # for cycle to make change randomly 100 times
    # (1,n), n for n-1 photos, (1,10), after change, 9 photos
    k+=1
    choice = 1
    for i in range(1,4):
      # take a random number as the choice
        changeandsave('fu',k,choice,img,i)
        choice += 1
        if choice > 3:
            choice =1
    choice = 4
    for i in range(1,7):
      # take a random number as the choice
        changeandsave('fu',k,choice,img,i+4)
        choice += 1
        if choice > 5:
            choice =4

