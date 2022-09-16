from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
import wget
import os
import time
import matplotlib.pyplot as plt
##______
##______
import requests
from io import BytesIO
from PIL import Image
import warnings
#__________
import urllib.request
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

from datetime import datetime
from flask import Flask, request, send_file


def make_photo(img_url):
    img = wget.download(img_url)
    model = create_model("Unet_2020-07-20")
    model.eval()
    image = load_rgb(img)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    cv2.imwrite('pop.jpg', image)
    
    warnings.filterwarnings('ignore')
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    cartoon_img = cv2.stylization(image, sigma_s=100, sigma_r=0.85)  
    cv2.imwrite('combined.jpg', cartoon_img)
    img = np.array(Image.open('combined.jpg'))

    fg_h, fg_w, _ = img.shape

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(cartoon_img)
    axes[1].imshow(img)

    background = np.array(Image.open('pop.jpg'))

    bg_h, bg_w, _ = background.shape

    # fit to fg width
    background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))

    bg_h, bg_w, _ = background.shape

    margin = (bg_h - fg_h) // 2

    if margin > 0:
        background = background[margin:-margin, :, :]
    else:
        background = cv2.copyMakeBorder(background, top=abs(margin), bottom=abs(margin), left=0, right=0, borderType=cv2.BORDER_REPLICATE)

    # final resize
    background = cv2.resize(background, dsize=(fg_w, fg_h))

    plt.figure(figsize=(12, 8))
    plt.imshow(background)




    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0).astype(float)

    alpha = alpha / 255. # (height, width)
    alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2) # (height, width, 3)

    foreground = cv2.multiply(alpha, img.astype(float))
    background = cv2.multiply(1. - alpha, background.astype(float))  

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(mask)
    axes[1].imshow(foreground.astype(np.uint8))
    axes[2].imshow(background.astype(np.uint8))
    result = cv2.add(foreground, background).astype(np.uint8)

    plt.figure(figsize=(12, 12))
    # fix_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    days = datetime.today()
    file_name = days.strftime('%Y-%m-%d-%H-%M-%S') + '-user_id.jpg'
    print(file_name)
    cv2.imwrite(file_name, result)   
    return send_file(file_name, mimetype='image/')




   
