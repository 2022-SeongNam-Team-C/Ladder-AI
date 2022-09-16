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
import requests
from io import BytesIO
from PIL import Image
# __________
import urllib.request
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model
from flask import Flask, request, send_file, Blueprint
from werkzeug.utils import secure_filename

from datetime import datetime

bp = Blueprint('model', __name__, url_prefix='/api/v1')

def make_photo(img_url):
    img = wget.download(img_url)   # image name
    model = create_model("Unet_2020-07-20")
    model.eval()
    image = load_rgb(img)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    cv2.imwrite('user1.jpg', image)    # insert name randomly
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]
    
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    mask1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (255, 255, 255)

    cv2.imwrite('back1.jpg', mask)
    dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (255, 0, 0)).astype(np.uint8), 0.2, 0)

    remove = mask1 + dst
    # print(remove.shape)  ###630, 1200, 3 ->255가 아닌 값들은 전부다 255로 바꿔버리고 255가 맞는것들은 0으로
    # plt.imshow(remove)

    cv2.imwrite('nox.jpg', remove)

    pix10 = np.array(remove)
    # print(pix10[300][500])

    re = 255 - mask1
    # plt.imshow(re)

    a = image + re
    # plt.imshow(a)

    cv2.imwrite('rs.jpg', a)
    cv2.imwrite('Me1.jpg', dst)

    img = cv2.imread('rs.jpg', 0)
    edges = cv2.Canny(img, 100, 200, 3)  ### 엣지 이미지 하나랑!

    img = cv2.imread("rs.jpg")  # Read image
    t_lower = 100  # Lower Threshold
    t_upper = 200  # Upper threshold
    aperture_size = 5  # Aperture size
    L2Gradient = True  # Boolean
    # Applying the Canny Edge filter with L2Gradient = True
    edge = cv2.Canny(img, t_lower, t_upper, L2gradient=L2Gradient)

    # plt.imshow(edge)

    mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) * (255, 255, 255)

    # print(edges.shape)

    ###plt.imshow(edges)

    edges10 = 255 - edge
    # plt.imshow(edges10)

    # print(edges10)

    cv2.imwrite('edge.jpg', edges)
    cv2.imwrite('e1.jpg', edges10)
    # plt.show()
    cv2.imwrite('M.jpg', edges10)


    background = cv2.imread('rs.jpg')
    overlay = cv2.imread('M.jpg')
    added_image = cv2.addWeighted(background, 0.3, overlay, 0.69, 0.1)

    cv2.imwrite('combined.jpg', added_image)

    plt.imshow(added_image)

    img = cv2.imread('rs.jpg', cv2.IMREAD_COLOR)
    cartoon_img = cv2.stylization(img, sigma_s=100, sigma_r=0.9)

    plt.imshow(cartoon_img)
    cv2.imwrite('kkkk.jpg', cartoon_img)

    c = cartoon_img + re

    plt.imshow(c)
    cv2.imwrite('peopleo.jpg', c)

    print(c.shape)  ### (700, 1050, 3)  [196 122 86]
    print(remove.shape)  ###(700, 1050, 3) [510, 366, 325]
    pix1 = np.array(c)
    pix2 = np.array(remove)

    print(pix1[300][500])
    print(pix2[300][500])

    background = cv2.imread('kkkk.jpg')
    overlay = cv2.imread('nox.jpg')

    ll = background + overlay


    add = cv2.addWeighted(background, 0.39, overlay, 0.56, 0.52)
    fix_img = cv2.cvtColor(add, cv2.COLOR_BGR2RGB)
    # plt.imshow(fix_img)
    days = datetime.today()
    file_name = days.strftime('%Y-%m-%d-%H-%M-%S') + '-user_id.jpg'
    print(file_name)
    cv2.imwrite(file_name, fix_img)   
    return send_file(file_name, mimetype='image/')

   
