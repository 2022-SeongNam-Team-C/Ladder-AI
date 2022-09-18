from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
import wget
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
##______
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
##______
import requests
from io import BytesIO
from PIL import Image
#__________
import urllib.request
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model
from datetime import datetime
from flask import send_file, send_from_directory
import torchvision.models as torch_model

# path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'weight.pth')
# weight = torch.load('./weight.pth', map_location='cpu')
weight = torch.load('./weight.pth')

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

def resize_mincrop(image):
    h, w, c = np.shape(image)
    if min(h, w) < 720:
        if h < w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

def resize_mask(image):
    h, w = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w]
    return image

def resize_minmask(image):
    h, w = np.shape(image)
    if min(h, w) < 720:
        if h < w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w]
    return image

def resize_select(image):
    h, w, c = np.shape(image)
    if min(h, w, c) > 720:
        return resize_crop(image)
    else:  
        return resize_mincrop(image)

def resize_select_mask(mask):
    h, w = np.shape(mask)
    np.shape(mask)
    if min(h, w) < 720:
        return resize_minmask(mask)
    else: 
         return resize_mask(mask)


#######################################################################################


def make_photo(img_url):
    img = wget.download(img_url)    # wget : image file name return
    model = create_model("Unet_2020-07-20")
    model.eval()
    image_file = load_rgb(img)    # load_rgb : image file return
    image_file = resize_select(image_file)   # return resize image file
    cv2.imwrite('pop.jpg', image_file)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image_file, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    mask10 = resize_minmask(mask)

###_____________________________________________________ ____________________________________________________
    class ResBlock(nn.Module):
        def __init__(self, num_channel):
            super(ResBlock, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, 3, 1, 1),
                nn.BatchNorm2d(num_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channel, num_channel, 3, 1, 1),
                nn.BatchNorm2d(num_channel))
            self.activation = nn.ReLU(inplace=True)

        def forward(self, inputs):
            output = self.conv_layer(inputs)
            output = self.activation(output + inputs)
            return output


    class DownBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(DownBlock, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))


        def forward(self, inputs):
            output = self.conv_layer(inputs)
            return output


    class UpBlock(nn.Module):
        def __init__(self, in_channel, out_channel, is_last=False):
            super(UpBlock, self).__init__()
            self.is_last = is_last
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, 1, 1))
            self.act = nn.Sequential(
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))
            self.last_act = nn.Tanh()


        def forward(self, inputs):
            output = self.conv_layer(inputs)
            if self.is_last:
                output = self.last_act(output)
            else:
                output = self.act(output)
            return output



    class SimpleGenerator(nn.Module):
        def __init__(self, num_channel=32, num_blocks=4):
            super(SimpleGenerator, self).__init__()
            self.down1 = DownBlock(3, num_channel)
            self.down2 = DownBlock(num_channel, num_channel*2)
            self.down3 = DownBlock(num_channel*2, num_channel*3)
            self.down4 = DownBlock(num_channel*3, num_channel*4)
            res_blocks = [ResBlock(num_channel*4)]*num_blocks
            self.res_blocks = nn.Sequential(*res_blocks)
            self.up1 = UpBlock(num_channel*4, num_channel*3)
            self.up2 = UpBlock(num_channel*3, num_channel*2)
            self.up3 = UpBlock(num_channel*2, num_channel)
            self.up4 = UpBlock(num_channel, 3, is_last=True)

        def forward(self, inputs):
            down1 = self.down1(inputs)
            down2 = self.down2(down1)
            down3 = self.down3(down2)
            down4 = self.down4(down3)
            down4 = self.res_blocks(down4)
            up1 = self.up1(down4)
            up2 = self.up2(up1+down3)
            up3 = self.up3(up2+down2)
            up4 = self.up4(up3+down1)
            return up4

    model = SimpleGenerator()
    model.load_state_dict(weight)
    #torch.save(model.state_dict(), 'weight.pth')
    model.eval()
    ##여기까지는 모델입니다.  
    raw_image = cv2.imread('pop.jpg')
        
    image = raw_image/127.5 - 1
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)
    output = model(image.float())
    output = output.squeeze(0).detach().numpy()
    output = output.transpose(1, 2, 0)
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imwrite('combined.jpg', output)

##______________________________________________________________________________________________________________

    img = np.array(Image.open('combined.jpg'))

    fg_h, fg_w, _ = img.shape

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    background = np.array(Image.open('pop.jpg'))
    bg_h, bg_w, _ = background.shape
    background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))
    bg_h, bg_w, _ = background.shape
    margin = (bg_h - fg_h) // 2

    if margin > 0:
        background = background[margin:-margin, :, :]
    else:
        background = cv2.copyMakeBorder(background, top=abs(margin), bottom=abs(margin), left=0, right=0, borderType=cv2.BORDER_REPLICATE)

    background = cv2.resize(background, dsize=(fg_w, fg_h))
    plt.figure(figsize=(12, 8))

    _, alpha = cv2.threshold(mask10, 0, 255, cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0).astype(float)

    alpha = alpha / 255. # (height, width)
    alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2) # (height, width, 3)

    foreground = cv2.multiply(alpha, img.astype(float))
    background = cv2.multiply(1. - alpha, background.astype(float))  

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    result = cv2.add(foreground, background).astype(np.uint8)

    plt.figure(figsize=(12, 12))
    # fix_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)    # result image
    # plt.imshow(fix_img)
    days = datetime.today()
    file_name = days.strftime('%Y-%m-%d-%H-%M-%S') + '-user_id.jpg'
    print(file_name)
    cv2.imwrite(file_name, result)   
    return send_file(file_name, mimetype='image/')
