from tqdm import tqdm
import net as nn
import torch
import pydicom
import pylibjpeg
from PIL import Image as im
from PIL import ImageDraw
import numpy as np
import cv2
from ipywidgets import *
import xnat

net = nn.Net()
net.load_weights('trained_data.npy')

def input_image(file):
    bboxes = []

    ds = pydicom.dcmread(file)
    ds.PhotometricInterprretation = 'YBR_FULL'
    image = ds.pixel_array

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image / np.amax(image)

    image = cv2.resize(image, (image.shape[0] // 3, image.shape[1] // 3))

    pil_im = im.fromarray(cv2.merge([image * 255, image * 255, image * 255]).astype(np.uint8))
    im_draw = ImageDraw.Draw(pil_im)

    print(image.shape)

    for x in tqdm(range(0, image.shape[0], 32)):
        for y in range(0, image.shape[1], 16):
            input_image = image[y: y + 16, x: x + 32]
            input_image = cv2.resize(input_image, (128, 64))
            input_image = torch.Tensor(input_image).to(nn.device).view(-1, 1, 128, 64)
            net_out = net(input_image)

            if torch.argmax(net_out) == 0:
                bboxes.append([x, y])

    corrected_bboxes = []

    for box in bboxes:
        if [box[0] + 32, box[1]] in bboxes or [box[0] + 32, box[1] + 16] in bboxes or [box[0], box[1] + 16] in bboxes or [box[0] - 32, box[1]] in bboxes or [box[0] - 32, box[1] - 16] in bboxes or [box[0], box[1] - 16] in bboxes or [box[0] - 32, box[1] + 16] in bboxes or [box[0] + 32, box[1] - 16]:
            corrected_bboxes.append(box)
            im_draw.rectangle((box[0], box[1], box[0] + 32, box[1] + 16), outline = 'red')

    pil_im.save('TempImages/Image.jpg')

    for box in bboxes:
        im_draw.rectangle((box[0], box[1], box[0] + 32, box[1] + 16), outline = 'red')
    pil_im.save('TempImages/Uncorrected.jpg')

    return corrected_bboxes

input_image('TempImages/0.dcm')
