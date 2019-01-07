#-*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import shutil
import base64
import ocr
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from airtest.aircv.utils import string_2_img


app = Flask(__name__)
result_dir = './test_result'


def prepare():
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # run one smoke test
    image_file = "demo/demo_detect.jpg"
    image = cv2.imread(image_file)
    res = detect(image)
    print(res)


def detect(image):
    t = time.time()
    result, image_framed = ocr.model(image)

    # # save output file for debug
    # output_file = os.path.join(result_dir, os.path.split(image_file)[-1])
    # Image.fromarray(image_framed).save(output_file)

    # print debug message
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print("Recognition Result:")
    for rect, text in result:
        print(text)
    return result


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/ocr", methods=['POST'])
def api_ocr():
    img = request.form["image"]
    img = base64.b64decode(img)
    image = string_2_img(img)
    res = detect(image)
    return jsonify(res)


if __name__ == "__main__":
    prepare()
    app.run("0.0.0.0")

