#-*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import json
import shutil
import ocr
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
image_files = glob('./test_images/*.*')


if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = cv2.imread(image_file)
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, os.path.split(image_file)[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")

        for rect, text in result:
            print(text)

        print(json.dumps(result))
