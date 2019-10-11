# -*- encoding:utf-8 -*-
# @Time    : 2019/10/7 22:42
# @Author  : gfjiang
# @Site    : 
# @File    : test_restful_api.py
# @Software: PyCharm
import requests


PyTorch_REST_API_URL = 'http://10.193.0.20:666/detect_hat'


def detect_hat(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        print(r['results'])
    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':
    detect_hat(r'D:\data\hat_detect\SHWD\VOC2028\JPEGImages\000332.jpg')
