# -*- encoding:utf-8 -*-
# @Time    : 2019/10/7 22:12
# @Author  : gfjiang
# @Site    : 
# @File    : deploy_model.py
# @Software: PyCharm
import flask
import io
import cv2.cv2 as cv
from PIL import Image
import numpy as np


app = flask.Flask(__name__)
model = None


# Just as an interface, you have to implement specific model code
class Model(object):

    def __init__(self, config_file, pth_file):
        self.config_file = config_file
        self.pth_file = pth_file
        self.model = None

    def detect(self, img):
        return img


def load_model():
    """Load the hat model.
    """
    global model
    # config_file = '../configs/hat/hatv2_cascade_rcnn_x101_32x4d_fpn_1x.py'
    # pth_file = 'work_dirs/hatv2_cascade_rcnn_x101_32x4d_fpn_1x/trainval/epoch_12.pth'
    config_file = '../configs/hat/hatv1_cascade_rcnn_r50_fpn_1x.py'
    pth_file = '../data/hatv1_cascade_rcnn_r50_fpn_1x/hat/epoch_12.pth'
    model = Model(config_file, pth_file)


@app.route("/detect", methods=["POST"])
def detect():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)

            results = model.detect(img)

            data['results'] = results

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)

