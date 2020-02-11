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
import cvtools
import os
import os.path as osp
from threading import Thread
from werkzeug.utils import secure_filename
import random


UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'}

model = None
log_save_root = None
logger = cvtools.get_logger('INFO', name='deploy_model')
PORT = 5000
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_image(filename, img, results=None):
    img_name = cvtools.get_time_str(form='%Y%m%d_%H%M%S_%f') + '.jpg'
    if filename:
        filename = filename.replace('\\', '/')
        if cvtools.is_image_file(filename):
            img_name = osp.basename(filename)
    img_file = osp.join(log_save_root, 'images', img_name)
    try:
        logger.info("filename: {}, saving image in {}".format(
            filename, img_file))
        if results:
            logger.info("detect results: {}".format(results))
        cvtools.imwrite(img, img_file)
    except Exception as e:
        logger.error(e)


@app.route("/detect", methods=["POST"])
def detect():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            try:
                # Read the image in PIL format
                img = flask.request.files.get("image").read()
                img = Image.open(io.BytesIO(img))
                img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

                results = model.detect(img)
                # 留给用户自己解析
                if hasattr(model, "prase_results"):
                    data['results'] = model.prase_results(results)
                else:
                    data[results] = results
                # data['results'] = list()
                # for cls_index, dets in enumerate(results):
                #     cls = model.CLASSES[cls_index]
                #     for det in dets:
                #         bbox = det[:4].astype(np.int).tolist()
                #         score = round(float(det[4]), 2)
                #         result = {'label': cls, 'bbox': bbox, 'score': score}
                #         data['results'].append(result)
                # Indicate that the request was a success.
                data["success"] = True

                # save image
                filename = flask.request.form.get("filename")
                t = Thread(target=save_image, args=[filename, img, data])
                t.start()

            except Exception as e:
                logger.info(e)
                print(e)
                data["error"] = e

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


@app.route('/')
def upload_test():
    # TODO: URL由参数注入
    return flask.render_template('upload.html', port=PORT)


class PicStr:
    # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
    def create_uuid(self):
        # 生成当前时间
        now_time = cvtools.get_time_str("%Y%m%d%H%M%S")
        random_num = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
        if random_num <= 10:
            random_num = str(0) + str(random_num)
        return now_time + str(random_num)


# 上传图片
@app.route('/up_image', methods=['POST'], strict_slashes=False)
def upload_image():
    file_dir = osp.join(log_save_root, app.config['UPLOAD_FOLDER'])
    if not osp.exists(file_dir):
        os.makedirs(file_dir)
    f = flask.request.files['image']
    if f and cvtools.is_image_file(f.filename):
        # Read the image in PIL format
        image = f.read()
        image = Image.open(io.BytesIO(image))
        img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)

        try:
            results = model.detect(img)
            # boxes, textes = [], []
            # for cls_index, dets in enumerate(results):
            #     cls = model.model.CLASSES[cls_index]
            #     for det in dets:
            #         bbox = det[:4].astype(np.int).tolist()
            #         score = round(float(det[4]), 2)
            #         boxes.append(bbox)
            #         textes.append(cls+'|'+str(score))
            # img = cvtools.draw_boxes_texts(img, boxes, textes)
            model.draw(img, results)

        except Exception as e:
            logger.error(e)
            print(e)

        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[1]
        new_filename = PicStr().create_uuid() + '.' + ext
        img_file = osp.join(log_save_root, 'images', new_filename)
        cvtools.imwrite(img, img_file)
        # t = Thread(target=save_image, args=[img_file, img])
        # t.start()
        f.save(os.path.join(file_dir, new_filename))
        # t = Thread(target=f.save, args=[os.path.join(file_dir, new_filename)])
        # t.start()

        # image_data = open(img_file, "rb").read()
        ret, buf = cv.imencode(".jpg", img)
        img_bin = Image.fromarray(np.uint8(buf)).tobytes()
        response = flask.make_response(img_bin)
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        return flask.jsonify({"error": 1001, "msg": "上传失败"})


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(log_save_root, app.config['UPLOAD_FOLDER'])
    if flask.request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(osp.join(file_dir, '%s' % filename), "rb").read()
            response = flask.make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        logger.info("show接口无post方法！")


def run_model(model_obj, port=5000, logfile=None):
    global model
    model = model_obj
    global PORT
    PORT = port
    if logfile is not None:
        logfile_split = cvtools.splitpath(logfile)
        global log_save_root
        log_save_root = logfile_split[0]
        global logger
        cvtools.makedirs(logfile)
        logger = cvtools.logger_file_handler(logger, logfile, mode='w')
    app.run(host='0.0.0.0', port=port)
