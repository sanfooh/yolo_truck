#encoding=utf-8
import os
import time
# import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
import base64
from io import BytesIO

from PIL import Image, ImageDraw

try:
    import cStringIO as StringIO
except ImportError:
    from io import StringIO

import urllib.request  
import exifutil
import importlib
import sys
importlib.reload(sys)
#sys.setdefaultencoding('utf-8') # add this to support Chinese in python2

# import caffe
import darknet

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/objdet_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif', 'tif', 'tiff'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

# fyk
def load_img(img_buffer):
    # image = caffe.io.load_image(string_buffer)
    pass
def disp_wait_msg(imagesrc):
    flask.render_template(
        'index.html', has_result=True,
        result=(False, '处理图片中...'),
        imagesrc=imagesrc
    )


def draw_rectangle(draw, coordinates, color, width=1, draw_ellipse=False):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        if draw_ellipse:
            draw.ellipse((rect_start, rect_end), outline=color)
        else:
            draw.rectangle((rect_start, rect_end), outline=color)


def draw_rectangles(image_pil,det_result):
    # draw rectangles
    draw = ImageDraw.Draw(image_pil)
    for idx, item in enumerate(det_result):
        x, y, w, h = item[2]
        half_w = w / 2
        half_h = h / 2
        box = (int(x - half_w+1), int(y - half_h+1), int(x + half_w+1), int(y + half_h+1))
        # draw.rectangle(box, outline=(0, 255, 0))
        draw_rectangle(draw,box,(0, 255, 0),width=2,draw_ellipse=True)
        # draw.ellipse(box, outline=(255, 0, 0))
        draw.text((x - half_w + 5, y - half_h + 5), str(idx + 1)+" : "+str(item[0]), fill=(0, 0, 150))
    del draw


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        # download
        raw_data = urllib.request.urlopen(imageurl).read()
        print(raw_data)
        string_buffer =  BytesIO(raw_data)
        #string_buffer = StringIO(raw_data)
      
        # image = load_img(string_buffer)
        image_pil = Image.open(string_buffer)
    
        filename = os.path.join(UPLOAD_FOLDER, 'tmp.jpg')
        with open(filename,'wb') as f:
            f.write(raw_data)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    print('2222222')
    results = app.clf.classify_image(filename.encode())
    print('33333')
    draw_rectangles(image_pil, results[1])
    print('444')
    new_img_base64 = embed_image_html(image_pil)
    return flask.render_template(
        'index.html', has_result=True, result=results, imagesrc=new_img_base64)
        # 'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image_pil = exifutil.open_oriented_pil(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    print("1111111")
    results = app.clf.classify_image(filename.encode())
    draw_rectangles(image_pil, results[1])
    print('3333')
    new_img_base64 = embed_image_html(image_pil)

    return flask.render_template(
        'index.html', has_result=True, result=results,
        imagesrc=new_img_base64
    )


def embed_image_html(image_pil):
    """Creates an image embedded in HTML base64 format."""
    size = (512, 512) # (256, 256)
    resized = image_pil.resize(size)
    
    buffered = BytesIO()
    resized.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue())    
    return 'data:image/png;base64,' + data.decode()


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
         'model_def_file': (
             '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
         'pretrained_model_file': (
             '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
         'mean_file': (
             '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
         'class_labels_file': (
             '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
         'bet_file': (
             '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    #  预先加载模型
    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        base_dir = b"/darknet/work/"
        self.net = darknet.load_net(base_dir + b"cfg/YOLO-obj.cfg",  b"/data/YOLO-obj.backup", 0)
        self.meta = darknet.load_meta(base_dir + b"cfg/obj.data")

    def classify_image(self, image_filename):
        try:
            starttime = time.time()
            results = darknet.detect(self.net, self.meta, image_filename)
            endtime = time.time()
            bet_result = [(str(idx+1)+' : '+str(v[0]), '%.5f' % v[1])
                          for idx, v in enumerate(results)]
            rtn = (True, results, bet_result, '%.3f' % (endtime - starttime))
        
            return rtn
        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    #app.clf.net.forward()
    app.run(debug=True, host='0.0.0.0')
    #if opts.debug:
    #    app.run(debug=True, host='0.0.0.0')
    #else:
    #    start_tornado(app)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
