""" Module docstring """
from flask import Flask, request
import tensorflow as tf


app = Flask('Image classifier')
resnet = tf.keras.applications.ResNet101()
with open('../data/imgnet_cats_ru.txt', encoding='utf-8') as f:
    cats = f.readlines()

cats_ru = [s.rstrip() for s in cats]


@app.route('/')
def home():
    """ Function docstring """
    return 'Home page'


@app.route('/classify', methods=['POST'])
def classify():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out, direction='DESCENDING')[0][:3].numpy()
    out = ', '.join([cats_ru[int(i)] for i in idxs])
    return out


if __name__ == '__main__':
    app.run(port=1234)
    input()
