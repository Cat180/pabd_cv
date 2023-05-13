from flask import Flask, request
import tensorflow as tf

app = Flask('Image classifier')
resnet = tf.keras.applications.ResNet101()
with open('../data/imgnet_cats_en.txt', encoding='utf-8') as f:
    cats = f.readlines()

categories = [s.rstrip() for s in cats]

@app.route('/')
def home():
    return "Home page"

@app.route('/classify', methods=['POST', 'GET'])
def classify():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out, direction='DESCENDING')[0][:3].numpy()
    out = ', '.join([categories[int(i)] for i in idxs])
    return out


if __name__ == '__main__':
    app.run(port=1779)         # Your personal 4 digits
    input()