from flask import Flask, request
import torch
import torchvision
from torchvision.models import resnet101,  ResNet101_Weights


app = Flask('Image classifier')


@app.route('/')
def home():
    return 'Home page'


if __name__ == '__main__':
    app.run(port=1675)
    input()
