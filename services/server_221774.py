""" Module docstring """

from flask import Flask

app = Flask('Image classifier')


@app.route('/')
def home():
    """ Function docstring """
    return 'Home page'


if __name__ == '__main__':
    app.run(port=1774)              # номер зачетки
    input()
