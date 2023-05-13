import io
import unittest
import PIL.Image
import requests
from requests import request


class MyTestCase(unittest.TestCase):
    def test_home(self):
        response = requests.request('GET', 'http://localhost:1234/')
        sample = response.content.decode()
        self.assertEqual(sample, 'Home page')  # add assertion here

    def test_classify(self):
        img = PIL.Image.open('../data/dog.jpg')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')

        with buffer as buf:
            buffer.seek(0)
            response = request('POST', 'http://localhost:1234/classify', data=buf)

        out = response.content.decode('utf-8')
        expected = 'келпи, Пембрук, Немецкая овчарка'

        self.assertEqual(out, expected)


if __name__ == '__main__':
    unittest.main()
