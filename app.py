from flask import Flask, request, jsonify
import os
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = torch.load('./modeling.pth', map_location='cpu')

@app.route('/')
def hello():
    return jsonify(model)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)