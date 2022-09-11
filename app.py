from flask import Flask, request, jsonify
import os
import torch

app = Flask(__name__)

model = torch.load('./ORGINAL_MODEL.pt', map_location='cpu')

@app.route('/')
def hello():
    print(model)
    return "모델 생성"

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)