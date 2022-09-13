from flask import Flask, request, send_file
import os
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = torch.load('./ORGINAL_MODEL.pt', map_location='cpu')

@app.route('/api/v1/images/result', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        print(model)
        return "모델 생성"
    if request.method == 'POST':
        file = request.files['img']
        fileName = file.filename
        file.save(secure_filename(file.filename))
        return send_file(fileName, mimetype='image/jpg')
        
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)