from flask import Flask, request
from api.model_module import make_photo
from datetime import datetime
import base64
import json


app = Flask(__name__) 
@app.route('/api/v1/images/result', methods=['POST'])
def hello():
    params = request.get_json() ## 이미지 url 받아오기
    print(params['img'])
    return make_photo(params['img'])
    
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
