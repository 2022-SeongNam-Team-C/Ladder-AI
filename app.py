from flask import Flask, request
from api.model_module import make_photo
from datetime import datetime
import base64
import json
from flask_restx import Api, Resource

app = Flask(__name__) 
api = Api(app, version=1.0, title="ladder api", description='ladder api docs', doc='/api-docs')  # Flask 객체에 Api 객체 등록
ladder_api = api.namespace('api/v1', description='ladder api docs')

@ladder_api.route('/converting-image')
class convertingImage(Resource):
    def post(self):
        params = request.get_json() ## 이미지 url 받아오기
        print('----------------AI 서버-------------------')
        print(params['img'])

        return make_photo(params['img'])
        

if __name__ == '__main__':
    app.run('0.0.0.0', port=5555, debug=True)
