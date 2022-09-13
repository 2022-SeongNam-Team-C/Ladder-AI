from flask import Flask, request
from models.model_module import make_photo

app = Flask(__name__) 
@app.route('/api/v1/images/result', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        # print(model)
        return "모델 생성"
    if request.method == 'POST':
        params = request.get_json()
        print(params['img'])
        return make_photo(params['img'])
        # return 'ok'
        
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)