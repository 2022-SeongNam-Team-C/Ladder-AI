from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from .models import model_module


app = Flask(__name__)

model = torch.load('./ORGINAL_MODEL.pt', map_location='cpu')

@app.route('/api/v1/images/result', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        print(model)
        return "모델 생성"
    if request.method == 'POST':
        #file = request.files['img']
        #fileName = file.filename
        #file.save(secure_filename(file.filename))
        # return send_file(fileName, mimetype='image/')
        model_module.make_photo('https://news.mt.co.kr/mtview.php?no=2015040510223180585')
        
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)