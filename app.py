from flask import Flask, request
from models.model_module import make_photo

app = Flask(__name__) 
@app.route('/api/v1/images/result', methods=['POST'])
def hello():


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)