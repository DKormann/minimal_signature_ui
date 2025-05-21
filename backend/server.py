

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)

@app.route('/api/hello', methods=['GET'])
def hello():
  return jsonify({'message': 'Hello, World!'})

@app.route('/api/message', methods=['GET'])
def message():
  return jsonify({'message': 'this is not a server message!'})

if __name__ == '__main__':
  app.run(debug=True)
