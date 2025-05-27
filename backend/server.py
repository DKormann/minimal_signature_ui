

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)

@app.route('/api/hello', methods=['GET'])
def hello():
  return jsonify({'message': 'Hello, World!'})

@app.route('/api/message', methods=['GET'])
def message():
  return jsonify({'message': 'this is not a server message!'})

@app.route('/api/list_documents', methods=['GET'])
def list_documents():
  return jsonify(os.listdir('./documents'))

@app.route('/api/get_document/<string:name>', methods=['GET'])
def get_document(name):
  try:
    return send_from_directory('./documents', name)
  except FileNotFoundError:
    return jsonify({'error': 'Document not found'}), 404
  

@app.route("/", methods = ["GET"])
def index(): return app.send_static_file('index.html')


if __name__ == '__main__':
  app.run(debug=True)
