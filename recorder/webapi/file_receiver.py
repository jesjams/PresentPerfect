# Save as file_receiver.py on friend's laptop
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = './received_recordings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'recording' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['recording']
    if file.filename != '':
        full_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_path)
        print(f"✅ Received: {file.filename}")
        print(f"📁 Saved to: {os.path.abspath(full_path)}")
        print(f"📏 File size: {os.path.getsize(full_path)} bytes") 
        return jsonify({'status': 'success'})
    
    return jsonify({'error': 'Empty file'}), 400

@app.route('/health', methods=['GET'])
def test_app():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=False)
    