from flask import Flask, request, jsonify
import json
import requests
from model import CaptionGenerator
from waitress import serve

app = Flask(__name__)

# Initialize Caption Generator
model = CaptionGenerator(model_dir='saved_model/', data_dir='saved_data/')


@app.route('/ask', methods=['GET'])
def process_image():
    body = request.get_json(force=True)
    
    # Get img base64 from request body
    img_base64 = body['img']

    # Remove Prefix if it exists
    if img_base64.startswith('data:image/jpeg;base64,'):
        img_base64 = img_base64[len('data:image/jpeg;base64,'):]
    
    # Generate caption
    try:
        caption = model.predict(img_base64)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return json.dumps({'status': 'ok'})


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
