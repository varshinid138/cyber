import os
from flask import Flask, render_template, request, jsonify
import requests
from werkzeug.utils import secure_filename
from model import process_file
import logging

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='web/static')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to fetch irrelevant sentences from Rasa API
def fetch_nlu_irrelevant(text, api_url='http://localhost:5005/model/parse'):
    try:
        request_body = {"text": text}
        response = requests.post(api_url, json=request_body)
        if response.status_code == 200:
            nlu_data = response.json()
            app.logger.debug(f"Rasa NLU response for '{text}': {nlu_data}")
            if 'intent' in nlu_data and nlu_data['intent']['name'] == 'irrelevant_sentence':
                app.logger.info(f"Sentence classified as irrelevant: '{text}'")
                return True
            else:
                app.logger.info(f"Sentence not classified as irrelevant: '{text}'")
        else:
            app.logger.error(f"Error response from Rasa NLU: {response.status_code} - {response.text}")
        return False
    except Exception as e:
        app.logger.error(f'Exception while fetching NLU data: {str(e)}')
        return False

# Function to save matched irrelevant data to a .txt file
def save_matched_data(matched_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in matched_data:
            file.write(f"{sentence}\n")
    app.logger.info(f"Saved {len(matched_data)} matched irrelevant sentences to {output_file}")

@app.route('/')

def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.error('No file selected for uploading')
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        app.logger.info(f'File saved at: {file_path}')

        try:
            # Step 1: Process the file and extract relevant, irrelevant, and flagged data
            relevant, irrelevant, flagged = process_file(file_path)
            app.logger.info(f"Extracted {len(irrelevant)} potentially irrelevant sentences from the file")
            
            # Step 2: Compare irrelevant sentences with Rasa NLU
            matched_irrelevant = []
            for sentence, _ in irrelevant:
                if fetch_nlu_irrelevant(sentence):
                    matched_irrelevant.append(sentence)
            
            app.logger.info(f"Found {len(matched_irrelevant)} sentences classified as irrelevant by Rasa NLU")
            
            # Step 3: Save matched sentences to a .txt file
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'matched_irrelevant_data.txt')
            save_matched_data(matched_irrelevant, output_file)
            
            # Prepare data for response
            data = {
                'matched_irrelevant': matched_irrelevant,
                'file': output_file
            }

            return jsonify({'status': 'success', 'data': data})

        except Exception as e:
            app.logger.error(f'Error processing file: {str(e)}')
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        app.logger.error('Allowed file types are txt, pdf, doc, docx')
        return jsonify({'error': 'Allowed file types are txt, pdf, doc, docx'}), 400

if __name__ == '__main__':
   app.run(debug=True, port=5000)