from flask import Blueprint, jsonify, request
import logging
from utils import  create_rag_pipeline, translate_text
main_bp = Blueprint('main', __name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from werkzeug.utils import secure_filename

import os
import sys

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

sys.path.append(os.path.abspath(os.path.join('../scripts')))


rag_chain = create_rag_pipeline()


@main_bp.route('/api/v1/file-upload', methods=['POST'])
def upload_file():
    response = {
        "data": None,
        "error": None
    }
    statusCode = 404
    try:
        if 'file' not in request.files:
            raise ValueError("No file part")

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        
            # extracted_data = DataExtractor.extract_data(filepath)
            rag_chain.add_datasource(filepath)

            response["data"] = "File added to vectorstore successfully"
       
            statusCode = 200
        else:
            response["error"] = "File format not supported"


    except Exception as error:
        logging.error(error)
        response['error'] = f"An error occured: {error}"
        statusCode = 404

    return jsonify(response), statusCode


@main_bp.route('/api/v1/chat', methods=['POST'])
def chat():

    data = request.json
    response = {
        "data" : None,
        "error" : None
    }
    statusCode = 404
    try:
        message = data['message']
        eng_text = translate_text(message, "am", "en")
        chain_response = rag_chain.invoke(eng_text)
        amh_text = translate_text(chain_response["answer"], "en", "am")

        response["data"] = amh_text
        statusCode = 200

    except Exception as error:
        logging.error(error)
        response['error'] = f"An error occured: {error}"
        statusCode = 404
    return jsonify(response), statusCode


@main_bp.route('/', methods=['GET'])
def base_get():
    response = {
        "data" : "The backend is working...",
        "error" : None
    }

    return jsonify(response), 200