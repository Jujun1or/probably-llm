# app.py
from flask import Flask, request, send_file, jsonify
import pandas as pd
from io import BytesIO
from model import Model

app = Flask(__name__)
model = Model()

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    sentiment = analyze_sentiment(text) 
    
    return jsonify({
        'sentiment': sentiment,
        'text_length': len(text)
    })

def analyze_sentiment(text):
    return "neutral"

@app.route('/api/analyze-csv', methods=['POST'])
def analyze_csv():
    try:
        app.logger.info("Headers: %s", request.headers)
        app.logger.info("Files: %s", request.files)
        
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        app.logger.info("Received file: %s, Size: %d", file.filename, len(file.read()))
        file.seek(0) 
        if not file.filename.endswith('.csv'):
            app.logger.error("Not a csv")
            return {'error': 'Invalid file format'}, 400
    
        # Чтение CSV
        df = pd.read_csv(file)
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return {'error': f'Error reading CSV: {str(e)}'}, 400

    df['sentiment'] = df['MessageText'].apply(analyze_sentiment)
    
    # Генерация CSV
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='results.csv'
    )