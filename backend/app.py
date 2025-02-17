from quart import Quart, request, send_file, jsonify
import httpx
import pandas as pd
from io import BytesIO
import tensorflow as tf
import re
import numpy as np
import asyncio
import html2text

h = html2text.HTML2Text()

def clean_text(text):
    # EXACT replication of training preprocessing
    text = re.sub(r'[^а-яА-ЯёЁ\s().,:-;?!]', ' ', text)
    text = re.sub(r'([().,:-;?!])', r' \1 ', text)  # Fixed regex (removed accidental 'p')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tf.config.set_visible_devices([], 'GPU')


loaded = tf.saved_model.load('model.tf')
infer = loaded.signatures['serving_default']

def predict_sentiment(text):
    cleaned = clean_text(text)
    input_tensor = tf.constant(([cleaned], ' '), dtype=tf.string)
    output = infer(input_tensor)
    logits = list(output.values())[0].numpy()[0]
    return ['Negative', 'Neutral', 'Positive'][logits.argmax()]

async def async_predict_sentiment(texts):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict_sentiment, texts)

app = Quart(__name__)

app.config['MAX_CONTENT_LENGTH'] = 200 *1000 * 1024

@app.route('/api/analyze', methods=['POST'])
async def analyze_text():
    try:
        data = await request.get_json()
        text = data['text'].strip()
        
        sentiments = await async_predict_sentiment(text) 
        app.logger.info(text)
        return jsonify({
            'sentiment': str(sentiments),
            'text_length': len(text)
        })
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-csv', methods=['POST'])
async def analyze_csv():
    try:
        # Асинхронное чтение файла
        files = await request.files
        if 'file' not in files:
            return jsonify({'error': 'No file part'}), 400
            
        file = files['file']
        file_contents = file.read()
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file format'}), 400

        # Синхронное чтение CSV в отдельном потоке
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, 
            pd.read_csv, 
            BytesIO(file_contents)
        )
        
        # Асинхронное предсказание для всех текстов
        texts = df['MessageText'].tolist()

        sentiments =[await async_predict_sentiment(h.handle(text)) for text in texts]
        df['sentiment'] = sentiments

        # Синхронная запись CSV в отдельном потоке
        output = BytesIO()
        await loop.run_in_executor(
            None,
            lambda: df.to_csv(output, index=False)
        )
        output.seek(0)
        
        return await send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            attachment_filename='results.csv'
        )
        
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500