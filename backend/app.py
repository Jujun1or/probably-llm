# app.py
from quart import Quart, request, send_file, jsonify
import pandas as pd
from io import BytesIO
import numpy as np
import asyncio
import html2text
import onnxruntime as ort
import re
import tensorflow as tf
import pickle


tf.config.set_visible_devices([], 'GPU') 

session = ort.InferenceSession(
    'model.onnx',
    providers=['CPUExecutionProvider'],  
)

h = html2text.HTML2Text()

with open('vocab.pkl', 'rb') as f:
        word_to_index = pickle.load(f)

def text_to_sequence(text, max_len=256):
    sequence = [word_to_index.get(word, 0) for word in text.split()] 
    # Pad/truncate to max_len
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))


def clean_text(text):
    text = re.sub(r'[^а-яА-ЯёЁ\s().,:-;?!]', ' ', text)
    text = re.sub(r'([().,:-;?!])', r' \1 ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    cleaned = clean_text(h.handle(text))
    input_name = session.get_inputs()[0].name
    inputs = {
        input_name: np.array([text_to_sequence(cleaned)], dtype = np.int32)
    }
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    return ['Negative', 'Neutral', 'Positive'][np.argmax(logits)]


async def async_predict_sentiment(text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict_sentiment, text)

app = Quart(__name__)

@app.route('/api/analyze', methods=['POST'])
async def analyze_text():
    try:
        data = await request.get_json()
        text = data['text'].strip()
        
        sentiment = await async_predict_sentiment(text)
        return jsonify({
            'sentiment': sentiment,
            'text_length': len(text)
        })
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-csv', methods=['POST'])
async def analyze_csv():
    try:
        files = await request.files
        if 'file' not in files:
            return jsonify({'error': 'No file part'}), 400

        file = files['file']
        
        # Читаем файл как синхронную операцию в executor
        loop = asyncio.get_event_loop()
        file_contents = await loop.run_in_executor(None, file.read)
        
        # Обработка CSV
        df = await loop.run_in_executor(
            None,
            lambda: pd.read_csv(BytesIO(file_contents))
        )
        
        # Пакетная обработка
        texts = df['MessageText'].tolist()
        results = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_cleaned = [clean_text(h.handle(text)) for text in batch]
            
            # Получаем имя входного тензора
            input_name = session.get_inputs()[0].name
            
            # Пакетный прогноз
            inputs = { input_name: np.array([text_to_sequence(batch_cleaned)], dtype = np.int32)}
            outputs = session.run(None, inputs)
            batch_results = [['Negative', 'Neutral', 'Positive'][np.argmax(p)]for p in outputs[0]]
            results.extend(batch_results)
        
        df['sentiment'] = results
        
        # Синхронная запись CSV
        output = BytesIO()
        await loop.run_in_executor(
            None,
            lambda: df.to_csv(output, index=False, encoding='utf-8')
        )
        output.seek(0)
        
        return await send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            attachment_filename='results.csv'
        )
        
    except Exception as e:
        app.logger.error(f"CSV Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run()