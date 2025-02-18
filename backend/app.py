# app.py
from quart import Quart, request, send_file, jsonify
import pandas as pd
from io import BytesIO
import numpy as np
import asyncio
import onnxruntime as ort
import re
import tensorflow as tf
import pickle
from threading import local
import concurrent.futures
from asyncio import Semaphore

CSV_SEMAPHORE = Semaphore(6)

thread_local = local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = ort.InferenceSession(
            'model.onnx',
            providers=['CPUExecutionProvider'],
        )
    return thread_local.session


tf.config.set_visible_devices([], 'GPU') 
executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)


with open('vocab.pkl', 'rb') as f:
        word_to_index = pickle.load(f)

def text_to_sequence(text, max_len=300):
    sequence = [word_to_index.get(word, 0) for word in text.split()] 
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))

def clean_text(text):
    try:
        # Original cleaning steps
        text = re.sub(r'[^а-яА-ЯёЁ?!()\s]', ' ', text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = text.replace('ё', 'е').replace('Ё', 'Е')
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'([?!()])', r' \1 ', text)  # Add spaces around punctuation
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()


    except UnicodeEncodeError:
        return ""

def predict_sentiment(text):
    session = get_session()
    cleaned = clean_text(text)
    input_name = session.get_inputs()[0].name
    inputs = {
        input_name: np.array([text_to_sequence(cleaned)], dtype = np.int32)
    }
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    return ['Negative', 'Neutral', 'Positive'][np.argmax(logits)]


async def async_predict_sentiment(text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, predict_sentiment, text)

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
    async with CSV_SEMAPHORE:
        try:
            files = await request.files
            if 'file' not in files:
                return jsonify({'error': 'No file part'}), 400

            file = files['file']
            loop = asyncio.get_event_loop()
            file_contents = await loop.run_in_executor(None, file.read)
            
            df = await loop.run_in_executor(
                None,
                lambda: pd.read_csv(BytesIO(file_contents))
            )
            texts = df['MessageText'].tolist()
            results = []
            batch_size = 128

            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            all_tasks = [
                async_predict_sentiment(text)
                for batch in batches
                for text in batch
            ]
            results = await asyncio.gather(*all_tasks)
            df['sentiment'] = results 
            output = BytesIO()
            await loop.run_in_executor(
                None,
                lambda: df.to_csv(
                output, 
                index=False, 
                encoding='utf-8',
                lineterminator='\n'  # Добавляем для корректных переносов строк
            )
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
