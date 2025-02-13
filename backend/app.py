# app.py
from flask import Flask, request, jsonify
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)