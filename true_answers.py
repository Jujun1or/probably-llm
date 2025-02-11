
# NEED
# pip install razdel
# pip install transformers
# pip install scipy

# Регулярные выражения потребуются для очистки текста
import re

# Библиотека razdel потребуется для разбиения текста на предложения
# перед отправкой в нейросеть
from razdel import sentenize

# Без torch невозможна работа с нейросетями
import torch

# Библиотека transformers нужна для работы с нейросетями-трансформерами,
# которые мы будем использовать для анализа тональности
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Библиотека matplotlib позволит построить графики кривых тональности
from matplotlib import pyplot as plt

# Фильтр Савицкого-Голея понадобится нам для обработки результатов,
# которая будет описана позже
from scipy.signal import savgol_filter

def clean_text(text: str) -> str:

    # Заменяем переносы строк на пробелы
    text = text.replace('\n', ' ')
    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    return cleaned_text
    
cleaned_text = clean_text("Дорогой дневник! У меня появилась сумка, которая учит меня шить :) Сегодня у нее отвалилось крепление для крепления.")

# Разбиваем текст на предложения и загружаем их в список
sentences = []
#for substring in list(sentenize(cleaned_text)):
#    sentences.append(substring.text)
sentences.append(cleaned_text)

# Загрузим модель с сайта HuggingFace и создадим ее экземпляр
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()
   
# Сложная функция, которая заставит модель работать
def estimate_sentiment(messages: list) -> list:
    sentiment_out = []
    for text in messages:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
            sentiment_out.append(proba.dot([-1, 0, 1]))
    return sentiment_out

import time

start = time.time()
sentiments = estimate_sentiment(sentences)
end = time.time()
print("Took", end - start, "ms")
print(round(sentiments[0], 2)) #print result
