
# NEED
# pip install razdel
# pip install transformers
# pip install scipy

# Регулярные выражения потребуются для очистки текста
import re

# Библиотека razdel потребуется для разбиения текста на предложения
# перед отправкой в нейросеть
from razdel import sentenize


import html2text
import csv 
# Без torch невозможна работа с нейросетями
import torch

# Библиотека transformers нужна для работы с нейросетями-трансформерами,
# которые мы будем использовать для анализа тональности
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Библиотека matplotlib позволит построить графики кривых тональности
#from matplotlib import pyplot as plt

# Фильтр Савицкого-Голея понадобится нам для обработки результатов,
# которая будет описана позже
from scipy.signal import savgol_filter

def clean_text(text: str) -> str:

    # Заменяем переносы строк на пробелы
    text = text.replace('\n', ' ')
    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    return cleaned_text


def read_file(filename) -> list:
    sentences = []
    with open(filename, 'r', encoding='utf-8-sig', newline='\n') as file:
        reader = csv.reader(file)
        header = list(next(reader)) 
        for items in reader:
            text = items[2]
            cleaned_text = clean_text(html2text.html2text(text))
            sentences.append(cleaned_text)
    return sentences

sentences = read_file('dataset_comments.csv')

 
# for substring in list(sentenize(cleaned_text)):
#    sentences.append(substring.text)
# Разбиваем текст на предложения и загружаем их в список
#for substring in list(sentenize(cleaned_text)):
#    sentences.append(substring.text)

# Загрузим модель с сайта HuggingFace и создадим ее экземпляр
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
#if torch.cuda.is_available():
#    model.cuda()

def convert_to_letter(val) -> str:
    if val < -0.33: return 'B' 
    return 'G' 
import time

# Сложная функция, которая заставит модель работать
def estimate_sentiment(messages: list):
    start = time.time()
    data = []
    for text in messages:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
            val = round(proba.dot([-1, 0, 1]), 4)
            data.append({'comment':text, 'value': val, 'letter': convert_to_letter(val)})
    end = time.time()
    print("Took", end - start, "sec")
    with open('output.csv', 'w+', encoding='utf-8-sig', newline='') as csvfile:
        fieldnames = ['comment', 'value', 'letter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


estimate_sentiment(sentences)
