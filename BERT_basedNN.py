import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import Dataset


# Загрузка данных из CSV
def load_data(filepath):
    df = pd.read_csv(filepath, delimiter=';', header=None, names=['label', 'text'])

    print("Уникальные значения в label:", df['label'].unique())  # Посмотрим, что там есть

    df['label'] = pd.to_numeric(df['label'], errors='coerce')  # Преобразуем к числу, ошибки -> NaN
    df = df.dropna(subset=['label'])  # Удаляем строки с NaN (если есть ошибки в данных)
    df['label'] = df['label'].astype(int)  # Теперь точно преобразуем в int

    df['label'] = df['label'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    return df


# Загрузка данных
df = load_data('C:/Users/rmedv/Downloads/Training_data_marked_balanced.csv')

# Разделение на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Преобразуем DataFrame в Dataset (библиотека datasets)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# Токенизация текста
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Преобразуем Dataset в формат PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загрузка предобученной модели BERT
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.to(device)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5)


# Функция для обучения
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)


# Функция для оценки
def evaluate(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)


# Основной цикл обучения
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Эпоха {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Тренировочные потери: {train_loss:.3f}")
    accuracy = evaluate(model, test_loader, device)
    print(f"Точность на тестовых данных: {accuracy * 100:.2f}%")

# Сохранение модели
model.save_pretrained('bert-finetuned')
tokenizer.save_pretrained('bert-finetuned')


# Функция предсказания
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    return preds.item()


# Пример использования
sentiment_labels = ["Отрицательный", "Нейтральный", "Положительный"]


def print_sentiment(text):
    prediction = predict_sentiment(text, model, tokenizer, device)
    print(f"Текст: '{text}' | Предсказанный класс: {sentiment_labels[prediction]}")


# Пример анализируемого текста
print_sentiment("Этот фильм ужасен")
print_sentiment("Этот фильм средненький")
print_sentiment("Этот фильм отличный!")