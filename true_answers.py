import random
import spacy
import time
import html2text
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext import datasets
import csv
import re

# Установка случайного начального состояния
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация спейси для русского языка
nlp = spacy.load('ru_core_news_sm')

# Функция для удаления HTML-тегов
def clean_html(text):
    return html2text.html2text(text)

# Функция очистки текста
def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def convertVal(val):
    if val <= 2: return -1
    elif val == 3: return 0
    return 1

# Функция чтения файла
def read_file(filename) -> list:
    sentences = []
    labels = []
    with open(filename, 'r', encoding='utf-8-sig', newline='\n') as file:
        reader = csv.reader(file, delimiter =';')
        header = list(next(reader))  # Пропускаем заголовок
        for items in reader:
            text = items[1]  
            cleaned_text = clean_text(html2text.html2text(text))
            sentences.append(cleaned_text)
            labels.append(convertVal(int(items[0])))  
    return sentences, labels

# Загрузка данных из CSV
sentences, labels = read_file('Training_data_marked (2).csv')

# Токенизация и подготовка данных с помощью torchtext
txt = data.Field(tokenize='spacy', tokenizer_language='ru_core_news_sm', include_lengths=True)
labels_field = data.LabelField(dtype=torch.float)

# Создание датасетов из CSV
fields = [('text', txt), ('label', labels_field)]
examples = [data.Example.fromlist([sentence, label], fields) for sentence, label in zip(sentences, labels)]
dataset = data.Dataset(examples, fields)

# Разделение на тренировочные и тестовые данные
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(seed))

# Ограничиваем размер словаря
num_words = 25_000
txt.build_vocab(train_data, max_size=num_words, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
labels_field.build_vocab(train_data)

# Параметры для пакетной обработки
batch_size = 64
train_itr, test_itr = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    device=device
)

# Модель RNN для анализа текста
class RNN(nn.Module):
    def __init__(self, word_limit, dimension_embedding, dimension_hidden, dimension_output, num_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(word_limit, dimension_embedding, padding_idx=pad_idx)
        self.rnn = nn.LSTM(dimension_embedding, dimension_hidden, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(dimension_hidden * 2, dimension_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, len_txt):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, len_txt.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

# Параметры модели
dimension_input = len(txt.vocab)
dimension_embedding = 100
dimension_hddn = 256
dimension_out = 1
layers = 2
bidirectional = True
droupout = 0.5
idx_pad = txt.vocab.stoi[txt.pad_token]

# Создание модели
model = RNN(dimension_input, dimension_embedding, dimension_hddn, dimension_out, layers, bidirectional, droupout, idx_pad)

# Функция для подсчета количества параметров
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Загружаем предобученные векторные представления слов
pretrained_embeddings = txt.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Обработка неизвестных и паддинг-слов
unique_id = txt.vocab.stoi[txt.unk_token]
model.embedding.weight.data[unique_id] = torch.zeros(dimension_embedding)
model.embedding.weight.data[idx_pad] = torch.zeros(dimension_embedding)

# Определение оптимизатора и функции потерь
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Перевод модели и критерия на устройство
model = model.to(device)
criterion = criterion.to(device)

# Функция для подсчета точности
def bin_acc(preds, y):
    predictions = torch.round(torch.sigmoid(preds))
    correct = (predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Функции для тренировки и оценки модели
def train(model, itr, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i in itr:
        optimizer.zero_grad()
        text, len_txt = i.text
        predictions = model(text, len_txt).squeeze(1)
        loss = criterion(predictions, i.label)
        acc = bin_acc(predictions, i.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(itr), epoch_acc / len(itr)

def evaluate(model, itr, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for i in itr:
            text, len_txt = i.text
            predictions = model(text, len_txt).squeeze(1)
            loss = criterion(predictions, i.label)
            acc = bin_acc(predictions, i.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(itr), epoch_acc / len(itr)

def epoch_time(start_time, end_time):
    used_time = end_time - start_time
    used_mins = int(used_time / 60)
    used_secs = int(used_time - (used_mins * 60))
    return used_mins, used_secs


# Основной цикл обучения
num_epochs = 5
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train(model, train_itr, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_itr, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    print(f'Эпоха: {epoch+1:02} | Время на эпоху: {epoch_mins}m {epoch_secs}s')
    print(f'\tТренировочные потери: {train_loss:.3f} | Тренировочная точность: {train_acc*100:.2f}%')
    print(f'\t Валидационные потери: {valid_loss:.3f} |  Валидационная точность: {valid_acc*100:.2f}%')

# Загрузка лучшей модели и проверка на тестовых данных
model.load_state_dict(torch.load('tut2-model.pt'))
test_loss, test_acc = evaluate(model, test_itr, criterion)
print(f'Тестовые потери: {test_loss:.3f} | Тестовая точность: {test_acc*100:.2f}%')

# Функция предсказания
def pred(model, sentence):
    model.eval()
    sentence = clean_html(sentence)  # Убираем HTML-теги из текста
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [txt.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# Пример использования
sent = ["Положительный", "Нейтральный", "Отрицательный"]
def print_sent(x):
    if x < 0.3:
        print(sent[0])
    elif 0.3 <= x <= 0.7:
        print(sent[1])
    else:
        print(sent[2])

# Пример анализируемого текста
print_sent(pred(model, "Этот фильм просто ужасен"))
print_sent(pred(model, "Этот фильм средненький"))
print_sent(pred(model, "Этот фильм отличный!"))
