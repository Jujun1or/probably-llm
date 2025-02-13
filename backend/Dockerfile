# Этап сборки
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

# Установка зависимостей в системные директории
RUN pip install --no-cache-dir -r requirements.txt

# Итоговый образ
FROM python:3.9-slim

WORKDIR /app

# Копируем зависимости из этапа builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Копируем исходный код
COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]