# Этап сборки
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
# Установка зависимостей
RUN pip install --upgrade pip && pip install --user --no-cache-dir -r requirements.txt --timeout=900

RUN apt-get update && apt-get install -y locales && apt-get install -y coreutils
RUN locale-gen en_US.UTF-8


FROM python:3.9-slim

WORKDIR /app

# Копируем зависимости
COPY --from=builder /root/.local /root/.local

# Копируем остальные файлы
COPY . .


ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH
ENV LANG en_US.UTF-8 
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Порт и команда запуска
EXPOSE 5000
CMD hypercorn app:app --bind 0.0.0.0:5000 --worker-class asyncio