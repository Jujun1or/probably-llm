# Этап сборки
FROM node:18 as builder

WORKDIR /app
COPY package*.json ./
COPY . .

# Установка зависимостей и сборка
RUN npm install
RUN npm run build

# Этап продакшена
FROM nginx:alpine

# Копируем собранные файлы
COPY --from=builder /app/build /usr/share/nginx/html

# Копируем конфиг Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Права доступа
RUN chmod -R 755 /usr/share/nginx/html

RUN mkdir -p /var/cache/nginx/client_temp && \
    chmod -R 755 /var/cache/nginx

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]