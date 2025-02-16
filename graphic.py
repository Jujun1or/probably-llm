import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Данные
epochs = list(range(1, 16))  # Эпохи с 1 по 15
train_loss = [1.5528, 1.0456, 0.9669, 0.9158, 0.8687, 0.8203, 0.7768, 0.7359, 0.7021, 0.6695, 0.6432, 0.6072, 0.5887, 0.5764, 0.5620]
train_accuracy = [0.3804, 0.4661, 0.5282, 0.5607, 0.5941, 0.6347, 0.6569, 0.6788, 0.6937, 0.7147, 0.7356, 0.7475, 0.7591, 0.7674, 0.7832]

# Создаем DataFrame для удобства
data = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': train_loss,
    'Train Accuracy': train_accuracy,
})

# Настройка стиля Seaborn
sns.set_style("whitegrid")

# График функции потерь
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Первый график
sns.lineplot(x='Epoch', y='Train Loss', data=data, marker='o', label='Loss function')
plt.title('Функция потерь (Loss) по эпохам', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Значение функции потерь', fontsize=12)
plt.legend()

# График точности
plt.subplot(1, 2, 2)  # Второй график
sns.lineplot(x='Epoch', y='Train Accuracy', data=data, marker='o', label='Accuracy')
plt.title('Точность (Accuracy) по эпохам', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.legend()

# Показать графики
plt.tight_layout()
plt.show()