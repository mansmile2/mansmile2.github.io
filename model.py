import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели
model = tf.keras.models.load_model('Car_model.h5')

def predict_plate(image_path):
    # Загрузка изображения
    image = Image.open(image_path)
    image = image.resize((150, 150))  # Измените размер согласно вашей модели
    image = np.array(image) / 255.0  # Нормализация изображения
    image = np.expand_dims(image, axis=0)  # Добавление батча

    # Предсказание
    prediction = model.predict(image)

    # Декодирование предсказания в номер
    plate_number = decode_prediction(prediction)
    
    return plate_number

def decode_prediction(prediction):
    # Преобразуем предсказание в строку
    # Например, если модель предсказывает каждый символ по очереди, мы можем просто взять индекс с максимальной вероятностью
    # Для простоты считаем, что модель выдаёт строку длиной 7 (например, ABC1234)

    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Алфавит для распознавания
    plate_number = ""

    for i in range(prediction.shape[1]):  # Для каждого символа
        char_index = np.argmax(prediction[0, i, :])  # Индекс символа с максимальной вероятностью
        plate_number += characters[char_index]  # Добавляем символ в строку

    return plate_number

def show_generated_image(prediction):
    # Например, если модель генерирует изображение, можно его отобразить
    generated_image = prediction[0]  # Пример, что модель выдает изображение как массив
    plt.imshow(generated_image)
    plt.axis('off')  # Убираем оси
    plt.show()
