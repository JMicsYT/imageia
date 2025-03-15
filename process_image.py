from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import time

# Инициализация моделей с улучшенными параметрами
object_detector = YOLO("yolov8x.pt")  # Используем более крупную модель YOLOv8x для лучшего распознавания
ocr_reader = easyocr.Reader(['ru'])  # OCR для русского языка

# Расширение словаря для более точного определения цветочных композиций
def translate_to_russian(english_name, is_flower_bouquet=False):
    """Перевод названий объектов с английского на русский с расширенным словарем."""
    translations = {
        "person": "человек",
        "bicycle": "велосипед",
        "car": "автомобиль",
        "motorcycle": "мотоцикл",
        "airplane": "самолет",
        "bus": "автобус",
        "train": "поезд",
        "truck": "грузовик",
        "boat": "лодка",
        "traffic light": "светофор",
        "fire hydrant": "пожарный гидрант",
        "stop sign": "знак стоп",
        "parking meter": "парковочный счетчик",
        "bench": "скамейка",
        "bird": "птица",
        "cat": "кот",
        "dog": "собака",
        "horse": "лошадь",
        "sheep": "овца",
        "cow": "корова",
        "elephant": "слон",
        "bear": "медведь",
        "zebra": "зебра",
        "giraffe": "жираф",
        "backpack": "рюкзак",
        "umbrella": "зонт",
        "handbag": "сумка",
        "tie": "галстук",
        "suitcase": "чемодан",
        "frisbee": "фрисби",
        "skis": "лыжи",
        "snowboard": "сноуборд",
        "sports ball": "спортивный мяч",
        "kite": "воздушный змей",
        "baseball bat": "бейсбольная бита",
        "baseball glove": "бейсбольная перчатка",
        "skateboard": "скейтборд",
        "surfboard": "доска для серфинга",
        "tennis racket": "теннисная ракетка",
        "bottle": "бутылка",
        "wine glass": "бокал вина",
        "cup": "чашка",
        "fork": "вилка",
        "knife": "нож",
        "spoon": "ложка",
        "bowl": "миска",
        "banana": "банан",
        "apple": "яблоко",
        "sandwich": "бутерброд",
        "orange": "апельсин",
        "broccoli": "брокколи",
        "carrot": "морковь",
        "hot dog": "хот-дог",
        "pizza": "пицца",
        "donut": "пончик",
        "cake": "торт",
        "chair": "стул",
        "couch": "диван",
        "potted plant": "комнатное растение",
        "bed": "кровать",
        "dining table": "обеденный стол",
        "toilet": "туалет",
        "tv": "телевизор",
        "laptop": "ноутбук",
        "mouse": "мышь",
        "remote": "пульт",
        "keyboard": "клавиатура",
        "cell phone": "мобильный телефон",
        "microwave": "микроволновка",
        "oven": "духовка",
        "toaster": "тостер",
        "sink": "раковина",
        "refrigerator": "холодильник",
        "book": "книга",
        "clock": "часы",
        "vase": "ваза",
        "scissors": "ножницы",
        "teddy bear": "плюшевый мишка",
        "hair drier": "фен",
        "toothbrush": "зубная щетка",
        # Дополнительные определения для цветов и букетов
        "flower": "цветок",
        "bouquet": "букет",
        "rose": "роза",
        "tulip": "тюльпан",
        "potted plant": "растение в горшке",
        "vase with flowers": "букет цветов",
    }

    # Переопределяем вазу как букет цветов, только если есть подтверждение
    if english_name.lower() == "vase" and is_flower_bouquet:
        return "букет цветов"
    return translations.get(english_name.lower(), english_name)

def preprocess_image(image_path):
    """Предобработка изображения для улучшения читаемости текста."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    # Преобразование в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Применение адаптивного порога
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    preprocessed_path = "preprocessed_" + os.path.basename(image_path)
    cv2.imwrite(preprocessed_path, thresh)
    return preprocessed_path

def correct_text(text):
    """Коррекция текста с помощью Яндекс.Спеллера."""
    if not text:
        return ""
    # Очистка текста
    cleaned_text = re.sub(r'[^а-яА-Я\s]', '', text)
    # Отправка запроса к Яндекс.Спеллеру
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {
        "text": cleaned_text,
        "lang": "ru",
        "options": 512  # Игнорировать заглавные буквы
    }
    try:
        response = requests.get(url, params=params)
        corrections = response.json()
        corrected_text = cleaned_text
        for correction in corrections:
            wrong = correction['word']
            correct = correction['s'][0] if correction['s'] else wrong
            corrected_text = corrected_text.replace(wrong, correct)
        return corrected_text
    except Exception as e:
        print(f"Ошибка при коррекции текста: {e}")
        return cleaned_text

def detect_flower_bouquet(image_path):
    """Дополнительная функция для определения букета цветов с более точной логикой."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Расширенные диапазоны HSV для различных цветов цветов
    pink_lower = np.array([140, 20, 50])
    pink_upper = np.array([170, 255, 255])
    white_lower = np.array([0, 0, 150])
    white_upper = np.array([180, 30, 255])
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])

    # Маски для разных цветов
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Объединение масок
    combined_mask = pink_mask + white_mask + red_mask1 + red_mask2 + yellow_mask

    # Морфологическая обработка для уменьшения шума
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Находим контуры цветных областей
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flower_area = 0
    total_pixels = img.shape[0] * img.shape[1]

    # Проверяем размер каждой области
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Минимальная площадь для цветочной области
            flower_area += area

    # Процент пикселей, соответствующих цветам
    flower_ratio = flower_area / total_pixels

    # Более строгий порог
    if flower_ratio > 0.15:  # 15% изображения должно содержать цветочные области
        return True
    return False

def generate_xml_metadata(image_path, objects, text, coordinates, master_xml_path="master_metadata.xml"):
    """
    Генерирует XML-файл метаданных в указанном формате и обновляет мастер-файл.
    Возвращает путь к XML-файлу.
    """
    image_name = os.path.basename(image_path)
    xml_path = os.path.splitext(image_path)[0] + ".xml"

    # Создаем структуру XML для текущего изображения
    root = ET.Element("metadata")
    image_elem = ET.SubElement(root, "image", name=image_name)

    # Добавление объектов с координатами
    for obj, coords in coordinates.items():
        obj_elem = ET.SubElement(image_elem, "object", 
                                name=obj, 
                                x=str(coords['x']), 
                                y=str(coords['y']), 
                                width=str(coords['width']), 
                                height=str(coords['height']))

    # Добавление текста
    text_elem = ET.SubElement(image_elem, "text")
    text_elem.text = text

    # Создаем форматированный XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Записываем данные текущего изображения в отдельный файл
    with open(xml_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(pretty_xml)

    # Обновляем мастер-файл
    try:
        if os.path.exists(master_xml_path):
            # Если мастер-файл существует, добавляем новое изображение
            master_tree = ET.parse(master_xml_path)
            master_root = master_tree.getroot()

            # Проверяем, есть ли уже такое изображение
            existing_image = None
            for img in master_root.findall(".//image[@name='%s']" % image_name):
                existing_image = img
                break

            if existing_image is not None:
                # Обновляем существующую запись
                parent = master_root
                parent.remove(existing_image)

            # Добавляем новую запись
            for child in root:
                master_root.append(child)
        else:
            # Если мастер-файл не существует, создаем новый
            master_root = ET.Element("metadata")
            for child in root:
                master_root.append(child)

        # Записываем обновленный мастер-файл
        master_rough_string = ET.tostring(master_root, 'utf-8')
        master_reparsed = minidom.parseString(master_rough_string)
        master_pretty_xml = master_reparsed.toprettyxml(indent="  ")

        with open(master_xml_path, "w", encoding="utf-8") as master_file:
            master_file.write(master_pretty_xml)

    except Exception as e:
        print(f"Ошибка при обновлении мастер-файла: {e}")

    return xml_path

def process_image(image_path, output_dir="processed"):
    """
    Обрабатывает изображение: обнаруживает объекты, распознает текст, записывает мета-данные.
    Возвращает объекты, текст, путь к обработанному изображению, путь к XML-файлу и координаты объектов.
    """
    # Создаем папку для обработанных изображений
    os.makedirs(output_dir, exist_ok=True)

    # 1. Обнаружение объектов с повышенным разрешением и порогом уверенности
    try:
        results = object_detector(image_path, conf=0.5, imgsz=1280)  # Увеличенное разрешение и сниженный порог
    except Exception as e:
        print(f"Ошибка при обнаружении объектов: {e}")
        results = [None]

    # Проверяем, есть ли обнаруженные объекты
    objects = []
    coordinates = {}
    detected_vase = False
    is_flower_bouquet = detect_flower_bouquet(image_path)

    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Получаем размеры изображения
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        # Сопоставляем классы и уверенности
        for i, cls in enumerate(results[0].boxes.cls):
            conf = results[0].boxes.conf[i]  # Получаем уверенность для текущего объекта
            class_name = object_detector.names[int(cls)]
            
            # Получаем координаты объекта
            box = results[0].boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Нормализуем координаты
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Проверяем, является ли объект вазой
            if class_name.lower() == "vase":
                detected_vase = True

            # Перевод на русский язык
            translated_name = translate_to_russian(class_name, is_flower_bouquet)
            if translated_name not in objects and conf > 0.5:
                objects.append(translated_name)
                
                # Сохраняем координаты объекта
                obj_id = f"{translated_name}_{i}"
                coordinates[obj_id] = {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'name': translated_name,
                    'confidence': float(conf)
                }

    # Дополнительная проверка: добавляем "букет цветов" только если обнаружена ваза и есть признаки цветов
    if detected_vase and is_flower_bouquet:
        if "букет цветов" not in objects:
            objects.append("букет цветов")

    # 2. Предобработка и распознавание текста
    try:
        preprocessed_path = preprocess_image(image_path)
        text_results = ocr_reader.readtext(preprocessed_path, text_threshold=0.4)
        raw_text = " ".join([result[1] for result in text_results if result])  # Проверка на None
        
        # Сохраняем координаты текста
        for i, (bbox, text, conf) in enumerate(text_results):
            if conf > 0.4:  # Фильтруем по уверенности
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
                text_id = f"text_{i}"
                coordinates[text_id] = {
                    'x': int(min(x1, x4)),
                    'y': int(min(y1, y2)),
                    'width': int(max(x2, x3) - min(x1, x4)),
                    'height': int(max(y3, y4) - min(y1, y2)),
                    'name': 'text',
                    'text': text,
                    'confidence': float(conf)
                }
    except Exception as e:
        print(f"Ошибка при обработке текста: {e}")
        raw_text = ""
    finally:
        # Удаляем временный предобработанный файл
        if 'preprocessed_path' in locals() and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    # 3. Коррекция текста
    corrected_text = correct_text(raw_text)

    # 4. Запись мета-данных
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()
        metadata = {
            "objects": objects,
            "text": corrected_text,
            "coordinates": coordinates
        }
        exif_data[270] = json.dumps(metadata)  # Тег 270 - описание
        image.save(output_path, exif=exif_data)
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        output_path = None  # Устанавливаем None, если сохранение не удалось

    # 5. Генерация XML
    xml_path = generate_xml_metadata(image_path, objects, corrected_text, coordinates) if output_path else None

    # Возвращаем пять значений, включая координаты
    return objects, corrected_text, output_path, xml_path, coordinates

def get_object_coordinates(file_path):
    """Получение координат объектов из EXIF метаданных изображения"""
    try:
        image = Image.open(file_path)
        exif_data = image.getexif()
        metadata = json.loads(exif_data.get(270, "{}"))
        return metadata.get("coordinates", {})
    except Exception as e:
        print(f"Ошибка при получении координат: {e}")
        return {}

def add_text_to_image(input_path, output_path, text, font_name="Arial", font_size=24, color="#000000", position=None):
    """Добавление текста на изображение с сохранением шрифта"""
    try:
        # Открываем изображение
        image = Image.open(input_path)
        draw = ImageDraw.Draw(image)
        
        # Пытаемся загрузить шрифт
        try:
            font = ImageFont.truetype(font_name, font_size)
        except:
            # Если не удалось загрузить указанный шрифт, используем стандартный
            font = ImageFont.load_default()
        
        # Преобразуем цвет из HEX в RGB
        if color.startswith('#'):
            color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        # Определяем позицию текста
        if position is None:
            position = {'x': 10, 'y': 10}
        
        # Рисуем текст на изображении
        draw.text((position['x'], position['y']), text, font=font, fill=color)
        
        # Сохраняем изображение
        image.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Ошибка при добавлении текста на изображение: {e}")
        return None

if __name__ == "__main__":
    # Тестовый запуск
    objects, text, output_path, xml_path, coordinates = process_image("test_image.jpg")
    print(f"Objects: {objects}")
    print(f"Text: {text}")
    print(f"Saved to: {output_path}")
    print(f"XML path: {xml_path}")
    print(f"Coordinates: {coordinates}")

