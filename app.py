from flask import Flask, request, render_template, send_file, g, jsonify, make_response, redirect, url_for
from process_image import process_image, get_object_coordinates, add_text_to_image
from search_engine import index_image, search_images
import os
import sqlite3
import json
import time
from datetime import datetime
from PIL import Image

# Удаляем старую базу данных при запуске (только для разработки)
if os.path.exists('images.db'):
    os.remove('images.db')

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
EDITED_FOLDER = "edited"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(EDITED_FOLDER, exist_ok=True)

# Максимальное количество изображений в истории
MAX_HISTORY_ITEMS = 10

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('images.db')
        # Create table with all required columns including coordinates
        g.db.execute("""
            CREATE TABLE IF NOT EXISTS images (
                file_path TEXT PRIMARY KEY,
                objects TEXT,
                text TEXT,
                coordinates TEXT,
                upload_date TEXT,
                xml_path TEXT
            )
        """)
        g.db.commit()
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

app.teardown_appcontext(close_db)

def get_history_from_cookie():
    """Получение истории загрузок из cookie"""
    history = request.cookies.get('image_history')
    if history:
        try:
            return json.loads(history)
        except:
            return []
    return []

def add_to_history(file_path, objects, text):
    """Добавление изображения в историю"""
    history = get_history_from_cookie()
    
    # Создаем запись для истории
    entry = {
        'file_path': file_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'preview': file_path,
        'objects': objects[:3] if objects else [],  # Сохраняем только первые 3 объекта для превью
    }
    
    # Проверяем, есть ли уже такой файл в истории
    history = [item for item in history if item['file_path'] != file_path]
    
    # Добавляем новую запись в начало списка
    history.insert(0, entry)
    
    # Ограничиваем количество записей
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[:MAX_HISTORY_ITEMS]
    
    return history

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files["image"]
        if file.filename == '':
            return redirect(request.url)
        
        # Сохраняем файл
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Обрабатываем изображение
        objects, text, processed_path, xml_path, coordinates = process_image(file_path, PROCESSED_FOLDER)
        
        # Сохраняем в базу данных
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO images (file_path, objects, text, coordinates, upload_date, xml_path) VALUES (?, ?, ?, ?, ?, ?)",
            (
        processed_path,
        json.dumps(objects),
        text,
        json.dumps(coordinates),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        xml_path
    )
        )
        db.commit()
        
        # Создаем ответ
        resp = make_response(render_template(
            "results.html",
            objects=objects,
            text=text,
            file_path=processed_path,
            xml_path=xml_path,
            coordinates=coordinates
        ))
        
        # Обновляем историю в cookie
        history = add_to_history(processed_path, objects, text)
        resp.set_cookie('image_history', json.dumps(history), max_age=30*24*60*60)  # 30 дней
        
        return resp
    
    # Получаем историю для отображения на странице загрузки
    history = get_history_from_cookie()
    return render_template("upload.html", history=history)

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    db = get_db()
    file_paths = search_images(query, db)
    return render_template("search_results.html", file_paths=file_paths, query=query)

@app.route("/download/<path:file_path>")
def download(file_path):
    return send_file(file_path, as_attachment=True)

@app.route("/download_xml/<path:file_path>")
def download_xml(file_path):
    return send_file(file_path, as_attachment=True)

@app.route("/get_coordinates/<path:file_path>")
def get_coordinates(file_path):
    """API для получения координат объектов на изображении"""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT coordinates FROM images WHERE file_path = ?", (file_path,))
    result = cursor.fetchone()
    
    if result and result[0]:
        return jsonify(json.loads(result[0]))
    return jsonify({})

@app.route("/save_edited_text", methods=["POST"])
def save_edited_text():
    """Сохранение отредактированного текста на изображении"""
    data = request.json
    file_path = data.get('file_path')
    text = data.get('text')
    font = data.get('font', 'Arial')
    font_size = data.get('font_size', 24)
    color = data.get('color', '#000000')
    position = data.get('position', {'x': 10, 'y': 10})
    
    if not file_path or not text:
        return jsonify({'success': False, 'error': 'Missing required parameters'})
    
    try:
        # Добавляем текст на изображение
        output_path = add_text_to_image(file_path, file_path, text, font, font_size, color, position)
        
        if output_path:
            # Обновляем запись в базе данных
            db = get_db()
            cursor = db.cursor()
            cursor.execute("UPDATE images SET text = ? WHERE file_path = ?", (text, file_path))
            db.commit()
            
            return jsonify({'success': True, 'file_path': file_path})
        else:
            return jsonify({'success': False, 'error': 'Failed to add text to image'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route("/remove_text", methods=["POST"])
def remove_text():
    """Удаление текста с изображения"""
    data = request.json
    file_path = data.get('file_path')
    
    if not file_path:
        return jsonify({'success': False, 'error': 'Missing file path'})
    
    try:
        # Открываем оригинальное изображение
        image = Image.open(file_path)
        
        # Создаем новое имя файла
        filename = os.path.basename(file_path)
        new_path = os.path.join(PROCESSED_FOLDER, f"clean_{filename}")
        
        # Сохраняем изображение без текста
        image.save(new_path)
        
        # Обновляем запись в базе данных
        db = get_db()
        cursor = db.cursor()
        cursor.execute("UPDATE images SET text = ? WHERE file_path = ?", (file_path,))
        db.commit()
        
        return jsonify({'success': True, 'file_path': new_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Очистка истории загрузок"""
    resp = make_response(redirect(url_for('upload')))
    resp.set_cookie('image_history', '', expires=0)
    return resp

if __name__ == "__main__":
    app.run(debug=True)

