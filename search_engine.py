import json
from PIL import Image
import datetime
import os

def index_image(file_path, conn):
    """Index image metadata into SQLite."""
    try:
        image = Image.open(file_path)
        exif_data = image.getexif()
        metadata = json.loads(exif_data.get(270, "{}"))
        objects = json.dumps(metadata.get("objects", []))
        text = metadata.get("text", "")
        coordinates = json.dumps(metadata.get("coordinates", {}))
        xml_path = os.path.splitext(file_path)[0] + ".xml"
        
        # Добавляем дату загрузки
        upload_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO images 
               (file_path, objects, text, coordinates, upload_date, xml_path) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_path, objects, text, coordinates, upload_date, xml_path)
        )
        conn.commit()
    except Exception as e:
        print(f"Ошибка при индексации изображения: {e}")

def search_images(query, conn):
    """Search images by query in SQLite."""
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM images WHERE objects LIKE ? OR text LIKE ?",
                   (f"%{query}%", f"%{query}%"))
    return [row[0] for row in cursor.fetchall()]

def get_recent_images(conn, limit=10):
    """Получение последних загруженных изображений."""
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, objects, text, upload_date FROM images ORDER BY upload_date DESC LIMIT ?", (limit,))
    results = []
    for row in cursor.fetchall():
        file_path, objects_json, text, upload_date = row
        try:
            objects = json.loads(objects_json)
        except:
            objects = []
        results.append({
            'file_path': file_path,
            'objects': objects,
            'text': text,
            'upload_date': upload_date
        })
    return results

if __name__ == "__main__":
    import sqlite3
    conn = sqlite3.connect("images.db")
    index_image("processed/test_image.jpg", conn)
    results = search_images("car", conn)
    print(f"Found images: {results}")
    conn.close()

