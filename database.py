
import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "crop_disease.db")


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.commit()

    finally:
        connection.close()


def create_user(name, email, password_hash):
    connection = get_connection()

    try:
        connection.execute("""
            INSERT INTO users (name, email, password, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            email.lower().strip(),
            password_hash,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def get_user_by_email(email):
    connection = get_connection()

    try:
        user = connection.execute("""
            SELECT *
            FROM users
            WHERE email = ?
        """, (email.lower().strip(),)).fetchone()

        return dict(user) if user else None

    finally:
        connection.close()


def get_user_by_id(user_id):
    connection = get_connection()

    try:
        user = connection.execute("""
            SELECT id, name, email, created_at
            FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

        return dict(user) if user else None

    finally:
        connection.close()


def save_prediction(filename, disease, confidence):
    connection = get_connection()

    try:
        connection.execute("""
            INSERT INTO predictions (
                filename, disease, confidence, created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            filename,
            disease,
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        connection.commit()

    finally:
        connection.close()


def get_all_predictions():
    connection = get_connection()

    try:
        records = connection.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """).fetchall()

        return [dict(record) for record in records]

    finally:
        connection.close()


def get_prediction_by_id(prediction_id):
    connection = get_connection()

    try:
        record = connection.execute("""
            SELECT *
            FROM predictions
            WHERE id = ?
        """, (prediction_id,)).fetchone()

        return dict(record) if record else None

    finally:
        connection.close()


def delete_prediction(prediction_id):
    connection = get_connection()

    try:
        cursor = connection.execute("""
            DELETE FROM predictions
            WHERE id = ?
        """, (prediction_id,))

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()


def get_dashboard_statistics():
    connection = get_connection()

    try:
        total = connection.execute("""
            SELECT COUNT(*) FROM predictions
        """).fetchone()[0]

        healthy = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE LOWER(disease) LIKE '%healthy%'
        """).fetchone()[0]

        diseased = total - healthy

        average_confidence = connection.execute("""
            SELECT AVG(confidence)
            FROM predictions
        """).fetchone()[0]

        disease_counts = connection.execute("""
            SELECT disease, COUNT(*) AS total
            FROM predictions
            GROUP BY disease
            ORDER BY total DESC
        """).fetchall()

        daily_counts = connection.execute("""
            SELECT DATE(created_at) AS day, COUNT(*) AS total
            FROM predictions
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """).fetchall()

        disease_counts = [dict(row) for row in disease_counts]
        daily_counts = [dict(row) for row in daily_counts]

        return {
            "total": total,
            "healthy": healthy,
            "diseased": diseased,
            "categories": len(disease_counts),
            "average_confidence": round(
                average_confidence or 0, 2
            ),
            "disease_counts": disease_counts,
            "daily_counts": daily_counts
        }

    finally:
        connection.close()


def export_predictions():
    connection = get_connection()

    try:
        records = connection.execute("""
            SELECT id, filename, disease, confidence, created_at
            FROM predictions
            ORDER BY id DESC
        """).fetchall()

        return [dict(record) for record in records]

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
    print("CropAI Database Initialized")
    print("Database location:", DATABASE_PATH)
