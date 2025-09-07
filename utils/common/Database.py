# utils/common/Database.py

import os
import duckdb
import pandas as pd
from typing import Dict, List, Any

from utils.common.other_utils import ensure_directory_exists

class DatabaseManager:
    """
    Універсальний менеджер для роботи з базою даних DuckDB.
    Надає модульний інтерфейс для підключення, створення таблиць,
    вставки, оновлення та вибірки даних.
    """
    def __init__(self, db_file: str, read_only: bool = False):
        """
        Ініціалізує менеджер бази даних.

        :param db_file: Шлях до файлу бази даних DuckDB.
        :param read_only: Відкрити базу даних у режимі лише для читання.
        """
        self.db_file = db_file
        self.read_only = read_only
        self.conn = None
        ensure_directory_exists(os.path.dirname(db_file))
        self.connect()

    def connect(self):
        """Підключається до бази даних DuckDB."""
        try:
            self.conn = duckdb.connect(database=self.db_file, read_only=self.read_only)
            print(f"✅ Підключено до DuckDB: '{self.db_file}'")
        except Exception as e:
            print(f"❌ Помилка підключення до DuckDB: {e}")
            self.conn = None

    def disconnect(self):
        """Закриває підключення до бази даних."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("🔌 Підключення до DuckDB закрито.")

    def execute(self, query: str, params: tuple = ()):
        """Виконує SQL-запит з параметрами."""
        if not self.conn:
            raise ConnectionError("Немає активного підключення до бази даних.")
        try:
            self.conn.execute(query, params)
        except Exception as e:
            print(f"❌ Помилка виконання запиту: {query}\n{e}")
            raise

    def fetch_df(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Виконує запит і повертає результат у вигляді pandas DataFrame."""
        if not self.conn:
            raise ConnectionError("Немає активного підключення до бази даних.")
        try:
            return self.conn.execute(query, params).fetchdf()
        except Exception as e:
            print(f"❌ Помилка отримання даних: {query}\n{e}")
            raise
            
    def table_exists(self, table_name: str) -> bool:
        """Перевіряє існування таблиці."""
        query = "SELECT 1 FROM information_schema.tables WHERE table_name = ?"
        result = self.fetch_df(query, (table_name,))
        return not result.empty

    def create_table(self, table_name: str, columns: Dict[str, str], unique_columns: List[str] = None):
        """
        Створює таблицю, якщо вона не існує.
        Без автоінкрементів (IDENTITY), сумісно з будь-якою версією DuckDB.
        """
        if not self.conn:
            raise ConnectionError("Немає активного підключення до бази даних.")

        # якщо вже є — просто вийти
        if self.table_exists(table_name):
            print(f"ℹ️ Таблиця '{table_name}' вже існує.")
            return

        unique_columns = unique_columns or []

        # формуємо визначення колонок як є, без підмін 'IDENTITY/GENERATED'
        col_defs = [f"{name} {dtype}" for name, dtype in columns.items()]

        # додаємо унікальні обмеження, якщо треба
        constraints = []
        if unique_columns:
            constraints.append(f"UNIQUE ({', '.join(unique_columns)})")

        sql = f"CREATE TABLE {table_name} ({', '.join(col_defs + constraints)});"
        self.execute(sql)
        print(f"✅ Таблицю '{table_name}' створено.")


    def insert_data(self, table_name: str, data: Dict[str, Any]):
        """Вставляє один рядок даних у таблицю."""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
        self.execute(query, tuple(data.values()))

    def update_data(self, table_name: str, data: Dict[str, Any], where_conditions: Dict[str, Any]):
        """Оновлює дані в таблиці."""
        set_clause = ', '.join([f"{col} = ?" for col in data.keys()])
        where_clause = ' AND '.join([f"{col} = ?" for col in where_conditions.keys()])
        
        values = list(data.values()) + list(where_conditions.values())
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause};"
        self.execute(query, tuple(values))

    def fetch_data_df(self, table_name: str, where_conditions: Dict = None, limit: int = None, order_by: str = None) -> pd.DataFrame:
        """Отримує дані з таблиці у вигляді DataFrame."""
        query = f"SELECT * FROM {table_name}"
        params = []

        if where_conditions:
            conditions = ' AND '.join([f"{col} = ?" for col in where_conditions.keys()])
            query += f" WHERE {conditions}"
            params.extend(where_conditions.values())

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit is not None:
            query += f" LIMIT {limit}"

        return self.fetch_df(query, tuple(params))

    def clear_all_tables(self):
        """Очищає всі таблиці в базі даних."""
        tables_df = self.fetch_df("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
        for table_name in tables_df['table_name']:
            self.execute(f"DELETE FROM {table_name};")
            print(f"🗑️ Таблицю '{table_name}' очищено.")
            
    def __del__(self):
        self.disconnect()