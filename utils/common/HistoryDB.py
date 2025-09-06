# utils/common/HistoryDB.py

import os
import json
import uuid
import pandas as pd
from utils.common.Database import DatabaseManager
from utils.common.other_utils import get_project_root

class HistoryDB:
    """
    Керує базою даних для зберігання історії угод, використовуючи DuckDB.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HistoryDB, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            db_path = os.path.join(get_project_root(), 'data', 'history.duckdb')
            self.db = DatabaseManager(db_file=db_path)
            self._create_tables()
            self.initialized = True

    def _create_tables(self):
        """Створює таблиці з типами даних DuckDB, якщо вони не існують."""
        trades_columns = {
            'id': 'UBIGINT', # Автоінкрементний первинний ключ
            'trade_uuid': 'VARCHAR',
            'exchange': 'VARCHAR',
            'symbol': 'VARCHAR',
            'direction': 'VARCHAR',
            'status': 'VARCHAR',
            'entry_time': 'TIMESTAMPTZ', # Спеціальний тип для часу з таймзоною
            'entry_price': 'DOUBLE',
            'exit_time': 'TIMESTAMPTZ',
            'exit_price': 'DOUBLE',
            'size': 'DOUBLE',
            'stop_loss': 'DOUBLE',
            'take_profit': 'DOUBLE',
            'leverage': 'INTEGER',
            'profit_loss': 'DOUBLE',
            'exit_reason': 'VARCHAR',
            'exchange_order_id': 'VARCHAR'
        }
        self.db.create_table('trades', trades_columns, unique_columns=['trade_uuid'])

        settings_columns = {
            'id': 'UBIGINT',
            'trade_uuid': 'VARCHAR',
            'settings_json': 'VARCHAR'
        }
        self.db.create_table('trade_settings', settings_columns)

    def record_trade(self, trade_data: dict, settings_data: dict):
        """Зберігає угоду та її налаштування в базу даних."""
        trade_uuid = str(uuid.uuid4())
        trade_data['trade_uuid'] = trade_uuid
        
        self.db.insert_data('trades', trade_data)
        
        settings_record = {
            'trade_uuid': trade_uuid,
            'settings_json': json.dumps(settings_data, indent=4)
        }
        self.db.insert_data('trade_settings', settings_record)
        
        print(f"✅ Угоду {trade_uuid} та її налаштування збережено в історію.")
        return trade_uuid

    def get_all_trades(self, limit=100) -> List[dict]:
        """Отримує всі угоди, відсортовані за часом, і повертає як список словників."""
        df = self.db.fetch_data_df('trades', order_by='entry_time DESC', limit=limit)
        return df.to_dict('records')

    def get_all_open_trades(self, limit=200) -> List[dict]:
        """Отримує всі відкриті угоди."""
        df = self.db.fetch_data_df(
            'trades',
            where_conditions={'status': 'open'},
            order_by='entry_time DESC',
            limit=limit
        )
        return df.to_dict('records')

    def get_settings_for_trade(self, trade_uuid: str):
        """Отримує налаштування для конкретної угоди."""
        df = self.db.fetch_data_df('trade_settings', where_conditions={'trade_uuid': trade_uuid}, limit=1)
        if not df.empty:
            return json.loads(df.iloc[0]['settings_json'])
        return None

    def update_trade_exit(self, trade_uuid: str, exit_time, exit_price: float, profit_loss: float, exit_reason: str):
        """Оновлює запис про угоду при її закритті."""
        update_data = {
            'status': 'closed',
            'exit_time': exit_time,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'exit_reason': exit_reason
        }
        where_conditions = {'trade_uuid': trade_uuid}
        
        self.db.update_data('trades', update_data, where_conditions)
        print(f"🔄 Запис про угоду {trade_uuid} оновлено. Причина: {exit_reason}.")