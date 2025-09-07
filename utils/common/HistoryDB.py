# utils/common/HistoryDB.py

import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Сумісно з твоєю інфраструктурою
from utils.common.Database import DatabaseManager

# Акуратно імпортуємо get_project_root; якщо немає — fallback на поточну директорію
try:
    from utils.common.other_utils import get_project_root
except Exception:
    def get_project_root() -> str:
        return os.getcwd()


# ---------- helpers ----------

def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "t", "yes", "y")
    if isinstance(v, (int, float)):
        return bool(v)
    return None

def _ms_to_ts(ms: Optional[int]):
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)
    except Exception:
        return None

def _to_ts(v):
    """Приводить до datetime з tz=UTC. Підтримує datetime з TZ, epoch ms/s, ISO-рядок."""
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        # якщо без TZ — вважаємо UTC
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        # epoch ms
        if isinstance(v, (int, float)) and v > 10_000_000_000:
            return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc)
        # epoch s
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        # ISO
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


class HistoryDB:
    """
    Керує базою даних для зберігання історії угод та екзек'юшнів, використовуючи DuckDB.
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

    # ---------- Schema ----------

    def _create_tables(self):
        """Створює таблиці з типами даних DuckDB, якщо вони не існують."""

        # РОЗШИРЕНА схема trades (повний набір ключів для відкритих і закритих угод)
        trades_columns = {
            'id': 'UBIGINT',                    # автоінкремент
            'trade_uuid': 'VARCHAR',            # локальний ключ (UUID), унікальний
            # ідентифікація
            'exchange': 'VARCHAR',
            'account': 'VARCHAR',
            'symbol': 'VARCHAR',
            'category': 'VARCHAR',              # linear, inverse, option...
            'margin_mode': 'VARCHAR',           # cross / isolated
            # напрям/стан
            'direction': 'VARCHAR',             # long/short
            'side': 'VARCHAR',                  # Buy/Sell (за бажанням)
            'status': 'VARCHAR',                # open/closed
            # entry (вхід)
            'entry_time': 'TIMESTAMPTZ',
            'entry_price': 'DOUBLE',
            'entry_qty': 'DOUBLE',
            'entry_value': 'DOUBLE',
            'entry_fee': 'DOUBLE',
            'entry_exec_ids': 'VARCHAR',        # JSON-рядок з масивом execId
            # exit (вихід, може бути None поки угода відкрита)
            'exit_time': 'TIMESTAMPTZ',
            'exit_price': 'DOUBLE',
            'exit_qty': 'DOUBLE',
            'exit_value': 'DOUBLE',
            'exit_fee': 'DOUBLE',
            'exit_exec_ids': 'VARCHAR',         # JSON-рядок з масивом execId
            # розмір/керування ризиком
            'size': 'DOUBLE',
            'stop_loss': 'DOUBLE',
            'take_profit': 'DOUBLE',
            'tp_order_id': 'VARCHAR',
            'sl_order_id': 'VARCHAR',
            # зв'язок з біржею
            'exchange_order_id': 'VARCHAR',     # головний orderId
            # результати
            'profit_loss': 'DOUBLE',
            'realized_pnl': 'DOUBLE',
            'unrealized_pnl': 'DOUBLE',
            'fees_total': 'DOUBLE',
            'exit_reason': 'VARCHAR',
            # метадані
            'strategy': 'VARCHAR',
            'timeframe': 'VARCHAR',
            'notes': 'VARCHAR',
            'created_at': 'TIMESTAMPTZ',
            'updated_at': 'TIMESTAMPTZ',
            'leverage': 'INTEGER'
        }
        self.db.create_table('trades', trades_columns, unique_columns=['trade_uuid'])

        # Налаштування до угод
        settings_columns = {
            'id': 'UBIGINT',
            'trade_uuid': 'VARCHAR',
            'settings_json': 'VARCHAR'
        }
        self.db.create_table('trade_settings', settings_columns)

        # Екзек'юшни (Bybit /v5/execution/list -> result.list[])
        executions_columns = {
            'id': 'UBIGINT',  # автоінкремент
            'exec_id': 'VARCHAR',  # унікальний ключ від біржі
            'symbol': 'VARCHAR',
            'order_type': 'VARCHAR',
            'underlying_price': 'VARCHAR',
            'order_link_id': 'VARCHAR',
            'order_id': 'VARCHAR',
            'stop_order_type': 'VARCHAR',
            'exec_time_ms': 'UBIGINT',
            'exec_time': 'TIMESTAMPTZ',
            'fee_currency': 'VARCHAR',
            'create_type': 'VARCHAR',
            'exec_fee_v2': 'VARCHAR',
            'fee_rate': 'DOUBLE',
            'trade_iv': 'VARCHAR',
            'block_trade_id': 'VARCHAR',
            'mark_price': 'DOUBLE',
            'exec_price': 'DOUBLE',
            'mark_iv': 'VARCHAR',
            'order_qty': 'DOUBLE',
            'order_price': 'DOUBLE',
            'exec_value': 'DOUBLE',
            'closed_size': 'DOUBLE',
            'exec_type': 'VARCHAR',
            'seq': 'UBIGINT',
            'side': 'VARCHAR',
            'index_price': 'DOUBLE',
            'leaves_qty': 'DOUBLE',
            'is_maker': 'BOOLEAN',
            'exec_fee': 'DOUBLE',
            'market_unit': 'VARCHAR',
            'exec_qty': 'DOUBLE',
            'extra_fees': 'VARCHAR',
            'category': 'VARCHAR'
        }
        self.db.create_table('executions', executions_columns, unique_columns=['exec_id'])

    # ---------- Trade API ----------

    def record_trade(self, trade_data: Dict[str, Any], settings_data: Optional[Dict[str, Any]] = None,
                     upsert_on: Optional[str] = None) -> str:
        """
        Запис або апдейт угоди (повний набір ключів). Підтримує open і closed.
        - trade_data: будь-які з полів таблиці `trades` (див. _create_tables)
        - settings_data: довільний dict, збережеться як JSON у trade_settings
        - upsert_on: якщо 'trade_uuid' або 'exchange_order_id' — виконується upsert по відповідному ключу.
        Повертає trade_uuid.
        """
        now = datetime.now(tz=timezone.utc)
        trade = dict(trade_data or {})

        # Нормалізація типів
        trade['entry_time'] = _to_ts(trade.get('entry_time'))
        trade['exit_time'] = _to_ts(trade.get('exit_time'))
        for f in ('entry_price','entry_qty','entry_value','entry_fee','exit_price','exit_qty','exit_value','exit_fee',
                  'size','stop_loss','take_profit','profit_loss','realized_pnl','unrealized_pnl','fees_total'):
            if f in trade:
                trade[f] = _to_float(trade.get(f))
        if 'leverage' in trade:
            trade['leverage'] = _to_int(trade.get('leverage'))

        # Масиви exec_ids збережемо як JSON-рядок
        for f in ('entry_exec_ids','exit_exec_ids'):
            if f in trade and isinstance(trade[f], (list, tuple)):
                trade[f] = json.dumps(list(trade[f]), ensure_ascii=False)

        # trade_uuid
        trade_uuid = trade.get('trade_uuid') or str(uuid.uuid4())
        trade['trade_uuid'] = trade_uuid

        # timestamps
        trade.setdefault('created_at', now)
        trade['updated_at'] = now

        # Визначаємо upsert-режим
        do_upsert = upsert_on in ('trade_uuid', 'exchange_order_id')
        where = None
        if do_upsert:
            key_val = trade.get(upsert_on)
            if key_val:
                # Чи існує запис?
                df = self.db.fetch_df(f"SELECT trade_uuid FROM trades WHERE {upsert_on} = ?", (key_val,))
                if not df.empty:
                    # оновлення
                    existing_uuid = df.iloc[0]['trade_uuid']
                    where = {'trade_uuid': existing_uuid}
                    # не даємо перезаписати чужий trade_uuid іншим
                    trade['trade_uuid'] = existing_uuid

        if where:
            # UPDATE
            data_to_update = {k: v for k, v in trade.items() if k != 'id'}
            self.db.update_data('trades', data_to_update, where)
        else:
            # INSERT
            self.db.insert_data('trades', trade)

        # settings (опційно)
        if settings_data is not None:
            settings_record = {
                'trade_uuid': trade['trade_uuid'],
                'settings_json': json.dumps(settings_data, ensure_ascii=False)
            }
            # простий upsert для settings — спробуємо оновити, якщо вже є
            df = self.db.fetch_df("SELECT 1 FROM trade_settings WHERE trade_uuid = ?", (trade['trade_uuid'],))
            if df.empty:
                self.db.insert_data('trade_settings', settings_record)
            else:
                self.db.update_data('trade_settings', settings_record, {'trade_uuid': trade['trade_uuid']})

        return trade['trade_uuid']

    def get_all_trades(self, limit: int = 100) -> List[dict]:
        """Отримує всі угоди, відсортовані за часом входу."""
        df = self.db.fetch_data_df('trades', order_by='entry_time DESC', limit=limit)
        return df.to_dict('records')

    def get_all_open_trades(self, limit: int = 200) -> List[dict]:
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
            try:
                return json.loads(df.iloc[0]['settings_json'])
            except Exception:
                return None
        return None

    def update_trade_exit(
        self,
        trade_uuid: str,
        exit_time=None,
        exit_price: Optional[float] = None,
        profit_loss: Optional[float] = None,
        exit_reason: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Оновлює запис про угоду при її закритті.
        Можна передати додаткові поля у extra_fields (наприклад exit_qty/exit_value/exit_fee/realized_pnl/fees_total).
        """
        update_data: Dict[str, Any] = {
            'status': 'closed',
            'exit_time': _to_ts(exit_time),
            'exit_price': _to_float(exit_price) if exit_price is not None else None,
            'profit_loss': _to_float(profit_loss) if profit_loss is not None else None,
            'exit_reason': exit_reason,
            'updated_at': datetime.now(tz=timezone.utc)
        }
        if extra_fields:
            # нормалізація базових числових
            for f in ('exit_qty','exit_value','exit_fee','realized_pnl','fees_total','unrealized_pnl','take_profit','stop_loss'):
                if f in extra_fields:
                    update_data[f] = _to_float(extra_fields.get(f))
            # exec_ids масиви
            for f in ('exit_exec_ids',):
                if f in extra_fields:
                    v = extra_fields[f]
                    if isinstance(v, (list, tuple)):
                        v = json.dumps(list(v), ensure_ascii=False)
                    update_data[f] = v
            # будь-які інші, як є
            for k, v in extra_fields.items():
                update_data.setdefault(k, v)

        where_conditions = {'trade_uuid': trade_uuid}
        self.db.update_data('trades', update_data, where_conditions)

    # ---------- Executions API ----------

    @staticmethod
    def _map_execution_row(item: Dict[str, Any], category: Optional[str]) -> Dict[str, Any]:
        """Нормалізує елемент result.list[] у рядок для таблиці executions."""
        exec_time_ms = _to_int(item.get('execTime'))
        row = {
            'exec_id': item.get('execId'),
            'symbol': item.get('symbol'),
            'order_type': item.get('orderType'),
            'underlying_price': item.get('underlyingPrice'),
            'order_link_id': item.get('orderLinkId'),
            'order_id': item.get('orderId'),
            'stop_order_type': item.get('stopOrderType'),
            'exec_time_ms': exec_time_ms,
            'exec_time': _ms_to_ts(exec_time_ms),
            'fee_currency': item.get('feeCurrency'),
            'create_type': item.get('createType'),
            'exec_fee_v2': item.get('execFeeV2'),
            'fee_rate': _to_float(item.get('feeRate')),
            'trade_iv': item.get('tradeIv'),
            'block_trade_id': item.get('blockTradeId'),
            'mark_price': _to_float(item.get('markPrice')),
            'exec_price': _to_float(item.get('execPrice')),
            'mark_iv': item.get('markIv'),
            'order_qty': _to_float(item.get('orderQty')),
            'order_price': _to_float(item.get('orderPrice')),
            'exec_value': _to_float(item.get('execValue')),
            'closed_size': _to_float(item.get('closedSize')),
            'exec_type': item.get('execType'),
            'seq': _to_int(item.get('seq')),
            'side': item.get('side'),
            'index_price': _to_float(item.get('indexPrice')),
            'leaves_qty': _to_float(item.get('leavesQty')),
            'is_maker': _to_bool(item.get('isMaker')),
            'exec_fee': _to_float(item.get('execFee')),
            'market_unit': item.get('marketUnit'),
            'exec_qty': _to_float(item.get('execQty')),
            'extra_fees': item.get('extraFees'),
            'category': category,
        }
        return row

    def save_execution(self, item: Dict[str, Any], category: Optional[str] = None) -> bool:
        """
        Зберігає один екзек'юшн. Повертає True якщо вставлено, False якщо вже існує exec_id.
        """
        row = self._map_execution_row(item, category)
        if not row.get('exec_id'):
            return False

        # Уникаємо дублікатів
        exists_df = self.db.fetch_df("SELECT 1 FROM executions WHERE exec_id = ?", (row['exec_id'],))
        if not exists_df.empty:
            return False

        self.db.insert_data('executions', row)
        return True

    def save_executions(self, items: List[Dict[str, Any]], category: Optional[str] = None) -> int:
        """
        Зберігає список екзек'юшнів. Повертає кількість нових вставок.
        """
        inserted = 0
        for it in items or []:
            try:
                if self.save_execution(it, category=category):
                    inserted += 1
            except Exception as e:
                print(f"⚠️ Не вдалось зберегти execution execId={it.get('execId')}: {e}")
        if inserted:
            print(f"✅ Збережено нових екзек'юшнів: {inserted}")
        else:
            print("ℹ️ Нових екзек'юшнів не знайдено.")
        return inserted

    def save_executions_from_bybit_response(self, api_json: Dict[str, Any]) -> int:
        """
        Приймає сирий JSON з /v5/execution/list і зберігає result.list[].
        """
        result = (api_json or {}).get('result') or {}
        category = result.get('category')
        items = result.get('list') or []
        return self.save_executions(items, category=category)

    def get_executions(self, symbol: Optional[str] = None, limit: int = 200) -> List[dict]:
        """
        Повертає останні екзек'юшни (опціонально по символу).
        """
        where = {'symbol': symbol} if symbol else None
        df = self.db.fetch_data_df('executions', where_conditions=where, order_by='exec_time DESC', limit=limit)
        return df.to_dict('records')
