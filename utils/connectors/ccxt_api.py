# utils/connectors/ccxt_api.py

import ccxt
import json
import os
import pandas as pd
import time

from utils.common.FileStructureManager import FileStructureManager

class CCXTExchangeAPI:
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name.lower()
        self.fsm = FileStructureManager()

        # Завантаження ключів
        self.keys_file = self.fsm.get_path("Settings_keys", is_file=True)
        if not self.keys_file:
            raise FileNotFoundError("Файл 'Settings_keys' не знайдено.")

        self.api_key = None
        self.api_secret = None
        self.client = None
        self.load_keys()

        # Папка збереження
        folder_key = f"data_{self.exchange_name.capitalize()}"
        data_base_dir = os.path.join(self.fsm.root_path, 'data', self.exchange_name.capitalize())

        if not os.path.exists(data_base_dir):
            print(f"[{self.exchange_name.upper()}] 📁 Створюємо папку: {data_base_dir}")
            os.makedirs(data_base_dir, exist_ok=True)
            self.fsm.scan_and_save_structure()

        self.data_base_dir = self.fsm.get_path(folder_key, is_file=False)

        if not self.data_base_dir:
            raise FileNotFoundError(f"Папка '{folder_key}' не знайдена навіть після створення.")

    def load_keys(self):
        with open(self.keys_file, 'r') as file:
            keys = json.load(file)
            exchange_keys = keys.get(self.exchange_name, {})
            self.api_key = exchange_keys.get("api_key")
            self.api_secret = exchange_keys.get("api_secret")
            if not self.api_key or not self.api_secret:
                raise ValueError(f"Ключі для '{self.exchange_name}' відсутні або неповні.")

    def connect(self):
        try:
            # нормалізуємо біржу під ф’ючерси
            exid = self.exchange_name
            if exid in ("binance", "binanceusdm", "binance-futures", "binance_futures"):
                exid = "binanceusdm"  # USDⓈ-M ф’ючерси
            opts = {}
            if exid == "bybit":
                # Bybit: працюємо зі свопами (ф’ючерси)
                opts = {"defaultType": "swap"}

            self.client = getattr(ccxt, exid)({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": opts,
            })
            self.client.load_markets()
            self.ccxt_id = exid  # збережемо фактичний id клієнта
            print(f"[{self.exchange_name.upper()}] ✅ Підключено як {self.ccxt_id}.")
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ❌ Помилка підключення: {e}")
            self.client = None


    def is_connected(self) -> bool:
        return self.client is not None

    def _get_market_symbol(self, symbol: str, params={}):
        """
        Конвертує стандартний символ (напр. 'ETH/USDT') у специфічний для біржі,
        якщо це необхідно (напр. для ф'ючерсів Bybit).
        """
        if self.exchange_name == 'bybit' and params.get('category') == 'linear':
            if ':' not in symbol:
                return f"{symbol}:USDT"
        return symbol

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['close_time', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ fetch_ohlcv помилка: {e}")
            return None

    def save_ohlcv(self, df: pd.DataFrame, timeframe: str, symbol: str):
        symbol_folder = symbol.replace("/", "_")
        save_path = os.path.join(self.data_base_dir, timeframe, symbol_folder)
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{symbol_folder}.csv")
        if 'close_time' not in df.columns:
            raise ValueError("DataFrame повинен містити колонку 'close_time'.")
        df = df.copy()
        df.loc[:, 'close_time'] = pd.to_datetime(df['close_time'], errors='coerce')
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if 'close_time' in existing_df.columns:
                existing_df['close_time'] = pd.to_datetime(existing_df['close_time'], errors='coerce')
                existing_df.loc[:, 'close_time'] = pd.to_datetime(existing_df['close_time'], errors='coerce')
        df.to_csv(file_path, index=False)
        print(f"[{self.exchange_name.upper()}] 💾 Збережено: {file_path}")

    def fetch_all_symbols(self):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        try:
            if not self.client.markets:
                self.client.load_markets()
            return list(self.client.markets.keys())
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ fetch_all_symbols помилка: {e}")
            return []

    def fetch_balance(self):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        try:
            if self.exchange_name == 'bybit':
                return self.client.fetch_balance(params={'accountType': 'UNIFIED'})
            else:
                return self.client.fetch_balance()
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ fetch_balance помилка: {e}")
            return {}
            
    def get_usdt_balance(self) -> float:
        if not self.is_connected():
            return 0.0
        try:
            balance_data = self.fetch_balance()
            usdt_info = balance_data.get('USDT', {})
            free_balance = usdt_info.get('free')
            return float(free_balance or 0.0)

        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання USDT балансу: {e}")
            return 0.0

    def update_symbol_data(self, symbol: str, standard_timeframes: list = None):
        for timeframe in standard_timeframes:
            try:
                print(f"[{self.exchange_name.upper()}] 🔄 Оновлення {symbol} {timeframe}...")
                symbol_folder = symbol.replace("/", "_")
                save_path = os.path.join(self.data_base_dir, timeframe, symbol_folder)
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, f"{symbol_folder}.csv")
                from_timestamp = None
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path)
                    if 'close_time' in existing_df.columns and not existing_df.empty:
                        last_time = pd.to_datetime(existing_df['close_time'].max())
                        from_timestamp = int(last_time.timestamp() * 1000)
                if from_timestamp:
                    new_ohlcv = self.client.fetch_ohlcv(symbol, timeframe, since=from_timestamp, limit=1000)
                else:
                    new_ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=1000)
                if new_ohlcv:
                    new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    new_df['close_time'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    self.save_ohlcv(new_df[['close_time', 'open', 'high', 'low', 'close', 'volume']], timeframe, symbol)
                else:
                    print(f"[{self.exchange_name.upper()}] ℹ️ Нових даних немає.")
            except Exception as e:
                print(f"[{self.exchange_name.upper()}] ⚠️ {symbol} {timeframe}: {e}")

    def updates_symbols_data(self, symbols: list, standard_timeframes: list):
        for idx, symbol in enumerate(symbols, 1):
            print(f"[{self.exchange_name.upper()}] [{idx}/{len(symbols)}] {symbol}")
            self.update_symbol_data(symbol, standard_timeframes)

    def set_leverage(self, symbol: str, leverage: int, params={}):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        if not self.client.has.get('setLeverage'):
            print(f"[{self.exchange_name.upper()}] ⚠️ Біржа не підтримує встановлення плеча через API.")
            return
        try:
            market_symbol = self._get_market_symbol(symbol, params)
            self.client.set_leverage(leverage, market_symbol, params)
            print(f"[{self.exchange_name.upper()}] ✅ Встановлено плече {leverage}x для {market_symbol}.")
        except Exception as e:
            error_message = str(e)
            if "leverage not modified" in error_message or "110043" in error_message:
                print(f"[{self.exchange_name.upper()}] ℹ️ Плече для {market_symbol} вже встановлено на {leverage}x. Продовжуємо.")
            else:
                print(f"[{self.exchange_name.upper()}] ❌ Справжня помилка встановлення плеча для {symbol}: {e}")
                raise e

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None, params={}, stop_loss: float = None, take_profit: float = None, leverage: int = None):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        order_params = params.copy()
        try:
            if self.exchange_name == 'bybit':
                order_params.update({'category': 'linear'})
            
            market_symbol = self._get_market_symbol(symbol, order_params)

            if leverage is not None:
                self.set_leverage(symbol, leverage, order_params)

            if self.exchange_name == 'bybit':
                if stop_loss:
                    order_params['stopLoss'] = str(stop_loss)
                if take_profit:
                    order_params['takeProfit'] = str(take_profit)
            
            order = self.client.create_order(market_symbol, order_type, side, amount, price, order_params)
            print(f"[{self.exchange_name.upper()}] ✅ Ордер створено: {order['id']}")
            return order
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ❌ Помилка створення ордера: {e}")
            raise e

    def fetch_positions(self, symbols: list, params={}):
        """
        Отримує інформацію про відкриті позиції для списку символів одним запитом.

        :param symbols: Список стандартних символів (наприклад, ['BTC/USDT', 'ETH/USDT']).
        :return: Словник, де ключ — символ, значення — дані позиції.
        """
        if not self.is_connected() or not self.client.has.get('fetchPositions'):
            return {}
        try:
            params = params.copy()
            if self.exchange_name == 'bybit':
                params.update({'category': 'linear'})

            market_symbols = [self._get_market_symbol(s, params) for s in symbols]
            symbol_map = {m: s for s, m in zip(symbols, market_symbols)}
            positions = self.client.fetch_positions(symbols=market_symbols, params=params)

            result = {}
            for pos in positions:
                size = pos.get('size', 0) or pos.get('contracts', 0)
                if size and float(size) > 0:
                    market_symbol = pos.get('symbol')
                    std_symbol = symbol_map.get(market_symbol, market_symbol)
                    result[std_symbol] = pos
            return result
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання позицій: {e}")
            return {}

    def close_position(self, symbol: str, params={}):
        if not self.is_connected():
            raise ConnectionError("Немає підключення до біржі.")
        try:
            if self.exchange_name == 'bybit':
                params.update({'category': 'linear'})
            
            market_symbol = self._get_market_symbol(symbol, params)
            positions = self.client.fetch_positions(symbols=[market_symbol], params=params)
            open_positions = [p for p in positions if p.get('contracts') is not None and float(p.get('contracts', 0)) > 0]
            if not open_positions:
                print(f"[{self.exchange_name.upper()}] ℹ️ Не знайдено відкритих позицій для {symbol}.")
                return None
            position_to_close = open_positions[0]
            amount = float(position_to_close['contracts'])
            side = 'sell' if position_to_close['side'] == 'long' else 'buy'
            print(f"[{self.exchange_name.upper()}] 🔄 Закриття {position_to_close['side']} позиції {symbol} розміром {amount}...")
            close_order = self.create_order(symbol, 'market', side, amount, params=params)
            return close_order
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ❌ Помилка закриття позиції: {e}")
            raise e
        
    def fetch_open_orders(self, symbol: str = None, since: int = None, limit: int = None, params: dict = None):
        """
        Відкриті ордери (ф’ючерси).
        - Bybit: за замовчуванням category='linear'
        - Якщо біржа вимагає symbol, а його не дали — зробимо обхід по всіх ринках.
        """
        if not self.is_connected() or not self.client.has.get('fetchOpenOrders'):
            return []

        params = (params or {}).copy()
        try:
            if self.exchange_name == 'bybit' and 'category' not in params:
                params['category'] = 'linear'

            if symbol:
                market_symbol = self._get_market_symbol(symbol, params)
                return self.client.fetch_open_orders(market_symbol, since=since, limit=limit, params=params)

            # деякі біржі дозволяють без символу
            try:
                return self.client.fetch_open_orders(None, since=since, limit=limit, params=params)
            except Exception:
                pass

            # фолбек: обійти всі ринки (повільніше, але надійно)
            self.client.load_markets()
            out = []
            for sym in self.client.symbols:
                ms = self._get_market_symbol(sym, params)
                try:
                    chunk = self.client.fetch_open_orders(ms, since=since, limit=limit, params=params)
                    if chunk:
                        out.extend(chunk)
                except Exception:
                    pass
                time.sleep(self.client.rateLimit / 1000.0)
            return out

        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання відкритих ордерів: {e}")
            return []
        
    def fetch_last_trades(self, symbol: str, limit: int = 50, params: dict = None, sort_ascending: bool = True):
        """
        Повертає останні N угод (fills) БЕЗ пагінації.
        - symbol: ф’ючерсний символ (наприклад, 'BTC/USDT')
        - limit: скільки останніх угод повернути (за замовчуванням 50)
        - Bybit: автоматично підставляється category='linear' якщо не задано в params
        - sort_ascending: відсортувати за часом зростаюче (True) або залишити як віддає біржа (False)
        """
        if not self.is_connected() or not self.client.has.get('fetchMyTrades'):
            print(f"[{self.exchange_name.upper()}] ⚠️ fetch_last_trades: немає підключення або метод не підтримується.")
            return []

        params = (params or {}).copy()
        try:
            # За умовчанням працюємо лише з ф’ючерсами
            if self.exchange_name == 'bybit' and 'category' not in params:
                params['category'] = 'linear'

            market_symbol = self._get_market_symbol(symbol, params)

            # --- ПОЧАТОК ЗМІН ---
            # Витягуємо 'since' з params, якщо він там є.
            # Метод .pop() отримує значення і видаляє ключ зі словника.
            since_timestamp = params.pop('since', None)
            
            # Тепер 'since' передається коректно, а 'params' не містить дублюючої інформації.
            trades = self.client.fetch_my_trades(market_symbol, since=since_timestamp, limit=limit, params=params) or []
            # --- КІНЕЦЬ ЗМІН ---
            
            if sort_ascending:
                trades.sort(key=lambda t: t.get('timestamp') or 0)

            return trades

        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання останніх {limit} угод для {symbol}: {e}")
            return []

        
    def fetch_my_trades(self, symbol: str, since: int = None, limit: int = 20, params: dict = None):
        """
        Історія угод користувача для конкретного ф’ючерсного символу.
        """
        if not self.is_connected() or not self.client.has.get('fetchMyTrades'):
            return []
        try:
            params = (params or {}).copy()
            if self.exchange_name == 'bybit' and 'category' not in params:
                params['category'] = 'linear'

            market_symbol = self._get_market_symbol(symbol, params)
            return self.client.fetch_my_trades(market_symbol, since=since, limit=limit, params=params)
        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання історії угод для {symbol}: {e}")
            return []


    def fetch_last_trades_all(self, total_limit: int = 200, per_symbol_limit: int = None, params: dict = None, symbols: list = None, only_active: bool = True, sort_ascending: bool = True):
        """
        Повертає ОСТАННІ total_limit угод по всіх ф’ючерсних символах.
        Працює швидко: бере невеликий хвіст з кожного символу, об’єднує, сортує, обрізає до N.

        - total_limit: скільки угод треба всього (по всіх символах)
        - per_symbol_limit: скільки тягнути з кожного символу (за замовч. = min(total_limit, 200))
        - params: додаткові параметри для біржі (для Bybit не потрібно — category=linear підставиться автоматично)
        - symbols: якщо задано, обходимо тільки ці символи (у форматі 'BTC/USDT', без ':USDT')
        - only_active: брати лише активні ринки
        - sort_ascending: чи відсортувати фінальний результат за часом ↑ (інакше залишити як «найновіші першими»)
        """
        if not self.is_connected() or not self.client.has.get('fetchMyTrades'):
            print(f"[{self.exchange_name.upper()}] ⚠️ fetch_last_trades_all: немає підключення або метод не підтримується.")
            return []

        p = (params or {}).copy()
        try:
            # Для Bybit за замовчуванням лінійні ф’ючерси
            if self.exchange_name == 'bybit' and 'category' not in p:
                p['category'] = 'linear'

            # Список символів
            self.client.load_markets()
            markets = self.client.markets
            if symbols is None:
                symbols = []
                for m in markets.values():
                    if not m.get('contract'):
                        continue  # беремо тільки деривативи
                    if only_active and not m.get('active', True):
                        continue
                    sym = m['symbol']
                    # Для Bybit передаємо у _get_market_symbol базовий вигляд 'BASE/QUOTE'
                    if self.exchange_name == 'bybit':
                        sym = sym.split(':')[0]
                    symbols.append(sym)
                symbols = list(dict.fromkeys(symbols))  # унікальні, збереження порядку

            # Скільки брати з кожного символу
            per_symbol_limit = per_symbol_limit or min(total_limit, 200)

            all_trades = []
            seen = set()  # для дедуплікації (id, symbol)
            for sym in symbols:
                # Використовуємо вже доданий «швидкий» метод по одному символу
                # (він сам підставляє Bybit linear і викликає _get_market_symbol)
                chunk = self.fetch_last_trades(sym, limit=per_symbol_limit, params=p, sort_ascending=False)
                for t in (chunk or []):
                    key = (t.get('id'), t.get('symbol'))
                    if key[0] is not None and key not in seen:
                        seen.add(key)
                        all_trades.append(t)
                    elif key[0] is None:
                        # якщо немає id — все одно додаємо (рідкісні випадки)
                        all_trades.append(t)

                # поважаємо rateLimit
                try:
                    time.sleep(self.client.rateLimit / 1000.0)
                except Exception:
                    pass

            # Глобальне сортування за часом ↓, обрізання до N
            all_trades.sort(key=lambda t: t.get('timestamp') or 0, reverse=True)
            out = all_trades[:max(0, int(total_limit))]

            if sort_ascending:
                out.sort(key=lambda t: t.get('timestamp') or 0)

            return out

        except Exception as e:
            print(f"[{self.exchange_name.upper()}] ⚠️ Помилка отримання останніх {total_limit} угод по всіх символах: {e}")
            return []

    def disconnect(self):
        self.client = None
        print(f"[{self.exchange_name.upper()}] 🔌 Відключено вручну.")