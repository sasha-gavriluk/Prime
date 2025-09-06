# ConnectionController.py
from utils.connectors.ccxt_api import CCXTExchangeAPI
from utils.common.SettingsLoader import SettingsLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta

class ConnectionController:
    def __init__(self):
        self.connections = {
            'Binance': CCXTExchangeAPI('binance'),
            "Bybit": CCXTExchangeAPI('bybit')
            # у майбутньому тут можна додати 'Bybit': BybitAPI()
        }

        self.settings_GUI = SettingsLoader("GUI")

    def connect(self, exchange_name):
        api = self.connections.get(exchange_name)
        if api:
            api.connect()
            return api.is_connected()
        return False

    def get_api(self, exchange_name):
        return self.connections.get(exchange_name)
    
    def get_connected_exchanges(self):
        """
        Повертає словник тільки підключених API-клієнтів.
        """
        return {name: api for name, api in self.connections.items() if api.is_connected()}
    
    def disconnect(self, exchange_name):
        api = self.connections.get(exchange_name)
        if api and hasattr(api, "disconnect"):
            api.disconnect()

    def _merge_pairs(self, new_pairs, key='all_pairs'):
        """
        Викачує існуючі пари за ключем `key`, зливає з `new_pairs` без дублів,
        зберігає оновлений список назад у налаштуваннях.
        """
        existing = self.settings_GUI.get_nested_setting(
            ['user_selections', key], []
        ) or []
        merged = sorted(set(existing) | set(new_pairs))
        self.settings_GUI.update_nested_setting(
            ['user_selections', key],
            merged
        )
        return merged

    def update_common_usdt_pairs(self, limit: int = 100):
        """
        Знаходить пари *.USDT, які є одночасно на всіх біржах,
        бере перші `limit` за алфавітом і зберігає їх у налаштуваннях GUI.
        """
        # 1) Переконаємося, що всі API підключені
        for api in self.connections.values():
            if not api.is_connected():
                api.connect()

        # 2) Збираємо множини USDT-пар із кожної біржі
        symbol_sets = []
        for name, api in self.connections.items():
            syms = api.fetch_all_symbols() or []
            usdt_syms = {s for s in syms if s.endswith('/USDT')}
            symbol_sets.append(usdt_syms)

        # 3) Знаходимо перетин
        if not symbol_sets:
            common = set()
        else:
            common = set.intersection(*symbol_sets)

        # 4) Сортуємо і обрізаємо до limit
        top_common = sorted(common)[:limit]

        # 5) Зберігаємо в SettingsLoader під ключем ['user_selections','selected_pairs']
        self.settings_GUI.update_nested_setting(
            ['user_selections', 'pairs_top_ustd'],
            top_common
        )

        # Мерджимо з all_pairs
        merged_all = self._merge_pairs(top_common, key='all_pairs')

        print(f"✅ Збережено {len(top_common)} спільних USDT-пар:", top_common)
        print(f"🔄 all_pairs merged ({len(merged_all)} total)")
        return top_common

    def get_balances_value_usdt(self, exchange_name):
        api = self.get_api(exchange_name)
        if api:
            return api.get_usdt_balance()
        return 0

    def get_open_orders(self, exchange_name):
        api = self.get_api(exchange_name)
        if api:
            return api.fetch_open_orders()
        return []

    def get_trade_history(self, exchange_name, total_limit: int = 200, per_symbol_limit: int = None, params: dict = None, symbols: list = None, only_active: bool = True, sort_ascending: bool = True):
        api = self.get_api(exchange_name)
        if api:
            return api.fetch_last_trades_all(total_limit=total_limit, per_symbol_limit=per_symbol_limit, params=params, symbols=symbols, only_active=only_active, sort_ascending=sort_ascending)
        return []

    def get_trade_history_for_last_month(self, exchange_name, symbols: list = None, total_limit: int = 500):
        """
        Отримує історію угод за останній місяць.

        Args:
            exchange_name (str): Назва біржі.
            symbols (list, optional): Список символів для перевірки. Defaults to None.
            total_limit (int, optional): Загальний ліміт угод. Defaults to 500.

        Returns:
            list: Список угод.
        """
        # 1. Визначаємо дату місяць тому
        one_month_ago = datetime.now() - relativedelta(months=1)
        
        # 2. Конвертуємо в timestamp у мілісекундах
        since_timestamp_ms = int(one_month_ago.timestamp() * 1000)
        
        # 3. Готуємо параметри для запиту
        history_params = {'since': since_timestamp_ms}
        
        print(f"Отримання історії угод з {one_month_ago.strftime('%Y-%m-%d %H:%M:%S')}...")
        
        # 4. Викликаємо існуючий метод з новими параметрами
        return self.get_trade_history(
            exchange_name,
            symbols=symbols,
            params=history_params,
            total_limit=total_limit,
            sort_ascending=True # Зазвичай історію хочуть бачити в хронологічному порядку
        )