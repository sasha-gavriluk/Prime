from utils.connectors.ConnectionController import ConnectionController
import datetime

from dateutil.relativedelta import relativedelta

def get_date_in_one_month():
    """
    Повертає об'єкт datetime, що представляє час рівно через один місяць від поточного.
    """
    # Отримуємо поточний час
    now = datetime.now()
    
    # Додаємо рівно один місяць
    future_date = now + relativedelta(months=1)
    
    return future_date

def test_connection_controller():
    cc = ConnectionController()

    # Тест підключення до Binance
    # assert cc.connect('Binance') == True, "Failed to connect to Binance"
    # binance_api = cc.get_api('Binance')
    # assert binance_api is not None, "Failed to get Binance API instance"
    # assert binance_api.is_connected() == True, "Binance API should be connected"

    # Тест підключення до Bybit
    assert cc.connect('Bybit') == True, "Failed to connect to Bybit"
    bybit_api = cc.get_api('Bybit')
    assert bybit_api is not None, "Failed to get Bybit API instance"
    assert bybit_api.is_connected() == True, "Bybit API should be connected"

    # Тест отримання підключених бірж
    connected_exchanges = cc.get_connected_exchanges()
    # assert 'Binance' in connected_exchanges, "Binance should be in connected exchanges"
    assert 'Bybit' in connected_exchanges, "Bybit should be in connected exchanges"

    # Тест оновлення спільних USDT пар
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT', 'DOGE/USDT', 'SOL/USDT', 'BNB/USDT']
    
    # --- ПОЧАТОК ЗМІН ---
    # Встановлюємо дату, з якої шукати угоди (наприклад, 1 серпня 2025 року)
    since_date = datetime.datetime(2025, 8, 1, 0, 0, 0)
    # Конвертуємо її в timestamp мілісекунди
    since_timestamp_ms = int(since_date.timestamp() * 1000)
    
    # Передаємо 'since' через словник 'params'
    history_params = {'since': since_timestamp_ms}
    history = cc.get_trade_history('Bybit', symbols=symbols, params=history_params, total_limit=500)
    # --- КІНЕЦЬ ЗМІН ---

    assert isinstance(history, list), "Updated pairs should be a list"

    # Тепер список не повинен бути порожнім, якщо угоди були після 1 серпня
    print(f"Знайдено {len(history)} угод.")
    print(history)
    assert len(history) > 0, "Trade history should not be empty if trades exist in the specified period"


    # Тест відключення від Binance
    # cc.disconnect('Binance')
    # assert not binance_api.is_connected(), "Binance API should be disconnected"

    # Тест відключення від Bybit
    cc.disconnect('Bybit')
    assert not bybit_api.is_connected(), "Bybit API should be disconnected"

if __name__ == "__main__":
    test_connection_controller()
    print("All tests passed!")