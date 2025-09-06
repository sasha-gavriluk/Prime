# utils/analysis/TradeMonitor.py

import threading
import time
import datetime
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from utils.common.HistoryDB import HistoryDB
from utils.connectors.ConnectionController import ConnectionController

class TradeMonitor:
    """
    Клас для моніторингу угод з пакетним отриманням цін та синхронізацією з біржею.
    """

    def __init__(self, connection_controller: ConnectionController, history_db: HistoryDB, check_interval_seconds: int = 30):
        self.connection_controller = connection_controller
        self.history_db = history_db
        self.check_interval = check_interval_seconds
        
        self._stop_event = threading.Event()
        self._thread = None
        self.log_callback = None

    def set_logger(self, log_callback):
        self.log_callback = log_callback

    def _log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        if self.log_callback:
            # Викликаємо логер в головному потоці Kivy
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self.log_callback(full_message))

    def start(self):
        if self._thread and self._thread.is_alive():
            self._log("⚠️ Моніторинг вже запущено.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._log("✅ Моніторинг угод запущено (з оптимізацією та синхронізацією).")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._log("🛑 Моніторинг угод зупинено.")

    def _monitor_loop(self):
        """Основний цикл, що перевіряє угоди."""
        while not self._stop_event.is_set():
            try:
                open_trades = self.history_db.get_all_open_trades()
                if not open_trades:
                    # Чекаємо наступну ітерацію, але дозволяємо перервати очікування
                    if self._stop_event.wait(self.check_interval):
                        break
                    continue

                self._log(f"🔍 Перевірка {len(open_trades)} відкритих угод...")
                
                # Групуємо угоди за біржею для пакетних запитів
                trades_by_exchange = defaultdict(list)
                for trade in open_trades:
                    trades_by_exchange[trade['exchange']].append(trade)

                # Обробляємо кожну біржу в окремому потоці, щоб не блокувати основний цикл
                with ThreadPoolExecutor(max_workers=len(trades_by_exchange)) as executor:
                    futures = [executor.submit(self._process_exchange_trades, ex, trs)
                               for ex, trs in trades_by_exchange.items()]
                    for future in futures:
                        if self._stop_event.is_set():
                            break
                        # Перекидаємо виключення, якщо вони виникли в робочих потоках
                        future.result()

            except Exception as e:
                self._log(f"❌ КРИТИЧНА ПОМИЛКА в циклі моніторингу: {e}")
            
            if self._stop_event.wait(self.check_interval):
                break

    def _process_exchange_trades(self, exchange_name: str, trades: List[Dict]):
        """Обробляє всі угоди для однієї біржі."""
        api = self.connection_controller.get_api(exchange_name)
        if not api or not api.is_connected(): return

        symbols = list(set(t['symbol'] for t in trades))
        
        try:
            # --- КРОК 1: Пакетний запит цін (вирішення проблеми "зависання") ---
            tickers = api.client.fetch_tickers(symbols)
            positions = api.fetch_positions(symbols)
            
            for trade in trades:
                symbol = trade['symbol']
                
                # --- КРОК 2: Синхронізація стану з біржею ---
                position = positions.get(symbol)
                if position is None:
                    self._handle_closed_on_exchange(trade, api)
                    continue

                # --- КРОК 3: Моніторинг ціни, якщо позиція активна ---
                if symbol in tickers and tickers[symbol].get('last') is not None:
                    current_price = tickers[symbol]['last']
                    self._log_trade_status(trade, current_price) # Детальний лог
                    self._check_sl_tp(trade, current_price)
                else:
                    self._log(f"  - ⚠️ Не вдалося отримати актуальну ціну для {symbol} на {exchange_name}.")

        except Exception as e:
            self._log(f"❌ Помилка при обробці угод на {exchange_name}: {e}")

    def _log_trade_status(self, trade: Dict, current_price: float):
        """Форматує та виводить детальний лог для однієї угоди."""
        direction = trade['direction'].upper()
        symbol = trade['symbol']
        entry = trade['entry_price']
        sl = trade['stop_loss']
        tp = trade['take_profit']
        
        size = trade['size']

        if trade['direction'] == 'buy':
            pnl = (current_price - entry) * size
        else:
            pnl = (entry - current_price) * size

        log_msg = (
            f"  - Слідкую за {direction} {symbol}: "
            f"Вхід={entry:.4f}, SL={sl:.4f}, TP={tp:.4f} | "
            f"Поточна ціна: {current_price:.4f} | P/L: ${pnl:.2f}"
        )
        self._log(log_msg)

    def _check_sl_tp(self, trade: Dict, current_price: float):
        """Перевіряє спрацювання SL/TP."""
        exit_price, exit_reason = None, None
        if trade['direction'] == 'buy':
            if current_price <= trade['stop_loss']: exit_price, exit_reason = trade['stop_loss'], 'Stop Loss'
            elif current_price >= trade['take_profit']: exit_price, exit_reason = trade['take_profit'], 'Take Profit'
        elif trade['direction'] == 'sell':
            if current_price >= trade['stop_loss']: exit_price, exit_reason = trade['stop_loss'], 'Stop Loss'
            elif current_price <= trade['take_profit']: exit_price, exit_reason = trade['take_profit'], 'Take Profit'

        if exit_price and exit_reason:
            self._close_trade_in_db(trade, exit_price, exit_reason)

    def _handle_closed_on_exchange(self, trade: Dict, api):
        """Обробляє угоди, які вже закриті на біржі."""
        self._log(f"  - СИНХРОНІЗАЦІЯ: Позиція {trade['symbol']} закрита на біржі. Шукаю історію...")
        
        # Шукаємо в історії угод біржі, щоб знайти реальну ціну виходу
        entry_time_ms = int(datetime.datetime.strptime(trade['entry_time'], "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        exchange_trades = api.fetch_my_trades(symbol=trade['symbol'], since=entry_time_ms, limit=10)
        
        closing_trade = None
        for ex_trade in reversed(exchange_trades): # Шукаємо з кінця, бо закриття буде пізніше
            # Умова закриття: протилежний напрямок та зв'язок по ID ордера (якщо є)
            is_closing_side = (ex_trade['side'] == 'sell' and trade['direction'] == 'buy') or \
                              (ex_trade['side'] == 'buy' and trade['direction'] == 'sell')
            
            if is_closing_side:
                closing_trade = ex_trade
                break
        
        if closing_trade:
            exit_price = closing_trade['price']
            exit_time_dt = datetime.datetime.fromtimestamp(closing_trade['timestamp'] / 1000)
            exit_time_str = exit_time_dt.strftime("%Y-%m-%d %H:%M:%S")
            self._log(f"  - Знайдено реальне закриття: Ціна={exit_price}, Час={exit_time_str}")
            self._close_trade_in_db(trade, exit_price, "Closed on Exchange (Synced)", exit_time_override=exit_time_str)
        else:
            # Якщо не знайшли, закриваємо по останній відомій ціні
            self._log(f"  - ⚠️ Не вдалося знайти точну угоду закриття в історії. Закриваю по приблизній ціні.")
            ticker = api.client.fetch_ticker(trade['symbol'])
            last_price = ticker.get('last', trade['entry_price'])
            self._close_trade_in_db(trade, last_price, "Closed on Exchange (Approx.)")

    def _close_trade_in_db(self, trade: Dict, exit_price: float, reason: str, exit_time_override: str = None):
        """Розраховує P/L та оновлює запис в базі даних."""
        if trade['direction'] == 'buy':
            pnl = (exit_price - trade['entry_price']) * trade['size']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['size']

        exit_time = exit_time_override or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history_db.update_trade_exit(
            trade_uuid=trade['trade_uuid'],
            exit_time=exit_time,
            exit_price=exit_price,
            profit_loss=pnl,
            exit_reason=reason
        )
        self._log(f" ✔️ Угоду {trade['symbol']} #{trade['trade_uuid'][:8]} закрито по {reason}. P/L: ${pnl:.2f}")