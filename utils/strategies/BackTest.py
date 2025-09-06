# file: NNUpdates/Backtesting/BackTest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import uuid

# --- Імпорти з вашого проєкту ---
# <--- ЗМІНА: Додано DecisionSettingsManager для завантаження дефолтних налаштувань
from utils.common.SettingsLoader import SettingsLoader, DecisionSettingsManager, FinancialSettingsManager, ProcessingSettingsBuilder
from gui.JobWindow import SignalConfigLoader, SignalProcessor, ResultFormatter 
from utils.common.FileStructureManager import FileStructureManager
from utils.nn.NeuralNetworkManager import NeuralNetworkManager

# --- Допоміжні класи (Portfolio, логер) залишаються без змін ---
class Portfolio:
    """Керує капіталом, позиціями та історією вартості портфеля."""
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {} # {trade_id: position_dict}
        self.history = [initial_capital]
        self.total_value = initial_capital

    def update_value(self, current_price: float):
        unrealized_pnl = 0
        for pos in self.positions.values():
            if pos['direction'] == 'buy':
                unrealized_pnl += (current_price - pos['entry_price']) * pos['size']
            elif pos['direction'] == 'sell':
                unrealized_pnl += (pos['entry_price'] - current_price) * pos['size']
        self.total_value = self.cash + unrealized_pnl
        self.history.append(self.total_value)

    def open_trade(self, direction: str, price: float, size: float, sl: float, tp: float) -> str:
        trade_id = str(uuid.uuid4())[:8]
        self.positions[trade_id] = {
            'trade_id': trade_id, 'size': size, 'entry_price': price, 
            'direction': direction, 'sl': sl, 'tp': tp
        }
        return trade_id

    def close_trade(self, trade_id: str, exit_price: float) -> float:
        position_to_close = self.positions.pop(trade_id, None)
        if position_to_close:
            pnl = 0
            if position_to_close['direction'] == 'buy':
                pnl = (exit_price - position_to_close['entry_price']) * position_to_close['size']
            elif position_to_close['direction'] == 'sell':
                pnl = (position_to_close['entry_price'] - exit_price) * position_to_close['size']
            self.cash += pnl
            return pnl
        return 0

def setup_backtest_logger(log_dir="logs/backtests", strategy_name="backtest"):
    """Налаштовує логер для запису процесу бектестування у файл."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"backtest_{strategy_name}_{timestamp}.log")
    logger = logging.getLogger(f"backtest_{strategy_name}_{timestamp}")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# --- Клас для конфігурації бектесту (без змін) ---
class BacktestConfigLoader:
    def __init__(self, mode: str = 'gui'):
        if mode not in ['gui', 'default']:
            raise ValueError("Режим має бути 'gui' або 'default'.")
        self.mode = mode
        self.fsm = FileStructureManager()

    def load(self) -> dict:
        print(f"Завантаження конфігурації бектесту в режимі: '{self.mode.upper()}'")
        if self.mode == 'gui':
            return self._load_from_gui()
        else:
            return self._load_from_defaults()

    def _load_from_gui(self) -> dict:
        gui_loader = SettingsLoader("GUI")
        return gui_loader.get_nested_setting(['user_selections'], {})

    def _load_from_defaults(self) -> dict:
        decision_mgr = DecisionSettingsManager()
        financial_mgr = FinancialSettingsManager()
        
        config = {
            'selected_pairs': ['BTC/USDT'],
            'selected_timeframes': list(decision_mgr.get_timeframe_weights().keys()),
            'tf_configurations': {tf: self._create_default_tf_config() for tf in decision_mgr.get_timeframe_weights().keys()},
            'selected_timeframe_weights': {tf: {'weight': w} for tf, w in decision_mgr.get_timeframe_weights().items()},
            'selected_metric_weights': {m: {'weight': w} for m, w in decision_mgr.get_metric_weights().items()},
            'financial_settings': financial_mgr.get_financial_settings(),
            'selected_strategies': {name: {'enabled': True, 'weight': 1.0} for name in self.fsm.get_all_strategy_names()}
        }
        return config

    def _create_default_tf_config(self) -> dict:
        proc_builder = ProcessingSettingsBuilder()
        return {
            'selected_indicators': proc_builder.get_indicator_settings(),
            'selected_patterns': proc_builder.get_pattern_settings(),
            'selected_algorithms': proc_builder.get_algorithm_settings()
        }

# --- Основний клас бектестера (З ВИПРАВЛЕННЯМ) ---
class Backtester:
    def __init__(self, initial_capital: float, nn_model_path: str = None):
        self.initial_capital = initial_capital
        self.nn_model_path = nn_model_path
        self.portfolio = Portfolio(self.initial_capital)
        self.trades = []
        self.logger = setup_backtest_logger()
        self.formatter = ResultFormatter()

    def _prepare_data(self, tf_data_paths: dict, limit_rows=None) -> tuple:
        self.logger.info("1. Синхронізація даних...")
        dfs = {}
        for tf, path in tf_data_paths.items():
            try:
                df = pd.read_csv(path, parse_dates=['close_time'])
                if limit_rows: df = df.tail(limit_rows)
                df = df.set_index('close_time').add_suffix(f'_{tf}')
                dfs[tf] = df
            except Exception as e:
                self.logger.error(f"Не вдалося завантажити дані для {tf}: {e}")
                return None, None, None
        
        master_df = pd.concat(dfs.values(), axis=1).ffill().dropna().reset_index()
        
        raw_data_by_tf, processed_data_by_tf = {}, {}
        for tf, path in tf_data_paths.items():
            df = pd.read_csv(path, parse_dates=['close_time'])
            if limit_rows: df = df.tail(limit_rows)
            raw_data_by_tf[tf] = df.copy()
            processed_data_by_tf[tf] = df.copy()

        self.logger.info(f"✅ Дані синхронізовано. Всього кроків для тестування: {len(master_df)}")
        return master_df, raw_data_by_tf, processed_data_by_tf

    def run(self, config: dict, tf_data_paths: dict, primary_tf: str, use_nn: bool = False, limit_rows=None):
        master_df, raw_data, proc_data = self._prepare_data(tf_data_paths, limit_rows)
        if master_df is None: return

        live_rl_trader = None
        if use_nn:
            if not self.nn_model_path or not os.path.exists(self.nn_model_path):
                self.logger.error("❌ Шлях до моделі NN не вказано або файл не знайдено!")
                return
            nn_manager = NeuralNetworkManager(model_path=self.nn_model_path)
            live_rl_trader = nn_manager.make_live_runner(initial_equity=self.initial_capital)
            self.logger.info("🧠 Нейромережа активована для бектесту.")

        # --- ПОЧАТОК ВИПРАВЛЕННЯ ---
        # Створюємо SignalConfigLoader і вручну заповнюємо його поля
        signal_config_loader = SignalConfigLoader(SettingsLoader("GUI")) # Dummy loader
        signal_config_loader.pairs = config.get('selected_pairs', [])
        signal_config_loader.active_timeframes = config.get('selected_timeframes', [])
        signal_config_loader.strategy_configs = config.get('selected_strategies', {})
        signal_config_loader.user_financial_settings = config.get('financial_settings', {})

        # Завантажуємо дефолтні налаштування агентів, щоб уникнути KeyError
        decision_settings = DecisionSettingsManager()
        default_agent_config = decision_settings.get_agent_configurations().get("default", {})
        fast_agent_config = decision_settings.get_agent_configurations().get("fast", default_agent_config)

        # Тепер створюємо повний словник agent_configs
        signal_config_loader.agent_configs = {
            "gui": config.get('tf_configurations', {}),
            "default": default_agent_config,
            "fast": fast_agent_config
        }
        # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---
        
        self.logger.info("2. Запуск симуляції...")
        for i in range(1, len(master_df)):
            current_row = master_df.iloc[i]
            current_price = current_row.get(f'close_{primary_tf}')
            if pd.isna(current_price): continue
            
            # 1. Оновлюємо портфель та перевіряємо закриття угод
            self.portfolio.update_value(current_price)
            for trade_id, trade in list(self.portfolio.positions.items()):
                exit_price, exit_reason = None, None
                if trade['direction'] == 'buy':
                    if current_row[f'low_{primary_tf}'] <= trade['sl']: exit_price, exit_reason = trade['sl'], 'SL'
                    elif current_row[f'high_{primary_tf}'] >= trade['tp']: exit_price, exit_reason = trade['tp'], 'TP'
                elif trade['direction'] == 'sell':
                    if current_row[f'high_{primary_tf}'] >= trade['sl']: exit_price, exit_reason = trade['sl'], 'SL'
                    elif current_row[f'low_{primary_tf}'] <= trade['tp']: exit_price, exit_reason = trade['tp'], 'TP'

                if exit_price:
                    pnl = self.portfolio.close_trade(trade_id, exit_price)
                    self._update_trade_log(trade_id, current_row['close_time'], exit_price, pnl)
                    self.logger.info(f"   - ЗАКРИТО {trade['direction'].upper()} #{trade_id} по {exit_reason}. P/L: ${pnl:.2f}")

            # 2. Логіка NN (якщо активна)
            if live_rl_trader:
                nn_result = live_rl_trader.on_new_candle(close=current_price, account_equity=self.portfolio.total_value)
                if nn_result:
                    self.logger.info("🤖 NN згенерувала нові параметри!")
                    signal_config_loader.user_financial_settings.update(nn_result['settings'])

            # 3. Генеруємо новий сигнал (тільки якщо немає відкритих позицій)
            if not self.portfolio.positions:
                current_raw_slice = {tf: df[df['close_time'] <= current_row['close_time']] for tf, df in raw_data.items()}
                current_proc_slice = {tf: df[df['close_time'] <= current_row['close_time']] for tf, df in proc_data.items()}

                processor = SignalProcessor(signal_config_loader)
                result = processor.generate("backtest", "SOME/PAIR", current_raw_slice, current_proc_slice)
                
                fin_brief = result.get('financial_briefing', {})
                if fin_brief.get("status") == "ok":
                    trade_params = fin_brief.get('trade_parameters', {})
                    direction = result['technical_decision']['direction']
                    
                    trade_id = self.portfolio.open_trade(
                        direction, current_price, trade_params['position_size_units'], 
                        trade_params['stop_loss_price'], trade_params['take_profit_price']
                    )
                    self._log_new_trade(trade_id, current_row['close_time'], direction, current_price, trade_params)
                    self.logger.info(f"   - ВІДКРИТО {direction.upper()} #{trade_id} | Вхід: {current_price:.2f}")

        self.logger.info("Бектест завершено.")
        return self.calculate_metrics()

    def _log_new_trade(self, trade_id, time, direction, price, params):
        self.trades.append({
            'trade_id': trade_id, 'entry_time': time, 'direction': direction,
            'entry_price': price, 'size': params.get('position_size_units', 0),
            'sl': params.get('stop_loss_price'), 'tp': params.get('take_profit_price'),
            'status': 'open'
        })
        
    def _update_trade_log(self, trade_id, time, price, pnl):
        for t in self.trades:
            if t['trade_id'] == trade_id:
                t.update({'status': 'closed', 'exit_time': time, 'exit_price': price, 'profit_loss': pnl})
                break

    def calculate_metrics(self):
        """
        Розраховує ключові метрики ефективності на основі закритих угод.
        """
        self.logger.info("\n3. Розрахунок метрик ефективності...")
        if not self.trades:
            self.logger.warning("Угод не було зроблено. Метрики не розраховуються.")
            return {"message": "No trades were made."}

        trades_df = pd.DataFrame(self.trades)
        
        # --- ПОЧАТОК ВИПРАВЛЕННЯ ---
        # Фільтруємо ТІЛЬКИ закриті угоди для розрахунку P/L метрик
        closed_trades_df = trades_df[trades_df['status'] == 'closed'].copy()
        
        total_trades = len(closed_trades_df)
        
        # Якщо закритих угод не було, встановлюємо фінансові метрики в 0
        if total_trades == 0:
            self.logger.warning("Жодна угода не була закрита під час бектесту. Фінансові метрики будуть нульовими.")
            winning_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            total_pnl = 0.0
        else:
            # Тепер ми гарантовано працюємо з DataFrame, де є колонка 'profit_loss'
            pnls = closed_trades_df['profit_loss']
            winning_trades_df = closed_trades_df[pnls > 0]
            losing_trades_df = closed_trades_df[pnls < 0]

            winning_trades = len(winning_trades_df)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            sum_wins = winning_trades_df['profit_loss'].sum()
            sum_losses = abs(losing_trades_df['profit_loss'].sum())
            
            profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')
            avg_win = sum_wins / winning_trades if winning_trades > 0 else 0.0
            avg_loss = sum_losses / len(losing_trades_df) if not losing_trades_df.empty else 0.0
            total_pnl = pnls.sum()
        # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

        # Загальні метрики портфеля розраховуються як і раніше
        total_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        portfolio_history = pd.Series(self.portfolio.history)
        rolling_max = portfolio_history.cummax()
        drawdown = (portfolio_history - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            "Start Capital": f"${self.initial_capital:,.2f}",
            "End Capital": f"${self.portfolio.total_value:,.2f}",
            "Total PnL": f"${total_pnl:,.2f}",
            "Total Return": f"{total_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Closed Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": total_trades - winning_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Average Win": f"${avg_win:.2f}",
            "Average Loss": f"${avg_loss:.2f}"
        }
        
        self.logger.info("\n--- Результати бектесту ---")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")
            
        return metrics
