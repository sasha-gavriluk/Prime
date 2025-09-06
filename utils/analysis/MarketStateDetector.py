# utils/analysis/MarketStateDetector.py

import pandas as pd
from utils.common.SettingsLoader import MarketStateSettingsManager

# --- Базовий клас для всіх правил ---
class StateRule:
    """Базовий клас для правила, що визначає стан ринку."""
    def __init__(self, data: pd.DataFrame, params: dict = None):
        self.data = data
        self.params = params if params is not None else {}

    def get_confidence(self) -> float:
        """Повертає впевненість (0.0 до 1.0), що ринок у цьому стані."""
        raise NotImplementedError("Метод get_confidence() має бути реалізований у підкласі.")

    def _check_columns(self, columns: list) -> bool:
        """Перевіряє наявність необхідних колонок у даних."""
        missing = [col for col in columns if col not in self.data.columns]
        if missing:
            # Це нормально, якщо індикатор вимкнений. Просто повертаємо 0.
            return False
        return True

# --- Конкретні реалізації правил (без змін) ---
class TrendState(StateRule):
    """Правило для визначення сили ТРЕНДУ."""
    def get_confidence(self) -> float:
        adx_period = self.params.get('adx_period', 14)
        adx_threshold = self.params.get('adx_threshold', 25) # Поріг для початку тренду
        adx_col = f'ADX_{adx_period}'

        if not self._check_columns([adx_col]):
            return 0.0

        adx = self.data[adx_col].iloc[-1]
        if pd.isna(adx) or adx < adx_threshold:
            return 0.0

        # Розраховуємо впевненість: лінійно зростає від adx_threshold до 60
        confidence = (adx - adx_threshold) / (60 - adx_threshold)
        return min(1.0, max(0.0, confidence))

class RangeState(StateRule):
    """Правило для визначення сили РЕНДЖУ (ФЛЕТУ)."""
    def get_confidence(self) -> float:
        adx_period = self.params.get('adx_period', 14)
        adx_threshold = self.params.get('adx_threshold', 20) # Поріг для початку флету
        adx_col = f'ADX_{adx_period}'

        if not self._check_columns([adx_col]):
            return 0.0

        adx = self.data[adx_col].iloc[-1]
        if pd.isna(adx) or adx > adx_threshold:
            return 0.0

        # Розраховуємо впевненість: чим нижчий ADX, тим сильніший флет
        confidence = 1.0 - (adx / adx_threshold)
        return min(1.0, max(0.0, confidence))

# --- Головний клас-детектор (ЗІ ЗМІНАМИ) ---
class MarketStateDetector:
    """Визначає домінуючий стан ринку на основі набору конфігурованих правил."""
    
    STATE_MAP = {
        'trend': TrendState,
        'range': RangeState,
    }

    def __init__(self, data: pd.DataFrame):
        # Створюємо копію, щоб не змінювати оригінальний DataFrame
        self.data = data.copy()

        self.settings_manager = MarketStateSettingsManager()
        self.config = {"states": self.settings_manager.get_state_configurations()}
        
        # **НОВА ЛОГІКА**: Перевіряємо та розраховуємо ADX, якщо потрібно
        self._ensure_adx_exists()

    def _ensure_adx_exists(self):
        """
        Перевіряє наявність ADX у даних. Якщо його немає, розраховує
        його для внутрішнього використання.
        """
        required_params = set()
        for state_config in self.config.get('states', []):
            params = state_config.get('params', {})
            if 'adx_period' in params:
                required_params.add(params['adx_period'])
        
        for period in required_params:
            adx_col = f'ADX_{period}'
            if adx_col not in self.data.columns:
                print(f"    [StateDetector] INFO: {adx_col} не знайдено. Розраховую його для аналізу стану ринку.")
                self._calculate_adx(period)

    def _calculate_adx(self, period: int):
        """
        Внутрішній метод для розрахунку ADX.
        Логіка скопійована з IndicatorProcessor.
        """
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            print("    [StateDetector] ERROR: Відсутні колонки high, low, close для розрахунку ADX.")
            return

        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        self.data[f'ADX_{period}'] = adx

    def get_dominant_state(self) -> tuple[str, float]:
        """
        Обчислює впевненість для кожного налаштованого стану і повертає стан з найвищим показником.
        :return: Кортеж (назва_стану, впевненість)
        """
        if not self.config or 'states' not in self.config:
            return 'unknown', 0.0

        confidences = {}
        for state_config in self.config.get('states', []):
            state_name = state_config.get('name')
            state_class = self.STATE_MAP.get(state_name)

            if state_class:
                instance = state_class(self.data, state_config.get('params'))
                confidences[state_name] = instance.get_confidence()

        if not confidences:
            return 'unknown', 0.0

        # Знаходимо стан з максимальною впевненістю
        dominant_state = max(confidences, key=confidences.get, default='unknown')
        max_confidence = confidences.get(dominant_state, 0.0)

        print(f"    [StateDetector] Confidences: { {k: f'{v:.2f}' for k, v in confidences.items()} } -> Dominant: {dominant_state}")
        return dominant_state, round(max_confidence, 3)