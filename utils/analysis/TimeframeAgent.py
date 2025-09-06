import pandas as pd
from utils.common.SettingsLoader import ProcessingSettingsBuilder
from utils.analysis.MarketStateDetector import MarketStateDetector

class TimeframeAgent:
    """
    Аналізує один таймфрейм на основі вже обробленого DataFrame.
    Логіка аналізу керується через конфігураційний словник.
    """

    def __init__(self, timeframe_name: str, processed_data: pd.DataFrame, agent_config: dict):
        """
        :param timeframe_name: Назва таймфрейму (напр. '5m').
        :param processed_data: DataFrame з усіма індикаторами та фічами.
        :param agent_config: Словник з налаштуваннями для цього агента.
        """
        self.timeframe = timeframe_name
        self.data = processed_data
        self.config = agent_config # Зберігаємо конфігурацію

        # Ініціалізуємо модуль патернів
        self.pattern_module = PatternAgent(processed_data)
        self.psb = ProcessingSettingsBuilder()

        self.market_state_detector = MarketStateDetector(self.data)

    def analyze_trend(self):
        # Використовуємо колонку, вказану в конфігурації
        trend_conf = self.config.get("trend", {})
        ema_column = trend_conf.get("column", "EMA_50") # За замовчуванням EMA_50 для зворотної сумісності

        if ema_column not in self.data.columns:
            return None # Немає даних для аналізу
            
        ema = self.data[ema_column]
        ema_slope = ema.diff().mean()
        if ema_slope > 0:
            return "up"
        elif ema_slope < 0:
            return "down"
        else:
            return "sideways"

    def analyze_smc_bias(self):
        if 'Market_Structure_Type' not in self.data.columns:
            return None, 0.0

        structure = self.data['Market_Structure_Type']
        hh = (structure == 'HH').sum()
        hl = (structure == 'HL').sum()
        ll = (structure == 'LL').sum()
        lh = (structure == 'LH').sum()

        total = hh + hl + ll + lh
        if total == 0:
            return None, 0.0

        up_score = (hh + hl) / total
        down_score = (ll + lh) / total

        if up_score > down_score:
            return "buy", up_score
        else:
            return "sell", down_score

    def analyze_retracement_quality(self):
        if 'Market_Structure_Point' not in self.data.columns:
            return 0.5

        swings = self.data['Market_Structure_Point']
        count_highs = (swings == 'swing_high').sum()
        count_lows = (swings == 'swing_low').sum()
        if count_highs + count_lows == 0:
            return 0.5

        ratio = min(count_highs, count_lows) / max(count_highs, count_lows)
        return ratio

    def analyze_near_support_resistance(self):
        near_support = self.data['Near_Support'].iloc[-1] if 'Near_Support' in self.data.columns else False
        near_resistance = self.data['Near_Resistance'].iloc[-1] if 'Near_Resistance' in self.data.columns else False
        return near_support, near_resistance

    def analyze_volatility(self):
        # Використовуємо колонку та поріг з конфігурації
        vol_conf = self.config.get("volatility", {})
        atr_column = vol_conf.get("column", "ATR_14")
        threshold = vol_conf.get("threshold", 0.3)

        if atr_column not in self.data.columns:
            return "normal"

        atr = self.data[atr_column]
        atr_change = atr.pct_change()
        if atr_change.iloc[-1] > threshold:
            return "high"
        else:
            return "normal"
        
    def analyze_fvg_quality(self, window=100):
        if 'FVG_Up' not in self.data.columns or 'FVG_Down' not in self.data.columns:
            return 0.5

        fvg_up = self.data['FVG_Up'].tail(window).sum()
        fvg_down = self.data['FVG_Down'].tail(window).sum()
        total = window

        frequency = (fvg_up + fvg_down) / total
        quality = max(0.0, 1.0 - frequency * 2)

        return round(quality, 3)

    def run_analysis(self):
        pattern_names = self.psb.get_pattern_settings()
        pattern_votes = {"buy": 0.0, "sell": 0.0}
 
        lookback = 3
        pattern_signals = {}

        for name in pattern_names:
            if name in self.data.columns and self.data[name].tail(lookback).any():
                logic = self.pattern_module.get_pattern_logic(name)
                direction = logic.get("direction")
                score = self.pattern_module.get_patterns_score(name)

                if direction in pattern_votes:
                    pattern_votes[direction] += score

                pattern_signals[name] = score

        pattern_score = round(sum(pattern_signals.values()) / len(pattern_signals), 3) if pattern_signals else 0.0
        print(f"--- Running Analysis for {self.timeframe} ---")
        dominant_state, confidence = self.market_state_detector.get_dominant_state()
        print(f"  [{self.timeframe}] Market State Detected: {dominant_state.upper()} (Strength: {confidence:.2f})")


        # Метрики
        result = {
            "timeframe": self.timeframe,
            "market_state": dominant_state, # <-- ДОДАНО СТАН РИНКУ
            "state_strength": confidence, # <-- ДОДАНО ВПЕВНЕНІСТЬ
            "trend": self.analyze_trend(),
            "volatility": self.analyze_volatility(),
            "retracement_quality": self.analyze_retracement_quality(),
            "smc_bias": None,
            "smc_confidence": 0.0,
            "near_support": False,
            "near_resistance": False,
            "fvg_quality": self.analyze_fvg_quality(),
            "pattern_score": pattern_score,
            "pattern_buy_score": round(pattern_votes["buy"], 3),
            "pattern_sell_score": round(pattern_votes["sell"], 3),
        }

        smc_bias, smc_confidence = self.analyze_smc_bias()
        result["smc_bias"] = smc_bias
        result["smc_confidence"] = smc_confidence

        ns, nr = self.analyze_near_support_resistance()
        result["near_support"] = ns
        result["near_resistance"] = nr

        # Додаємо активні патерни окремими полями
        for pat, score in pattern_signals.items():
            result[pat] = score

        return result

class PatternAgent:
    """
    Відповідає за детекцію свічкових патернів
    і зважену агрегацію їхніх сигналів.
    """
    def __init__(self, data: pd.DataFrame,):
        self.data = data

        self.score = 0.8
        self.hammer_score = self.score
        self.engulfing_score = self.score
        self.inverted_hammer_score = self.score
        self.shooting_star_score = self.score
        self.morning_star_score = self.score
        self.evening_star_score = self.score
        self.piercing_pattern_score = self.score
        self.dark_cloud_cover_score = self.score
        self.three_white_soldiers_score = self.score
        self.three_black_crows_score = self.score

        self.pattern_logic = {
            'Hammer': {'direction': 'buy'},
            'Shooting_Star': {'direction': 'sell'},
            'Engulfing': {'direction': 'reversal'},
            'Morning_Star': {'direction': 'buy'},
            'Evening_Star': {'direction': 'sell'},
            'Piercing_Pattern': {'direction': 'buy'},
            'Dark_Cloud_Cover': {'direction': 'sell'},
            'Three_White_Soldiers': {'direction': 'buy'},
            'Three_Black_Crows': {'direction': 'sell'},
        }
        
    def get_patterns_score(self, pattern_name: str):
        if pattern_name in self.data.columns:
            return getattr(self, f"{pattern_name.lower()}_score")
        else:
            return 0
        
    def get_pattern_logic(self, pattern_name: str) -> dict:
        return self.pattern_logic.get(pattern_name, {})