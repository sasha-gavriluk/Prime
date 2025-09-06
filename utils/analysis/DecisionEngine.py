# utils/DecisionEngine.py

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import numpy as np # <-- Додайте імпорт numpy

class DecisionEngine:
    """
    DecisionEngine об'єднує аналіз кількох TimeframeAgent-ів, враховує новини
    і формує фінальне торгове рішення, динамічно адаптуючись до стану ринку.
    """

    def __init__(self, timeframe_weights: dict, metric_weights: dict, strategy_weights: dict):
        self.tf_weights = timeframe_weights
        self.base_metric_weights = metric_weights
        self.strategy_weights = strategy_weights

        self.profiles = {
            'trend': {'smc_confidence': 1.5, 'pattern_score': 0.7, 'trend_bonus': 0.5},
            'range': {'smc_confidence': 0.8, 'pattern_score': 1.5, 'trend_bonus': 0.1}
        }

        # NEW: параметри штрафів/бонусів
        self.sr_penalty = 0.35      # штраф за «покупку під опором» чи «продаж над підтримкою»
        self.neutral_band = 0.08    # поріг «сірої зони», щоб легше давати neutral

    # NEW: зручна утиліта
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


    def _get_market_state(self, analysis_results: Dict) -> str:
        """
        Визначає загальний стан ринку шляхом зваженого голосування
        між усіма таймфреймами.
        :param analysis_results: Словник з результатами аналізу від TimeframeAgent.
        :return: 'trend' або 'range'.
        """
        state_scores = {'trend': 0.0, 'range': 0.0}

        for tf, metrics in analysis_results.items():
            tf_weight = self.tf_weights.get(tf, 1.0)
            state = metrics.get('market_state')
            strength = metrics.get('state_strength', 0.0)

            if state in state_scores:
                # Голос кожного ТФ = його вага * впевненість у стані
                state_scores[state] += tf_weight * strength

        # Визначаємо переможця
        if not state_scores or state_scores['trend'] >= state_scores['range']:
            return 'trend'
        else:
            return 'range'
        
    def _apply_sr_penalty(self, metrics: Dict, buy_score: float, sell_score: float) -> Tuple[float, float]:
        near_sup = metrics.get('near_support', False)
        near_res = metrics.get('near_resistance', False)
        # якщо близько до опору — зменшуємо buy; якщо близько до підтримки — зменшуємо sell
        if near_res:
            buy_score *= (1.0 - self.sr_penalty)
        if near_sup:
            sell_score *= (1.0 - self.sr_penalty)
        return buy_score, sell_score


    def _calculate_news_score(self, articles: list, max_age_hours: int = 48, decay_factor: float = 0.95) -> float:
        if not articles:
            return 0.0

        total_raw_score = 0.0
        now_utc = datetime.now(timezone.utc)

        for article in articles:
            published_struct = article.get("published")
            if not published_struct: continue
            
            published_dt = datetime.fromtimestamp(time.mktime(published_struct), tz=timezone.utc)
            age_hours = (now_utc - published_dt).total_seconds() / 3600
            
            if age_hours > max_age_hours: continue

            time_decay_weight = decay_factor ** age_hours
            impact_score = article.get('impact_score', 0.0)
            total_raw_score += impact_score * time_decay_weight
        
        # НОРМАЛІЗАЦІЯ: перетворюємо сумарний бал у діапазон [-1, 1]
        normalized_score = np.tanh(total_raw_score / 5.0) # Ділення на 5 робить функцію менш чутливою
        
        print(f"[DecisionEngine] Raw News Score: {total_raw_score:.2f} -> Normalized: {normalized_score:.2f}")
        return normalized_score

    def aggregate_metrics(self, analysis_results: dict):
        # Цей метод залишається без змін
        aggregated, total_weights = {}, {}
        for tf, metrics in analysis_results.items():
            tf_weight = self.tf_weights.get(tf, 1.0)
            for key, value in metrics.items():
                if not isinstance(value, (int, float)): continue
                metric_weight = self.base_metric_weights.get(key, 1.0)
                weight = tf_weight * metric_weight
                if value is None: continue
                if key not in aggregated:
                    aggregated[key], total_weights[key] = 0.0, 0.0
                aggregated[key] += value * weight
                total_weights[key] += weight
        for key in aggregated:
            if total_weights[key] > 0: aggregated[key] /= total_weights[key]
        return aggregated
    
    def _aggregate_strategy_signals(self, strategy_signals: Dict) -> tuple[float, float]:
        """
        Агрегує сигнали від усіх активних стратегій.
        
        :param strategy_signals: Словник {'назва_стратегії': 'buy'/'sell'/'hold'}
        :return: Кортеж (strategy_buy_score, strategy_sell_score)
        """
        buy_score = 0.0
        sell_score = 0.0

        if not strategy_signals:
            return 0.0, 0.0

        print("\n--- 🧠 Агрегація сигналів від стратегій ---")
        for strategy_name, signal in strategy_signals.items():
            # Отримуємо вагу для цієї стратегії, або дефолтну, якщо не задано
            weight = self.strategy_weights.get(strategy_name, self.strategy_weights.get("default", 1.0))
            
            if signal == 'buy':
                buy_score += weight
                print(f"  - Стратегія '{strategy_name}' додає {weight} до купівлі.")
            elif signal == 'sell':
                sell_score += weight
                print(f"  - Стратегія '{strategy_name}' додає {weight} до продажу.")
        
        return buy_score, sell_score


    def count_votes(self, analysis_results: dict, key: str):
        # Цей метод залишається без змін
        vote_counter = {}
        for tf, metrics in analysis_results.items():
            tf_weight = self.tf_weights.get(tf, 1.0)
            value = metrics.get(key)
            if value is None: continue
            vote_counter[value] = vote_counter.get(value, 0) + tf_weight
        return vote_counter

    def make_decision(self, analysis_results: Dict, news_articles: Optional[list], strategy_signals: Optional[Dict] = None):
        # 0) визначення стану ринку (як у тебе є)
        market_state = self._get_market_state(analysis_results)

        # профіль ваг під стан
        current_metric_weights = dict(self.base_metric_weights)
        if market_state == 'trend':
            current_metric_weights['smc_confidence'] = current_metric_weights.get('smc_confidence', 1.0) * 1.2
            current_metric_weights['pattern_buy_score'] = current_metric_weights.get('pattern_buy_score', 1.0) * 0.9
            current_metric_weights['pattern_sell_score'] = current_metric_weights.get('pattern_sell_score', 1.0) * 0.9
        else:
            current_metric_weights['pattern_buy_score'] = current_metric_weights.get('pattern_buy_score', 1.0) * 1.2
            current_metric_weights['pattern_sell_score'] = current_metric_weights.get('pattern_sell_score', 1.0) * 1.2

        total_buy_score = 0.0
        total_sell_score = 0.0
        total_weight = 0.0

        agree_bonus = 0.0
        agree = 0
        disagree = 0
        for tf, m in analysis_results.items():
            if m.get('smc_bias') == 'buy': agree += 1
            elif m.get('smc_bias') == 'sell': disagree += 1
        # бонус пропорційний різниці
        agree_bonus = max(0, agree - disagree) * 0.02  # 0.02 — дуже маленький крок
        total_buy_score += agree_bonus
        total_weight += 0.02 

        # 1) збирання від таймфреймів (твоя логіка + наші S/R-штрафи)
        for tf, metrics in analysis_results.items():
            tf_weight = self.tf_weights.get(tf, 1.0)

            # смc у потрібну сторону
            smc_buy_component = 0.0
            smc_sell_component = 0.0
            if metrics.get('smc_bias') == 'buy':
                smc_buy_component = metrics.get('smc_confidence', 0.0) * current_metric_weights.get('smc_confidence', 1.0)
            elif metrics.get('smc_bias') == 'sell':
                smc_sell_component = metrics.get('smc_confidence', 0.0) * current_metric_weights.get('smc_confidence', 1.0)

            buy_part = smc_buy_component + metrics.get('pattern_buy_score', 0.0) * current_metric_weights.get('pattern_buy_score', 1.0)
            sell_part = smc_sell_component + metrics.get('pattern_sell_score', 0.0) * current_metric_weights.get('pattern_sell_score', 1.0)

            # NEW: штрафи по близькості до рівнів
            buy_part, sell_part = self._apply_sr_penalty(metrics, buy_part, sell_part)

            total_buy_score += buy_part * tf_weight
            total_sell_score += sell_part * tf_weight
            total_weight += tf_weight

        # 2) сигнали стратегій (як у тебе)
        if strategy_signals:
            strategy_buy_score, strategy_sell_score = self._aggregate_strategy_signals(strategy_signals)
            total_buy_score += strategy_buy_score
            total_sell_score += strategy_sell_score
            total_weight += sum(self.strategy_weights.get(s, 1.0) for s in strategy_signals.keys())

        # 3) новини (як у тебе)
        if self.base_metric_weights.get('news_impact', 0) > 0 and news_articles:
            news_weight = self.base_metric_weights.get('news_impact', 1.0)
            news_score = self._calculate_news_score(news_articles)  # в діапазоні [-1, 1]
            if news_score > 0:
                total_buy_score += news_score * news_weight
            elif news_score < 0:
                total_sell_score += (-news_score) * news_weight
            total_weight += news_weight


        if total_weight == 0:
            return {"direction": "neutral", "confidence": 0.0, "reason": "Немає даних для аналізу."}

        # 4) нормалізація та «м’якший» конфіденс
        final_buy = total_buy_score / total_weight
        final_sell = total_sell_score / total_weight
        diff = final_buy - final_sell
        # NEW: переводимо різницю через сигмоїду — отримуємо плавний conf в [0..1]
        confidence = float(self._sigmoid(diff))  # >0.5 buy-бік, <0.5 sell-бік
        strength = abs(diff)

        # NEW: нейтральна «сіра зона», щоб не давати buy/sell на мікро-перевазі
        if abs(diff) < self.neutral_band:
            return {"direction": "neutral", "confidence": 0.0, "reason": "Сигнал у сірій зоні.", "market_state": market_state}

        direction = "buy" if diff > 0 else "sell"
        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "reason": f"Перевага {direction}.",
            "market_state": market_state,
            # дод. інфа (зручно для відладки)
            "raw": {"final_buy": final_buy, "final_sell": final_sell, "diff": diff}
        }
