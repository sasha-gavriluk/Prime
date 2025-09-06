# utils/FinancialContext.py

from typing import Dict, List, Union
# Припускаємо, що FinancialAdvisor знаходиться в папці utils
from utils.financial.FinancialAdvisor import FinancialAdvisor 

class FinancialContext:
    """
    Цей клас є мостом між технічним аналізом (сигналами) та фінансовим
    плануванням. Він використовує FinancialAdvisor для надання конкретних
    рекомендацій по кожній угоді на основі налаштувань користувача та режиму торгівлі.
    """

    def __init__(self, user_settings: Dict, advisor: FinancialAdvisor, trade_mode: str = 'spot'):
        """
        Ініціалізує контекст на основі налаштувань користувача.

        :param user_settings: Словник з налаштуваннями.
        :param advisor: Екземпляр класу FinancialAdvisor.
        :param trade_mode: Режим торгівлі ('spot' або 'futures').
        """
        self.settings = user_settings
        self.advisor = advisor
        self.trade_mode = trade_mode
        
        # Загальні налаштування
        self.total_capital = self.settings.get("total_capital", 0.0)
        self.risk_pct = self.settings.get("default_risk_per_trade_pct", 1.0)
        self.open_trades = self.settings.get("open_trades", [])
        
        # Специфічні налаштування для ф'ючерсів
        if self.trade_mode == 'futures':
            self.leverage = self.settings.get("leverage", 10)

    def generate_financial_briefing(
        self,
        signal_data: Dict,
        current_price: float,
        atr_value: float = None,
        signal_duration: float = None
    ) -> Dict:
        """
        Створює повний фінансовий "брифінг" для отриманого торгового сигналу,
        адаптований під обраний режим торгівлі (spot або futures).
        """
        direction = signal_data.get('direction')
        if not direction or direction == 'neutral':
            return {"status": "no_action", "reason": "Нейтральний сигнал, фінансовий аналіз не проводиться."}

        # 1. Розрахунок рівнів Stop Loss та Take Profit (універсальний для обох режимів)
        sl_tp_info = self._calculate_sl_tp(current_price, direction, atr_value, signal_duration)
        if "error" in sl_tp_info:
            return {"status": "error", "reason": sl_tp_info["error"]}
        stop_loss_price = sl_tp_info['stop_loss_price']

        # 2. Розрахунок позиції та специфічних даних залежно від режиму
        if self.trade_mode == 'spot':
            position_info = self.advisor.calculate_position_size(
                capital=self.total_capital,
                risk_per_trade_pct=self.risk_pct,
                entry_price=current_price,
                stop_loss_price=stop_loss_price
            )
        elif self.trade_mode == 'futures':
            position_info = self.advisor.calculate_futures_position_size(
                capital=self.total_capital,
                risk_per_trade_pct=self.risk_pct,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                leverage=self.leverage
            )
            # Додаємо розрахунок ціни ліквідації
            liquidation_info = self.advisor.calculate_liquidation_price(
                entry_price=current_price,
                leverage=self.leverage,
                direction=direction
            )
            position_info.update(liquidation_info) # Об'єднуємо інформацію
        else:
            return {"status": "error", "reason": f"Невідомий режим торгівлі: {self.trade_mode}"}

        if "error" in position_info:
            return {"status": "error", "reason": position_info["error"]}
        
        # Зберігаємо напрямок угоди для подальшого використання
        position_info["direction"] = direction

        # 3. Оцінка ризику для портфеля
        portfolio_risk_info = self._evaluate_portfolio_risk(position_info)

        # 4. Формування фінального брифінгу
        briefing = {
            "status": "ok",
            "trade_mode": self.trade_mode,
            "signal": signal_data,
            "trade_parameters": {
                "entry_price": current_price,
                **sl_tp_info,
                **position_info
            },
            "portfolio_impact": portfolio_risk_info,
            "commentary": self._generate_commentary(position_info, portfolio_risk_info)
        }
        
        return briefing

    def _calculate_sl_tp(self, current_price, direction, atr_value, signal_duration):
        """Внутрішній метод для розрахунку SL/TP з резервною логікою."""
        sl_tp_params = {
            "entry_price": current_price,
            "direction": direction,
            "risk_reward_ratio": self.settings.get("risk_reward_ratio", 2.0)
        }
        if atr_value and atr_value > 0:
            sl_tp_params["atr_value"] = atr_value
            sl_tp_params["atr_multiplier"] = self.settings.get("atr_multiplier", 1.5)
            if signal_duration and signal_duration > 0:
                sl_tp_params["signal_duration"] = signal_duration
        else:
            default_sl_pct = self.settings.get("default_stop_loss_pct")
            if not default_sl_pct:
                return {"error": "ATR недоступний, і не задано резервний stop_loss_pct."}
            sl_tp_params["stop_loss_pct"] = default_sl_pct
        sl_tp_info = self.advisor.calculate_sl_tp_levels(**sl_tp_params)

        if atr_value and "take_profit_price" in sl_tp_info:
            duration_info = self.advisor.estimate_signal_duration(
                entry_price=current_price,
                target_price=sl_tp_info["take_profit_price"],
                atr_value=atr_value
            )
            if "estimated_periods" in duration_info:
                sl_tp_info["estimated_duration_bars"] = duration_info["estimated_periods"]
        return sl_tp_info

    def _evaluate_portfolio_risk(self, position_info):
        """Внутрішній метод для оцінки ризику портфеля."""
        potential_new_trade = {"risk_per_trade_usd": position_info["risk_per_trade_usd"]}
        all_potential_trades = self.open_trades + [potential_new_trade]
        return self.advisor.evaluate_portfolio_risk(
            total_capital=self.total_capital,
            open_trades=all_potential_trades
        )

    def _generate_commentary(self, pos_info: Dict, port_risk_info: Dict) -> str:
        """Генерує текстовий коментар-рекомендацію."""
        comment = f"Ризик на угоду: ${pos_info['risk_per_trade_usd']} ({self.risk_pct}% від капіталу).\n"
        
        if self.trade_mode == 'futures':
            comment += (
                f"Розмір позиції: ${pos_info['position_size_usd']:,} ({pos_info['leverage']}x). "
                f"Необхідна маржа: ${pos_info['margin_required_usd']:.2f}.\n"
            )
            # Додаємо попередження про ліквідацію
            liq_price = pos_info.get('liquidation_price')
            sl_price = pos_info.get('stop_loss_price')
            if liq_price and sl_price:
                direction = pos_info.get('direction', 'buy')
                is_safe = (direction == 'buy' and sl_price > liq_price) or \
                          (direction == 'sell' and sl_price < liq_price)
                if not is_safe:
                     comment += f"⚠️ НЕБЕЗПЕЧНО! Ваш SL (${sl_price}) знаходиться ближче за ціну ліквідації (${liq_price}).\n"

        if port_risk_info['risk_level'] == 'high':
            comment += (
                f"⚠️ УВАГА: Сукупний ризик по портфелю ({port_risk_info['total_capital_at_risk_pct']:.2f}%) є високим."
            )
        return comment


if __name__ == '__main__':
    # --- Демонстрація використання FinancialContext для Ф'ЮЧЕРСІВ ---
    advisor = FinancialAdvisor()
    
    user_futures_config = {
        "total_capital": 1000.0,
        "default_risk_per_trade_pct": 2.0,
        "leverage": 20,
        "default_stop_loss_pct": 5.0, # Резервний SL, якщо ATR недоступний
        "open_trades": []
    }

    # Ініціалізуємо FinancialContext в режимі 'futures'
    financial_context_futures = FinancialContext(
        user_settings=user_futures_config, 
        advisor=advisor,
        trade_mode='futures'
    )

    raw_signal = {'direction': 'buy', 'confidence': 0.78}
    
    print("--- Генерація Ф'ЮЧЕРСНОГО фінансового брифінгу ---")
    briefing = financial_context_futures.generate_financial_briefing(
        signal_data=raw_signal,
        current_price=68000.0,
        atr_value=550.0 # Припустимо, є валідне значення ATR
    )

    if briefing.get("status") == "ok":
        print(f"Режим: {briefing['trade_mode'].upper()}")
        print(f"Сигнал: {briefing['signal']['direction'].upper()}, Впевненість: {briefing['signal']['confidence']:.2f}")
        
        print("\n--- Параметри ф'ючерсної угоди ---")
        params = briefing['trade_parameters']
        for key, value in params.items():
            print(f"  - {key.replace('_', ' ').capitalize()}: {value}")
        
        print("\n--- Рекомендація ---")
        print(briefing['commentary'])
    else:
        print(f"Помилка: {briefing.get('reason')}")

