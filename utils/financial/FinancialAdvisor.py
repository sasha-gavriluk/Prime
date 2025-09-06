import numpy as np
import math
from typing import List, Dict, Union, Tuple

class FinancialAdvisor:
    """
    Надає набір інструментів для фінансового планування, управління ризиками
    та аналізу інвестиційних стратегій.

    Цей клас розроблено як самостійний модуль, що не залежить від
    конкретної реалізації торгової логіки чи джерел даних.
    """

    def __init__(self):
        """Ініціалізація фінансового радника."""
        # У майбутньому тут можна буде зберігати стан, наприклад, історію операцій.
        pass

    # --- РОЗДІЛ 1: УПРАВЛІННЯ РИЗИКАМИ В УГОДІ ---

    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує оптимальний розмір позиції для СПОТОВОЇ торгівлі.
        """
        if risk_per_trade_pct <= 0 or risk_per_trade_pct > 100:
            return {"error": "Відсоток ризику має бути в діапазоні (0, 100]."}
        if entry_price <= 0 or stop_loss_price <= 0 or capital <= 0:
            return {"error": "Капітал та ціни мають бути додатними числами."}
        if entry_price == stop_loss_price:
            return {"error": "Ціна входу не може дорівнювати ціні стоп-лосу."}

        risk_amount = capital * (risk_per_trade_pct / 100.0)
        price_risk_per_unit = abs(entry_price - stop_loss_price)

        if price_risk_per_unit == 0:
             return {"error": "Нульовий ризик на одиницю активу."}

        position_size_units = risk_amount / price_risk_per_unit
        position_value = position_size_units * entry_price

        return {
            "position_size_units": round(position_size_units, 8),
            "position_value_usd": round(position_value, 2),
            "risk_per_trade_usd": round(risk_amount, 2),
        }

    def calculate_sl_tp_levels(
        self,
        entry_price: float,
        direction: str,
        stop_loss_pct: float = None,
        risk_reward_ratio: float = 2.0,
        atr_value: float = None,
        atr_multiplier: float = 1.5,
        signal_duration: float = None
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує рівні Stop Loss та Take Profit.
        :param signal_duration: Очікувана тривалість сигналу у кількості періодів.
        """
        if direction not in ['buy', 'sell']:
            return {"error": "Напрямок має бути 'buy' або 'sell'."}

        sl_price = 0.0
        if stop_loss_pct:
            if direction == 'buy':
                sl_price = entry_price * (1 - stop_loss_pct / 100.0)
            else: # sell
                sl_price = entry_price * (1 + stop_loss_pct / 100.0)
        elif atr_value and atr_value > 0:
            # Використовуємо тривалість сигналу, якщо вона надана
            duration = signal_duration if signal_duration and signal_duration > 0 else 1
            move = atr_value * atr_multiplier * duration
            if direction == 'buy':
                sl_price = entry_price - move
            else: # sell
                sl_price = entry_price + move
        else:
            return {"error": "Необхідно надати stop_loss_pct або atr_value."}

        risk_per_unit = abs(entry_price - sl_price)
        profit_target_per_unit = risk_per_unit * risk_reward_ratio

        if direction == 'buy':
            tp_price = entry_price + profit_target_per_unit
        else: # sell
            tp_price = entry_price - profit_target_per_unit

        return {
            "stop_loss_price": round(sl_price, 4),
            "take_profit_price": round(tp_price, 4),
            "risk_reward_ratio": risk_reward_ratio
        }
    
    def estimate_signal_duration(
        self,
        entry_price: float,
        target_price: float,
        atr_value: float
    ) -> Dict[str, Union[float, str]]:
        """
        Оцінює кількість періодів, необхідних для досягнення цільової ціни
        на основі ATR.
        """
        if atr_value is None or atr_value <= 0:
            return {"error": "atr_value має бути додатним."}
        distance = abs(target_price - entry_price)
        if distance == 0:
            return {"error": "Ціна входу та цільова ціна співпадають."}
        periods = distance / atr_value
        return {"estimated_periods": round(periods, 2)}

    # --- РОЗДІЛ 2: ПЛАНУВАННЯ КАПІТАЛУ ТА ЗРОСТАННЯ ---

    def calculate_compound_growth(
        self,
        initial_capital: float,
        periods: int,
        avg_return_per_period_pct: float,
        contribution_per_period: float = 0
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Розраховує зростання капіталу за рахунок складного відсотка.

        :param initial_capital: Початковий капітал.
        :param periods: Кількість періодів (напр., місяців, років).
        :param avg_return_per_period_pct: Середня дохідність за період у відсотках.
        :param contribution_per_period: Сума, що додається до капіталу кожен період.
        :return: Словник з кінцевим капіталом та історією зростання.
        """
        if initial_capital < 0 or periods <= 0:
            return {"error": "Початковий капітал та кількість періодів мають бути додатними."}

        history = [initial_capital]
        current_capital = initial_capital
        return_multiplier = 1 + (avg_return_per_period_pct / 100.0)

        for _ in range(periods):
            current_capital += contribution_per_period
            current_capital *= return_multiplier
            history.append(round(current_capital, 2))

        total_growth_pct = ((history[-1] - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0

        return {
            "final_capital": history[-1],
            "total_periods": periods,
            "total_growth_pct": round(total_growth_pct, 2),
            "growth_history": history
        }

    def plan_portfolio_diversification(
        self,
        risk_profile: str,
        available_asset_classes: List[str] = ['stocks', 'bonds', 'crypto', 'real_estate']
    ) -> Dict[str, Union[float, str]]:
        """
        Пропонує базовий план диверсифікації портфеля за профілем ризику.

        :param risk_profile: Профіль ризику ('conservative', 'moderate', 'aggressive').
        :param available_asset_classes: Список доступних класів активів.
        :return: Словник з рекомендованим розподілом у відсотках.
        """
        allocations = {
            'conservative': {'stocks': 20, 'bonds': 60, 'crypto': 5, 'real_estate': 15},
            'moderate': {'stocks': 40, 'bonds': 40, 'crypto': 10, 'real_estate': 10},
            'aggressive': {'stocks': 60, 'bonds': 15, 'crypto': 20, 'real_estate': 5}
        }

        if risk_profile not in allocations:
            return {"error": "Невідомий профіль ризику. Використовуйте 'conservative', 'moderate' або 'aggressive'."}

        plan = allocations[risk_profile]
        # Фільтруємо план, залишаючи тільки доступні класи активів
        filtered_plan = {asset: pct for asset, pct in plan.items() if asset in available_asset_classes}

        # Нормалізуємо відсотки, якщо деякі класи активів були видалені
        total_pct = sum(filtered_plan.values())
        if total_pct > 0 and total_pct != 100:
            factor = 100 / total_pct
            filtered_plan = {asset: round(pct * factor, 1) for asset, pct in filtered_plan.items()}

        return {
            "risk_profile": risk_profile,
            "recommended_allocation": filtered_plan
        }
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: str,
        margin_type: str = 'isolated'
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує приблизну ціну ліквідації для ф'ючерсної позиції.

        :param entry_price: Ціна входу в позицію.
        :param leverage: Кредитне плече (напр., 10 для 10x).
        :param direction: Напрямок угоди ('buy' або 'sell').
        :param margin_type: Тип маржі ('isolated' або 'cross'). Для cross розрахунок складніший.
        :return: Словник з ціною ліквідації.
        """
        if leverage <= 0:
            return {"error": "Кредитне плече має бути додатним."}
        if direction not in ['buy', 'sell']:
            return {"error": "Напрямок має бути 'buy' або 'sell'."}

        # Формула для ізольованої маржі. Для cross-маржі потрібно знати весь баланс.
        # Коефіцієнт підтримки маржі (maintenance margin rate) зазвичай залежить від біржі та розміру позиції.
        # Для спрощення беремо середнє значення, наприклад, 0.5% (0.005).
        maintenance_margin_rate = 0.005 
        
        liquidation_price = 0.0
        if direction == 'buy':
            # Ціна ліквідації для Long = Ціна входу * (1 - (1 / Плече) + Коеф. підтримки)
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
        else: # sell
            # Ціна ліквідації для Short = Ціна входу * (1 + (1 / Плече) - Коеф. підтримки)
            liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate)

        return {
            "liquidation_price": round(liquidation_price, 4),
            "leverage": leverage,
            "margin_type": margin_type,
            "comment": "Це приблизна ціна. Реальна може відрізнятись залежно від біржі."
        }

    def calculate_futures_position_size(
        self,
        capital: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує розмір ф'ючерсної позиції у USDT та в одиницях активу.

        :param capital: Загальний капітал, доступний для торгівлі.
        :param risk_per_trade_pct: Відсоток капіталу, яким ви готові ризикнути.
        :param entry_price: Ціна входу.
        :param stop_loss_price: Ціна стоп-лосу.
        :param leverage: Кредитне плече.
        :return: Словник з деталями позиції.
        """
        if entry_price == stop_loss_price:
            return {"error": "Ціна входу не може дорівнювати ціні стоп-лосу."}
            
        risk_amount_usd = capital * (risk_per_trade_pct / 100.0)
        price_change_pct = abs(entry_price - stop_loss_price) / entry_price
        
        if price_change_pct == 0:
            return {"error": "Нульовий ризик на одиницю активу."}

        # Розмір позиції (з плечем) = Сума ризику / (% зміни ціни до SL)
        position_size_usd = risk_amount_usd / price_change_pct
        
        # Маржа, необхідна для відкриття такої позиції
        margin_required = position_size_usd / leverage
        
        # Розмір позиції в одиницях базового активу
        position_size_units = position_size_usd / entry_price

        return {
            "position_size_usd": round(position_size_usd, 2),
            "position_size_units": round(position_size_units, 8),
            "margin_required_usd": round(margin_required, 2),
            "risk_per_trade_usd": round(risk_amount_usd, 2),
            "leverage": leverage
        }

    # --- РОЗДІЛ 3: АНАЛІЗ ЕФЕКТИВНОСТІ ---

    def analyze_trade_history(
        self,
        trade_history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Аналізує історію угод та розраховує ключові метрики ефективності.
        """
        if not trade_history:
            return {"error": "Історія угод порожня."}

        pnls = [trade['pnl'] for trade in trade_history]
        total_trades = len(pnls)
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')

        equity_curve = np.cumsum([10000] + pnls)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown_pct = np.max(drawdowns) * 100

        return {
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "average_win_usd": round(avg_win, 2),
            "average_loss_usd": round(avg_loss, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_pnl_usd": round(sum(pnls), 2)
        }
    
    # --- РОЗДІЛ 4: ПРОСУНУТИЙ РИСК-МЕНЕДЖМЕНТ ТА СТРАТЕГІЯ ---

    def calculate_dynamic_risk_pct(
        self,
        win_rate_pct: float,
        avg_win_to_loss_ratio: float,
        kelly_fraction: float = 0.5
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує динамічний відсоток ризику за допомогою спрощеного критерію Келлі.

        :param win_rate_pct: Історичний відсоток прибуткових угод (0-100).
        :param avg_win_to_loss_ratio: Середнє співвідношення прибутку до збитку (R:R).
        :param kelly_fraction: Частка від повного критерію Келлі для зменшення агресивності (0.1-1.0).
        :return: Словник з рекомендованим відсотком ризику.
        """
        if not (0 < win_rate_pct <= 100):
            return {"error": "Win Rate має бути в діапазоні (0, 100]."}
        if avg_win_to_loss_ratio <= 0:
            return {"error": "Співвідношення середнього прибутку до збитку має бути додатним."}
        if not (0 < kelly_fraction <= 1.0):
            return {"error": "Частка Келлі має бути в діапазоні (0, 1.0]."}

        W = win_rate_pct / 100.0
        R = avg_win_to_loss_ratio
        
        # Формула Келлі: K% = W - (1 - W) / R
        kelly_percentage = W - ((1 - W) / R)

        if kelly_percentage <= 0:
            return {
                "recommended_risk_pct": 0.0,
                "comment": "Стратегія є збитковою на дистанції. Ризик не рекомендується."
            }

        recommended_risk = kelly_percentage * 100 * kelly_fraction

        return {
            "full_kelly_pct": round(kelly_percentage * 100, 2),
            "recommended_risk_pct": round(recommended_risk, 2),
            "comment": f"Рекомендовано ризикувати не більше {round(recommended_risk, 2)}% капіталу на угоду."
        }

    def calculate_breakeven_trigger_price(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: str,
        profit_factor_for_breakeven: float = 1.0
    ) -> Dict[str, Union[float, str]]:
        """
        Розраховує ціну, при досягненні якої варто перемістити стоп-лосс в беззбиток.

        :param entry_price: Ціна входу.
        :param stop_loss_price: Початкова ціна стоп-лосу.
        :param direction: Напрямок угоди ('buy' або 'sell').
        :param profit_factor_for_breakeven: Коефіцієнт, що показує, у скільки разів
                                            поточний прибуток має перевищувати початковий ризик.
        :return: Словник з ціною для переведення в беззбиток.
        """
        initial_risk_per_unit = abs(entry_price - stop_loss_price)
        
        if direction == 'buy':
            trigger_price = entry_price + (initial_risk_per_unit * profit_factor_for_breakeven)
        elif direction == 'sell':
            trigger_price = entry_price - (initial_risk_per_unit * profit_factor_for_breakeven)
        else:
            return {"error": "Напрямок має бути 'buy' або 'sell'."}

        return {
            "breakeven_trigger_price": round(trigger_price, 4),
            "comment": f"При досягненні ціни {round(trigger_price, 4)} перемістіть SL на {entry_price}."
        }

    def evaluate_portfolio_risk(
        self,
        total_capital: float,
        open_trades: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Оцінює сукупний ризик для всього портфеля з відкритими угодами.

        :param total_capital: Загальний капітал портфеля.
        :param open_trades: Список відкритих угод. Кожна угода - словник з ключем 'risk_per_trade_usd'.
        :return: Словник з оцінкою загального ризику.
        """
        if not open_trades:
            return {
                "total_capital_at_risk_usd": 0.0,
                "total_capital_at_risk_pct": 0.0,
                "risk_level": "none"
            }
            
        total_risk_usd = sum(trade.get('risk_per_trade_usd', 0) for trade in open_trades)
        total_risk_pct = (total_risk_usd / total_capital) * 100 if total_capital > 0 else 0
        
        risk_level = "low"
        if 5 < total_risk_pct <= 10:
            risk_level = "medium"
        elif total_risk_pct > 10:
            risk_level = "high"
            
        return {
            "total_capital_at_risk_usd": round(total_risk_usd, 2),
            "total_capital_at_risk_pct": round(total_risk_pct, 2),
            "risk_level": risk_level,
            "number_of_open_trades": len(open_trades)
        }

    # --- РОЗДІЛ 5: ФІНАНСОВЕ ПЛАНУВАННЯ ТА ЦІЛІ ---

    def estimate_time_to_goal(
        self,
        initial_capital: float,
        target_capital: float,
        avg_return_per_period_pct: float,
        contribution_per_period: float = 0
    ) -> Dict[str, Union[int, str]]:
        """
        Розраховує приблизну кількість періодів для досягнення фінансової цілі.

        :param initial_capital: Початковий капітал.
        :param target_capital: Цільова сума капіталу.
        :param avg_return_per_period_pct: Середня дохідність за період у відсотках.
        :param contribution_per_period: Сума, що додається до капіталу кожен період.
        :return: Словник з кількістю періодів та коментарем.
        """
        if target_capital <= initial_capital:
            return {"periods_to_goal": 0, "comment": "Ціль вже досягнута або нижча за початковий капітал."}
        if avg_return_per_period_pct <= 0 and contribution_per_period <= 0:
            return {"error": "Ціль недосяжна без позитивної дохідності або внесків."}

        periods = 0
        current_capital = initial_capital
        return_multiplier = 1 + (avg_return_per_period_pct / 100.0)

        # Обмеження на 100 років (1200 місяців), щоб уникнути нескінченного циклу
        while current_capital < target_capital and periods < 1200:
            current_capital += contribution_per_period
            current_capital *= return_multiplier
            periods += 1
            
        if periods >= 1200:
             return {"error": "Розрахунок перевищив 100 років. Ціль може бути недосяжною."}

        return {
            "periods_to_goal": periods,
            "comment": f"Приблизно {periods} періодів для досягнення цілі."
        }
        
    def calculate_required_win_rate(
        self,
        risk_reward_ratio: float
    ) -> Dict[str, float]:
        """
        Розраховує мінімальний відсоток прибуткових угод (Win Rate) для беззбитковості.

        :param risk_reward_ratio: Співвідношення ризику до прибутку (R:R).
        :return: Словник з необхідним Win Rate.
        """
        if risk_reward_ratio <= 0:
            return {"error": "Співвідношення ризику до прибутку має бути додатним."}
            
        # Формула: Breakeven Win Rate = 1 / (1 + R:R)
        required_rate = 1 / (1 + risk_reward_ratio)
        
        return {
            "risk_reward_ratio": risk_reward_ratio,
            "breakeven_win_rate_pct": round(required_rate * 100, 2)
        }