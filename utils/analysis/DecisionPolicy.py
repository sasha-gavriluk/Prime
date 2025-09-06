# utils/analysis/DecisionPolicy.py

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

@dataclass
class HysteresisCooldownPolicy:
    conf_margin: float = 0.07       # наскільки новий конф. має перевищити старий, щоб flipнути
    cooldown_bars: int = 3          # мін. кількість барів між змінами напрямку
    prefer_hold_when_uncertain: bool = True

    last_decision: str = "neutral"
    last_confidence: float = 0.0
    last_change_bar_idx: Optional[int] = None

    def decide(
        self,
        proposed_direction: str,
        proposed_confidence: float,
        current_bar_idx: int
    ) -> Tuple[str, float]:
        """
        Повертає (direction, confidence) після застосування гістерезису та кулдауну.
        current_bar_idx — монотонно зростаючий лічильник барів (наприклад, індекс рядка).
        """
        # 1) Немає історії — приймаємо перший адекватний сигнал
        if self.last_change_bar_idx is None:
            self.last_decision = proposed_direction
            self.last_confidence = proposed_confidence
            self.last_change_bar_idx = current_bar_idx
            return self.last_decision, self.last_confidence

        # 2) Якщо кулдаун ще триває — тримаємо рішення
        bars_since_change = current_bar_idx - self.last_change_bar_idx
        if bars_since_change < self.cooldown_bars:
            return self.last_decision, self.last_confidence

        # 3) Якщо пропонують flip — вимагаємо запас по впевненості
        if proposed_direction != self.last_decision:
            # Якщо нейтраль пропонують — можна прийняти без margin (щоб «зупинити кровотечу»)
            if proposed_direction == "neutral":
                self.last_decision = "neutral"
                self.last_confidence = 0.0
                self.last_change_bar_idx = current_bar_idx
                return self.last_decision, self.last_confidence

            # В інші боки — потрібен conf_margin
            if proposed_confidence >= (self.last_confidence + self.conf_margin):
                self.last_decision = proposed_direction
                self.last_confidence = proposed_confidence
                self.last_change_bar_idx = current_bar_idx
                return self.last_decision, self.last_confidence
            else:
                # Недостатній запас — або тримаємо попередній напрям, або hold
                if self.prefer_hold_when_uncertain:
                    return "neutral", 0.0
                return self.last_decision, self.last_confidence

        # 4) Той самий напрям → оновлюємо впевненість і йдемо далі
        self.last_confidence = proposed_confidence
        return self.last_decision, self.last_confidence
