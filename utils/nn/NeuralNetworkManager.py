# NeuralNetworkManager.py
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque


# =========================
# Константи та утиліти
# =========================

ActionArray = np.ndarray
ObsArray = np.ndarray

# Діапазони для мапінгу 10 безперервних дій моделі у твої налаштування
ACTION_RANGES = {
    0: (0.5, 5.0),   # default_risk_per_trade_pct (%)
    1: (1.0, 3.0),   # risk_reward_ratio
    2: (1.0, 50.0),  # leverage (int)
    3: (0.5, 2.0),   # smc_confidence
    4: (0.5, 2.0),   # pattern_score
    5: (0.5, 2.0),   # state_strength
    6: (-1.0, 1.0),  # contrarian_enabled (поріг >0)
    7: (0.5, 2.0),   # tf_5m
    8: (0.8, 2.5),   # tf_1h
    9: (1.0, 3.0),   # tf_4h
}

# Parameters adjusted by the neural network.
CONTROLLED_PARAMETERS = [
    "default_risk_per_trade_pct",
    "risk_reward_ratio",
    "leverage",
    "smc_confidence",
    "pattern_score",
    "state_strength",
    "contrarian_enabled",
    "tf_5m",
    "tf_1h",
    "tf_4h",
]


def get_controlled_parameters() -> list[str]:
    """Return a copy of parameters managed by the neural network."""
    return CONTROLLED_PARAMETERS.copy()


def _robust_z_to_unit(v: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Перетворення уніваріатного ряду в [-1, 1] через робастне Z (median/MAD) + кліп.
    Повертає масив тієї ж довжини.
    """
    x = pd.Series(np.asarray(v, dtype="float64"), copy=False)
    eps = 1e-9
    med = float(x.median())
    mad = float((x - med).abs().median())
    z = (x - med) / (1.4826 * max(mad, eps))
    z = np.asarray(np.clip(z, -3.0, 3.0) / 3.0, dtype=np.float32)
    return z


def _denorm_unit_to_range(u: float, lo: float, hi: float) -> float:
    """Мапінг із [-1,1] у [lo,hi]."""
    return float(lo + (float(u) + 1.0) * 0.5 * (hi - lo))


def _safe_std(a: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.std(a))


def _lazy_import_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        return PPO
    except Exception as e:
        raise ImportError(
            "Потрібна бібліотека stable_baselines3 для роботи з .zip моделлю. "
            "Встанови: pip install stable-baselines3[extra]"
        ) from e


# =========================
# КЛАС 1: LiveRLTrader
# =========================

class LiveRLTrader:
    """
    Лайв-адаптер для RL-моделі (SB3 PPO) під реальні свічки.

    - Тримає буфер спостережень форми (window_size, 4), де 4 канали:
        [нормована equity, rolling Sharpe-like (на equity), signal_confidence (на ціні),
         market_state (волатильність/режим)]
    - Раз на кожні `step_length` нових свічок формує слайс, обчислює канали,
      викликає model.predict(obs) і повертає словник із дією, налаштуваннями та діагностикою.

    Очікування:
      * Модель навчена з тими самими window_size та step_length (або еквівалентною логікою).
      * Дія моделі — вектор із 10 значень у [-1, 1].
    """

    def __init__(
        self,
        model_or_path: Union[str, Path, Any],
        device: str = "auto",
        window_size: int = 30,
        step_length: int = 12,
        inertia: float = 0.8,
        initial_equity: float = 10_000.0,
    ):
        """
        :param model_or_path: шлях до SB3 .zip або вже завантажена PPO-модель.
        :param device: "auto" | "cpu" | "cuda" | "cuda:0" | ...
        :param window_size: довжина буфера спостережень (рядів).
        :param step_length: скільки свічок становить один "крок" прийняття рішення.
        :param inertia: інерція/згладження дії [0..1], ближче до 1 — інертніші дії.
        :param initial_equity: стартове equity для ініціалізації буфера.
        """
        self.window_size = int(window_size)
        self.step_length = int(step_length)
        self.inertia = float(inertia)
        self.device = device

        # завантаження/прив’язка моделі
        if isinstance(model_or_path, (str, Path)):
            PPO = _lazy_import_sb3()
            self.model = PPO.load(str(model_or_path), device=device)
        else:
            # очікуємо SB3 PPO instance
            self.model = model_or_path

        # буфери
        self.eq_hist: deque[float] = deque([float(initial_equity)] * self.window_size, maxlen=self.window_size)
        self.obs_buf: deque[np.ndarray] = deque(
            [np.zeros(4, dtype=np.float32) for _ in range(self.window_size)],
            maxlen=self.window_size,
        )
        self.slice_closes: deque[float] = deque(maxlen=self.step_length)
        self.prev_action: np.ndarray = np.zeros(10, dtype=np.float32)

    # ---------- Публічні методи ----------

    def reset_buffers(self, initial_equity: Optional[float] = None) -> None:
        """Очистити буфери, опційно задати нове стартове equity."""
        init_eq = float(initial_equity) if initial_equity is not None else (self.eq_hist[-1] if self.eq_hist else 10_000.0)
        self.eq_hist.clear()
        self.eq_hist.extend([init_eq] * self.window_size)
        self.obs_buf.clear()
        self.obs_buf.extend([np.zeros(4, dtype=np.float32) for _ in range(self.window_size)])
        self.slice_closes.clear()
        self.prev_action[:] = 0.0

    def on_new_candle(
        self,
        close: float,
        account_equity: Optional[float] = None,
        deterministic: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Викликається на кожній новій свічці (на закритті бару).
        Коли назбиралося step_length свічок — рахує ознаки, робить прогноз і повертає результат.

        :param close: ціна закриття бару
        :param account_equity: поточне equity акаунта (якщо None — використовуємо попереднє)
        :param deterministic: детерміноване рішення політики SB3
        :return: dict{action, settings, obs, diagnostics} або None, якщо ще не дозібрався слайс
        """
        self.slice_closes.append(float(close))
        if account_equity is not None:
            self.eq_hist.append(float(account_equity))

        if len(self.slice_closes) < self.step_length:
            return None  # ще чекаємо слайс

        # ---- 1) Фічі на останньому слайсі цін ----
        df = pd.DataFrame({"close": list(self.slice_closes)})
        rets = df["close"].pct_change().dropna()
        vol = float(rets.std()) if len(rets) else 0.0
        drift = float(rets.mean()) if len(rets) else 0.0
        eps = 1e-12
        sharpe_like = drift / (vol + eps)
        signal_confidence = float(np.tanh(sharpe_like))  # [-1..1]

        # market_state: робастна стандартизація волатильності
        vol_series = rets.rolling(max(2, len(rets) // 3)).std().dropna()
        if len(vol_series) >= 5:
            q1, q3 = np.quantile(vol_series, [0.25, 0.75])
            iqr = max(q3 - q1, eps)
            center = (q1 + q3) / 2.0
            market_state = float(np.clip((vol - center) / (iqr / 1.349 + eps), -3.0, 3.0) / 3.0)
        else:
            market_state = float(np.tanh(vol * 50.0))

        # ---- 2) Фічі на equity ----
        eq = pd.Series(list(self.eq_hist), dtype="float64")
        log_eq = np.log(eq.clip(lower=1e-9))
        norm_eq_series = _robust_z_to_unit(log_eq)  # [-1..1] на всьому вікні
        norm_eq = float(norm_eq_series[-1])
        rets_eq = eq.pct_change().dropna()
        if len(rets_eq) >= 3:
            sh_like_eq = float(rets_eq.mean() / (rets_eq.std() + 1e-9))
        else:
            sh_like_eq = 0.0
        norm_sharpe = float(np.clip(sh_like_eq, -3.0, 3.0) / 3.0)

        # ---- 3) Формуємо спостереження та робимо прогноз ----
        obs_vec = np.array([norm_eq, norm_sharpe, signal_confidence, market_state], dtype=np.float32)
        self.obs_buf.append(obs_vec)
        obs: ObsArray = np.asarray(self.obs_buf, dtype=np.float32)  # (window_size, 4)

        action, _ = self.model.predict(obs, deterministic=deterministic)  # очікуємо форму (10,)
        action = np.asarray(action, dtype=np.float32)
        settings = self._map_action_to_settings(action)

        # Готуємо наступний слайс
        self.slice_closes.clear()

        diagnostics = {
            "norm_eq": norm_eq,
            "norm_sharpe": norm_sharpe,
            "signal_confidence": signal_confidence,
            "market_state": market_state,
        }

        return {"action": action, "settings": settings, "obs": obs, "diagnostics": diagnostics}

    # ---------- Внутрішні методи ----------

    def _map_action_to_settings(self, action: ActionArray) -> Dict[str, Any]:
        """
        Згладжує дію (інерція) та мапить її у діапазони твоїх налаштувань.
        Очікує action у [-1, 1] довжини 10.
        """
        if action.shape[0] != 10:
            raise ValueError(f"Очікував action довжини 10, отримав {action.shape[0]}")

        # Інерція (експоненційне згладження)
        smooth = self.inertia * self.prev_action + (1.0 - self.inertia) * action
        self.prev_action = smooth

        # Мапінг у твої параметри
        out = {
            "default_risk_per_trade_pct": np.clip(_denorm_unit_to_range(smooth[0], *ACTION_RANGES[0]), 0.5, 5.0),
            "risk_reward_ratio": np.clip(_denorm_unit_to_range(smooth[1], *ACTION_RANGES[1]), 1.0, 3.0),
            "leverage": int(np.clip(round(_denorm_unit_to_range(smooth[2], *ACTION_RANGES[2])), 1, 50)),
            "smc_confidence": np.clip(_denorm_unit_to_range(smooth[3], *ACTION_RANGES[3]), 0.5, 2.0),
            "pattern_score": np.clip(_denorm_unit_to_range(smooth[4], *ACTION_RANGES[4]), 0.5, 2.0),
            "state_strength": np.clip(_denorm_unit_to_range(smooth[5], *ACTION_RANGES[5]), 0.5, 2.0),
            "contrarian_enabled": float(1.0 if smooth[6] > 0.0 else 0.0),
            "tf_5m": np.clip(_denorm_unit_to_range(smooth[7], *ACTION_RANGES[7]), 0.5, 2.0),
            "tf_1h": np.clip(_denorm_unit_to_range(smooth[8], *ACTION_RANGES[8]), 0.8, 2.5),
            "tf_4h": np.clip(_denorm_unit_to_range(smooth[9], *ACTION_RANGES[9]), 1.0, 3.0),
        }
        return out


# =========================
# КЛАС 2: NeuralNetworkManager
# =========================

class NeuralNetworkManager:
    """
    Універсальний менеджер моделі:
      - завантажує SB3 PPO .zip або чистий PyTorch nn.Module (.pt/.pth),
      - дає єдиний інтерфейс summary()/predict(),
      - уміє створити LiveRLTrader для лайв-даних.

    Рекомендований формат для твоєї RL-моделі: **SB3 .zip**.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: str = "auto"):
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.device: str = device
        self.model: Optional[Any] = None
        self.backend: Optional[str] = None  # "sb3" | "torch"

    # ---------- Життєвий цикл ----------

    def load(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> Any:
        """
        Завантажити модель. Якщо шлях не передано — використовує self.model_path.
        Повертає завантажений об’єкт моделі.
        """
        if model_path is not None:
            self.model_path = Path(model_path)
        if device is not None:
            self.device = device

        if self.model is not None:
            return self.model

        if self.model_path is None:
            # fallback: іграшкова модель (щоб не падати в інтеграціях, де немає файлу)
            self.model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
            self.backend = "torch"
            return self.model

        suffix = self.model_path.suffix.lower()
        if suffix == ".zip":
            PPO = _lazy_import_sb3()
            self.model = PPO.load(str(self.model_path), device=self.device)
            self.backend = "sb3"
        else:
            # «чистий» PyTorch
            obj = torch.load(str(self.model_path), map_location=self.device if self.device in ("cpu", "mps") else None)
            if isinstance(obj, nn.Module):
                self.model = obj.to(self.device if self.device in ("cpu", "mps") else "cpu")
                self.model.eval()
                self.backend = "torch"
            elif isinstance(obj, dict):
                raise ValueError(
                    "Знайдено state_dict. Потрібна точна архітектура моделі, щоб викликати load_state_dict(...)."
                )
            else:
                raise ValueError("Невідомий формат моделі (очікував nn.Module або SB3 .zip).")
        return self.model

    def unload(self) -> None:
        """Вивантажити модель із пам’яті."""
        self.model = None
        self.backend = None

    def is_ready(self) -> bool:
        """Чи завантажена модель."""
        return self.model is not None

    # ---------- Інформація/інференс ----------

    def summary(self) -> None:
        """Надрукувати короткий опис моделі/політики."""
        if not self.is_ready():
            print("Модель не завантажена.")
            return
        if self.backend == "sb3":
            print(self.model.policy)  # тип політики PPO/архітектура
        else:
            print(self.model)

    def predict(self, obs: ObsArray, deterministic: bool = True) -> ActionArray:
        """
        Зробити прогноз на одному спостереженні.
        Для SB3: повертає дію (np.ndarray довжини 10).
        Для torch: повертає вихід мережі (np.ndarray).
        """
        if not self.is_ready():
            raise RuntimeError("Модель не завантажена. Виклич load().")

        if self.backend == "sb3":
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32)

        # Чистий torch: прогін уперед
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device="cpu")
            if x.ndim == 1:
                x = x.unsqueeze(0)
            out = self.model(x).detach().cpu().numpy()
        return out

    # ---------- Лайв-режим ----------

    def make_live_runner(
        self,
        window_size: int = 30,
        step_length: int = 12,
        inertia: float = 0.8,
        initial_equity: float = 10_000.0,
    ) -> LiveRLTrader:
        """
        Створити лайв-адаптер для реального стріму свічок.
        Працює лише з SB3-моделлю (.zip). Якщо модель ще не завантажена — завантажить.
        """
        if not self.is_ready():
            self.load()

        if self.backend != "sb3":
            raise RuntimeError(
                "LiveRLTrader очікує SB3 PPO .zip модель. "
                "Завантаж .zip через manager.load(...)."
            )

        # Передаємо сам шлях (щоб LiveRLTrader міг коректно робити predict без env)
        path = str(self.model_path) if self.model_path is not None else None
        if not path:
            # Якщо немає шляху (наприклад, модель інжектнули як об’єкт) — передамо саму модель
            model_or_path = self.model
        else:
            model_or_path = path

        return LiveRLTrader(
            model_or_path=model_or_path,
            device=self.device,
            window_size=window_size,
            step_length=step_length,
            inertia=inertia,
            initial_equity=initial_equity,
        )
