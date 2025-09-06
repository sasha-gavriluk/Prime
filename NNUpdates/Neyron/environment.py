# Файл: Updates/Neyron/environment.py (Фінальна версія)

import gymnasium as gym
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional
from collections import deque

# Імпорти з вашого проєкту
from utils.common.SettingsLoader import DecisionSettingsManager, FinancialSettingsManager
from NNUpdates.Neyron.skills.loader import apply_skills, parse_manifest

from NNUpdates.Backtesting.simplified_backtester import SimplifiedBacktester

class SettingDropout:
    """Manages random on/off switching of settings during training."""
    def __init__(self, p_off: float = 0.0, p_midflip: float = 0.0,
                 hold_steps_min: int = 1, hold_steps_max: int = 1,
                 eval_mode: bool = False):
        self.p_off = float(p_off)
        self.p_midflip = float(p_midflip)
        self.hold_min = int(hold_steps_min)
        self.hold_max = int(hold_steps_max)
        self.eval_mode = bool(eval_mode)
        self._mask: Optional[np.ndarray] = None
        self._counter = 0
        self._hold = self.hold_min

    def reset(self, base_mask: np.ndarray) -> np.ndarray:
        size = len(base_mask)
        self._mask = np.ones(size, dtype=np.float32)
        if not self.eval_mode:
            for i in range(size):
                if random.random() < self.p_off:
                    self._mask[i] = 0.0
        self._counter = 0
        self._hold = random.randint(self.hold_min, self.hold_max)
        return self._mask

    def step(self) -> np.ndarray:
        if self.eval_mode or self._mask is None:
            return self._mask if self._mask is not None else np.array([])
        self._counter += 1
        if self._counter >= self._hold and len(self._mask):
            if random.random() < self.p_midflip:
                idx = random.randrange(len(self._mask))
                self._mask[idx] = 1.0 - self._mask[idx]
            self._counter = 0
            self._hold = random.randint(self.hold_min, self.hold_max)
        return self._mask

class TradingEnv(gym.Env):
    """
    RL-середовище з "кривими": (equity, rolling_sharpe_like, signal_confidence, market_state).
    Всі спостереження нормуються у [-1, 1]. Дії керують 10 параметрами з інерцією.
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        market_season_files: List[str],
        initial_capital: float = 10_000.0,
        step_length: int = 24,
        window_size: int = 60,
        fee_pct: float = 0.0005,
        slip_pct: float = 0.0002,
        action_inertia: float = 0.8,
        skills_manifest: Optional[str] = None,
        profile_path: Optional[str] = None,
        eval_mode: bool = False,
        p_off: float = 0.0,
        p_midflip: float = 0.0,
        hold_steps_min: int = 10,
        hold_steps_max: int = 30,
    ):
        super().__init__()

        # --- Базові налаштування ---
        self.market_season_files = list(market_season_files)
        self.initial_capital = float(initial_capital)
        self.step_length = int(step_length)
        self.window_size = int(window_size)
        self.fee_pct = float(fee_pct)
        self.slip_pct = float(slip_pct)
        self.action_inertia = float(action_inertia)
        self.skills_manifest = skills_manifest
        self.profile_path = profile_path
        self.eval_mode = bool(eval_mode)
        self.p_off = float(p_off)
        self.p_midflip = float(p_midflip)
        self.hold_steps_min = int(hold_steps_min)
        self.hold_steps_max = int(hold_steps_max)

        self.manual_mask = np.ones(10, dtype=np.float32)
        self.current_mask = self.manual_mask.copy()
        self.dropout = None
        if self.p_off > 0 or self.p_midflip > 0:
            self.dropout = SettingDropout(self.p_off, self.p_midflip,
                                          self.hold_steps_min, self.hold_steps_max,
                                          self.eval_mode)

        # --- Простори дій та спостережень ---
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.window_size, 4), dtype=np.float32)

        # --- Тіньові налаштування ---
        self.fin_settings_manager = FinancialSettingsManager()
        self.dec_settings_manager = DecisionSettingsManager()
        self.shadow_settings = self._load_initial_settings()
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # --- Стан епізоду ---
        self.obs_buffer = deque(maxlen=self.window_size)
        self.equity_history = None
        self.processed_data = None
        self.backtester = None
        self.current_step_index = 0

    # ---------- Життєвий цикл епізоду ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1) Обираємо сезон
        season_path = random.choice(self.market_season_files)
        self.manual_mask = np.ones(10, dtype=np.float32)

        # 2) Завантажуємо/підготовлюємо дані сезону
        # Якщо маєте власний DataProcessingManager — використайте його тут.
        # dpm = DataProcessingManager(data=pd.read_csv(season_path))
        # df = dpm.process_all()
        df = pd.read_csv(season_path)
        # Нормалізація назв колонок (критично для 'close')
        df.columns = [c.strip().lower() for c in df.columns]
        if 'close' not in df.columns:
            raise ValueError(f"CSV '{season_path}' must contain 'close' column (after lowercasing).")
        self.processed_data = df.reset_index(drop=True)

        manifest_path = self.profile_path or self.skills_manifest
        if manifest_path:
            self.processed_data = apply_skills(self.processed_data, manifest_path)
            self.manual_mask = self._mask_from_manifest(manifest_path)
        drop_mask = self.dropout.reset(self.manual_mask) if self.dropout else np.ones_like(self.manual_mask)
        self.current_mask = self.manual_mask * drop_mask

        # 3) Ініціалізуємо бектестер (один раз на епізод)
        self.backtester = SimplifiedBacktester(self.processed_data, fee_pct=self.fee_pct, slip_pct=self.slip_pct)

        # 4) Скидаємо стан портфеля і лічильники
        self.portfolio_equity = self.initial_capital
        self.current_step_index = 0
        self.equity_history = [self.initial_capital]

        # 5) Стартовий буфер спостережень (заповнюємо нулями)
        self.obs_buffer.clear()
        for _ in range(self.window_size):
            self.obs_buffer.append(np.zeros(4, dtype=np.float32))

        # 6) Нульова дія на старті
        self.prev_action = np.zeros_like(self.prev_action, dtype=np.float32)

        return np.array(self.obs_buffer, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        # 1) Застосовуємо дію з маскою та інерцією
        action = np.asarray(action, dtype=np.float32)
        mask = self.manual_mask
        if self.dropout:
            drop_mask = self.dropout.step()
            mask = self.manual_mask * drop_mask
        self.current_mask = mask
        action = action * mask
        self._apply_action_with_inertia(action)

        # 2) Індекси слайсу
        start_idx = self.current_step_index * self.step_length
        end_idx = start_idx + self.step_length

        # Якщо дані закінчились — термінуємо епізод
        if start_idx >= len(self.processed_data) - 1:
            obs = np.array(self.obs_buffer, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # 3) Швидкий бектест на поточному слайсі
        sim = self.backtester.run_slice(start_idx, end_idx, self.shadow_settings, equity=self.portfolio_equity)

        # 4) Оновлюємо портфель
        prev_equity = self.portfolio_equity
        self.portfolio_equity = max(0.0, self.portfolio_equity + sim["pnl"])
        self.equity_history.append(self.portfolio_equity)

        # 5) Обчислюємо винагороду
        reward = self._calculate_reward(prev_equity, sim, action)

        # 6) Оновлюємо буфер спостережень (криві)
        self._update_observation_buffer(sim)

        # 7) Крок + завершення
        self.current_step_index += 1
        terminated = (end_idx >= len(self.processed_data) - 1) or (self.portfolio_equity <= 0.0)
        truncated = False

        obs = np.array(self.obs_buffer, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), {}

    # ---------- Допоміжні методи ----------

    def _load_initial_settings(self) -> dict:
        """Завантажує початкові налаштування у RAM (без IO у step)."""
        settings: Dict = {}
        # Фінансові
        fin = self.fin_settings_manager.get_financial_settings()
        if isinstance(fin, dict):
            settings.update(fin)
        # Ваги таймфреймів/метрик (може знадобитися у ваших логіках)
        tf = getattr(self.dec_settings_manager, "get_timeframe_weights", lambda: {})()
        mw = getattr(self.dec_settings_manager, "get_metric_weights", lambda: {})()
        if isinstance(tf, dict):
            settings.update(tf)
        if isinstance(mw, dict):
            settings.update(mw)
        return settings
    
    def _mask_from_manifest(self, manifest_path: str) -> np.ndarray:
        skills, _, tf_flags = parse_manifest(manifest_path)
        mask = np.ones(self.action_space.shape[0], dtype=np.float32)
        for sc in skills:
            if sc.name == 'pattern_engulfing' and not sc.on:
                mask[4] = 0.0
        tf_index_map = {'TF_M5_on': 7, 'TF_1H_on': 8, 'TF_4H_on': 9}
        for key, idx in tf_index_map.items():
            val = tf_flags.get(key)
            if val is not None and str(val).lower() in {'0', 'false'}:
                mask[idx] = 0.0
        return mask

    def _apply_action_with_inertia(self, action: np.ndarray):
        """
        Інерція: new = inertia*prev + (1-inertia)*action.
        Денормалізація + clamp у фізичні межі. Зберігаємо у self.shadow_settings.
        """
        inertia = self.action_inertia
        smoothed = inertia * self.prev_action + (1.0 - inertia) * action
        self.prev_action = smoothed

        # --- Мапінг 10 параметрів ---
        # 0) ризик на угоду, 0.5..5.0 (%)
        self.shadow_settings['default_risk_per_trade_pct'] = float(np.clip(self._denormalize(smoothed[0], 0.5, 5.0), 0.5, 5.0))
        # 1) RR 1.0..3.0
        self.shadow_settings['risk_reward_ratio'] = float(np.clip(self._denormalize(smoothed[1], 1.0, 3.0), 1.0, 3.0))
        # 2) плече 1..50
        self.shadow_settings['leverage'] = int(np.clip(self._denormalize(smoothed[2], 1.0, 50.0), 1, 50))

        # 3-5) ваги метрик 0.5..2.0
        self.shadow_settings['smc_confidence'] = float(np.clip(self._denormalize(smoothed[3], 0.5, 2.0), 0.5, 2.0))
        self.shadow_settings['pattern_score']   = float(np.clip(self._denormalize(smoothed[4], 0.5, 2.0), 0.5, 2.0))
        self.shadow_settings['state_strength']  = float(np.clip(self._denormalize(smoothed[5], 0.5, 2.0), 0.5, 2.0))

        # 6) контртренд on/off — збережемо як вагу у [0,1]
        self.shadow_settings['contrarian_enabled'] = float(1.0 if smoothed[6] > 0 else 0.0)

        # 7-9) ваги ТФ
        self.shadow_settings['tf_5m'] = float(np.clip(self._denormalize(smoothed[7], 0.5, 2.0), 0.5, 2.0))
        self.shadow_settings['tf_1h'] = float(np.clip(self._denormalize(smoothed[8], 0.8, 2.5), 0.8, 2.5))
        self.shadow_settings['tf_4h'] = float(np.clip(self._denormalize(smoothed[9], 1.0, 3.0), 1.0, 3.0))

    def _calculate_reward(self, prev_equity: float, sim_results: dict, raw_action: np.ndarray) -> float:
        """
        Економічна винагорода: лог-прибуток - витрати - штраф за ривки дій.
        """
        eps = 1e-12
        if prev_equity <= 0.0:
            return 0.0

        # лог-повернення
        ret = (self.portfolio_equity - prev_equity) / (prev_equity + eps)
        reward = np.log1p(ret)  # стабільніше за сирий ret

        # витрати (fees+slippage) у частках еквіті
        fees = float(sim_results.get("fees", 0.0))
        slip = float(sim_results.get("slippage", 0.0))
        reward -= 0.1 * ((fees + slip) / (prev_equity + eps))  # λ = 0.1

        # плавність дій (штраф за різкі зміни)
        action_change = np.linalg.norm(raw_action - self.prev_action)
        reward -= 0.01 * action_change  # α = 0.01

        # клік і межі
        if not np.isfinite(reward):
            reward = -1.0
        reward = float(np.clip(reward, -5.0, 5.0))
        return reward

    def _update_observation_buffer(self, sim_results: dict):
        """
        Формуємо 4 криві:
          1) нормована крива equity ([-1,1]),
          2) rolling "Sharpe-like" за останні window_size повернень ([-1,1]),
          3) signal_confidence зі слайсу ([-1,1]),
          4) market_state зі слайсу ([-1,1]).
        """
        # 1) equity -> лог, робастний z-score, стиск у [-1,1]
        eq = pd.Series(self.equity_history[-self.window_size:])
        log_eq = np.log(eq.clip(lower=1e-9))
        norm_eq = self._to_unit_range_from_z(log_eq)

        # 2) rolling Sharpe-like на хвості
        rets = eq.pct_change().dropna()
        if len(rets) >= 3:
            sharpe_like = rets.mean() / (rets.std() + 1e-9)
        else:
            sharpe_like = 0.0
        norm_sharpe = float(np.clip(sharpe_like, -3.0, 3.0) / 3.0)

        # 3) 4) з бектестера
        signal_confidence = float(np.clip(sim_results.get("signal_confidence", 0.0), -1.0, 1.0))
        market_state = float(np.clip(sim_results.get("market_state", 0.0), -1.0, 1.0))

        obs_vec = np.array([norm_eq, norm_sharpe, signal_confidence, market_state], dtype=np.float32)
        self.obs_buffer.append(obs_vec)

    # ---------- Утіліти ----------

    @staticmethod
    def _denormalize(value: float, min_val: float, max_val: float) -> float:
        """Перетворює значення з [-1, 1] у [min_val, max_val]."""
        return float(min_val + (float(value) + 1.0) * 0.5 * (max_val - min_val))

    @staticmethod
    def _to_unit_range_from_z(series: pd.Series) -> float:
        """
        Робастне z-score на вікні, потім стискаємо до [-1,1]:
        z = (x - median) / (1.4826 * MAD + eps), потім clip [-3,3] і /3.
        Повертаємо ОСТАННЄ значення вікна.
        """
        eps = 1e-9
        if len(series) == 0:
            return 0.0
        x = series.values.astype(np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        z = (x - med) / (1.4826 * mad)
        z_last = float(z[-1])
        return float(np.clip(z_last, -3.0, 3.0) / 3.0)
