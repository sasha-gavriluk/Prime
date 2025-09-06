# Файл: NNUpdates/Neyron/agent.py (Оновлена версія)

import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

# Імпортуємо наші кастомні класи
from NNUpdates.Neyron.environment import TradingEnv
from NNUpdates.Neyron.model import HybridExtractor

class TradingAgent:
    """
    Керує процесом навчання та використання натренованої моделі.
    """
    def __init__(self, env: TradingEnv, log_path: str = "data/NNUpdates/Neyron/logs", model_save_path: str = "data/NNUpdates/Neyron/models"):
        self.env = env
        self.log_path = log_path
        self.model_save_path = model_save_path
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Створюємо словник для політики, де вказуємо наш кастомний extractor
        policy_kwargs = dict(
            features_extractor_class=HybridExtractor,
            features_extractor_kwargs=dict(features_dim=128), # Розмір виходу нашого LSTM
            net_arch=dict(pi=[64, 64], vf=[64, 64]) # Архітектура для Actor (pi) і Critic (vf) голів
        )
        
        # Ініціалізація PPO моделі
        self.model = PPO(
            "MlpPolicy", # Ми використовуємо MlpPolicy, але з кастомним extractor'ом
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.log_path,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            learning_rate=3e-4,
            device="cuda" if th.cuda.is_available() else "cpu"
        )
        print(f"🤖 Агент PPO створено. Навчання буде на: {self.model.device}")

    def train(self, total_timesteps: int = 100000):
        """Запускає процес навчання."""
        print(f"\n🔥 Початок навчання на {total_timesteps} кроків...")
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.model_save_path,
            name_prefix="trading_model"
        )
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        final_model_path = os.path.join(self.model_save_path, "final_trading_model.zip")
        self.model.save(final_model_path)
        print(f"✅ Навчання завершено. Фінальну модель збережено в: {final_model_path}")

    def load_model(self, path: str):
        """Завантажує натреновану модель."""
        self.model = PPO.load(path, env=self.env)
        print(f"🧠 Модель успішно завантажено з: {path}")

    def run_inference(self, episodes: int = 1):
        """Запускає натреновану модель для демонстрації."""
        print("\n🎬 Запуск демонстрації натренованої моделі...")
        for ep in range(episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
            print(f"   - Епізод {ep+1}: Фінальна винагорода (Sharpe Ratio) = {total_reward:.4f}")