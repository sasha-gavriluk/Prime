# Файл: NNUpdates/Neyron/model.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HybridExtractor(BaseFeaturesExtractor):
    """
    Кастомний екстрактор ознак для SB3.
    Приймає флетнені спостереження (batch, 30*4), робить reshape у (batch, 30, 4),
    проганяє через LSTM і повертає вектор фіч розміром features_dim.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        # ВАЖЛИВО: тут передаємо features_dim у батьківський клас
        super().__init__(observation_space, features_dim)
        # observation_space має форму (timesteps, n_features), напр. (30, 4)
        self.timesteps, self.n_features = observation_space.shape

        hidden_size = 128

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        # Лінійний шар у розмір _features_dim, який встановив BaseFeaturesExtractor
        self.linear = nn.Linear(hidden_size, self._features_dim)
        self.act = nn.ReLU()

    def forward(self, observations: Tensor) -> Tensor:
        """
        observations: тензор форми (batch, timesteps*n_features), бо MlpPolicy все флетнить.
        Перетворюємо у (batch, timesteps, n_features), далі LSTM -> останній крок -> Linear -> ReLU.
        """
        B = observations.shape[0]
        # Переконуємось, що тип float32
        x = observations.view(B, self.timesteps, self.n_features)
        lstm_out, _ = self.lstm(x)            # (B, T, hidden)
        last = lstm_out[:, -1, :]             # (B, hidden)
        feats = self.act(self.linear(last))   # (B, _features_dim)
        return feats
