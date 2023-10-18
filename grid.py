from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import random
from typing import Any, SupportsFloat

# action - (hydro, batt)
# observation - (time, batt_level, price, solar, hydro_done)
class Microgrid(Env):
    def __init__(self) -> None:
        self.action_space = Box(low=np.array([0, 0]), high=np.array([1, 1]))
        self.observation_space = Box(low=np.array([0, 0, -1, 0, 0]), high=np.array([1, 1, 10, 2, 1]))
        self.state = np.array([0, 0.2, 1, 0, 0])
        self.sim_length = 2880
        self.day_length = 288

        # temp hardcoded values
        self.timestep = 0
        self.dt = 5/60 # 5 mins
        self.solar_power = np.ones(self.sim_length)
        self.power_price = np.ones(self.sim_length)
        self.batt_cpct = 2.5
        self.batt_pwr = 0.3*self.batt_cpct
        self.price_norm = 77
        self.solar_norm = 0.35
        self.hydro_cost = 0.06
        self.hydro_prod = 100
        self.hydro_penalty = 10
        self.hydro_amount = 1000
        
    def step(self, action: tuple[float, float]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        power_price = self.power_price[self.ts_start + self.timestep]
        solar_power = self.solar_power[self.ts_start + self.timestep]

        # don't procude hydro after quota is reached
        action[0] = min(action[0], (1 - self.state[4]) * self.hydro_amount / self.hydro_prod / self.dt)
        batt_pwr = action[1] * 2 - 1
        # don't take energy from empty battery
        batt_pwr = min(self.state[1] / self.dt, batt_pwr)
        # don't put energy into a full battery
        batt_pwr = max(batt_pwr, (self.state[1] - 1) / self.dt)
        action_pwr = action * [self.hydro_prod*self.hydro_cost, self.batt_pwr]
        grid_load = (action_pwr[0] - action_pwr[1] - solar_power) * self.dt
        reward = power_price * grid_load / self.price_norm
        self.state[0] += 1/self.day_length
        if self.state[0] >= 1:
            self.state[0] -= 1
            self.state[4] = 0
        self.state[1] -= batt_pwr*self.dt
        self.state[2] = min(10, power_price / self.price_norm)
        self.state[3] = min(2, solar_power / self.solar_norm)
        self.state[4] += action[0] * self.hydro_prod / self.hydro_amount * self.dt
        self.timestep += 1
        done = self.timestep == self.sim_length
        info = {}
        if done:
            reward -= self.hydro_penalty * self.hydro_amount * (1 - self.state[4])
        return self.state, reward, done, done, info
    
    def render(self):
        pass
    
    def reset(self, data=None, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.state = np.array([0, 0.2, 1, 0, 0])
        self.timestep = 0
        if data is not None:
            self.data = data
        rng = random.Random(seed)
        self.ts_start = rng.randint(10000, len(self.data)-30000)
        self.power_price = self.data['price']
        self.solar_power = self.data['power']
        info = {'ts_start': self.ts_start}
        return self.state, info