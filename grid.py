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
        self.observation_space = Box(low=np.array([0, 0, -5, 0, 0]), high=np.array([1, 1, 10, 2, 1]))
        self.state = np.array([0, 0.2, 1, 0, 0])
        self.sim_length = 288*5
        self.day_length = 288

        # temp hardcoded values
        self.timestep = 0
        self.dt = 5/60 # 5 mins
        self.batt_cpct = 2.5
        self.batt_pwr = 0.3*self.batt_cpct
        self.price_norm = 77
        self.solar_norm = 0.35
        self.hydro_cost = 0.06
        self.hydro_prod = 100
        self.hydro_penalty = 10
        self.hydro_amount = 1000
        self.verbose = False
        
    def step(self, action: tuple[float, float]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        power_price = self.power_price[self.ts_start + self.timestep]
        power_price = max(power_price, 0)
        solar_power = self.solar_power[self.ts_start + self.timestep]

        if self.verbose:
            print(*self.state, ' ', *action)

        action_batt_energy = action[1] * self.batt_pwr * self.dt
        action_batt_energy = max(-self.state[1] * self.batt_cpct, action_batt_energy)
        action_batt_energy = min((1-self.state[1]) * self.batt_cpct, action_batt_energy)
        self.state[1] += action_batt_energy / self.batt_cpct

        action[0] = max(0, min(action[0], 1))
        action_hydro_energy = action[0] * self.hydro_prod * self.hydro_cost * self.dt
        self.state[4] += action_hydro_energy / (self.hydro_amount * self.hydro_cost)
        self.state[4] = min(1, self.state[4])

        grid_load = action_hydro_energy + action_batt_energy - solar_power*self.dt
        reward = -power_price * grid_load / self.price_norm
        self.state[0] += 1/self.day_length
        if self.state[0] >= 1:
            self.state[0] -= 1
            self.state[4] = 0
        self.state[2] = min(10, power_price / self.price_norm)
        self.state[3] = min(2, solar_power / self.solar_norm)
        self.timestep += 1

        done = self.timestep == self.sim_length
        info = {}
        if done:
            reward -= self.hydro_penalty * self.hydro_amount * (1 - self.state[4])
            reward += self.state[1] * self.batt_cpct * power_price
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
        self.ts_start = self.ts_start // self.day_length * self.day_length
        self.power_price = self.data['MW']
        self.solar_power = self.data['power']
        info = {'ts_start': self.ts_start}
        return self.state, info