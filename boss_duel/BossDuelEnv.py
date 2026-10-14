import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BossDuelEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        # 游戏参数
        self.max_health = 100
        self.boss_health_max = 200
        self.max_energy = 100
        self.max_stamina = 100
        self.max_steps = 50

        # 动作空间：4个离散动作
        self.action_space = spaces.Discrete(4)
        # 观测空间：8维向量，范围0~1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # Boss 动作概率（轻击、重击、蓄力）
        self.boss_probs = [0.4, 0.5, 0.1]

        # 伤害值
        self.agent_normal_damage = 10
        self.agent_ult_damage = 30
        self.boss_light_damage = 10
        self.boss_heavy_damage = 25

        # 动作消耗/收益
        self.stamina_cost_dodge = 20
        self.energy_gain_normal = 10
        self.energy_gain_dodge = 5
        self.energy_gain_block = 15

        self.reset()

    def _get_state_with_boss_action(self, boss_action):
        boss_onehot = np.zeros(3)
        boss_onehot[boss_action] = 1
        state = np.concatenate([
            boss_onehot,
            [self.agent_health / self.max_health],
            [self.boss_health / self.boss_health_max],
            [self.agent_energy / self.max_energy],
            [self.agent_stamina / self.max_stamina],
            [self.boss_charge]
        ])
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 必须调用，用于设置随机种子
        self.agent_health = self.max_health
        self.agent_energy = 0
        self.agent_stamina = self.max_stamina
        self.boss_charge = 0
        self.boss_health = self.boss_health_max
        self.steps = 0
        boss_action = np.random.choice([0, 1, 2], p=self.boss_probs)
        self.state = self._get_state_with_boss_action(boss_action)
        info = {}  # 可以放一些调试信息
        return self.state, info

    def step(self, action):
        self.steps += 1

        boss_action = np.argmax(self.state[:3])
        is_boss_charged = (self.boss_charge == 1)
        self.boss_charge = 0

        reward = 0
        agent_attack_damage = 0
        agent_taken_damage = 0

        # ========== 先处理 Boss 动作 ==========
        if boss_action == 2:  # Boss 蓄力
            self.boss_charge = 1
            # 打断判断：大招且能量≥一半
            if action == 2 and self.agent_energy >= self.max_energy / 2 - 1e-5:
                self.boss_charge = 0
                reward += 10

        # ========== 处理 Agent 动作 ==========
        # 大招
        if action == 2:
            if self.agent_energy >= self.max_energy / 2 - 1e-5:
                damage = self.agent_ult_damage
                if np.random.rand() < 0.2:
                    damage = int(damage * 1.5)
                agent_attack_damage = damage
                reward += damage * 0.1
                reward += 5
                self.agent_energy = max(0, self.agent_energy - self.max_energy / 2)
            else:
                # 能量不足，转为普攻
                action = 3

        # 普攻（主动或转来）
        if action == 3:
            damage = self.agent_normal_damage + np.random.randint(-2, 3)
            agent_attack_damage = damage
            reward += damage * 0.1
            self.agent_energy = min(self.max_energy, self.agent_energy + self.energy_gain_normal)

        # 应用 Agent 伤害
        self.boss_health -= agent_attack_damage

        # ========== 处理 Boss 攻击 ==========
        if boss_action != 2:
            base_damage = self.boss_heavy_damage if boss_action == 0 else self.boss_light_damage
            if is_boss_charged:
                base_damage = int(base_damage * 2)

            if action == 0:  # 格挡
                if is_boss_charged:
                    actual_damage = base_damage
                else:
                    actual_damage = int(base_damage * 0.2)
                self.agent_energy = min(self.max_energy, self.agent_energy + self.energy_gain_block)

            elif action == 1:  # 闪避
                if self.agent_stamina >= self.stamina_cost_dodge:
                    self.agent_stamina -= self.stamina_cost_dodge
                    if is_boss_charged:
                        actual_damage = base_damage
                    else:
                        actual_damage = 0
                        reward += 1
                    self.agent_energy = min(self.max_energy, self.agent_energy + self.energy_gain_dodge)
                else:
                    actual_damage = base_damage
            else:
                actual_damage = base_damage

            agent_taken_damage = actual_damage
            self.agent_health -= agent_taken_damage
            reward -= agent_taken_damage * 0.2

        # 体力自然恢复
        self.agent_stamina = min(self.max_stamina, self.agent_stamina + 5)

        # 边界限制
        self.agent_health = max(0, min(self.max_health, self.agent_health))
        self.boss_health = max(0, min(self.boss_health_max, self.boss_health))
        self.agent_energy = max(0, min(self.max_energy, self.agent_energy))
        self.agent_stamina = max(0, min(self.max_stamina, self.agent_stamina))

        # 终止判断（gymnasium 用 terminated 和 truncated）
        terminated = False
        truncated = False
        if self.agent_health <= 0 or self.boss_health <= 0:
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        # 奖励处理
        if self.agent_health <= 0:
            reward -= 200
        elif self.boss_health <= 0:
            reward += 200
        elif truncated:
            reward += (self.agent_health - self.boss_health) / 10

        # 生成下一状态
        if not (terminated or truncated):
            next_boss_action = np.random.choice([0, 1, 2], p=self.boss_probs)
            next_state = self._get_state_with_boss_action(next_boss_action)
        else:
            next_state = np.zeros(8, dtype=np.float32)

        self.state = next_state

        info = {
            'boss_action': boss_action,
            'agent_action': action,
            'damage_dealt': agent_attack_damage,
            'damage_taken': agent_taken_damage
        }

        return next_state, reward, terminated, truncated, info