import numpy as np
import random
import scipy.stats
from collections import deque

# 设置随机种子
np.random.seed(0)
random.seed(0)

# ================= 全局参数 =================
AREA_SIZE = 1500
NUM_AGV = 10
NUM_UAV = 4
NUM_TIMESLOTS = 100
TIMESLOT_DURATION = 1
UAV_HEIGHT = 60
UAV_SERVICE_RANGE = 300
AGV_SPEED_MAX = 15
AGV_SPEED_MIN = 10
AGV_DIRECTION_CHANGE = np.pi / 50
TASK_SIZE_MIN = 0.3e6
TASK_SIZE_MAX = 10e6
UAV_MAX_COMPUTE = 5e9
CPU_CYCLE_PER_BIT = 1000
RESULT_SIZE = 0.1e6
FIXED_STEP_PENALTY = -100

LOCAL_TASK_THRESHOLD = 0.5e6
AGV_LOCAL_COMPUTE = 0.8e6

AGV_TX_POWER = 1
UAV_TX_POWER = 5
BANDWIDTH = 5e6
NOISE_POWER = 1e-9
PATH_LOSS_AT_1M = 1e-5
SAFETY_DISTANCE = 20

MAX_AGV_SELECT = 5
CONTINUOUS_ACTION_DIM = 2
DISCRETE_ACTION_DIM = NUM_AGV


# ================= AGV 类 =================
class AGV:
    def __init__(self, id):
        self.id = id
        self.position = np.array([random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)])
        self.speed = random.uniform(AGV_SPEED_MIN, AGV_SPEED_MAX)
        self.direction = random.uniform(0, 2 * np.pi)
        self.task_size = 0
        self.next_position = self.position
        self.positions = [self.position.copy()]

        self.local_compute = AGV_LOCAL_COMPUTE
        self.is_local_task = False
        self.local_delay = 0.0
        self.associated_uav = None
        self.offload_delay = 0.0

    def generate_task(self):
        self.task_size = random.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)
        self.offload_delay = 0.0
        self.associated_uav = None
        self.is_local_task = self.task_size <= LOCAL_TASK_THRESHOLD
        if self.is_local_task:
            self.local_delay = (self.task_size * CPU_CYCLE_PER_BIT) / self.local_compute
        else:
            self.local_delay = 0.0

    def move(self):
        perturbation = random.uniform(-AGV_DIRECTION_CHANGE, AGV_DIRECTION_CHANGE)
        self.direction += perturbation
        dx = self.speed * TIMESLOT_DURATION * np.cos(self.direction)
        dy = self.speed * TIMESLOT_DURATION * np.sin(self.direction)
        next_x = self.position[0] + dx
        next_y = self.position[1] + dy

        if next_y >= AREA_SIZE - SAFETY_DISTANCE:
            next_y = AREA_SIZE - SAFETY_DISTANCE
            self.direction = np.pi - self.direction
        elif next_y <= SAFETY_DISTANCE:
            next_y = SAFETY_DISTANCE
            self.direction = np.pi - self.direction
        elif next_x <= SAFETY_DISTANCE:
            next_x = SAFETY_DISTANCE
            self.direction = -self.direction
        elif next_x >= AREA_SIZE - SAFETY_DISTANCE:
            next_x = AREA_SIZE - SAFETY_DISTANCE
            self.direction = -self.direction

        self.next_position = np.array([next_x, next_y])

    def update_position(self):
        self.position = self.next_position
        self.positions.append(self.position.copy())


# ================= UAV 类 =================
class UAV:
    def __init__(self, id):
        self.id = id
        predefined_positions = [[455, 469], [1048, 1202], [1065, 378], [255, 1170], [250, 250]]
        self.position = np.array(predefined_positions[self.id])
        self.positions = [self.position.copy()]
        self.compute_available = UAV_MAX_COMPUTE
        self.service_range = UAV_SERVICE_RANGE
        self.height = UAV_HEIGHT
        self.max_compute = UAV_MAX_COMPUTE
        self.velocity = np.array([0.0, 0.0])

    def move(self, cont_action):
        max_step = 5.0
        dx = cont_action[0] * max_step
        dy = cont_action[1] * max_step
        self.velocity = 0.7 * self.velocity + 0.3 * np.array([dx, dy])

        speed = np.linalg.norm(self.velocity)
        if speed > max_step:
            self.velocity = self.velocity * (max_step / speed)

        self.position[0] = np.clip(self.position[0] + self.velocity[0], 0, AREA_SIZE)
        self.position[1] = np.clip(self.position[1] + self.velocity[1], 0, AREA_SIZE)

        if self.position[0] < 10 or self.position[0] > AREA_SIZE - 10:
            self.velocity[0] = -0.5 * self.velocity[0]
        if self.position[1] < 10 or self.position[1] > AREA_SIZE - 10:
            self.velocity[1] = -0.5 * self.velocity[1]

        self.positions.append(self.position.copy())

    def in_service_range(self, agv_position):
        x_in_range = abs(agv_position[0] - self.position[0]) <= self.service_range
        y_in_range = abs(agv_position[1] - self.position[1]) <= self.service_range
        return x_in_range and y_in_range


def get_attraction_points(env):
    if NUM_UAV <= 4:
        points = []
        for i in range(2):
            for j in range(2):
                x = AREA_SIZE * (i + 0.5) / 2
                y = AREA_SIZE * (j + 0.5) / 2
                points.append(np.array([x, y]))
        return points[:NUM_UAV]
    else:
        points = []
        grid_size = int(np.ceil(np.sqrt(NUM_UAV)))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < NUM_UAV:
                    x = AREA_SIZE * (i + 0.5) / grid_size
                    y = AREA_SIZE * (j + 0.5) / grid_size
                    points.append(np.array([x, y]))
        return points


# ================= 环境类 =================
class Environment:
    def __init__(self):
        self.agvs = [AGV(i) for i in range(NUM_AGV)]
        self.uavs = [UAV(i) for i in range(NUM_UAV)]
        self.time_slot = 0
        self.CPU_CYCLE_PER_BIT = CPU_CYCLE_PER_BIT
        self.RESULT_SIZE = RESULT_SIZE
        self.PATH_LOSS_AT_1M = PATH_LOSS_AT_1M
        self.BANDWIDTH = BANDWIDTH
        self.NOISE_POWER = NOISE_POWER
        self.AGV_TX_POWER = AGV_TX_POWER
        self.UAV_TX_POWER = UAV_TX_POWER

        self.reward_history = []
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_alpha = 0.05
        self.last_rewards = deque(maxlen=20)
        self.reward_difficulty = 1.0
        self.historical_rewards = []
        self.avg_reward_window = deque(maxlen=10)

    def get_closest_uav_for_agv(self, agv):
        min_dist = float('inf')
        closest_uav = None
        for uav in self.uavs:
            dist = np.linalg.norm(agv.position - uav.position)
            if dist <= uav.service_range and dist < min_dist:
                min_dist = dist
                closest_uav = uav
        return closest_uav

    def reset(self):
        self.__init__()
        return self.get_state()

    def step(self, actions):
        raw_rewards = np.zeros(NUM_UAV)
        done = False

        for agv in self.agvs:
            agv.generate_task()
            agv.move()
            if not agv.is_local_task:
                agv.associated_uav = self.get_closest_uav_for_agv(agv)

        for uav_id, (cont_action, disc_action) in enumerate(actions):
            uav = self.uavs[uav_id]
            uav.compute_available = UAV_MAX_COMPUTE
            uav.move(cont_action)

            agv_ids_in_range = [
                agv.id for agv in self.agvs
                if uav.in_service_range(agv.position)
                   and (not agv.is_local_task)
                   and (agv.associated_uav is not None)
                   and (agv.associated_uav.id == uav.id)
            ]
            selected_agv_ids = []

            if agv_ids_in_range:
                action_values = disc_action[agv_ids_in_range]
                k = min(MAX_AGV_SELECT, len(agv_ids_in_range))
                top_indices = np.argsort(action_values)[::-1][:k]
                selected_agv_ids = [agv_ids_in_range[i] for i in top_indices]

            coverage_reward = len(selected_agv_ids) / NUM_AGV * 20

            closest_attraction_dist = float('inf')
            attraction_points = get_attraction_points(self)
            for point in attraction_points:
                dist = np.sqrt(((uav.position - point) ** 2).sum())
                closest_attraction_dist = min(closest_attraction_dist, dist)
            attraction_reward = 10 * np.exp(-0.02 * closest_attraction_dist)

            for agv_id in selected_agv_ids:
                agv = self.agvs[agv_id]
                compute_needed = agv.task_size * CPU_CYCLE_PER_BIT
                compute_allocated = min(compute_needed, uav.compute_available)
                uav.compute_available -= compute_allocated

                in_range_next = uav.in_service_range(agv.next_position)
                total_delay = self.calculate_total_delay(uav, agv, compute_allocated, in_range_next)
                agv.offload_delay = total_delay
                delay_reward = 100 * np.exp(-0.001 * total_delay * self.reward_difficulty)
                raw_rewards[uav_id] += delay_reward

            raw_rewards[uav_id] += (coverage_reward + attraction_reward) / self.reward_difficulty
            resource_used = (UAV_MAX_COMPUTE - uav.compute_available) / UAV_MAX_COMPUTE
            raw_rewards[uav_id] += 5 * resource_used / self.reward_difficulty

        for agv in self.agvs:
            agv.update_position()

        rewards = self.normalize_rewards(raw_rewards)
        avg_reward = np.mean(rewards)
        self.last_rewards.append(avg_reward)
        self.time_slot += 1
        if self.time_slot >= NUM_TIMESLOTS:
            done = True

        return self.get_state(), rewards, done

    def calculate_channel_gain(self, distance):
        return self.PATH_LOSS_AT_1M / (distance ** 2)

    def calculate_data_rate(self, channel_gain, is_upload=True):
        power = self.AGV_TX_POWER if is_upload else self.UAV_TX_POWER
        snr = (power * channel_gain) / self.NOISE_POWER
        snr = max(snr, 1e-10)
        return self.BANDWIDTH * np.log2(1 + snr)

    def calculate_total_delay(self, uav_u, agv_k, compute_allocated, in_range_next):
        try:
            distance = max(1.0, np.sqrt(
                (uav_u.position[0] - agv_k.position[0]) ** 2 +
                (uav_u.position[1] - agv_k.position[1]) ** 2 +
                uav_u.height ** 2
            ))
            channel_gain = self.calculate_channel_gain(distance)
            upload_rate = max(1.0, self.calculate_data_rate(channel_gain, is_upload=True))
            upload_delay = agv_k.task_size / upload_rate

            compute_allocated = max(1e-10, compute_allocated)
            compute_delay = (agv_k.task_size * CPU_CYCLE_PER_BIT) / compute_allocated

            download_rate = max(1.0, self.calculate_data_rate(channel_gain, is_upload=False))
            download_delay = RESULT_SIZE / download_rate

            total_delay = upload_delay + compute_delay + download_delay

            if not in_range_next:
                target_uav = self.find_uav_in_range(agv_k.next_position)
                if target_uav:
                    migration_distance = max(1.0, np.sqrt(
                        (uav_u.position[0] - target_uav.position[0]) ** 2 +
                        (uav_u.position[1] - target_uav.position[1]) ** 2
                    ))
                    migration_channel_gain = self.calculate_channel_gain(migration_distance)
                    migration_rate = max(1.0, self.calculate_data_rate(migration_channel_gain, is_upload=False))
                    migration_delay = RESULT_SIZE / migration_rate
                    total_delay += migration_delay
                else:
                    return 1000.0
            return min(1000.0, total_delay)
        except Exception as e:
            return 1000.0

    def find_uav_in_range(self, position):
        for uav in self.uavs:
            if uav.in_service_range(position):
                return uav
        return None

    def normalize_rewards(self, rewards):
        raw_reward_mean = np.mean(rewards)
        self.historical_rewards.append(raw_reward_mean)
        if not self.reward_history:
            self.reward_history.extend(rewards)
            return rewards

        self.avg_reward_window.append(raw_reward_mean)
        avg_reward = np.mean(list(self.avg_reward_window))

        episode_num = len(self.historical_rewards)
        boost_factor = min(1.5, 1.0 + 0.01 * (episode_num - 50)) if episode_num > 50 else 1.0

        self.reward_history.extend(rewards)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        history_array = np.array(self.reward_history)
        new_mean = np.mean(history_array)
        new_std = max(1.0, np.std(history_array))

        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * new_mean
        self.reward_std = (1 - self.reward_alpha) * self.reward_std + self.reward_alpha * new_std

        normalized_rewards = boost_factor * (rewards - self.reward_mean) / self.reward_std
        stabilized_rewards = []
        for r in normalized_rewards:
            r_stable = np.clip(r, -3, 5)
            if r_stable > 2:
                r_stable = 2 + np.log(r_stable - 1)
            elif r_stable < -2:
                r_stable = -2 - np.log(-r_stable - 1)
            stabilized_rewards.append(r_stable)

        return np.clip(np.array(stabilized_rewards) + 3, 0.1, 10.0)

    def is_converged(self, threshold=0.05):
        if len(self.last_rewards) < 20 or len(self.historical_rewards) < 100:
            return False
        if len(self.historical_rewards) >= 30:
            recent_history = self.historical_rewards[-30:]
            x = np.arange(len(recent_history))
            slope, _, _, _, _ = scipy.stats.linregress(x, recent_history)
            if slope < 0.01:
                reward_array = np.array(list(self.last_rewards))
                mean = np.mean(reward_array)
                std = np.std(reward_array)
                cv = std / max(1e-5, abs(mean))
                return cv < threshold
        return False

    def get_state(self):
        state = []
        for uav in self.uavs:
            state.append(self.get_observation(uav))
        return state

    def get_observation(self, uav):
        observed_agvs = []
        for agv in self.agvs:
            if uav.in_service_range(agv.position) and (not agv.is_local_task) and (agv.associated_uav is not None) and (agv.associated_uav.id == uav.id):
                observed_agvs.append({
                    'id': agv.id, 'position': agv.position, 'task_size': agv.task_size
                })
        return {
            'uav_id': uav.id, 'uav_position': uav.position,
            'compute_available': uav.compute_available, 'observed_agvs': observed_agvs
        }

    def set_reward_difficulty(self, difficulty):
        self.reward_difficulty = difficulty