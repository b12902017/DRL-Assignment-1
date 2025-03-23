import numpy as np
import random
from simple_custom_taxi_env import SimpleTaxiEnv
import pickle

# Training parameters
episodes = 20000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_end = 0.05
decay_rate = 0.9999

env = SimpleTaxiEnv()
q_table = {}
rewards_per_episode = []
action_space = list(
    range(6)
)  # 0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff


def get_state(obs, memory):

    def sign(x):
        return (x > 0) - (x < 0)

    taxi_row, taxi_col = obs[0], obs[1]
    obstacle_and_flags = obs[10:]

    stations = memory.get("stations", [])
    phase = memory.get("phase", 0)
    picked = memory.get("picked_stations", set())
    dropped = memory.get("dropped_stations", set())
    passenger_look = obstacle_and_flags[-2]
    destination_look = obstacle_and_flags[-1]

    """
    if memory.get("previous_action") == 4 and phase == 0:
        for station in stations:
            if (taxi_row, taxi_col) == station:
                picked.add(station)
    if memory.get("previous_action") == 5 and phase == 1:
        for station in stations:
            if (taxi_row, taxi_col) == station:
                picked.add(station)'
    """

    for s in stations:
        if s in picked or (phase == 1 and s in dropped):
            continue

        sy, sx = s
        dist = abs(sy - taxi_row) + abs(sx - taxi_col)
        if dist == 1:
            if phase == 0 and passenger_look == 0:
                picked.add(s)
            elif phase == 1 and destination_look == 0:
                dropped.add(s)

    if phase == 0 and len(picked) == 4:
        memory["phase"] = 1
        memory["dropped_stations"] = set()
    else:
        memory["phase"] = phase

    memory["picked_stations"] = picked

    rel_dirs = []
    for s in stations:
        dy = sign(s[0] - taxi_row)
        dx = sign(s[1] - taxi_col)
        rel_dirs.extend([dy, dx])

    picked_flags = [int(s in picked) for s in stations] if phase == 0 else [0] * 4
    dropped_flags = [int(s in dropped) for s in stations] if phase == 1 else [0] * 4

    state = tuple(
        rel_dirs
        + picked_flags
        + dropped_flags
        + list(obstacle_and_flags)
        + [memory["phase"]]
    )
    return state


def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


for episode in range(episodes):
    obs, _ = env.reset()
    station_coords = obs[2:10]
    stations = [(station_coords[i], station_coords[i + 1]) for i in range(0, 8, 2)]

    memory = {
        "phase": 0,
        "picked_stations": set(),
        "dropped_stations": set(),
        "stations": stations,
        "previous_action": None,
    }

    state = get_state(obs, memory)
    done = False
    total_reward = 0
    steps = 0

    taxi_pos = (obs[0], obs[1])
    prev_picked_len = 0
    prev_dropped_len = 0

    while not done:
        if state not in q_table:
            q_table[state] = np.zeros(len(action_space))

        if np.random.random() < epsilon:
            action = random.choice(action_space)
        else:
            action = int(np.argmax(q_table[state]))

        obs, reward, done, _ = env.step(action)
        memory["previous_action"] = action
        next_state = get_state(obs, memory)

        shaped_reward = 0

        if action == 4 and reward > -1:
            shaped_reward += 10

        new_taxi_pos = (obs[0], obs[1])
        current_targets = [
            s
            for s in stations
            if (
                s not in memory["picked_stations"]
                if memory["phase"] == 0
                else s not in memory["dropped_stations"]
            )
        ]

        """
        if current_targets:
            old_dists = [manhattan(taxi_pos, s) for s in current_targets]
            new_dists = [manhattan(new_taxi_pos, s) for s in current_targets]
            if min(new_dists) < min(old_dists):
                shaped_reward += 0.3'
        """
        """
        new_picked_len = len(memory["picked_stations"])
        new_dropped_len = len(memory["dropped_stations"])
        if new_picked_len > prev_picked_len:
            shaped_reward += 2
        if new_dropped_len > prev_dropped_len:
            shaped_reward += 2
        prev_picked_len = new_picked_len
        prev_dropped_len = new_dropped_len
        """

        taxi_pos = new_taxi_pos

        reward += shaped_reward

        total_reward += reward
        steps += 1

        if next_state not in q_table:
            q_table[next_state] = np.zeros(len(action_space))

        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state

    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon_end, epsilon * decay_rate)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(
            f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}"
        )

# Save Q-table
"""
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("Q-table saved to q_table.pkl")'
"""

keys = list(q_table.keys())
values = np.array(list(q_table.values()))
np.savez_compressed("q_table.npz", keys=keys, values=values)
