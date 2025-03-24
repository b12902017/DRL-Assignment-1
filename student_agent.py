# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import pickle
import json



with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

"""
data = np.load("q_table.npz", allow_pickle=True)
keys = data["keys"]
values = data["values"]

q_table = {tuple(k): v for k, v in zip(keys, values)}
"""

action_space = list(range(6))

"""
for (state, action_values) in q_table.items():
    print(f"State: {state}")
    print(f"Q-values: {action_values}")
"""


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

    if memory.get("previous_action") == 4 and phase == 0:
        for station in stations:
            if (taxi_row, taxi_col) == station:
                picked.add(station)
    if memory.get("previous_action") == 5 and phase == 1:
        for station in stations:
            if (taxi_row, taxi_col) == station:
                picked.add(station)

    for s in stations:
        if (phase == 0 and s in picked) or (phase == 1 and s in dropped):
            continue

        sy, sx = s
        dist = abs(sy - taxi_row) + abs(sx - taxi_col)
        if dist <= 1:
            if phase == 0 and passenger_look == 0:
                picked.add(s)
            elif phase == 1 and destination_look == 0:
                dropped.add(s)
        if dist > 1:
            if phase == 0 and passenger_look == 1:
                picked.add(s)
            elif phase == 1 and destination_look == 1:
                dropped.add(s)

    if phase == 0 and len(picked) == 4:
        memory["phase"] = 1
        memory["dropped_stations"] = set()

    rel_dirs = []
    for s in stations:
        dy = sign(s[0] - taxi_row)
        dx = sign(s[1] - taxi_col)
        if (phase == 0 and s in picked) or (phase == 1 and s in dropped):
            rel_dirs.extend([None, None])
        else:
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


def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    station_coords = obs[2:10]
    stations = [(station_coords[i], station_coords[i + 1]) for i in range(0, 8, 2)]

    if not hasattr(get_action, "memory"):
        get_action.memory = {
            "phase": 0,
            "picked_stations": set(),
            "dropped_stations": set(),
            "stations": stations,
            "previous_action": None,
        }

    memory = get_action.memory
    state = get_state(obs, memory)

    if state in q_table:
        action = int(np.argmax(q_table[state]))
    else:
        action = random.choice(range(6))

    print(f"State: {state}")
    print(f"Action: {action}")

    memory["previous_action"] = action
    return action
    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
