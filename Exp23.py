import numpy as np

q_table = np.zeros((3, 2))

learning_rate = 0.1
discount = 0.9
epsilon = 0.15

for episode in range(500):

    state = np.random.randint(0, 3)

    for step in range(20):

        if np.random.random() < epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(q_table[state])

        if (state == 2 and action == 1) or (
            state == 0 and action == 0
        ):
            reward = 2
        else:
            reward = -1

        next_state = np.clip(
            state + np.random.choice([-1, 0, 1]),
            0, 2
        )

        q_table[state, action] += learning_rate * (
            reward
            + discount * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state

print("Learned Q Table:")
print(q_table)

for state, name in enumerate(["Low", "Medium", "High"]):
    action = np.argmax(q_table[state])

    if action == 0:
        print(name, "Traffic: Short Green Signal")
    else:
        print(name, "Traffic: Long Green Signal")
