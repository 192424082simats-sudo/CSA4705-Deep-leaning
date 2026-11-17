learning_rates = [0.001, 0.1, 0.9, 1.1]

for lr in learning_rates:

    w = 10

    for i in range(30):
        gradient = 2 * (w - 3)
        w = w - lr * gradient

    loss = (w - 3) ** 2

    print("Learning Rate:", lr)
    print("Final Weight:", w)
    print("Final Loss:", loss)
