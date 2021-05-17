from gym.envs.toy_text.blackjack import BlackjackEnv
import sys


def print_obs(obs):
    print(f"Sum: {obs[0]}")


def episode(env):
    in_process = True
    obs = env.reset()
    print_obs(obs)

    while in_process:
        print("You move")
        a = sys.stdin.readline()
        a = a[0]

        while not a.isnumeric():
            print("Try another command.")
            a = sys.stdin.readline()[0]

        obs, reward, done, _ = env.step(int(a))
        print_obs(obs)

        in_process = not done
        if done:
            if reward > 0:
                print("WINNER!")
            else:
                print("LOSER!")


def start():
    counter = 0
    env = BlackjackEnv()
    print("ACTIONS:\n\t0 - stand\n\t1 - hit\n-------------------\n")
    while True:
        print("\nGame #" + str(counter))
        episode(env)
        counter += 1


start()