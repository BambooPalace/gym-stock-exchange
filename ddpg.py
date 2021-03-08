import gym
import gym_exchange
from collections import Counter, defaultdict
import time
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy


if __name__ == '__main__':
    # Create and wrap the environment
    # env = gym.make('game-stock-exchange-v0')
    env = gym.make('game-stock-exchange-continuous-v0')
    env = DummyVecEnv([lambda: env])

    model = DDPG(MlpPolicy, env, verbose=1)
    # model = A2C.load("a2c_gym_exchange_continuous", env=env)
    model.learning_rate = 1e-7

    # Train the agent
    model.learn(total_timesteps=100000) # default=1000000

    # Save the agent
    model.save("ddpg_gym_exchange_continuous")
    # del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = DDPG.load("ddpg_gym_exchange_continuous", env=env)

    # Enjoy trained agent
    obs = env.reset()
    actions = Counter()
    pnl = defaultdict(float)
    total_rewards = 0.0

    for i in range(300):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        actions[action[0].item()] += 1
        pnl[action[0].item()] += rewards
        total_rewards += rewards
        if dones:
            break

    print('actions : {}'.format(actions))
    print('rewards : {}'.format(total_rewards))
