import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")

def play(agent, env):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not (done or truncated):
        env.render()
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print("total reward episodes is {}".format(total_reward))

def train(agent, env, n_episodes): # è il “ciclo di allenamento” completo: fa giocare l’agente per tanti episodi e, alla fine di ogni episodio, lo aggiorna con REINFORCE
    for episode in range(n_episodes):
        done = False
        truncated = False
        obs, info = env.reset()
        total_reward = 0
        rewards = []
        observations = []
        actions = []
        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            observations.append(obs)
            actions.append(action)
        
            obs = next_obs
            total_reward += reward
        
            if (done or truncated):
                agent.train(observations, rewards, actions)
                print("total reward after {} episodes is {}".format(episode, total_reward))