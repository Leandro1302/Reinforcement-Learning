import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
from utils import  train, env, play


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(30, activation="relu")
        self.dense2 = tf.keras.layers.Dense(30, activation="relu")
        self.out = tf.keras.layers.Dense(2, activation="softmax") 

    def call(self, input_data):
        x = tf.convert_to_tensor(input_data)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x

class ReinforceAgent():
    def __init__(self):
        self.model = Model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.gamma = 1 # no discounted future
        
    def act(self, obs):
        probs = self.model(np.array([obs])) 
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32) 
        action = dist.sample()
        return int(action.numpy()[0])
    
    def a_loss(self, prob, action, reward): # L = -(log(π(a | s))) * G 
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss
    
    def train(self, observations, rewards, actions):
        sum_reward = 0
        discounted_rewards = []
        rewards.reverse()
        for reward in rewards:
            sum_reward = reward + self.gamma * sum_reward
        discounted_rewards.reverse()
        
        for obs, reward, action in zip(observations, discounted_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([obs]), training=True) 
                loss = self.a_loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables) 
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    
reinforce_agent = ReinforceAgent()
train(reinforce_agent, env, 200)
play(reinforce_agent, env)
    