import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

# Create the CarRacing environment
env = gym.make('CarRacing-v2')

log_dir = './ppo_car_racing_tf/'
os.makedirs(log_dir, exist_ok=True)

# Define the Actor-Critic model
class ActorCritic(tf.keras.Model):
    def __init__(self, action_space):
        super(ActorCritic, self).__init__()
        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')

        # For continuous action spaces, we use a tanh activation
        self.actor_output = tf.keras.layers.Dense(action_space.shape[0], activation='tanh')
        self.critic_output = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)

        action = self.actor_output(x)  # Output action (continuous)
        value = self.critic_output(x)  # Output value estimate
        return action, value


# Define the PPO agent class
class PPOAgent:
    def __init__(self, env, model, gamma=0.99, lam=0.95, epsilon=0.2, learning_rate=1e-4):
        self.env = env
        self.model = model
        self.gamma = gamma  # Discount factor
        self.lam = lam  # GAE lambda
        self.epsilon = epsilon  # Clipping parameter for PPO
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.buffer = []
        self.max_timesteps = 1000000

    def collect_experience(self):
        state = self.env.reset()
        done = False
        episode_rewards = 0
        episode_length = 0
        
        while not done:
            state =  np.expand_dims(state, axis=0)  # Add batch dimension
            action_probs, _ = self.model(state)
            action = np.random.choice(self.env.action_space.n, p=action_probs.numpy().flatten())  # Sample action

            next_state, reward, done, truncated, info = self.env.step(action)
            self.buffer.append([state, action, reward, done, next_state])
            state = next_state
            episode_rewards += reward
            episode_length += 1
            
            if len(self.buffer) >= 32:  # Mini-batch size for updates
                self.update_model()

        return episode_rewards, episode_length

    def update_model(self):
        # Process buffer to compute advantages and perform PPO update
        pass

# Create model and agent
action_space = env.action_space
model = ActorCritic(action_space)
agent = PPOAgent(env, model)

# Train the model
total_timesteps = 1000000
timesteps = 0
while timesteps < total_timesteps:
    rewards, length = agent.collect_experience()
    timesteps += length
    if timesteps % 10000 == 0:
        print(f"Timesteps: {timesteps}, Total Rewards: {rewards}")

# Save the trained model
model.save('ppo_car_racing_tf')



# Load the trained model
model = tf.keras.models.load_model(log_dir + "ppo_car_racing_tf")

# Test the trained agent
obs = env.reset()
done = False

for _ in range(1000):
    action_probs, _ = model(np.expand_dims(obs, axis=0))
    action = np.argmax(action_probs.numpy())  # Take the action with the highest probability
    obs, reward, done, truncated, info = env.step(action)
    
    env.render()  # Render the environment
    if done or truncated:
        obs = env.reset()

env.close()
