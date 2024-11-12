import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

# Create the CarRacing environment
env = gym.make('CarRacing-v2')

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
        state, _ = self.env.reset()  # The state should be reset with gymnasium
        done = False
        episode_rewards = 0
        episode_length = 0
        
        while not done:
            state = np.expand_dims(state, axis=0)  # Add batch dimension (needed for conv layers)
            
            # Convert the state to float32 and normalize it (pixel values between 0 and 1)
            state = state.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Feed the state to the model and get the action and value
            action, _ = self.model(state)
            action = np.squeeze(action, axis=0)  # Remove batch dimension from the action

            # Ensure action is of type float32 and clip it to the range [-1, 1]
            action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)

            # Debugging: Print the action to verify it is within the valid range
            print("Action:", action)

            # Take action in the environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Ensure `next_state` is consistent with state shape (it should be (96, 96, 3))
            next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension for processing
            next_state = next_state.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Store the transition (state, action, reward, next_state, done)
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


log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

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
