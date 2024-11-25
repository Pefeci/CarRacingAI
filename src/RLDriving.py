import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

from keras.saving import register_keras_serializable

env = gym.make('CarRacing-v3')

@register_keras_serializable()
class ActorCritic(tf.keras.Model):
    def __init__(self, action_space, **kwargs): 
        super(ActorCritic, self).__init__(**kwargs)
        self.action_space = action_space

        #self.action_space = tf.constant(action_space, dtype=tf.int32)
        
        if isinstance(action_space, gym.spaces.Box):
            self.action_space = {
                "low": action_space.low.tolist(),
                "high": action_space.high.tolist(),
                "shape": action_space.shape,
                "dtype": str(action_space.dtype),
            }
        else:
            raise ValueError("Unsupported action_space type. Expected gymnasium.spaces.Box.")
        
        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')

        # For continuous action spaces, we use a tanh activation
        self.steering_output = tf.keras.layers.Dense(1, activation='tanh')  # [-1, 1]
        self.acceleration_output = tf.keras.layers.Dense(1, activation='sigmoid')  # [0, 1]
        self.brake_output = tf.keras.layers.Dense(1, activation='sigmoid')  # [0, 1]
        #self.actor_output = tf.keras.layers.Dense(action_space.shape[0], activation='tanh')
        self.critic_output = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)

        steering = self.steering_output(x)  # Range [-1, 1]
        acceleration = self.acceleration_output(x)  # Range [0, 1]
        brake = self.brake_output(x)  # Range [0, 1]

        action = tf.concat([steering, acceleration, brake], axis=-1)

        #action = self.actor_output(x)  # Output action (continuous)
        value = self.critic_output(x)  # Output value estimate
        return action, value
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "action_space": self.action_space
        })
        return config

    @classmethod
    def from_config(cls, config):
        action_space_dict = config.pop("action_space")      
        action_space = gym.spaces.Box(
            low=np.array(action_space_dict["low"], dtype=action_space_dict["dtype"]),
            high=np.array(action_space_dict["high"], dtype=action_space_dict["dtype"]),
            shape=tuple(action_space_dict["shape"]),
            dtype=action_space_dict["dtype"]
        )
        return cls(action_space=action_space, **config)

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
        truncated = False
        episode_rewards = 0
        episode_length = 0
        grass_counter = 0
        
        while not done and not truncated:
            if state.ndim == 3:  # Shape is (height, width, channels), needs batch dimension
                state = np.expand_dims(state, axis=0)
            elif state.ndim == 5:  # Extra dimension present, remove it
                state = np.squeeze(state, axis=1)

            
            # Convert the state to float32 and normalize it (pixel values between 0 and 1)
            state = state.astype(np.float32) / 255.0  # Normalize to [0, 1]
            #print(state.shape)
            # Feed the state to the model and get the action and value

            action, _ = self.model(state)
            action = np.squeeze(action, axis=0)  # Remove batch dimension from the action
            action = np.array(action).astype(np.float32)
            steering, acceleration, brake = tf.split(action, 3, axis=-1)
            steering = tf.clip_by_value(steering, -1.0, 1.0)
            acceleration = tf.clip_by_value(acceleration, 0.0, 1.0)
            brake = tf.clip_by_value(brake, 0.0, 1.0)
            action = tf.concat([steering, acceleration, brake], axis=-1)

            # Ensure action is of type float32 and clip it to the range [-1, 1]
            #action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)

            action = action.numpy().astype(np.float64)

            # Debugging: Print the action to verify it is within the valid range

            # Take action in the environment
            next_state, reward, done, truncated, info = self.env.step(action)
            #print(reward)

            
            # Ensure `next_state` is consistent with state shape (it should be (96, 96, 3))
            if next_state.ndim == 3:  # Shape is (height, width, channels), needs batch dimension
                next_state = np.expand_dims(next_state, axis=0)
            elif next_state.ndim == 5:  # Extra dimension present, remove it
                next_state = np.squeeze(next_state, axis=1)


            next_state = next_state.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Store the transition (state, action, reward, next_state, done)
            #reward += 0.01 * episode_rewards
            self.buffer.append([state, action, reward, done, next_state])

            state = next_state
            last_episode_rewards = episode_rewards
            episode_rewards += reward
            episode_length += 1

            if last_episode_rewards > episode_rewards:
                grass_counter += 1
                if grass_counter >= 129:
                    print(f'shit {grass_counter}')
                    done = True
            else:
                grass_counter = 0

            
            if len(self.buffer) >= 32:  # Mini-batch size for updates
                loss = self.update_model()
                print(f'Updating model loss: {loss} and episode length: {episode_length} with reward: {episode_rewards} current action: {action}')

        return episode_rewards, episode_length

    def update_model(self):
        # Convert the buffer to numpy arrays for easier processing
        states = np.array([transition[0] for transition in self.buffer], dtype=np.float32)
        actions = np.array([transition[1] for transition in self.buffer], dtype=np.float32)
        rewards = np.array([transition[2] for transition in self.buffer], dtype=np.float32)
        dones = np.array([transition[3] for transition in self.buffer], dtype=np.float32)
        next_states = np.array([transition[4] for transition in self.buffer], dtype=np.float32)

        if states.ndim == 3:  # Shape is (height, width, channels), needs batch dimension
            states = np.expand_dims(states, axis=0)
        elif states.ndim == 5:  # Extra dimension present, remove it
            states = np.squeeze(states, axis=1)

        if next_states.ndim == 3:  # Shape is (height, width, channels), needs batch dimension
            next_states = np.expand_dims(next_states, axis=0)
        elif next_states.ndim == 5:  # Extra dimension present, remove it
            next_states = np.squeeze(next_states, axis=1)

        # Normalize pixel values (important for image input)
        states /= 255.0
        next_states /= 255.0

        # Get the value predictions for current and next states
        _, values = self.model(states)
        _, next_values = self.model(next_states)

        values = np.squeeze(values.numpy(), axis=-1)
        next_values = np.squeeze(next_values.numpy(), axis=-1)

        # Compute Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]

            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute the target values for the critic
        targets = advantages + values

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Convert data to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        # Define the PPO loss function
        with tf.GradientTape() as tape:
            # Get the current policy and value predictions
            new_actions, new_values = self.model(states)
            new_values = tf.squeeze(new_values, axis=-1)

            # Clip actions to valid ranges for CarRacing-v3
            steering, acceleration, brake = tf.split(new_actions, 3, axis=-1)
            steering = tf.clip_by_value(steering, -1.0, 1.0)
            acceleration = tf.clip_by_value(acceleration, 0.0, 1.0)
            brake = tf.clip_by_value(brake, 0.0, 1.0)
            new_actions = tf.concat([steering, acceleration, brake], axis=-1)

            # Compute the probability ratio
            old_actions = tf.stop_gradient(actions)
            prob_ratio = tf.reduce_sum(new_actions * old_actions, axis=-1) / (
                        tf.reduce_sum(old_actions * old_actions, axis=-1) + 1e-8)

            # Clipped surrogate loss
            clipped_ratio = tf.clip_by_value(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped_ratio * advantages))

            # Critic loss (Mean Squared Error)
            critic_loss = tf.reduce_mean(tf.square(targets - new_values))

            # Total loss (combine actor and critic losses)
            total_loss = policy_loss + 0.5 * critic_loss

        # Apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Clear the buffer after the update
        self.buffer = []

        # Optionally, return the loss for logging
        return total_loss.numpy()




if __name__ == '__main__':
    # Create model and agent

    action_space = env.action_space
    model = ActorCritic(action_space)
    agent = PPOAgent(env, model)

    # Train the model
    total_timesteps = 1000 #1000000
    timesteps = 0
    while timesteps < total_timesteps:
        rewards, length = agent.collect_experience()
        print(f'{length} timesteps collected and {rewards} rewards')
        timesteps += length
        if timesteps % 10000 == 0:
            print(f"Timesteps: {timesteps}, Total Rewards: {rewards}")

    # Save the trained model
    model.save('ppo_car_racing_tf.keras')


    #log_dir = "logs/"
    #os.makedirs(log_dir, exist_ok=True)

    # Load the trained model
    model = tf.keras.models.load_model("ppo_car_racing_tf.keras")

    # Test the trained agent
    env = gym.make('CarRacing-v3', render_mode='human')
    obs = env.reset()
    done = False

obs = env.reset()  # Reset environment
done = False

while not done:
    try:
        # Debugging: Print the type and content of the observation
        print("Type of obs:", type(obs))
        print("Content of obs:", obs)

        # If obs is a tuple, extract the first element as the actual observation
        if isinstance(obs, tuple):
            obs = obs[0]  # Adjust index based on the structure of the tuple

        # Convert to numpy array and ensure correct dtype
        obs = np.array(obs, dtype=np.float32)

        # Check if the observation has the expected dimensions
        if len(obs.shape) != 3:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        # Add batch dimension
        obs = np.expand_dims(obs, axis=0)  # Shape should now be (1, 96, 96, 3)

    except Exception as e:
        print("Error processing observation:", e)
        break


    # Get action probabilities from the model
    '''
    action_probs, _ = model(obs)
    print("Action probabilities:", action_probs)

    # Sample an action (replace with your action logic)
    action = np.argmax(action_probs)  # Example: choose the action with the highest probability

    # Step the environment with the chosen action
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}") '''


    # Get action probabilities from the model
    action_probs, _ = model(obs)
    print("Action probabilities:", action_probs)

    # Sample an action (replace with your action logic)
    action = np.argmax(action_probs)  # Example: choose the action with the highest probability

    # Step the environment with the chosen action
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")

    for _ in range(1000):
        action_probs, _ = model(np.expand_dims(obs, axis=0))
        action = np.argmax(action_probs.numpy())  # Take the action with the highest probability
        obs, reward, done, truncated, info = env.step(action)

        env.render()  # Render the environment
        if done or truncated:
            obs = env.reset()

    env.close()
