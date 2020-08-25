import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


GAMMA = 0.99
GAMMA_REWARD = 0.05
learning_rate = 0.01
MAX_STEPS_PER_EPISODE = 10000
env = gym.make("CartPole-v1")
EPS = np.finfo(np.float32).eps.item()

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape
print("Size of Action Space -> {}".format(num_actions))
print(env.action_space)

sample_action = env.action_space.sample()
print("Sample Action: ", sample_action)

def demo():
    state = env.reset()
    state1 = tf.convert_to_tensor(state)
    print(state1.shape)
    state1 = tf.expand_dims(state1, 0)
    print(state1.shape)
    # for i in range(MAX_STEPS_PER_EPISODE):
    #     env.render()
    #     action = env.action_space.sample()
    #     state, reward, done, _ = env.step(action)
    #     print(f"Step = {i+1}, State = {state}, reward = {reward}, done = {done}")


def train():
    num_inputs = 4
    num_actions = 2
    num_hidden = 128

    inputs = layers.Input(shape=(num_inputs,))
    hidden_layer = layers.Dense(num_hidden, activation="relu")(inputs)
    # hidden_layer = layers.Dense(num_hidden, activation="relu")(hidden_layer)
    action = layers.Dense(num_actions, activation="softmax")(hidden_layer)
    critic = layers.Dense(1)(hidden_layer)

    model = keras.Model(inputs=inputs, outputs=[action, critic])

    huber_loss = keras.losses.Huber()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    episode_count = 0
    
    critic_value_history = []
    action_probs_history = []
    rewards_history = []

    
    running_reward = 0.0
    print("Start Trainning.....")
    while True:
        state = env.reset()
        episode_reward = 0.0
        with tf.GradientTape() as tape:
            for timestep in range(1, MAX_STEPS_PER_EPISODE):
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0,0])

                np_action_probs = keras.backend.eval(action_probs)
                action = np.random.choice(num_actions, p=np.squeeze(np_action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                state, reward, done, _ = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break
            # print("Vizualization done")
            # print(episode_reward)
            running_reward = GAMMA_REWARD * episode_reward + (1 - GAMMA_REWARD) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + GAMMA * discounted_sum
                returns.append(discounted_sum)
            returns = np.array(returns[::-1])
            returns = (returns - np.mean(returns)) / (np.std(returns) + EPS)
            returns = returns.tolist()
            # print("Calculating return done")

            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)

                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # print("Gradienting... ")
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            ### Clear history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
        
        episode_count += 1
        if episode_count % 10 == 0:
            print(f"Running raward: {running_reward:.2f} at episode {episode_count}")

        if running_reward > 195:
            print(f"Solved at {episode_count}!")
            break

    model.save("model/cart_pole_v1")        

def inference():
    model = keras.models.load_model("model/cart_pole_v1")
    state = env.reset()
    step = 0
    while True:
        step += 1
        env.render()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs, critic_value = model(state)
        np_action_probs =  keras.backend.eval(action_probs)

        action = np.argmax(action_probs)
        state, reward, done, _ = env.step(action)
        print(f"Step = {step}, State = {state}, reward = {reward}, done = {done}")
        if done:
            break


if __name__ == "__main__":
    # demo()
    # train()
    inference()

