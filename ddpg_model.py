from env_generator import make_env
train_env = make_env(visualization=True)
from rl.agents.ddpg import DDPGAgent
import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, \
Concatenate, Conv3D, MaxPooling2D, Conv2D
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

WINDOW_LENGTH = 5

nb_actions= train_env.action_space.shape[0]
stride_time = train_env.state_dim[1] -1 -2

# ------------------ Actor Model ------------------
actor = Sequential()
actor.add(Conv3D(2, kernel_size=(4,3,1),
                 input_shape= (WINDOW_LENGTH,) + train_env.observation_space.shape,
                 activation='relu', data_format="channels_first"))
actor.add(Conv3D(20, kernel_size=(1,3,49)))
actor.add(Conv3D(1, kernel_size=(1,1,1)))
actor.add(Flatten())
actor.add(Dense(5, activation='softmax'))
print(actor.summary())

# ------------------ Critic Model ------------------

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + \
                          train_env.observation_space.shape,
                          name='observation_input')
x = Conv3D(2, kernel_size=(4,3,1),
                 input_shape= (WINDOW_LENGTH,) + train_env.observation_space.shape,
                 activation='relu', data_format="channels_first") \
                 (observation_input)
x = Conv3D(20, kernel_size=(1,3,49))(x)
x = Conv3D(1, kernel_size=(1,1,1))(x)

flattened_observation = Flatten()(x)

x = Concatenate()([action_input, flattened_observation])
x = Dense(1, activation='linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                          theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions,
                  actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100,
                  nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99,
                  target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(train_env, nb_steps=1000, visualize=False,
          verbose=2, nb_max_episode_steps=100)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format("abc"), overwrite=True)
