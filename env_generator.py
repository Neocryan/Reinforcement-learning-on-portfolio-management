import pandas as pd
from management.environments.portfolio import PortfolioEnv
from management.wrappers import SoftmaxActions, \
    TransposeHistory, ConcatStates
import gym
from subprocess import Popen
import pandas
import os
from collections import deque
class DeepRLWrapper(gym.Wrapper):
    def __init__(self, env, draw=False,draw_interval = 20):
        super().__init__(env)
        self.render_on_reset = False
        self.plot = draw
        self.plot_weight = []
        self.plot_reward = deque(maxlen=100)
        self.plot_price1 = deque(maxlen=100)
        self.plot_price2 = deque(maxlen=100)
        self.plot_price3 = deque(maxlen=100)
        self.plot_price4 = deque(maxlen=100)
        self.total_history = deque(maxlen=3000)
        self.plot_interval = draw_interval
        self.plot_t = 0
        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]
        self.total_reward = 0
        self.name = 'PortfolioEnv'
        self.success_threshold = 2


    def normalize_state(self, state):
        return state

    def step(self, action):

        state, reward, done, info = self.env.step(action)
        reward *= 1e4  # often reward scaling is important sooo...
        if self.plot:
            self.total_reward += reward
            self.total_history.append(self.total_reward)
            self.plot_weight = [info['weight_USD'],info['weight_BTC'],info['weight_ETH'],
                                info['weight_LTC'],info['weight_XRP']]
            self.plot_reward.append(reward)
            self.plot_price1.append(info['price_BTC'])
            self.plot_price2.append(info['price_ETH'])
            self.plot_price3.append(info['price_LTC'])
            self.plot_price4.append(info['price_XRP'])
            if self.plot_t == 0:
                Popen('python ./viz.py', shell=True)

            self.plot_t += 1
            if self.plot_t % self.plot_interval == 0:
                with open('log.txt', 'w') as aa:
                    aa.write('{}//{}//{}//{}//{}//{}//{}'.format(self.plot_weight,  list(self.plot_reward),
                                                                 list(self.plot_price1) , list(self.plot_price2),
                                                                 list(self.plot_price3) , list(self.plot_price4),
                                                                 list(self.total_history)))
        return state, reward, done, info

    def reset(self):
        # here's a roundabout way to get it to plot on reset
        if self.render_on_reset:
            self.env.render('notebook')
        try:
            os.remove('log.txt')
        except:
            print('ain\'t remove the log')
            pass

        return self.env.reset()


def make_env(data='default_train', step=1000, visualization=False,draw_interval = 20):
    '''
    :usage : from env_generator import make_env
    :param data: should be a pd dataframe, or 'default_train', 'or default_test'
    :param step: number of steps in each episode
    :param visualization: Wanna draw the result out in the real time?
    :return: a gym environment
    '''

    if type(data) == str:
        if data == 'default_train':
            df = pd.read_hdf('data/df_train.hf', key='train')
        elif data == 'default_test':
            df = pd.read_hdf('data/df_test.hf', key='test')
        else:
            raise EnvironmentError("data should be should be a pd dataframe, or 'default_train', 'or default_test'")

    else:
        assert type(data) == pandas.core.frame.DataFrame
        df = data
    env = PortfolioEnv(df=df, steps=step, output_mode='EIIE')
    env = TransposeHistory(env)
    env = ConcatStates(env)
    env = SoftmaxActions(env)  # softmax --> action ouptut sum up to 1
    env = DeepRLWrapper(env, draw=visualization,draw_interval = draw_interval)
    return env
