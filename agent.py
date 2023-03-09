import copy
import random
from collections import deque

from helpers import get_rewards_from_game_state
from networks import QNetworkModel
import torch
import torch.nn as nn
import numpy as np

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device('cpu')
print(f"Using {device} device")

# Set hyper parameters
START_LEARNING_RATE = 0.05
# START_LEARNING_RATE = .0001
REDUCE_LR_FACTOR = 0.99995
MIN_LEARNING_RATE = 0.0001
GAMMA = 0.99
# START_EPSILON = 0.001
START_EPSILON = 0.1
REDUCE_EPSILON_FACTOR = 0.99995
MIN_EPSILON = 0.001
NUM_INPUTS = 75
MOVES = ['up', 'down', 'left', 'right']
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000

is_test = False

class Agent:
    def __init__(self):
        self.online_network = QNetworkModel()
        self.load_model_if_found()
        self.target_network = copy.deepcopy(self.online_network)
        self.online_network.to(device)
        self.target_network.to(device)
        self.prev_action = 0
        self.prev_state = torch.tensor(np.zeros((1, 2, 5, 5), dtype='f'), device=device)

        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = START_EPSILON
        self.epoch = 0
        self.score = 0
        self.best_score = 0
        self.cumulative_score = 0
        self.lr = START_LEARNING_RATE
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RAdam(self.online_network.parameters(), lr=self.lr)


    def load_model_if_found(self):
        try:
            saved_model = copy.deepcopy(self.online_network)
            saved_model.load_state_dict(torch.load('saved_model.pt'))
            self.online_network = saved_model
            print('Using saved model')
        except:
            print('No model found, using new model')

    def get_next_move(self, input_tensor):
        # use a more stochastic policy
        if is_test or np.random.random() > self.epsilon:
            prediction = self.target_network(input_tensor.to(device))
            move_index = torch.argmax(prediction).item()
        else:
            move_index = np.random.randint(0, 4)
        move = MOVES[move_index]
        self.prev_action = move_index
        return move

    def collect_experience(self, state, game_state, dead: bool):
        global is_test
        if game_state['turn'] < 1:
            return
        rewards = get_rewards_from_game_state(game_state, dead)
        self.score += rewards
        self.replay_buffer.append((self.prev_state, self.prev_action, rewards, state, dead))
        if not dead:
            # del self.prev_state
            # because prev_state is same when dead
            self.prev_state = state
        self.replay(game_state['turn'])
        if dead:
            # reduce epsilon by REDUCE_EPSILON_FACTOR
            self.epsilon *= REDUCE_EPSILON_FACTOR
            if self.epsilon <= MIN_EPSILON:
                self.epsilon = MIN_EPSILON

            # reduce learning rate:
            self.lr *= REDUCE_LR_FACTOR
            if self.lr <= MIN_LEARNING_RATE:
                self.lr = MIN_LEARNING_RATE

            if self.epoch > 0 and self.epoch % 10 == 0:
                # del self.target_network
                self.target_network = copy.deepcopy(self.online_network).to(device)
                torch.save(self.target_network.state_dict(), 'saved_model.pt')
            if is_test:
                print(f"Test Score: {self.score} ")
                is_test = False
            if self.epoch > 0 and self.epoch % 100 == 0:
                print(
                    f"Epoch: {self.epoch} "
                    f"Best Score So Far: {self.best_score} "
                    f"Avg Score: {self.cumulative_score / 100} "
                    f"Epsilon: {self.epsilon} "
                    f"LR: {self.lr}")
                with open('score.log', 'a') as f:
                    f.write(f"\n{self.cumulative_score / 100}")
                self.cumulative_score = 0
                is_test = True
            self.epoch += 1
            if self.score > self.best_score:
                self.best_score = self.score
                torch.save(self.online_network.state_dict(), 'saved_best_model.pt')
            self.cumulative_score += self.score
            self.score = 0

    def replay(self, turn):
        if turn < 1:
            return
        if BATCH_SIZE > len(self.replay_buffer):
            return
        else:
            sample_size = BATCH_SIZE
        prev_states, prev_actions, rewards, states, deads = self.sample_experiences(sample_size)
        q_values = self.online_network(states.to(device))
        targets = self.target_network(prev_states.to(device))
        target_batch = None
        for i in range(0, sample_size):
            max_q_value = torch.max(q_values[i]).item()
            target = torch.clone(targets[i]).to(device)
            target = target.reshape(1, 4)
            if deads[i]:
                # if training is done then we know our Q value
                target[0][prev_actions[i]] = rewards[i]
            else:
                # else use online network prediction
                target[0][prev_actions[i]] = rewards[i] + max_q_value * GAMMA
            if i == 0:
                target_batch = target
            else:
                target_batch = torch.cat((target_batch, target))
            # del target
        self.train(q_values, target_batch)
        del targets
        del states
        del prev_actions
        del rewards
        del deads
        del prev_states
        del q_values
        del target_batch
        torch.cuda.empty_cache()
    def train(self, q_values, target_batch):
        torch.autograd.set_detect_anomaly(True)
        self.online_network.train()
        loss = self.loss_fn(q_values, target_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del loss

    def sample_experiences(self, sample_size):
        first = True
        if sample_size <= len(self.replay_buffer):
            for i in range(0, sample_size):
                if first:
                    prev_states = self.replay_buffer[i][0]
                    prev_actions = [self.replay_buffer[i][1]]
                    rewards = [self.replay_buffer[i][2]]
                    states = self.replay_buffer[i][3]
                    deads = [self.replay_buffer[i][4]]
                    first = False
                else:
                    prev_states = torch.cat((prev_states, self.replay_buffer[i][0]))
                    prev_actions.append(self.replay_buffer[i][1])
                    rewards.append(self.replay_buffer[i][2])
                    states = torch.cat((states, self.replay_buffer[i][3]))
                    deads.append(self.replay_buffer[i][4])
        else:
            start_index = random.randrange(0, len(self.replay_buffer) - sample_size)
            for i in range(start_index + 1, start_index + sample_size):
                if first:
                    prev_states = self.replay_buffer[i][0]
                    prev_actions = [self.replay_buffer[i][1]]
                    rewards = [self.replay_buffer[i][2]]
                    states = self.replay_buffer[i][3]
                    deads = [self.replay_buffer[i][4]]
                    first = False
                else:
                    prev_states = torch.cat((prev_states, self.replay_buffer[i][0]))
                    prev_actions.append(self.replay_buffer[i][1])
                    rewards.append(self.replay_buffer[i][2])
                    states = torch.cat((states, self.replay_buffer[i][3]))
                    deads.append(self.replay_buffer[i][4])
        return prev_states, prev_actions, rewards, states, deads
