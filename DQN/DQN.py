import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import env


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001,
                 batch_size=64, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Q-Network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action (exploration)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()  # Best action (exploitation)

    def train(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (for exploration-exploitation trade-off)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Main training loop
def train_dqn(agent, env, f_num,episodes, target_update_freq=10):
    for episode in range(episodes):
        starttime = time.strftime("%Y-%m-%d %H_%M_%S")  # 时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
        path = f'./logs_{f_num}/'
        writer = SummaryWriter(log_dir=path + starttime[:] + '_episode_{}'.format(episode))
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        data = {
            'episode': [],
            'step': [],
            'loss': [],
            'total_profit': [],
            'reward': [],
            'leader_profit': [],
            'followers_profit': [],
            'resource1_satisfaction_rate': [],
            'resource2_satisfaction_rate': [],
            'all_resource__rate': [],
        }

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(state, {'leader': action})
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            l_profit = 0
            l_Rs1_rate = 0
            l_Rs2_rate = 0
            A_R_rate = 0
            fs_profit = []
            # 每隔100步输出结果并记录到TensorBoard
            # 计算信息增益的总和或平均值，确保其为一个标量
            total_profit = env.get_total_profit()
            if step == 0 or step % 50 == 0:
                l_profit = 0
                l_Rs1_rate = 0
                l_Rs2_rate = 0
                A_R_rate = []
                fs_profit = []
                for i in range(f_num + 1):
                    state_array = state.reshape((f_num + 1, 5))
                    # print(state_array)
                    # print(state_array)
                    if i == 0:
                        # 领导者节点的值
                        leader_node = env.get_leader_node()
                        l_profit = state_array[i][4]
                        # 计算领导者的资源满足率
                        l_Rs1_rate = state_array[i][2] / leader_node.delay_insensitive_size
                        l_Rs2_rate = state_array[i][3] / leader_node.delay_sensitive_size
                        # print("leader profit:", state_array[i][4])
                    else:
                        f_profit = state_array[i][2] + state_array[i][3]
                        # print(f_profit)
                        all_space = state_array[i][0] + state_array[i][1]
                        f_node = env.get_follower_node(i - 1)
                        A_R_rate.append((all_space + f_node.used_storage_space) / f_node.storage_space)
                        fs_profit.append(f_profit)
                print(
                    f'Episode: {episode}, Step: {step}, Total Profit: {total_profit}, Reward: {reward}')
                writer.add_scalar('Total Profit', total_profit, step)
                writer.add_scalar('Reward', reward, step)
                # 领导者资源满足率
                writer.add_scalar("resource1_satisfaction_rate", l_Rs1_rate, step)
                writer.add_scalar("resource2_satisfaction_rate", l_Rs2_rate, step)
                writer.add_scalar("leader_profit", l_profit,  step)
                for follower_id in range(f_num):
                    # print(fs_profit)
                    writer.add_scalar(f'follower_{follower_id}_profit', fs_profit[follower_id], step)
                    writer.add_scalar(f'follower_{follower_id}_rate', A_R_rate[follower_id], step)
            # 记录数据
            data['episode'].append(episode)
            data['step'].append(step)
            # data['loss'].append(total_loss.item())
            data['total_profit'].append(total_profit)
            data['reward'].append(reward)
            data['leader_profit'].append(l_profit)
            data['followers_profit'].append(fs_profit)
            data['resource1_satisfaction_rate'].append(fs_profit)
            data['resource2_satisfaction_rate'].append(fs_profit)

            writer.close()

            state = next_state
            total_reward += reward

            step += 1
            if step > 6000:
                done = True

        # Update the target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")



if __name__ == "__main__":
    node_state_dim = 5
    num_followers = 10  # 追随者数量
    num_agents = num_followers + 1
    # 环境的维度跟追随者的数量(领导者数加追随者数)相关
    state_dim = node_state_dim * (num_followers + 1)
    # state_dim = 20  # 假设环境状态维度为20，
    leader_input_dim = state_dim  # 领导者的输入维度等于环境状态维度
    # 追随者的局部观察应为， 自身的状态，交易期望，领导者资源的以满足空间
    follower_input_dim = node_state_dim + 3
    action_dim = 3  # 动作空间维度

    # 利润历史列表的长度
    list_len = 10
    # 判断是否结束的方差值
    variance_num = 10

    # env = Environment(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents)
    env = env.EdgeEnv(state_dim=state_dim, num_agents=num_agents,
                      node_state_dim=node_state_dim, profit_list_len=list_len,
                      variance=variance_num)

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    # Train the agent
    train_dqn(agent, env, num_followers,episodes=20)
