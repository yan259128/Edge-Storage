import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import env
from torch.utils.tensorboard import SummaryWriter


# 超网络
class HyperNetwork(nn.Module):
    def __init__(self, leader_action_dim, follower_value_dim, hidden_dim=64):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(leader_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, follower_value_dim)

    def forward(self, leader_action):
        x = self.fc1(leader_action)
        x = self.fc2(x)
        # x = torch.relu(x)
        x = self.fc3(x)
        weights = torch.relu(self.fc4(x))
        return weights


# GRU前缀的MLP网络，用于领导者和追随者
class LFMCONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(LFMCONetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, output_dim)

    def forward(self, x, h=None):
        x, h = self.gru(x, h)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        return x, h

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents + 2
        self.hyper_w1 = HyperNetwork(state_dim, self.num_agents * 32)
        self.hyper_w2 = HyperNetwork(state_dim, 32 * 1)
        self.hyper_b1 = HyperNetwork(state_dim, 32)
        self.hyper_b2 = HyperNetwork(state_dim, 1)

    def forward(self, agent_qs, state):
        # 生成 W1, W2, b1, b2
        w1 = self.hyper_w1(state).view(-1, self.num_agents, 32)
        w2 = self.hyper_w2(state).view(-1, 32, 1)
        b1 = self.hyper_b1(state).view(-1, 1, 32)
        b2 = self.hyper_b2(state).view(-1, 1, 1)

        # 执行混合
        q_tot = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        q_tot = torch.bmm(torch.relu(q_tot), w2) + b2
        q_tot = q_tot.view(-1, 1)
        return q_tot


# 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions_list, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        leader_actions = torch.tensor([actions['leader'] for actions in actions_list], dtype=torch.int64)
        follower_actions = {
            f'follower_{i}': torch.tensor([actions[f'follower_{i}'] for actions in actions_list], dtype=torch.int64)
            for i in range(len(actions_list[0]) - 1)}

        return states, leader_actions, follower_actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# LFMCO算法类
class LFMCO:
    def __init__(self, leader_input_dim, follower_input_dim, action_dim, num_followers, state_dim, gamma=0.99,
                 epsilon=0.1, lr=0.0005, lambda_ig=0.05):
        self.leader_net = LFMCONetwork(leader_input_dim, action_dim)
        self.follower_net = LFMCONetwork(follower_input_dim, action_dim)  # 确保 GRU 的 input_size 正确
        self.hyper_net = HyperNetwork(action_dim, action_dim)
        self.mixing_net = MixingNetwork(num_agents=num_followers + 1, state_dim=state_dim)
        self.optimizer = optim.RMSprop(list(self.leader_net.parameters()) +
                                       list(self.follower_net.parameters()) +
                                       list(self.hyper_net.parameters()) +
                                       list(self.mixing_net.parameters()), lr=lr)
        self.num_followers = num_followers
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ig = lambda_ig

        # 初始化数据记录器
        self.data = {
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
        # 获得当前的时间
        # starttime = time.strftime("%Y-%m-%d %H %M %S")  # 时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
        # 添加TensorBoard记录器
        self.writer = None

    def train(self, episodes, batch_size, replay_buffer, env):
        for episode in range(episodes):
            starttime = time.strftime("%Y-%m-%d %H_%M_%S")  # 时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
            path = f'./logs_{self.num_followers}/'
            self.writer = SummaryWriter(log_dir=path + starttime[:] + '_episode_{}'.format(episode))
            state = env.reset()
            done = False
            step = 0

            while not done:
                leader_input = self.get_leader_input(state)  # 获取领导者输入
                leader_values, _ = self.leader_net(leader_input)  # 使用领导者网络处理
                leader_action = self.select_action(leader_values)

                actions = {'leader': leader_action}
                follower_values_list = []
                for follower_id in range(self.num_followers):
                    # print(f'Follower: {follower_id}')
                    follower_input = self.get_follower_input(state, follower_id)  # 获取追随者输入
                    follower_values, _ = self.follower_net(follower_input)  # 使用追随者网络处理
                    follower_values_list.append(follower_values)

                    follower_action = self.select_action(follower_values)
                    actions[f'follower_{follower_id}'] = follower_action

                next_state, reward, done = env.step(state, actions)
                replay_buffer.add(state, actions, reward, next_state, done)

                if len(replay_buffer) >= batch_size:
                    states, leader_actions, follower_actions, rewards, next_states, dones = replay_buffer.sample(
                        batch_size)
                    # print(states)

                    # 获得每个状态的领导者的q值
                    first_leader_input = self.get_leader_input(states)
                    first_leader_values, _ = self.leader_net(first_leader_input)

                    losses = []
                    information_gains = []
                    joint_values_list = []

                    for follower_id in range(self.num_followers):
                        follower_input = self.get_follower_input(states, follower_id)
                        original_values, _ = self.follower_net(follower_input)
                        # print(original_values.shape)

                        leader_action_one_hot = self.one_hot_encode(leader_actions, leader_values.shape[-1])
                        weights = self.hyper_net(leader_action_one_hot)
                        follower_values = torch.sum(weights * original_values, dim=-1)
                        joint_values_list.append(follower_values)

                    # agent_value = leader_values[0].detach().numpy().tolist()
                    # print(agent_value)
                    follower_value_list = []
                    follower_values_list = []
                    #
                    for j in range(batch_size):
                        for i in range(self.num_followers):
                            follower_value_list.append(joint_values_list[i][j].item())
                        first_agent_value = first_leader_values[0][j].detach().numpy().tolist()
                        followers_value = first_agent_value + follower_value_list
                        # print(followers_value)
                        follower_values_list.append(followers_value)
                        follower_value_list = []

                    agent_values = torch.Tensor(follower_values_list)

                    # agent_values = torch.cat(leader_values, joint_values_list, dim=1)

                    # print(states.shape)
                    Q_tot = self.mixing_net(agent_values, states)

                    # 计算目标 Q 值
                    with torch.no_grad():
                        next_leader_input = self.get_leader_input(next_states)
                        next_leader_values, _ = self.leader_net(next_leader_input)
                        next_joint_values_list = []


                        for follower_id in range(self.num_followers):
                            next_follower_input = self.get_follower_input(next_states, follower_id)
                            next_original_values, _ = self.follower_net(next_follower_input)
                            next_weights = self.hyper_net(leader_action_one_hot)  # 注意这里用的也是 leader_action_one_hot
                            next_follower_values = torch.sum(next_weights * next_original_values, dim=-1)
                            next_joint_values_list.append(next_follower_values)

                        # print(next_joint_values_list)
                        # print(next_leader_values[0].shape)
                        next_follower_value_list = []
                        next_follower_values_list = []
                        for j in range(batch_size):
                            for i in range(self.num_followers):
                                next_follower_value_list.append(next_joint_values_list[i][j].item())
                            next_agent_value = next_leader_values[0][j].detach().numpy().tolist()
                            followers_value = next_agent_value + next_follower_value_list
                            # print(followers_value)
                            next_follower_values_list.append(followers_value)
                            next_follower_value_list = []
                        # print(next_follower_values_list)
                        next_agent_values = torch.Tensor(next_follower_values_list)
                        next_Q_tot = self.mixing_net(next_agent_values, next_states)
                        y_tot = rewards + self.gamma * next_Q_tot * (1 - dones)

                    # 计算 TD 误差
                    loss = self.calculate_loss(Q_tot, y_tot)
                    losses.append(loss)

                    # 计算信息增益
                    original_entropy = self.calculate_entropy(original_values)
                    new_entropy = self.calculate_entropy(follower_values)
                    information_gain = original_entropy - new_entropy
                    information_gains.append(information_gain)

                    total_loss = sum(losses) - self.lambda_ig * sum(information_gains)
                    # 确保 total_loss 是标量
                    if total_loss.dim() > 0:
                        total_loss = total_loss.mean()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

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
                        for i in range(self.num_followers+1):
                            state_array = state.reshape((self.num_followers+1, 5))
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
                                all_space = state_array[i][0] + state_array[i][1]
                                f_node = env.get_follower_node(i-1)
                                A_R_rate.append((all_space+f_node.used_storage_space)/f_node.storage_space)
                                fs_profit.append(f_profit)
                                # print("follower id: ", i-1, "profit: ", f_profit)
                        print(
                            f'Episode: {episode}, Step: {step}, Loss: {total_loss.item()}, Total Profit: {total_profit}, Reward: {reward}')
                        # nodes = env.get_sorted_node()
                        # node_list = {}
                        # for tup in nodes:
                        #     node_list[tup[0]] = tup[1].transaction_expectations
                        # print("node", node_list)
                        # profit_list = env.get_agent_profit()
                        # print("leader profit: ", profit_list['leader'])
                        # print('follower profit:  ', profit_list['follower_1'])
                        # print('follower profit:  ', profit_list['follower_2'])

                        self.writer.add_scalar('Loss', total_loss.item(), episode * len(replay_buffer) + step)
                        self.writer.add_scalar('Total Profit', total_profit,
                                               episode * len(replay_buffer) + step)
                        self.writer.add_scalar('Reward', reward, episode * len(replay_buffer) + step)
                        # 领导者资源满足率
                        self.writer.add_scalar("resource1_satisfaction_rate", l_Rs1_rate,
                                               episode * len(replay_buffer) + step)
                        self.writer.add_scalar("resource2_satisfaction_rate", l_Rs2_rate,
                                               episode * len(replay_buffer) + step)
                        self.writer.add_scalar("leader_profit", l_profit, episode * len(replay_buffer) + step)
                        for follower_id in range(self.num_followers):
                            self.writer.add_scalar(f'follower_{follower_id}_profit', fs_profit[follower_id], episode * len(replay_buffer) + step)
                            self.writer.add_scalar(f'follower_{follower_id}_rate', A_R_rate[follower_id], episode * len(replay_buffer) + step)


                    # 记录数据
                    self.data['episode'].append(episode)
                    self.data['step'].append(step)
                    self.data['loss'].append(total_loss.item())
                    self.data['total_profit'].append(total_profit)
                    self.data['reward'].append(reward)
                    self.data['leader_profit'].append(l_profit)
                    self.data['followers_profit'].append(fs_profit)
                    self.data['resource1_satisfaction_rate'].append(fs_profit)
                    self.data['resource2_satisfaction_rate'].append(fs_profit)

                    self.writer.close()

                state = next_state
                step += 1
                if step > 2000:
                    done = True

    def get_leader_input(self, state):
        # 领导者的输入包括全局状态
        if state.ndim == 2:
            leader_input = state
        else:
            leader_input = state  # 示例中只使用状态
        # print(leader_input.shape)
        return torch.tensor(leader_input, dtype=torch.float32).unsqueeze(0)
        # print("leader_input", leader_input.shape)


    def get_follower_input(self, state, follower_id):
        # 将 numpy array 转换为 torch tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        node = env.get_follower_node(follower_id)
        # 如果 state 是二维的，表示批量数据
        if state.ndim == 2:  # [batch_size, state_dim]
            batch_size = state.size(0)
            state_array = state.view(batch_size, self.num_followers + 1, 5)
            leader_state = state_array[:, 0, :]
            follower_state = state_array[:, follower_id + 1, :]

            # 生成附加状态，扩展到批量大小
            add_follower_state = torch.stack([
                torch.full((batch_size, 1), node.transaction_expectations, dtype=torch.float32),
                leader_state[:, 2:3],  # 取第2列
                leader_state[:, 3:4]  # 取第3列
            ], dim=-1).view(batch_size, -1)

            follower_state_array = torch.cat([follower_state, add_follower_state], dim=1)

        # 如果 state 是一维的，表示单个样本
        elif state.ndim == 1:  # [state_dim]
            state_array = state.view(self.num_followers + 1, 5)
            leader_state = state_array[0, :]
            follower_state = state_array[follower_id + 1, :]

            # 生成附加状态
            add_follower_state = torch.tensor([
                node.transaction_expectations,
                leader_state[2],
                leader_state[3]
            ], dtype=torch.float32)

            follower_state_array = torch.cat([follower_state, add_follower_state], dim=0).unsqueeze(0)

        else:
            raise ValueError("Invalid state dimension")
        # print(follower_state_array.unsqueeze(1).shape)

        return follower_state_array.unsqueeze(1)  # 保证返回维度为 [batch_size, sequence_length, input_dim]

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_values.shape[-1])
        else:
            return q_values.argmax(dim=-1).item()

    def one_hot_encode(self, action, action_dim):
        one_hot = torch.zeros(action_dim)
        one_hot[action] = 1
        return one_hot

    def calculate_loss(self, values, targets):
        loss_fn = nn.MSELoss()
        return loss_fn(values, targets)

    def calculate_entropy(self, q_values):
        probabilities = torch.softmax(q_values, dim=-1)
        log_probabilities = torch.log(probabilities + 1e-9)
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
        return entropy


# 主程序
if __name__ == "__main__":
    # 每个节点的状体空间为5
    node_state_dim = 5
    num_followers = 5  # 追随者数量
    num_agents = num_followers + 1
    # 环境的维度跟追随者的数量(领导者数加追随者数)相关
    state_dim = node_state_dim * (num_followers + 1)
    # state_dim = 20  # 假设环境状态维度为20，
    leader_input_dim = state_dim  # 领导者的输入维度等于环境状态维度
    # 追随者的局部观察应为， 自身的状态，交易期望，领导者资源的以满足空间
    follower_input_dim = node_state_dim + 3
    action_dim = 3  # 动作空间维度
    replay_buffer_capacity = 10000  # 经验回放池的容量

    # 利润历史列表的长度
    list_len = 10
    # 判断是否结束的方差值
    variance_num = 10

    lfmco = LFMCO(leader_input_dim, follower_input_dim, action_dim, num_followers, state_dim)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    # env = Environment(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents)
    env = env.EdgeEnv(state_dim=state_dim, num_agents=num_agents,
                      node_state_dim=node_state_dim, profit_list_len=list_len,
                      variance=variance_num)

    # 训练模型
    lfmco.train(episodes=20, batch_size=32, replay_buffer=replay_buffer, env=env)
