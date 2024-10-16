# 边缘节点博弈的环境
import random

import numpy as np
import node


class EdgeEnv:
    def __init__(self, state_dim, num_agents, node_state_dim, profit_list_len, variance):
        self.profit_list = {}
        self.state = None
        self.leader_node = None
        self.follower_node = {}
        # 追随者的数量
        self.follow_num = num_agents - 1
        # 每个节点的状态空间
        self.node_state_dim = node_state_dim
        # 节点的状态包括： 节点id, 当前利润， 领导者的报价， 提供的存储资源， 剩余存储资源比例
        self.state_dim = state_dim
        # 节点数量为领导者数加追随者数
        self.num_agents = num_agents
        # 历史列表的长度
        self.profit_list_len = profit_list_len
        # 利润的历史列表
        self.profit_list_history = None
        # 判断是否结束的值
        self.variance = variance
        self.sorted_node = None
        # 进行环境的初始化
        # self.reset()

    def reset(self):
        self.profit_list = {}
        self.profit_list_history = []
        # 初始化节点状态
        self.leader_node = node.LeaderNode()
        for i in range(self.follow_num):
            self.follower_node[f'follower_{i}'] = node.FollowerNode(i)
        # 初始化状态 二维
        self.state = np.zeros((self.num_agents, self.node_state_dim))
        # self.state[0] = self.leader_node.price
        # print(self.state.shape)
        # 将二维转换为一维
        self.state = self.state.flatten()
        # print(self.state)
        return self.state

    def step(self, state, actions):
        """
        领导者状态为： 报价1 敏感型数据报价2 资源1已满足的空间 资源2已满足的空间 领导者的利润
        追随者状态为： 空间1 敏感型数据空间2 提供资源1所得利润 提供资源2所得利润 资源利用率
        """
        # 保存领导者与追随者的下一个状态
        leader_next_state = []
        followers_next_state = []
        # 提供的总空间 敏感型
        followers_delay_sensitive_space = 0
        # 提供的总空间 非敏感型
        followers_delay_insensitive_space = 0
        # 保存领导者的利润
        # leader_profit = 0
        # 保存追随者的利润 字典保存
        # follower_profit = {}
        # actions为一个字典
        # 通过执行动作来更新环境状态
        # 节点状态数组
        state_array = state.reshape((self.num_agents, self.node_state_dim))
        # 追随者的动作根据领导者而改变
        follower_actions = {}
        for i in range(self.follow_num):
            follower_actions[f'follower_{i}'] = random.randint(0, 2)
        actions.update(follower_actions)
        for k, action in actions.items():
            if k == 'leader':
                if action == 0:
                    # 领导者报价降低
                    reduce_value = np.random.uniform(1, 5)
                    if state_array[0][2] > self.leader_node.delay_insensitive_size:
                        self.leader_node.price -= reduce_value
                    else:
                        pass
                    if state_array[0][3] > self.leader_node.delay_sensitive_size:
                        self.leader_node.delay_sensitive_price -= reduce_value
                    else:
                        pass
                    # 边界值判断
                    if self.leader_node.price <= self.leader_node.storage_cost:
                        self.leader_node.price = self.leader_node.storage_cost
                    if self.leader_node.delay_sensitive_price <= self.leader_node.storage_cost:
                        self.leader_node.delay_sensitive_price = self.leader_node.storage_cost
                if action == 1:
                    # 领导者报价不变
                    pass
                if action == 2:
                    add_value = np.random.uniform(1, 5)
                    # 当自己空间的已经满足，则不增加报价
                    if state_array[0][2] < self.leader_node.delay_insensitive_size:
                        self.leader_node.price += add_value
                    else:
                        pass
                    if state_array[0][3] < self.leader_node.delay_sensitive_size:
                        self.leader_node.delay_sensitive_price += add_value
                    else:
                        pass
                    # 边界值判断
                    if self.leader_node.price > self.leader_node.user_price:
                        self.leader_node.price = self.leader_node.user_price
                    if self.leader_node.delay_sensitive_price > self.leader_node.user_price:
                        self.leader_node.delay_sensitive_price = self.leader_node.user_price
            else:
                for follower_id, follower_state in self.follower_node.items():
                    if k == follower_id:
                        # print(k)
                        if action == 0:
                            reduce_space = np.random.randint(1, 2)
                            # 追随者资源减少
                            self.follower_node[follower_id].delay_sensitive_space -= reduce_space
                            self.follower_node[follower_id].delay_insensitive_space -= reduce_space
                            if self.follower_node[follower_id].delay_sensitive_space < 0:
                                self.follower_node[follower_id].delay_sensitive_space = 0
                            if self.follower_node[follower_id].delay_insensitive_space < 0:
                                self.follower_node[follower_id].delay_insensitive_space = 0
                            # 判断领导者价格是否高于自己的成本
                            if self.leader_node.price < self.follower_node[follower_id].storage_cost:
                                # 价格低于成本，则将提供的空间置为零
                                self.follower_node[follower_id].delay_insensitive_space = 0
                            if self.leader_node.delay_sensitive_price < self.follower_node[follower_id].storage_cost:
                                self.follower_node[follower_id].delay_sensitive_space = 0
                            # 状态的更新
                            self.follower_node[follower_id].total_provide_space = (
                                    self.follower_node[follower_id].delay_sensitive_space +
                                    self.follower_node[follower_id].delay_insensitive_space)

                        if action == 1:
                            # 追随者资源不变
                            pass
                        if action == 2:
                            add_space = np.random.randint(1, 2)
                            # 追随者资源增加
                            # print("id", follower_id, "total_spcae", self.follower_node[follower_id].total_provide_space)
                            self.follower_node[follower_id].delay_sensitive_space += add_space
                            self.follower_node[follower_id].delay_insensitive_space += add_space
                            # 状态的更新
                            self.follower_node[follower_id].total_provide_space = (
                                    self.follower_node[follower_id].delay_sensitive_space +
                                    self.follower_node[follower_id].delay_insensitive_space)
                            remain_space = (self.follower_node[follower_id].storage_space
                                            - self.follower_node[follower_id].total_provide_space
                                            - self.follower_node[follower_id].used_storage_space)
                            # print("id", follower_id, "limit_remain_space", self.follower_node[follower_id].limit_remain_space)
                            # print("id", follower_id, "remain_space", remain_space)
                            exceed_spcae = remain_space - self.follower_node[follower_id].limit_remain_space
                            # print("id", follower_id, "exceed_spcae", exceed_spcae)
                            if exceed_spcae < 0:
                                self.follower_node[follower_id].delay_insensitive_space += exceed_spcae
                                if self.follower_node[follower_id].delay_insensitive_space < 0:
                                    # 时延敏感数据减少提供量
                                    self.follower_node[follower_id].delay_sensitive_space += self.follower_node[
                                        follower_id].delay_insensitive_space
                                    if self.follower_node[follower_id].delay_sensitive_space < 0:
                                        self.follower_node[follower_id].delay_sensitive_space = 0
                                    # 将时延不敏感资源提供的空间置为0
                                    self.follower_node[follower_id].delay_insensitive_space = 0
                            # 判断领导者价格是否高于自己的成本
                            if self.leader_node.price < self.follower_node[follower_id].storage_cost:
                                # 价格低于成本，则将提供的空间置为零
                                self.follower_node[follower_id].delay_insensitive_space = 0
                            if self.leader_node.delay_sensitive_price < self.follower_node[follower_id].storage_cost:
                                self.follower_node[follower_id].delay_sensitive_space = 0
                            # 状态的更新
                            self.follower_node[follower_id].total_provide_space = (
                                    self.follower_node[follower_id].delay_sensitive_space +
                                    self.follower_node[follower_id].delay_insensitive_space)

        # 统计总提供的值
        for node in self.follower_node.values():
            followers_delay_sensitive_space += node.delay_sensitive_space
            followers_delay_insensitive_space += node.delay_insensitive_space
        # print("followers_delay_sensitive_space", followers_delay_sensitive_space)
        # 进行利润的计算
        leader_profit, follower_profit1, follower_profit2 = self.calc_profit(followers_delay_insensitive_space,
                                                                             followers_delay_sensitive_space)
        # 保存历史利润
        if len(self.profit_list_history) < self.profit_list_len:
            profit = leader_profit + sum(follower_profit1.values()) + sum(follower_profit2.values())
            self.profit_list_history.append(profit)
        else:
            profit = leader_profit + sum(follower_profit1.values()) + sum(follower_profit2.values())
            # 先删再加
            del self.profit_list_history[0]
            self.profit_list_history.append(profit)
        # print("follower_profit1", follower_profit1)
        # print("follower_profit2", follower_profit2)

        # 进行状态的更新
        leader_next_state.extend((self.leader_node.price, self.leader_node.delay_sensitive_price,
                                  followers_delay_insensitive_space, followers_delay_sensitive_space,
                                  leader_profit))
        # print(leader_next_state)
        # 追随者状态更新
        for node_id, node_info in self.follower_node.items():
            # follower_state = []
            # 计算资源利用率
            utilization_rate = self.calc_utilization_rate(node_id)
            followers_next_state.extend((self.follower_node[node_id].delay_insensitive_space,
                                         self.follower_node[node_id].delay_sensitive_space,
                                         follower_profit1[node_id], follower_profit2[node_id],
                                         utilization_rate))
            # followers_next_state.append(follower_state)
        # print(len(followers_next_state))
        # print(followers_next_state)

        # print("leader state:", leader_next_state)
        # print(self.leader_node.delay_sensitive_size+self.leader_node.delay_insensitive_size)
        # sorted_nodes = sorted(self.follower_node.items(), key=lambda x: x[1].transaction_expectations, reverse=True)
        # print(sorted_nodes)
        # for i in sorted_nodes:
        #     print(i[1].transaction_expectations)

        next_state = np.array(leader_next_state + followers_next_state)
        # next_state_array = next_state.reshape((self.num_agents, self.node_state_dim))
        state_array = state.reshape((self.num_agents, self.node_state_dim))
        leader_reward, follower_reward = self.calc_reward(state_array)
        reward = leader_reward + sum(follower_reward.values())
        # print("reward: ", reward)
        # 设置
        # reward = np.random.rand()  # 随机生成奖励
        if np.var(self.profit_list_history) < self.variance and len(self.profit_list_history) >= self.profit_list_len:
            if state_array[0][2] + state_array[0][3] >= self.leader_node.need_storage_space:
                done = True
            else:
                done = False
        else:
            done = False
        # done = np.random.choice([0, 1], p=[0.95, 0.05])  # 设定一个简单的终止条件
        # print("next_state: ", next_state)
        return next_state, reward, done

    def get_agent_state(self, agent_id):
        # 获取特定智能体的状态
        return self.state  # 在简单情况下，所有智能体看到的状态是相同的

    def calc_profit(self, insensitive_space, sensitive_space):
        # 计算利润
        # leader_profit = 0
        follower_profit1 = {}
        follower_profit2 = {}
        # 进行节点的排序
        sorted_nodes = sorted(self.follower_node.items(), key=lambda x: x[1].transaction_expectations, reverse=True)
        # 排序后的节点
        self.sorted_node = sorted_nodes
        # 提供的空间大于所需空间
        if insensitive_space > self.leader_node.delay_insensitive_size:
            # print("不敏感空间大于所需")
            # 领导者的利润
            leader_profit = (self.leader_node.user_price - self.leader_node.price) * self.leader_node.delay_insensitive_size
            # 协作者的利润
            residue_space = 0
            for tup in sorted_nodes:
                residue_space += tup[1].delay_insensitive_space
                if residue_space <= self.leader_node.delay_insensitive_size:
                    follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * tup[1].delay_insensitive_space
                else:
                    # print("need space", self.leader_node.delay_insensitive_size)
                    # print("residue space", residue_space)
                    # print("tup",tup[0], "space", tup[1].delay_insensitive_space)
                    space_taken = self.leader_node.delay_insensitive_size - residue_space + tup[1].delay_insensitive_space
                    # print("mid space_token", space_taken)
                    if space_taken <= 0:
                        space_taken = 0
                    follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * space_taken
        else:
            # print("不敏感空间小于所需")
            leader_profit = (self.leader_node.user_price - self.leader_node.price) * insensitive_space
            for tup in sorted_nodes:
                follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * tup[1].delay_insensitive_space

        if sensitive_space > self.leader_node.delay_sensitive_size:
            # print("敏感空间大于所需")
            # print("sensitive_space", sensitive_space)
            # print("self.leader_node.delay_sensitive_size", self.leader_node.delay_sensitive_size)
            # 领导者的利润
            leader_profit += (self.leader_node.user_price - self.leader_node.delay_sensitive_price) * self.leader_node.delay_sensitive_size
            # 协作者的利润
            residue_space = 0
            for tup in sorted_nodes:
                # print(tup[0], tup[1].delay_sensitive_space)
                residue_space += tup[1].delay_sensitive_space
                # print("residue_space",residue_space)
                if residue_space <= self.leader_node.delay_sensitive_size:
                    profit = (self.leader_node.delay_sensitive_price - tup[1].storage_cost) * tup[
                        1].delay_sensitive_space
                    # print(profit)
                    follower_profit2[tup[0]] = profit
                else:
                    space_taken = self.leader_node.delay_sensitive_size - residue_space + tup[1].delay_sensitive_space
                    # print("space_token", space_taken)
                    if space_taken <= 0:
                        space_taken = 0
                    follower_profit2[tup[0]] = (self.leader_node.delay_sensitive_price - tup[
                        1].storage_cost) * space_taken
        else:
            # print("敏感空间小于所需")
            leader_profit += (self.leader_node.user_price - self.leader_node.delay_sensitive_price) * sensitive_space
            for tup in sorted_nodes:
                follower_profit2[tup[0]] = (self.leader_node.delay_sensitive_price - tup[1].storage_cost) * tup[
                    1].delay_sensitive_space


        # 将每次的利润进行更新
        # for k,v in follower_profit1.items():
        # print(follower_profit1)
        self.profit_list['leader'] = leader_profit
        self.profit_list['follower_1'] = follower_profit1
        self.profit_list['follower_2'] = follower_profit2
        # print(self.profit_list['leader'])
        return leader_profit, follower_profit1, follower_profit2

    def calc_reward(self, state: np.array):
        # 计算领导者的reward 考虑利润
        """
        对于领导者
        判断资源的满足率，并将其作为参数，乘以领导者的利润
        对于追随者
        计算空间剩余率，当剩余的空间小于10%时，将reward置为负值
        """
        total_get_space = state[0][2] + state[0][3]

        # 计算每个节点的reward
        # 计算领导者节点的奖励 ， 将资源的满足率作为领导者节点的奖励
        if state[0][2] < self.leader_node.delay_insensitive_size:
            insensitive_satisfaction_rate = ((self.leader_node.delay_insensitive_size - state[0][2])
                                             / self.leader_node.delay_insensitive_size)
        else:
            insensitive_satisfaction_rate = - ((state[0][2] - self.leader_node.delay_insensitive_size)
                                               / self.leader_node.delay_insensitive_size)
        # 延迟敏感数据
        if state[0][3] < self.leader_node.delay_sensitive_size:
            sensitive_satisfaction_rate = ((self.leader_node.delay_sensitive_size - state[0][3])
                                           / self.leader_node.delay_sensitive_size)
        else:
            sensitive_satisfaction_rate = - ((state[0][3] - self.leader_node.delay_sensitive_size)
                                             / self.leader_node.delay_sensitive_size)
        # if insensitive_satisfaction_rate < 0:
        #     print("state[0][3]", state[0][3])
        #     print("leader_node.delay_sensitive_size", self.leader_node.delay_sensitive_size)
        #     print("insensitive_satisfaction_rate", insensitive_satisfaction_rate)
        # if sensitive_satisfaction_rate < 0:
        #     print("sensitive_satisfaction_rate", sensitive_satisfaction_rate)
        reward_coefficient = (insensitive_satisfaction_rate + sensitive_satisfaction_rate) / 2
        # leader_reward = reward_coefficient * state[0][4]
        leader_reward = state[0][4] / 10000
        # print("reward_coefficient", reward_coefficient)
        # 计算追随者的奖励
        follower_reward = {}
        for i in range(self.follow_num):
            index = i + 1
            if state[index][4] > 0.9:
                # print(f'follower_{i}', "资源利用率", state[index][4])
                f_reward = - (state[index][4] - 0.9)
            else:
                f_reward = (0.9 - state[index][4]) / 0.9
            follower_reward[f'follower_{i}'] = f_reward
            # follower_reward[f'follower_{i}'] = f_reward * (state[0][2] + state[0][3])
        # print(follower_reward)
        return leader_reward, follower_reward

    def calc_utilization_rate(self, node_id):
        # 计算资源利用率
        using_space = (self.follower_node[node_id].delay_insensitive_space
                       + self.follower_node[node_id].delay_sensitive_space
                       + self.follower_node[node_id].used_storage_space)
        return using_space / self.follower_node[node_id].storage_space

    def get_follower_node(self, node_id):
        # 获得节点的信息
        return self.follower_node[f'follower_{node_id}']

    def get_leader_node(self):
        return self.leader_node

    def get_total_profit(self):
        index = len(self.profit_list_history)
        return self.profit_list_history[index - 1]

    def get_agent_profit(self):
        # 返回利润列表
        return self.profit_list

    def get_sorted_node(self):
        return self.sorted_node
