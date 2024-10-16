import numpy as np
import node


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.pre_leader_state = None
        self.action_dim = 3  # 设置智能体的动作维度
        # 节点数量为领导者数加追随者数
        self.agent_num = 16
        self.leader_node = None
        self.follower_node = {}
        # 追随者的数量
        self.follow_num = self.agent_num - 1
        # 每个节点的状态空间
        self.node_state_dim = 5
        # 节点的状态包括： 节点id, 当前利润， 领导者的报价， 提供的存储资源， 剩余存储资源比例
        self.state_dim = self.node_state_dim * self.agent_num
        # 历史列表的长度
        self.profit_list_len = 10
        # 利润的历史列表
        self.profit_list_history = None
        # 判断是否结束的值
        self.variance = 10
        self.sorted_node = None

        self.obs_dim = self.state_dim  # 设置智能体的观测维度

        self.step_index = 0

        self.pre_reward = None

        self.agent_state_array = None

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        self.profit_list = {}
        self.profit_list_history = []
        self.state = []
        # 初始化节点状态
        self.leader_node = node.LeaderNode()
        for i in range(self.follow_num):
            self.follower_node[f'follower_{i}'] = node.FollowerNode(i)
        # 初始化状态 二维
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(self.state_dim,))
            self.state.append(sub_obs)

        # print(self.state)
        # self.step_index = 0
        self.pre_leader_state = [0, 0, 0, 0, 0]
        self.pre_reward = 0
        self.agent_state_array = np.zeros((self.agent_num, self.node_state_dim))
        return self.state

        # sub_agent_obs = []
        # for i in range(self.agent_num):
        #     sub_obs = np.random.random(size=(14,))
        #     sub_agent_obs.append(sub_obs)
        # # print(sub_agent_obs)
        # return sub_agent_obs

    def step(self, actions):

        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        """

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []

        # 保存领导者与追随者的下一个状态
        leader_next_state = []
        followers_next_state = []
        # 提供的总空间 敏感型
        followers_delay_sensitive_space = 0
        # 提供的总空间 非敏感型
        followers_delay_insensitive_space = 0

        # print(actions[0])
        for i in range(len(actions)):
            # 获得智能体的动作
            action = actions[i].tolist().index(1)
            # if i ==0:
                # print("leader action",action)
            # 从领导者开始执行动作
            if i == 0:
                # 领导者
                if action == 0:
                    # 领导者报价降低
                    reduce_value = np.random.uniform(1, 5)
                    if self.pre_leader_state[2] > self.leader_node.delay_insensitive_size:
                        self.leader_node.price -= reduce_value
                    else:
                        pass
                    if self.pre_leader_state[3] > self.leader_node.delay_sensitive_size:
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
                    if self.pre_leader_state[2] < self.leader_node.delay_insensitive_size:
                        self.leader_node.price += add_value
                    else:
                        pass
                    if self.pre_leader_state[3] < self.leader_node.delay_sensitive_size:
                        self.leader_node.delay_sensitive_price += add_value
                    else:
                        pass
                    # 边界值判断
                    if self.leader_node.price > self.leader_node.user_price:
                        self.leader_node.price = self.leader_node.user_price
                    if self.leader_node.delay_sensitive_price > self.leader_node.user_price:
                        self.leader_node.delay_sensitive_price = self.leader_node.user_price
            else:
                node_index = f'follower_{i - 1}'
                # 追随者
                if action == 0:
                    reduce_space = np.random.randint(1, 2)
                    # 追随者资源减少
                    self.follower_node[node_index].delay_sensitive_space -= reduce_space
                    self.follower_node[node_index].delay_insensitive_space -= reduce_space
                    if self.follower_node[node_index].delay_sensitive_space < 0:
                        self.follower_node[node_index].delay_sensitive_space = 0
                    if self.follower_node[node_index].delay_insensitive_space < 0:
                        self.follower_node[node_index].delay_insensitive_space = 0
                    # 判断领导者价格是否高于自己的成本
                    if self.leader_node.price < self.follower_node[node_index].storage_cost:
                        # 价格低于成本，则将提供的空间置为零
                        self.follower_node[node_index].delay_insensitive_space = 0
                    if self.leader_node.delay_sensitive_price < self.follower_node[node_index].storage_cost:
                        self.follower_node[node_index].delay_sensitive_space = 0
                    # 状态的更新
                    self.follower_node[node_index].total_provide_space = (
                            self.follower_node[node_index].delay_sensitive_space +
                            self.follower_node[node_index].delay_insensitive_space)
                if action == 1:
                    # 追随者资源不变
                    pass
                if action == 2:
                    add_space = np.random.randint(1, 2)
                    # 追随者资源增加
                    # print("id", follower_id, "total_spcae", self.follower_node[follower_id].total_provide_space)
                    self.follower_node[node_index].delay_sensitive_space += add_space
                    self.follower_node[node_index].delay_insensitive_space += add_space
                    # 状态的更新
                    self.follower_node[node_index].total_provide_space = (
                            self.follower_node[node_index].delay_sensitive_space +
                            self.follower_node[node_index].delay_insensitive_space)
                    remain_space = (self.follower_node[node_index].storage_space
                                    - self.follower_node[node_index].total_provide_space
                                    - self.follower_node[node_index].used_storage_space)
                    # print("id", follower_id, "limit_remain_space", self.follower_node[follower_id].limit_remain_space)
                    # print("id", follower_id, "remain_space", remain_space)
                    exceed_spcae = remain_space - self.follower_node[node_index].limit_remain_space
                    # print("id", follower_id, "exceed_spcae", exceed_spcae)
                    if exceed_spcae < 0:
                        self.follower_node[node_index].delay_insensitive_space += exceed_spcae
                        if self.follower_node[node_index].delay_insensitive_space < 0:
                            # 时延敏感数据减少提供量
                            self.follower_node[node_index].delay_sensitive_space += self.follower_node[
                                node_index].delay_insensitive_space
                            if self.follower_node[node_index].delay_sensitive_space < 0:
                                self.follower_node[node_index].delay_sensitive_space = 0
                            # 将时延不敏感资源提供的空间置为0
                            self.follower_node[node_index].delay_insensitive_space = 0
                    # 判断领导者价格是否高于自己的成本
                    if self.leader_node.price < self.follower_node[node_index].storage_cost:
                        # 价格低于成本，则将提供的空间置为零
                        self.follower_node[node_index].delay_insensitive_space = 0
                    if self.leader_node.delay_sensitive_price < self.follower_node[node_index].storage_cost:
                        self.follower_node[node_index].delay_sensitive_space = 0
                    # 状态的更新
                    self.follower_node[node_index].total_provide_space = (
                            self.follower_node[node_index].delay_sensitive_space +
                            self.follower_node[node_index].delay_insensitive_space)

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

        # 进行状态的更新
        leader_next_state.extend((self.leader_node.price, self.leader_node.delay_sensitive_price,
                                  followers_delay_insensitive_space, followers_delay_sensitive_space,
                                  leader_profit))
        # leader_next_state_array = np.array(leader_next_state)
        # print(leader_next_state)
        # 追随者状态更新
        for node_index, node_info in self.follower_node.items():
            # follower_state = []
            # 计算资源利用率
            utilization_rate = self.calc_utilization_rate(node_index)
            followers_next_state.extend((self.follower_node[node_index].delay_insensitive_space,
                                         self.follower_node[node_index].delay_sensitive_space,
                                         follower_profit1[node_index], follower_profit2[node_index],
                                         utilization_rate))
            # followers_next_state.append(np.array(follower_state))
        agent_obs = leader_next_state + followers_next_state
        agent_obs_array = np.array(agent_obs)
        # 计算奖励
        next_state = np.array(leader_next_state + followers_next_state)
        next_state_array = next_state.reshape((self.agent_num, self.node_state_dim))
        self.agent_state_array = next_state_array
        leader_reward, follower_reward = self.calc_reward(next_state_array)
        sub_agent_reward.append([leader_reward])
        for i in range(self.agent_num - 1):
            sub_agent_reward.append([follower_reward[f'follower_{i}']])
        # 判断是否结束
        if (np.var(self.profit_list_history) < self.variance and len(self.profit_list_history) >= self.profit_list_len):
            if leader_next_state[2] + leader_next_state[3] >= self.leader_node.need_storage_space:
                done = True
        else:
            done = False
        sub_agent_info = []
        # a = []
        for i in range(self.agent_num):
            sub_agent_obs.append(agent_obs_array)
            # sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(done)
            sub_agent_info.append({})
            # a.append(done)
        # print(a)
        # print(sub_agent_done)
        # print(self.step_index)
        # print(leader_next_state)
        self.pre_leader_state = leader_next_state
        self.step_index += 1
        # if self.step_index % 50 == 0:
        #     print(agent_obs)
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

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
        # 空间大于所需空间
        if insensitive_space > self.leader_node.delay_insensitive_size:
            # 领导者的利润
            leader_profit = (
                                        self.leader_node.user_price - self.leader_node.price) * self.leader_node.delay_insensitive_size
            # 协作者的利润
            residue_space = 0
            for tup in sorted_nodes:
                residue_space += tup[1].delay_insensitive_space
                if residue_space <= self.leader_node.delay_insensitive_size:
                    follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * tup[
                        1].delay_insensitive_space
                else:
                    # print("need space", self.leader_node.delay_insensitive_size)
                    # print("residue space", residue_space)
                    # print("tup",tup[0], "space", tup[1].delay_insensitive_space)
                    space_taken = self.leader_node.delay_insensitive_size - residue_space + tup[
                        1].delay_insensitive_space
                    # print("mid space_token", space_taken)
                    if space_taken <= 0:
                        space_taken = 0
                    follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * space_taken
        else:
            leader_profit = (self.leader_node.user_price - self.leader_node.price) * insensitive_space
            for tup in sorted_nodes:
                follower_profit1[tup[0]] = (self.leader_node.price - tup[1].storage_cost) * tup[
                    1].delay_insensitive_space

        if sensitive_space > self.leader_node.delay_sensitive_size:
            # print("sensitive_space", sensitive_space)
            # print("self.leader_node.delay_sensitive_size", self.leader_node.delay_sensitive_size)
            # 领导者的利润
            leader_profit += (
                                         self.leader_node.user_price - self.leader_node.delay_sensitive_price) * self.leader_node.delay_sensitive_size
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
            leader_profit += (self.leader_node.user_price - self.leader_node.delay_sensitive_price) * sensitive_space
            for tup in sorted_nodes:
                follower_profit2[tup[0]] = (self.leader_node.delay_sensitive_price - tup[1].storage_cost) * tup[
                    1].delay_sensitive_space

        # print("follower_profit2", follower_profit2)
        self.profit_list['leader'] = leader_profit
        self.profit_list['follower_1'] = follower_profit1
        self.profit_list['follower_2'] = follower_profit2
        # print(self.profit_list['leader'])
        return leader_profit, follower_profit1, follower_profit2

    def calc_reward(self, next_state: np.array):
        # 计算每个节点的reward
        # 计算领导者节点的奖励 ， 将资源的满足率作为领导者节点的奖励
        if next_state[0][2] < self.leader_node.delay_insensitive_size:
            insensitive_satisfaction_rate = ((self.leader_node.delay_insensitive_size - next_state[0][2])
                                             / self.leader_node.delay_insensitive_size)
        else:
            insensitive_satisfaction_rate = ((next_state[0][2] - self.leader_node.delay_insensitive_size)
                                             / self.leader_node.delay_insensitive_size)
        # 延迟敏感数据
        if next_state[0][3] < self.leader_node.delay_sensitive_size:
            sensitive_satisfaction_rate = ((self.leader_node.delay_sensitive_size - next_state[0][3])
                                           / self.leader_node.delay_sensitive_size)
        else:
            sensitive_satisfaction_rate = ((next_state[0][3] - self.leader_node.delay_sensitive_size)
                                           / self.leader_node.delay_sensitive_size)
        leader_reward = (insensitive_satisfaction_rate + sensitive_satisfaction_rate) / 2
        # 计算追随者的奖励
        follower_reward = {}
        for i in range(self.follow_num):
            index = i + 1
            # using_sapce = self.follower_node[f'follower_{i}'].used_storage_space
            # if next_state[index][2] != 0:
            #     using_sapce += self.follower_node[f'follower_{i}'].delay_insensitive_space
            # if next_state[index][3] != 0:
            #     using_sapce += self.follower_node[f'follower_{i}'].delay_sensitive_space
            # # 计算真正的资源利用率
            # utilization_rate = using_sapce / self.follower_node[f'follower_{i}'].storage_space
            if next_state[index][4] > 0.9:
                f_reward = - (next_state[index][4] - 0.9)
            else:
                f_reward = (0.9 - next_state[index][4]) / 0.9
            follower_reward[f'follower_{i}'] = f_reward * 10
            # follower_reward[f'follower_{i}'] = f_reward * 10
        # print(follower_reward)
        self.pre_reward = leader_reward + sum(follower_reward.values())
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
        # print("env core profit", self.profit_list_history[index - 1])
        return self.profit_list_history[index - 1]

    def get_agent_profit(self):
        # 返回利润列表
        return self.profit_list

    def get_sorted_node(self):
        return self.sorted_node

    def get_reward(self):
        return self.pre_reward

    def get_state(self):
        return self.agent_state_array
