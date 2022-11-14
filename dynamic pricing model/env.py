import re
import numpy as np
import json
import tqdm


class Env():
    def __init__(self, all_data, user_details, mean_loads, std_loads, adj_fac, power_price_whole, alpha, beta):
        self.data = all_data
        self.adj_fac = adj_fac
        self.data_num = len(all_data)
        self.user_class = user_details['user_class']
        self.user_price_max = user_details['max_price']
        self.user_price_min = user_details['min_price']
        self.power_price_whole = power_price_whole
        self.user_price_ave = user_details['ave_price']
        self.rank_price = self.user_price_ave

        self.peak_time = user_details['peak_time']
        self.peace_time = user_details['peace_time']
        self.valley_time = user_details['valley_time']
        self.max_loads = user_details['max_load']
        self.min_loads = user_details['min_load']
        self.ave_loads = user_details['ave_loads']
        self.mean_loads = mean_loads
        self.std_loads = std_loads

        self.action_space = [0.02, 0.03, -0.06, 0.09, 0.1, -0.03, 0.06, -0.09, -0.02, -0.1]
        self.coefficient = [-0.05, -0.1, -0.2]  #调控系数因子，[低谷期，过渡期，高峰期]
        self.alpha = alpha
        self.beta = beta
        self.action_num = len(self.action_space)
        self.features = 3
        return

    def reset(self, user_num):
        self.user_num = user_num
        self.index = 0
        self.current_total_load = self.data[user_num][0]
        self.current_adj_load = self.adj_fac*self.data[user_num][0]
        self.current_user_price = self.user_price_ave
        self.current_coe = self.coefficient[0]
        self.day_done = False
        current_state = [self.current_total_load, self.current_adj_load, self.current_user_price]
        return current_state

    def step(self, action):
        current_action = self.action_space[action]  #根据动作更新当前的价格
        pre_user_price = self.current_user_price
        current_user_price = self.current_user_price*(1+current_action)

        if current_user_price < self.user_price_min:  #价格在一定区间内波动
            self.current_user_price = self.user_price_min
        elif current_user_price > self.user_price_max:
            self.current_user_price = self.user_price_max
        else:
            self.current_user_price = current_user_price

        if self.index in self.valley_time:  #不同的时间段又价格对用电量的影响不同
            self.current_coe = self.coefficient[0]
            self.rank_price = self.user_price_min
        elif self.index in self.peace_time:
            self.current_coe = self.coefficient[1]
            self.rank_price = self.user_price_ave
        elif self.index in self.peak_time:
            self.current_coe = self.coefficient[2]
            self.rank_price = self.user_price_max

        reward, total_actual_load = self.reward(pre_user_price)  #根据当前动作计算奖励值

        self.index = self.index + 1
        if self.index == 48:
            self.user_num = self.user_num + 1
            self.index = 0
            self.day_done = True
        else:
            self.current_total_load = self.data[self.user_num][self.index]  # 更新当前所需总负荷值
            self.current_adj_load = self.adj_fac * self.current_total_load  # 更新当前所需可调节负荷值

        current_state = [self.current_total_load, self.current_adj_load, self.current_user_price]  #更新当前的状态

        return current_state, reward, self.day_done, self.current_user_price, self.rank_price, total_actual_load

    def reward(self, pre_user_price):
        adj_actual_load = (1 + self.current_coe * ((self.current_user_price - self.rank_price) / self.user_price_ave)) * self.adj_fac * self.current_total_load  #可调节实际负荷

        total_actual_load = (1-self.adj_fac)*self.current_total_load + adj_actual_load

        if self.user_class == 'A':
            fac_reward = (self.current_total_load - total_actual_load) * (
                        self.current_total_load - total_actual_load) * self.alpha

            if fac_reward < 0:
                fac_reward = 0.0
            else:
                fac_reward = fac_reward

            if total_actual_load == self.current_total_load:
                reward = -0.5
            else:
                reward = ((self.current_user_price - self.power_price_whole) * total_actual_load -
                          (self.rank_price - self.power_price_whole) * self.current_total_load) / self.std_loads - fac_reward
        elif self.user_class == 'B':
            fac_reward = (self.current_total_load - total_actual_load) * 0.01  #类型二最优

            if fac_reward < 0:
                fac_reward = 0.0
            else:
                fac_reward = fac_reward

            if total_actual_load == self.current_total_load:
                reward = -0.5
            else:
                if self.current_total_load < self.min_loads:
                    reward = ((total_actual_load - self.current_total_load) / (self.ave_loads - self.current_total_load)) * 10
                elif self.current_total_load > self.max_loads:
                    reward = ((total_actual_load - self.current_total_load) / (self.ave_loads - self.current_total_load)) * 10
                else:
                    if adj_actual_load > self.max_loads:
                        reward = -0.5
                    else:
                        reward = (((self.current_user_price - self.power_price_whole) * adj_actual_load -
                                  (self.rank_price - self.power_price_whole) * self.current_total_load) / (self.power_price_whole * adj_actual_load)) - fac_reward
        else:
            fac_reward = (self.current_total_load - total_actual_load) * 0.025

            if fac_reward < 0:
                fac_reward = 0.0
            else:
                fac_reward = fac_reward

            if total_actual_load == self.current_total_load:
                reward = -0.5
            else:
                if self.current_total_load > self.max_loads:
                    reward = ((total_actual_load - self.current_total_load) / (self.ave_loads - self.current_total_load)) * 10
                else:
                    if total_actual_load > self.max_loads:
                        reward = -1.0
                    else:
                        reward = ((self.current_user_price - self.power_price_whole) * total_actual_load -
                                  (self.rank_price - self.power_price_whole) * self.current_total_load) / self.std_loads - fac_reward

        return reward, total_actual_load


class Reader():
    def __init__(self, read_data_path):
        self.read_data_path = read_data_path
        return

    def read_all_user_loads(self):
        print("正在读取数据！")
        all_user_loads = {}
        with open(self.read_data_path, 'r', encoding='utf-8') as f_read:
            lines = f_read.readlines()
            for k,line in enumerate(tqdm.tqdm(lines)):
                if k == 200:
                    pass
                data_json = json.loads(line)
                user_num = int(list(data_json.keys())[0])
                user_loads = list(data_json.values())[0]

                if user_num not in all_user_loads.keys():
                    all_user_loads[user_num] = user_loads
                else:
                    all_user_loads[user_num].extend(user_loads)
        f_read.close()
        user_numbers = len(all_user_loads.keys())
        print("数据读取完成，共读取到%s个用户的数据！"%str(user_numbers))
        return all_user_loads

    def read_user_class(self, read_data_path):
        all_user_class = {}
        with open(read_data_path, 'r', encoding='utf-8') as f_read:
            lines = f_read.readlines()
            for line in tqdm.tqdm(lines):
                user_num = int(line.split('\t')[0])
                user_class = int(line.split('\t')[1].strip())

                if user_class not in all_user_class.keys():
                    all_user_class[user_class] = [user_num]
                else:
                    all_user_class[user_class].append(user_num)
        f_read.close()

        return all_user_class

    def create_train_data(self, all_user_loads, all_user_class, user_class):
        all_data = []
        all_user_nums = all_user_class[user_class]

        if user_class == 2:
            for user in all_user_nums:
                user_loads = all_user_loads[user]

                for user_load in user_loads:
                    day_user_load = user_load
                    all_data.append(day_user_load)
        else:
            for user in all_user_nums:
                user_loads = all_user_loads[user]

                for user_load in user_loads:
                    day_user_load = user_load
                    all_data.append(day_user_load)
                    break

        all_data = np.array(all_data)

        mean_load = np.mean(all_data)
        std_load = np.std(all_data)

        train_data = all_data[:int(0.95*len(all_data))]
        test_data = all_data[int(0.95*len(all_data)):]

        print('train_x shape: {}'.format(train_data.shape))
        print('test_x shape: {}'.format(test_data.shape))

        return train_data,test_data,mean_load,std_load,all_data

    def write_data(self, train_data, test_data, all_data):
        np.save('data/train_data.npy', train_data)
        np.save('data/test_data.npy', test_data)
        np.save('data/all_data.npy', all_data)
        return