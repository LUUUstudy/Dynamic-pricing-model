import re
import rlmodel
import env
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib as mpl
import random

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

READ_DATA_PATH = '../data/all_user_loads.json'
CLASS_RESULT_PATH = '../data/all_user_class.csv'

class_num = 0  #训练用户的类型

def predict():
    step = 0

    if class_num == 0:
        data_path = 'data(类型一)'
    elif class_num == 1:
        data_path = 'data(类型二)'
    else:
        data_path = 'data(类型三)'

    if not os.path.exists(os.path.join(data_path, 'train_data.npy')):
        reader = env.Reader(READ_DATA_PATH)
        all_user_loads = reader.read_all_user_loads()
        all_user_class = reader.read_user_class(CLASS_RESULT_PATH)
        train_data, test_data, mean_load, std_load, all_data = reader.create_train_data(all_user_loads, all_user_class,
                                                                                        class_num)
        reader.write_data(train_data, test_data, all_data)
    else:
        train_data = np.load(os.path.join(data_path, 'train_data.npy'))
        test_data = np.load(os.path.join(data_path, 'test_data.npy'))
        all_data = np.load(os.path.join(data_path, 'all_data.npy'))
        print('test_x shape: {}'.format(test_data.shape))
        mean_load = np.mean(all_data)
        std_load = np.std(all_data)

    test_data = np.array([test_data[random.randint(0, len(test_data)-1)]])

    user_details_path = '../data/user' + str(class_num + 1) + '.json'

    with open(user_details_path, 'r', encoding='utf-8') as f_read:
        line = f_read.readline()
        user_details = json.loads(line)
    f_read.close()

    all_price = []
    all_price.append(user_details['max_price'])
    all_price.append(user_details['min_price'])
    all_price.append(user_details['ave_price'])

    mean_price = np.mean(np.array(all_price))
    std_price = np.std(np.array(all_price))

    Env = env.Env(all_data=test_data, user_details=user_details, mean_loads=mean_load, std_loads=std_load, adj_fac=0.6,
                  power_price_whole=9, alpha=0.8, beta=0.1)

    Agent = rlmodel.DeepQNetwork(Env.action_num, Env.features,
                                 learning_rate=0.001,
                                 reward_decay=0.9,
                                 e_greedy=1,
                                 replace_target_iter=200,
                                 memory_size=2000,
                                 output_graph=False)

    Agent.restore(class_num)

    for i in range(len(test_data)):
        x = []
        load = []
        max_load = []
        min_load = []
        ave_load = []
        actual_load = []
        price = []
        rank_prices = []
        actual_adj_sy = []
        rank_sy = []
        unsat = []
        sy = []
        step = 0
        state = Env.reset(user_num=i)
        while True:
            x.append(step)
            load.append(state[0]*0.001)
            max_load.append(user_details['max_load'])
            min_load.append(user_details['min_load'])
            ave_load.append(user_details['ave_loads'])
            state = np.array(state)

            obversation = []

            obversation.append((state[0] - mean_load) / (std_load))
            obversation.append((state[1] - mean_load) / (std_load))
            obversation.append((state[2] - mean_price) / (std_price))

            obversation = np.array(obversation)

            action = Agent.choose_action(obversation)  # 根据当前的观测状态选择动作

            state_, reward, day_done, current_price, rank_price, current_actual_load = Env.step(action)  # 根据当前选择的动作改变环境值
            state_ = np.array(state_)

            actual_load.append(current_actual_load*0.001)

            price.append(current_price)
            rank_prices.append(rank_price)

            actual_adj_sy.append(current_actual_load*current_price*0.001)
            rank_sy.append(state[0]*rank_price*0.001)
            sy.append(current_actual_load*current_price*0.001 - state[0]*rank_price*0.001)

            if state[0] - current_actual_load > 0:
                unsat.append(((state[0] - current_actual_load)*(state[0] - current_actual_load)*0.8) / state[0])
            else:
                unsat.append(0)

            state = state_

            step = step + 1

            if day_done:
                break

        act_pro_after_adj = np.sum(actual_adj_sy)
        rank_pro = np.sum(rank_sy)

        print("调价后的实际收益: %s" % str(act_pro_after_adj))

        print("分段式电费的收益: %s" % str(rank_pro))

        print("用户不满意度: %s" % str(np.mean(np.array(unsat))))

        plt.subplot(3, 1, 1)
        plt.bar(x, load, label='实际需求电荷', fc='b')
        plt.plot(x, actual_load, color='yellow', lw=1.5, label='调节后的实际电荷')
        plt.ylabel('电荷（KWh）')
        plt.xticks([])
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(x, actual_adj_sy, color='m', label='实际调节后收益')
        plt.plot(x, rank_sy, color='b', label='分段式电价收益')
        plt.ylabel('收益（爱尔兰磅）')
        plt.xticks([])
        plt.legend()

        plt.subplot(3, 1, 3)

        plt.plot(x, price, 'g*-', linewidth=0.8, markersize='3', label='动态调节价格')
        plt.plot(x, rank_prices, color='b', lw=0.5, label='分段式价格')
        plt.ylabel('电价（爱尔兰磅/KWh）')
        plt.legend()
        plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                    26, 28, 30, 32, 34, 36, 38, 40,
                    42, 44, 46, 48])
        plt.xlabel('时间（/30min）')
        plt.show()

    return

def test(data_path, test_data, model_path, user_details_path):
    all_data = np.load(os.path.join(data_path, 'all_data.npy'))
    test_data = test_data
    mean_load = np.mean(all_data)
    std_load = np.std(all_data)

    with open(user_details_path, 'r', encoding='utf-8') as f_read:
        line = f_read.readline()
        user_details = json.loads(line)
    f_read.close()

    all_price = []
    all_price.append(user_details['max_price'])
    all_price.append(user_details['min_price'])
    all_price.append(user_details['ave_price'])

    mean_price = np.mean(np.array(all_price))
    std_price = np.std(np.array(all_price))

    Env = env.Env(all_data=test_data, user_details=user_details, mean_loads=mean_load, std_loads=std_load, adj_fac=0.6,
                  power_price_whole=9, alpha=0.8, beta=0.1)

    Agent = rlmodel.DeepQNetwork(Env.action_num, Env.features,
                                 learning_rate=0.001,
                                 reward_decay=0.9,
                                 e_greedy=1,
                                 replace_target_iter=200,
                                 memory_size=2000,
                                 output_graph=False)

    Agent.test_restore(model_path)

    for i in range(len(test_data)):
        x = []
        load = []
        max_load = []
        min_load = []
        ave_load = []
        actual_load = []
        price = []
        rank_prices = []
        actual_adj_sy = []
        rank_sy = []
        unsat = []
        step = 0
        state = Env.reset(user_num=i)
        while True:
            x.append(step)
            load.append(state[0]*0.001)
            max_load.append(user_details['max_load'])
            min_load.append(user_details['min_load'])
            ave_load.append(user_details['ave_loads'])
            state = np.array(state)

            obversation = []

            obversation.append((state[0] - mean_load) / (std_load))
            obversation.append((state[1] - mean_load) / (std_load))
            obversation.append((state[2] - mean_price) / (std_price))

            obversation = np.array(obversation)

            action = Agent.choose_action(obversation)  # 根据当前的观测状态选择动作

            state_, reward, day_done, current_price, rank_price, current_actual_load = Env.step(action)  # 根据当前选择的动作改变环境值
            state_ = np.array(state_)

            actual_load.append(current_actual_load*0.001)

            price.append(current_price)
            rank_prices.append(rank_price)

            actual_adj_sy.append(current_actual_load*current_price*0.001)
            rank_sy.append(state[0]*rank_price*0.001)

            if state[0] - current_actual_load > 0:
                unsat.append(((state[0] - current_actual_load)*(state[0] - current_actual_load)*0.8) / state[0])
            else:
                unsat.append(0)

            state = state_

            step = step + 1

            if day_done:
                break

        act_pro_after_adj = np.sum(actual_adj_sy)
        rank_pro = np.sum(rank_sy)

        print("调价后的实际收益: %s"%str(act_pro_after_adj))

        print("分段式电费的收益: %s"%str(rank_pro))

        print("用户不满意度: %s"%str(np.mean(np.array(unsat))))

        plt.subplot(3, 1, 1)
        plt.bar(x, load, label='实际需求电荷', fc='b')
        plt.plot(x, actual_load, color='yellow', lw=1.5, label='调节后的实际电荷')
        plt.ylabel('电荷（KWh）')
        plt.xticks([])
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(x, actual_adj_sy, color='m', label='实际调节后收益')
        plt.plot(x, rank_sy, color='b', label='分段式电价收益')
        plt.ylabel('收益（爱尔兰磅）')
        plt.xticks([])
        plt.legend()

        plt.subplot(3, 1, 3)

        plt.plot(x, price, 'g*-', linewidth=0.8, markersize='3', label='动态调节价格')
        plt.plot(x, rank_prices, color='b', lw=0.5, label='分段式价格')
        plt.ylabel('电价（爱尔兰磅/KWh）')
        plt.legend()
        plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                    26, 28, 30, 32, 34, 36, 38, 40,
                    42, 44, 46, 48])
        plt.xlabel('时间（/30min）')
        plt.show()

    return

def main():
    predict()
    return

if __name__ == '__main__':
    main()