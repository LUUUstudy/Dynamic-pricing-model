import re
import rlmodel
import env
import numpy as np
import matplotlib.pyplot as plt
import json
import os

READ_DATA_PATH = '../data/all_user_loads.json'
CLASS_RESULT_PATH = '../data/all_user_class.csv'

class_num = 1  #训练用户的类型

def train():
    if class_num == 0 or class_num == 2:
        epoch_num = 10
    else:
        epoch_num = 25
    sample_num = 5
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
        train_data, test_data, mean_load, std_load, all_data = reader.create_train_data(all_user_loads, all_user_class, class_num)
        reader.write_data(train_data, test_data, all_data)
    else:
        train_data = np.load(os.path.join(data_path, 'train_data.npy'))
        test_data = np.load(os.path.join(data_path, 'test_data.npy'))
        all_data = np.load(os.path.join(data_path, 'all_data.npy'))

        if class_num == 0 or class_num == 2:
            train_data = train_data[:500]

        print('train_x shape: {}'.format(train_data.shape))
        print('test_x shape: {}'.format(test_data.shape))
        mean_load = np.mean(all_data)
        std_load =np.std(all_data)

    train_data_num = len(train_data)

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

    rewards = []

    Env = env.Env(all_data=train_data, user_details=user_details, mean_loads=mean_load, std_loads=std_load, adj_fac=0.6,
              power_price_whole=9, alpha=0.8, beta=0.01)

    Agent = rlmodel.DeepQNetwork(Env.action_num, Env.features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.6,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.01,
                      output_graph=False)

    for epoch in range(epoch_num):
        print("正在训练第%s轮！"%str(epoch+1))
        user_num = 0
        while True:
            for sample in range(sample_num):
                state = Env.reset(user_num=user_num)  # 对一天的数据采样重置环境状态
                while True:
                    state = np.array(state)
                    obversation = []

                    obversation.append((state[0]-mean_load)/(std_load))  #标准化
                    obversation.append((state[1]-mean_load)/(std_load))
                    obversation.append((state[2]-mean_price)/(std_price))

                    obversation = np.array(obversation)

                    action = Agent.choose_action(obversation)  #根据当前的观测状态选择动作

                    state_,reward,day_done,_,_,_ = Env.step(action)  #根据当前选择的动作改变环境值
                    state_ = np.array(state_)
                    obversation_ = []

                    obversation_.append((state_[0] - mean_load) / (std_load))
                    obversation_.append((state_[1] - mean_load) / (std_load))
                    obversation_.append((state_[2] - mean_price) / (std_price))

                    obversation_ = np.array(obversation_)

                    Agent.store_transition(obversation, action, reward, obversation_)  #将当前的数据存入经验池

                    if (step>2000) and (step%5==0):
                        Agent.learn()  #更新策略网络

                    state = state_  #更新当前的状态

                    if day_done:
                        break

                    step = step + 1

            user_num = user_num + 1

            if user_num >= train_data_num:
                break

    Agent.save(class_num)

    return

def main():
    train()
    return

if __name__ == '__main__':
    main()