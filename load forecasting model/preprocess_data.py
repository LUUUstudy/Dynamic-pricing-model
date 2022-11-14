import re
import numpy as np
import json
import tqdm
import os

READ_DATA_PATH = '../data/all_user_loads.json'
CLASS_RESULT_PATH = '../data/all_user_class.csv'

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

        for user in all_user_nums:
            user_loads = all_user_loads[user]

            for user_load in user_loads:
                day_user_load = user_load
                all_data.extend(day_user_load)
            break

        all_data = np.array(all_data)

        mean_load = np.mean(all_data)
        std_load = np.std(all_data)
        all_data = (all_data - mean_load) / std_load

        all_data = all_data[:, np.newaxis]

        all_x = []
        all_y = []

        time_step = 20
        for i in range(len(all_data) - time_step - 1):
            x = all_data[i: i + time_step]
            y = all_data[i + time_step]
            all_x.append(x.tolist())
            all_y.append(y.tolist())

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        train_x = all_x[:int(0.95*len(all_x))]
        train_y = all_y[:int(0.95*len(all_y))]

        test_x = all_x[int(0.95*len(all_x)):]
        test_y = all_y[int(0.95*len(all_y)):]

        return train_x,train_y,test_x,test_y,all_data,mean_load,std_load

    def com_mean_std(self, all_data):
        mean_load = np.mean(all_data)
        std_load = np.std(all_data)
        return mean_load,std_load

    def write_data(self, train_x, train_y, test_x, test_y, data_path):
        np.save(os.path.join(data_path, 'train_x.npy'), train_x)
        np.save(os.path.join(data_path, 'train_y.npy'), train_y)
        np.save(os.path.join(data_path, 'test_x.npy'), test_x)
        np.save(os.path.join(data_path, 'test_y.npy'), test_y)
        return