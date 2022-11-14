import re
import json
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

READ_DATA_PATH = 'data/all_user_loads.json'
CLASS_RESULT_PATH = 'data/all_user_class.csv'

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

def plot():
    reader = Reader(READ_DATA_PATH)
    all_user_loads = reader.read_all_user_loads()
    all_user_class = reader.read_user_class(CLASS_RESULT_PATH)

    user_nums = all_user_class.keys()

    for user_class in user_nums:
        users = all_user_class[user_class]
        mean_loads = []

        f, ax = plt.subplots()
        for user in users:
            user_loads = all_user_loads[user]

            for user_load in user_loads:
                lines = plt.plot(np.multiply(0.001, user_load), 'b')
                plt.setp(lines, color='b', linewidth=0.3)

                mean_loads.append(list(np.multiply(0.001, user_load)))
                break

            # break

        plt.xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44], rotation=45)

        datas = ('0:00', '2:00', '4:00', '6:00', '8:00', '10:00',
                 '12:00', '14:00', '16:00', '18:00',
                 '20:00', '22:00')

        mean_loads = np.array(mean_loads)
        mean_loads = np.mean(mean_loads, axis=0)

        plt.plot(mean_loads, color='r', lw=1)

        ax.set_xticklabels(datas)

        plt.xlabel('times/30min')
        plt.ylabel('loads/KW')

        plt.show()

        # break
    return

def user_detials():
    reader = Reader(READ_DATA_PATH)
    all_user_loads = reader.read_all_user_loads()
    all_user_class = reader.read_user_class(CLASS_RESULT_PATH)

    user_nums = all_user_class.keys()

    for user in user_nums:
        users = all_user_class[user]
        mean_loads = []

        f, ax = plt.subplots()
        for each_user in users:
            user_loads = all_user_loads[each_user]

            for user_load in user_loads:
                mean_loads.append(user_load)
                break

        mean_loads = np.array(mean_loads)
        mean_loads = np.mean(mean_loads, axis=0)
        max_loads = int(np.max(mean_loads))
        min_loads = int(np.min(mean_loads))
        ave_loads = int(np.mean(mean_loads))

        if user == 0:
            user_class = 'A'
            max_price = 20
            min_price = 12
            ave_price = 14

            peak_time = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                         36, 37, 38, 39, 40, 41, 42, 43]
            peace_time = [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 44, 45, 46, 47]
            valley_time = [3, 4, 5, 6, 7]

            user_1 = {
                'user_class': user_class,
                'max_price': max_price,
                'min_price': min_price,
                'ave_price': ave_price,
                'peak_time': peak_time,
                'peace_time': peace_time,
                'valley_time': valley_time,
                'max_load': max_loads,
                'min_load': min_loads,
                'ave_loads': ave_loads,
            }

            if not os.path.exists('data/user1.json'):
                with open('data/user1.json', 'a', encoding='utf-8') as f_write:
                    json.dump(user_1, f_write)

        elif user == 1:
            user_class = 'B'
            max_price = 26
            min_price = 11
            ave_price = 13.5

            peak_time = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            peace_time = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            valley_time = [0, 1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

            user_2 = {
                'user_class': user_class,
                'max_price': max_price,
                'min_price': min_price,
                'ave_price': ave_price,
                'peak_time': peak_time,
                'peace_time': peace_time,
                'valley_time': valley_time,
                'max_load': max_loads,
                'min_load': min_loads,
                'ave_loads': ave_loads,
            }

            if not os.path.exists('data/user2.json'):
                with open('data/user2.json', 'a', encoding='utf-8') as f_write:
                    json.dump(user_2, f_write)
        else:
            user_class = 'C'
            max_price = 32
            min_price = 10
            ave_price = 13

            peak_time = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            peace_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
            valley_time = [43, 44, 45, 46, 47]

            user_2 = {
                'user_class': user_class,
                'max_price': max_price,
                'min_price': min_price,
                'ave_price': ave_price,
                'peak_time': peak_time,
                'peace_time': peace_time,
                'valley_time': valley_time,
                'max_load': max_loads,
                'min_load': min_loads,
                'ave_loads': ave_loads,
            }

            if not os.path.exists('data/user3.json'):
                with open('data/user3.json', 'a', encoding='utf-8') as f_write:
                    json.dump(user_2, f_write)
    return

def main():
    plot()
    # user_detials()
    return

if __name__ == '__main__':
    main()