import re
import numpy as np
from sklearn.cluster import KMeans
import os
import json
from sklearn.metrics import silhouette_score
import tqdm
import pickle

READ_DATA_PATH = 'data/all_user_loads.json'
WRITE_DATA_PATH = 'data'
CLASS_RESULT_PATH = 'data/all_user_class.csv'

Cluster_num = 3  #聚类中心的个数

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

    def read_user_features(self, read_data_path):
        read_feature_path = read_data_path + '/' + 'all_user_features.npy'
        read_user_path = read_data_path + '/' + 'all_user_nums.npy'
        all_user_features = np.load(read_feature_path)
        all_user_nums = np.load(read_user_path)
        return all_user_features, all_user_nums

class Day_Features():
    def __init__(self, all_user_loads):
        self.all_user_loads = all_user_loads
        return

    def create_user_features(self):
        all_user_nums = list(self.all_user_loads.keys())
        all_user_features = []
        all_users = []

        for user in tqdm.tqdm(all_user_nums):
            user_loads = self.all_user_loads[user]

            user_ave_loads = self.get_day_max_min_ave_load(user_loads, flag='ave')  #获取用户的日平均负荷

            user_max_loads = self.get_day_max_min_ave_load(user_loads, flag='max')  #获取用户的日最大负荷

            user_min_loads = self.get_day_max_min_ave_load(user_loads, flag='min')  #获取用户的日最小负荷

            if user_max_loads == 0:
                continue

            user_load_rate = int((user_ave_loads / user_max_loads) * 100)  #获取用户的日负荷率

            user_day_loads = self.get_day_loads(user_loads)  #获取用户的日用电量

            user_day_dif = self.get_day_max_min_dif(user_loads)  #获取用户的日峰谷差

            user_dif_rate = int((user_day_dif / user_max_loads) * 100)  #获取用户的日峰谷差率

            user_valley_loads = self.get_valley_loads(user_loads)  #获取用户平时段用电量

            user_valley_rate = int((user_valley_loads / user_day_loads) * 100)  #获取用户平时段用电百分比

            user_fearutes = [user_ave_loads, user_load_rate, user_valley_rate]
            all_user_features.append(user_fearutes)
            all_users.append(user)

        all_user_features = np.array(all_user_features)
        all_users = np.array(all_users)

        user_nums = len(all_user_features)
        print("共读取%s个用户特征！"%str(user_nums))

        return all_user_features, all_users

    def write_user_features(self, all_user_features, all_user_nums, write_data_path):
        write_features_path = write_data_path + '/' + 'all_user_features.npy'
        write_users_path = write_data_path + '/' + 'all_user_nums.npy'
        np.save(write_features_path, all_user_features)
        np.save(write_users_path, all_user_nums)
        return

    #获取用户日最大、最小、平均负荷
    def get_day_max_min_ave_load(self, user_loads, flag=None):  #user_num: 用户id，返回用户平均一天的最大、最小、平均负荷，[1]
        day_value_loads = []
        for day_loads in user_loads:
            if flag == 'max':
                day_value_loads.append(np.max(np.array(day_loads)))
            elif flag == 'ave':
                day_value_loads.append(np.mean(np.array(day_loads)))
            elif flag == 'min':
                day_value_loads.append(np.min(np.array(day_loads)))

            break

        loads = int(np.mean(np.array(day_value_loads)))
        return loads

    #获取用户负荷日峰谷差
    def get_day_max_min_dif(self, user_loads):  #user_num: 用户id，返回用户平均每天的日峰谷差，[1]
        all_day_diff = []
        for day_losds in user_loads:
            all_day_diff.append(np.max(np.array(day_losds)) - np.min(np.array(day_losds)))

            break

        loads = int(np.mean(np.array(all_day_diff)))
        return loads

    #构造用户日用电量
    def get_day_loads(self, user_loads):
        all_day_loads = []
        for day_loads in user_loads:
            all_day_loads.append(np.sum(np.array(day_loads)))
            break

        loads = int(np.mean(np.array(all_day_loads)))
        return loads

    #构造用户平时段用电量
    def get_valley_loads(self, user_loads):
        each_day_loads = []
        all_valley_loads = []

        for day_loads in user_loads:
            for i,loads in enumerate(day_loads):
                if i<20 or i>40:
                    each_day_loads.append(loads)
            all_valley_loads.append(np.sum(np.array(each_day_loads)))
            break

        peace_loads = np.mean(np.array(all_valley_loads))

        return peace_loads

class Month_Features():
    def __init__(self, all_user_loads):
        self.all_user_loads = all_user_loads
        return



class Classifier():
    def __init__(self, cluster_num=Cluster_num):
        self.num = cluster_num
        return

    def cluster(self, input):
        self.cluster_obj = KMeans(n_clusters=self.num, random_state=1).fit(input)
        y_pred = self.cluster_obj.labels_
        print("聚类已完成！")
        return y_pred

    def classify(self, input):
        y_pre = self.cluster_obj.predict(input)
        print(y_pre)
        return y_pre

    def devel(self, input, y):
        score = silhouette_score(input, y)
        print("轮廓系数为：%s"%str(score))
        return

    def save(self):
        with open('classification_model/user_classification_model.pickle', 'wb') as f_write:
            pickle.dump(self.cluster_obj, f_write)
        f_write.close()
        return

    def restore(self):
        with open('classification_model/user_classification_model.pickle', 'rb') as f_read:
            self.cluster_obj = pickle.load(f_read)
        f_read.close()
        return

def main():
    reader = Reader(read_data_path=READ_DATA_PATH)

    if not os.path.exists(WRITE_DATA_PATH + '/' + 'all_user_features.npy'):
        all_user_loads = reader.read_all_user_loads()
        feature = Day_Features(all_user_loads)
        print("正在构造用户特征！")
        all_user_features, all_user_nums = feature.create_user_features()
        print("用户特征构造完成")
        feature.write_user_features(all_user_features, all_user_nums, WRITE_DATA_PATH)
    else:
        all_user_features, all_user_nums = reader.read_user_features(WRITE_DATA_PATH)


    # 对用户进行聚类
    classifier = Classifier(cluster_num=Cluster_num)
    y_pred = classifier.cluster(input=all_user_features)
    y_pred = np.array(y_pred)
    user_class = {}
    for y in y_pred:
        if y not in user_class.keys():
            user_class[y] = 1
        else:
            user_class[y] = user_class[y] + 1
    print("每一类中的样本数量：")
    print(user_class)
    score = classifier.devel(input=all_user_features, y=y_pred)

    classifier.save()
    print("聚类模型已保存！")

    class_result_path = CLASS_RESULT_PATH

    if len(all_user_nums) != len(y_pred):
        print("error!")

    if not os.path.exists(class_result_path):
        with open(class_result_path, 'a', encoding='utf-8') as f_write:
            for user,y in zip(all_user_nums, y_pred):
                f_write.write(str(user))
                f_write.write('\t')
                f_write.write(str(y))
                f_write.write('\n')
        f_write.close()

    return

if __name__ == '__main__':
    main()