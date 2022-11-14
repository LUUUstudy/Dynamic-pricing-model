import re
import json
import tqdm
import numpy as np
import sys
from sklearn.cluster import KMeans
import pickle
import tensorflow as tf
import time
import os
sys.path.append('负荷预测模型/')
sys.path.append('动态调价模型/')
from 负荷预测模型 import predict_load
from 动态调价模型 import predict_price

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

    def read_test_users(self, all_user_loads, all_user_class):  #获取每一类中的一个用户三天的用电负荷数据
        user_nums = all_user_class.keys()
        user_day_loads = {}

        for user_class in user_nums:
            users = all_user_class[user_class]
            day_loads = []

            for k,user in enumerate(users):

                if k == 0:
                    day_loads = []
                    user_loads = all_user_loads[user]

                    for i,load in enumerate(user_loads):
                        if i == 3:
                            break
                        day_loads.append(load)

                    break

            if user_class not in user_day_loads.keys():
                user_day_loads[user_class] = day_loads

        return user_day_loads

class Features():
    def __init__(self):
        return

    def create_features(self, user_day_loads):
        user_nums = user_day_loads.keys()
        all_user_features = []

        for user_num in user_nums:
            user_loads = user_day_loads[user_num]
            user_features = []

            for user_load in user_loads:
                ave_load = self.get_ave_load(user_load)  #获取用户日平均负荷
                max_load = self.get_max_load(user_load)  #获取用户最大负荷
                load_rate = int((ave_load / max_load) * 100)  #获取用户日负荷率
                sum_load = self.get_sum_loads(user_load)  #获取用户日用电负荷
                peace_load = self.get_peace_loads(user_load)  #获取用户平时段用电负荷
                peace_rate = int((peace_load / sum_load) * 100)  #获取用户平时段用电百分比

                user_features = [ave_load, load_rate, peace_rate]
                break
            all_user_features.append(user_features)
        all_user_features = np.array(all_user_features)

        return all_user_features

    def get_max_load(self, day_loads):
        max_load = int(np.max(np.array(day_loads)))
        return max_load

    def get_ave_load(self, day_loads):
        ave_load = int(np.mean(np.array(day_loads)))
        return ave_load

    def get_sum_loads(self, day_loads):
        sum_loads = np.sum(np.array(day_loads))
        return sum_loads

    def get_peace_loads(self, day_loads):
        peace_loads = []
        for i,load in enumerate(day_loads):
            if i<20 or i>40:
                peace_loads.append(load)
        peace_load = np.sum(np.array(peace_loads))
        return peace_load

class Classifier():
    def __init__(self):
        self.cluster_obj = None
        return

    def store(self):
        with open('classification_model/user_classification_model.pickle', 'rb') as f_read:
            self.cluster_obj = pickle.load(f_read)
        f_read.close()
        return

    def predict(self, user_features):
        y_pred = self.cluster_obj.predict(user_features)
        return y_pred

def main():
    print("正在准备测试用户数据！")
    reader = Reader(read_data_path=READ_DATA_PATH)
    all_user_loads = reader.read_all_user_loads()
    all_user_class = reader.read_user_class(read_data_path=CLASS_RESULT_PATH)
    user_day_loads = reader.read_test_users(all_user_loads=all_user_loads, all_user_class=all_user_class)
    print("测试用户数据准备完毕！\n")
    time.sleep(3)

    print("********** 测试用户分类！ *************")
    print("正在构造测试用户特征！")
    features = Features()
    all_user_features = features.create_features(user_day_loads=user_day_loads)
    print("测试用户特征构造完毕！")

    print("正在使用已有用户聚类模型对测试用户进行分类")
    classifier = Classifier()
    classifier.store()
    y_pred = classifier.predict(all_user_features)
    print("测试用户用特征分类完毕！\n")
    print("=====================")

    for i,y in enumerate(y_pred):
        print("用户%s属于第%s类用户！"%(str(i), str(y+1)))
    print("=====================\n")

    time.sleep(6)

    print("\n********** 处理第一类用户，负荷预测及动态定价！*************\n")
    user_loads = user_day_loads[0]

    print("--------- 负荷预测模型测试！-----------")
    all_day_loads = sum(user_loads, [])  #[3,48]->[1,144]

    all_day_loads = np.array(all_day_loads)
    all_day_loads = all_day_loads[:, np.newaxis]

    test_x = []
    test_y = []
    time_step = 20
    for i in range(len(all_day_loads) - time_step):
        x = all_day_loads[i: i + time_step]
        y = all_day_loads[i + time_step]
        test_x.append(x)
        test_y.append(y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    read_data_path = 'data/all_user_loads.json'
    class_result_path = 'data/all_user_class.csv'
    model_path = '负荷预测模型/model(第一类)'
    predict_load.test(user_class=0, test_x=test_x, test_y=test_y, read_data_path=read_data_path, class_result_path=class_result_path, model_path=model_path)
    print("--------- 负荷模型测试完毕！-----------")
    tf.reset_default_graph()

    print("\n--------- 动态调价模型测试！-----------")
    all_data_path = '动态调价模型/data(类型一)'
    user_day_load = [user_loads[2]]  #获取用户一天的用电负荷数据
    model_path = '动态调价模型/rlmodel(类型一)'
    user_details_path = 'data/user1.json'
    predict_price.test(data_path=all_data_path, test_data=user_day_load, model_path=model_path, user_details_path=user_details_path)
    print("--------- 动态调价模型测试完毕！-----------")
    tf.reset_default_graph()

    print("\n********** 处理第二类用户，负荷预测及动态定价！*************\n")
    user_loads = user_day_loads[1]

    print("--------- 负荷预测模型测试！-----------")
    all_day_loads = sum(user_loads, [])  # [3,48]->[1,144]

    all_day_loads = np.array(all_day_loads)
    all_day_loads = all_day_loads[:, np.newaxis]

    test_x = []
    test_y = []
    time_step = 20
    for i in range(len(all_day_loads) - time_step):
        x = all_day_loads[i: i + time_step]
        y = all_day_loads[i + time_step]
        test_x.append(x)
        test_y.append(y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    read_data_path = 'data/all_user_loads.json'
    class_result_path = 'data/all_user_class.csv'
    model_path = '负荷预测模型/model(第二类)'
    predict_load.test(user_class=1, test_x=test_x, test_y=test_y, read_data_path=read_data_path,
                                  class_result_path=class_result_path, model_path=model_path)
    print("--------- 负荷模型测试完毕！-----------")
    tf.reset_default_graph()

    print("\n--------- 动态调价模型测试！-----------")
    all_data_path = '动态调价模型/data(类型二)'
    user_day_load = [user_loads[1]]  # 获取用户一天的用电负荷数据
    model_path = '动态调价模型/rlmodel(类型二)'
    user_details_path = 'data/user2.json'
    predict_price.test(data_path=all_data_path, test_data=user_day_load, model_path=model_path,
                       user_details_path=user_details_path)
    print("--------- 动态调价模型测试完毕！-----------")
    tf.reset_default_graph()

    print("\n********** 处理第三类用户，负荷预测及动态定价！*************\n")
    user_loads = user_day_loads[2]

    print("--------- 负荷预测模型测试！-----------")
    all_day_loads = sum(user_loads, [])  # [3,48]->[1,144]

    all_day_loads = np.array(all_day_loads)
    all_day_loads = all_day_loads[:, np.newaxis]

    test_x = []
    test_y = []
    time_step = 20
    for i in range(len(all_day_loads) - time_step):
        x = all_day_loads[i: i + time_step]
        y = all_day_loads[i + time_step]
        test_x.append(x)
        test_y.append(y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    read_data_path = 'data/all_user_loads.json'
    class_result_path = 'data/all_user_class.csv'
    model_path = '负荷预测模型/model(第三类)'
    predict_load.test(user_class=2, test_x=test_x, test_y=test_y, read_data_path=read_data_path,
                                  class_result_path=class_result_path, model_path=model_path)
    print("--------- 负荷模型测试完毕！-----------")
    tf.reset_default_graph()

    print("\n--------- 动态调价模型测试！-----------")
    all_data_path = '动态调价模型/data(类型三)'
    user_day_load = [user_loads[0]]  # 获取用户一天的用电负荷数据
    model_path = '动态调价模型/rlmodel(类型三)'
    user_details_path = 'data/user3.json'
    predict_price.test(data_path=all_data_path, test_data=user_day_load, model_path=model_path,
                       user_details_path=user_details_path)
    print("--------- 动态调价模型测试完毕！-----------")
    tf.reset_default_graph()

    return

if __name__ == '__main__':
    main()