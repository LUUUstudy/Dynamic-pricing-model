import re
import tensorflow as tf
import preprocess_data
import model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

READ_DATA_PATH = '../data/all_user_loads.json'
CLASS_RESULT_PATH = '../data/all_user_class.csv'

def predict():
    user_class = 2  # 待预训练用户类型
    time_step = 20  # 负荷预测步长
    input_size = 1
    output_size = 1
    rnn_unit = 64
    lr = 0.0001
    batch_size = 1

    y_predict = []
    y_ture = []
    mape = []

    if user_class == 0:
        data_path = 'data(第一类)'
        save_dir = 'model(第一类)'
    elif user_class == 1:
        data_path = 'data(第二类)'
        save_dir = 'model(第二类)'
    else:
        data_path = 'data(第三类)'
        save_dir = 'model(第三类)'

    if not os.path.exists(os.path.join(data_path, 'train_x.npy')):
        reader = preprocess_data.Reader(READ_DATA_PATH)
        all_user_loads = reader.read_all_user_loads()
        all_user_class = reader.read_user_class(CLASS_RESULT_PATH)
        train_x,train_y,test_x,test_y,all_data,mean_load,std_load = reader.create_train_data(all_user_loads, all_user_class, user_class)
        reader.write_data(train_x,train_y,test_x,test_y,data_path)
    else:
        reader = preprocess_data.Reader(READ_DATA_PATH)
        all_user_loads = reader.read_all_user_loads()
        all_user_class = reader.read_user_class(CLASS_RESULT_PATH)
        train_x, train_y, test_x, test_y, all_data ,mean_load,std_load= reader.create_train_data(all_user_loads, all_user_class, user_class)
        train_x = np.load(os.path.join(data_path, 'train_x.npy'))
        train_y = np.load(os.path.join(data_path, 'train_y.npy'))
        test_x = np.load(os.path.join(data_path, 'test_x.npy'))
        test_y = np.load(os.path.join(data_path, 'test_y.npy'))

    test_x = test_x[:144]
    test_y = test_y[:144]

    Model = model.Model(time_step=time_step, input_size=input_size, output_size=output_size, rnn_unit=rnn_unit, lr=lr)

    pred, _ = Model.network(batch_size=batch_size, layer_num=1)

    save = tf.train.Saver(tf.global_variables())
    save_graph = os.path.join(save_dir, 'best_validation.meta')
    save_model = os.path.join(save_dir, 'best_validation')

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_graph)
        new_saver.restore(sess, save_model)
        start = 0
        end = start + batch_size

        while end < len(test_x):
            pred_ = sess.run([pred], feed_dict={Model.X:test_x[start:end]})
            y_predict.append(pred_[0][0][0] * std_load + mean_load)
            y_ture.append(test_y[start][0] * std_load + mean_load)

            mape.append(abs(((test_y[start][0]*std_load+mean_load)-(pred_[0][0][0]*std_load+mean_load)) / (test_y[start][0]*std_load+mean_load)))

            start = start + batch_size
            end = start + 1

        figure_name = '第' + str(user_class+1) + '类用户负荷预测曲线'
        plt.figure()
        plt.plot(list(range(len(y_predict))), y_predict, color='b', lw=1, label='Predict loads')
        plt.plot(list(range(len(y_ture))), y_ture, color='r', lw=1, label='True loads')
        plt.title(figure_name)
        plt.xlabel('时间 / (30min)')
        plt.ylabel('负荷 (W)')
        plt.legend()
        plt.show()

    mape = np.array(mape)
    mape_value = (np.sum(mape) / len(mape))
    print("预测评估指标MAPE:%s"%str(mape_value))

    return

def test(user_class, test_x, test_y, read_data_path, class_result_path, model_path):
    user_class = user_class  # 待预训练用户类型
    time_step = 20  # 负荷预测步长
    input_size = 1
    output_size = 1
    rnn_unit = 64
    lr = 0.0001
    batch_size = 1

    y_predict = []
    y_ture = []
    mape = []

    save_dir = model_path

    reader = preprocess_data.Reader(read_data_path)
    all_user_loads = reader.read_all_user_loads()
    all_user_class = reader.read_user_class(class_result_path)
    _,_,_,_, all_data ,mean_load,std_load= reader.create_train_data(all_user_loads, all_user_class, user_class)
    test_x = test_x
    test_y = test_y

    test_x = (test_x - mean_load) / std_load
    test_y = (test_y - mean_load) / std_load

    Model = model.Model(time_step=time_step, input_size=input_size, output_size=output_size, rnn_unit=rnn_unit, lr=lr)

    pred, _ = Model.network(batch_size=batch_size, layer_num=1)

    save = tf.train.Saver(tf.global_variables())
    save_graph = os.path.join(save_dir, 'best_validation.meta')
    save_model = os.path.join(save_dir, 'best_validation')

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_graph)
        new_saver.restore(sess, save_model)
        start = 0
        end = start + batch_size

        while end < (len(test_x)+1):
            pred_ = sess.run([pred], feed_dict={Model.X: test_x[start:end]})
            y_predict.append(pred_[0][0][0] * std_load + mean_load)
            y_ture.append(test_y[start][0] * std_load + mean_load)

            mape.append(abs(((test_y[start][0] * std_load + mean_load) - (pred_[0][0][0] * std_load + mean_load)) / (
                        test_y[start][0] * std_load + mean_load)))

            start = start + batch_size
            end = start + 1

        figure_name = '第' + str(user_class + 1) + '类用户负荷预测曲线'
        plt.figure()
        plt.plot(list(range(len(y_predict))), y_predict, color='b', lw=1, label='Predict loads')
        plt.plot(list(range(len(y_ture))), y_ture, color='r', lw=1, label='True loads')
        plt.title(figure_name)
        plt.xlabel('时间 / (30min)')
        plt.ylabel('负荷 (W)')
        plt.legend()
        plt.show()

    mape = np.array(mape)
    mape_value = (np.sum(mape) / len(mape))
    print("预测评估指标MAPE:%s" % str(mape_value))

    return

def main():
    predict()
    return

if __name__ == '__main__':
    main()