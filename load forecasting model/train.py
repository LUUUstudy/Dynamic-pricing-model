import re
import tensorflow as tf
import preprocess_data
import model
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #使用cpu
READ_DATA_PATH = '../data/all_user_loads.json'
CLASS_RESULT_PATH = '../data/all_user_class.csv'

def train():
    user_class = 2  #待预训练用户类型
    time_step = 20  #负荷预测步长
    input_size = 1
    output_size = 1
    rnn_unit = 64
    lr = 0.0001
    batch_size = 16

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
        train_x,train_y,test_x,test_y,all_data,_,_ = reader.create_train_data(all_user_loads, all_user_class, user_class)
        mean_load,std_load = reader.com_mean_std(all_data)
        reader.write_data(train_x,train_y,test_x,test_y,data_path)
    else:
        train_x = np.load(os.path.join(data_path, 'train_x.npy'))
        train_y = np.load(os.path.join(data_path, 'train_y.npy'))
        test_x = np.load(os.path.join(data_path, 'test_x.npy'))
        test_y = np.load(os.path.join(data_path, 'test_y.npy'))

    Model = model.Model(time_step=time_step, input_size=input_size, output_size=output_size, rnn_unit=rnn_unit, lr=lr)

    pred,_ = Model.network(batch_size=batch_size, layer_num=1)

    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Model.Y, [-1])))
    train_op = tf.train.AdamOptimizer(Model.lr).minimize(loss)

    saver = tf.train.Saver()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start = 0
        end = start + batch_size
        best_loss = np.inf

        for step in range(50000):
            if end < len(train_x):
                _,loss_ = sess.run([train_op,loss], feed_dict={Model.X:train_x[start:end], Model.Y:train_y[start:end]})
                start = start + batch_size
                end = end + batch_size
            else:
                start = 0
                end = start + batch_size

            if step%500 == 0:
                start_val = 0
                end_val = start_val + batch_size
                total_loss = 0
                num = 0

                while (end_val < len(test_x)):
                    loss_temp = sess.run([loss], feed_dict={Model.X:test_x[start_val:end_val], Model.Y:test_y[start_val:end_val]})
                    loss_temp = tf.squeeze(loss_temp)
                    total_loss = total_loss + sess.run(loss_temp)

                    start_val = start_val + batch_size
                    end_val = end_val + batch_size
                    num = num + 1

                loss_val = total_loss / num
                print('setp: {}      train loss: {}      validation loss: {}'.format(step, loss_, loss_val))
                if loss_val < best_loss:
                    print('best_loss: {} -> {}, save model.'.format(best_loss, loss_val))
                    best_loss = loss_val
                    saver.save(sess=sess, save_path=save_path)

            step = step + 1

    return

def main():
    train()
    return

if __name__ == '__main__':
    main()