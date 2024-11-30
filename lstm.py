import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam

def custom_loss(y_true, y_pred, w):
    num_classes = len(w)
    loss = 0.0
    for i in range(num_classes):
        mask = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), i), tf.float32)
        num_samples = tf.reduce_sum(mask) + 1e-6
        pi = tf.reduce_sum(y_pred[:, i] * mask) / num_samples
        pi = tf.clip_by_value(pi, 1e-6, 1.0)  # 限制 pi 的范围
        loss += w[i] * (1 - pi) ** 3 * tf.math.log(pi)
    return -loss

def lstm_model(time_step,num_types,w):
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(time_step ,6),return_sequences=True))
    model.add(LSTM(units=64, return_sequences=False))
    #开始分类
    #model.add(TimeDistributed(Dense(units=32, activation='relu')))
    #model.add(TimeDistributed(Dense(units=16, activation='softmax')))

    #model.add((Dense(units=32, activation='relu')))
    model.add((Dense(units=num_types, activation='softmax')))
    #不是独热用sparse_categorical_crossentropy
    model.compile(loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, w),
                  optimizer=Adam(learning_rate=1e-4),metrics=['accuracy'])
    return model