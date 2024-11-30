from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.layers import Input, Bidirectional, Attention, Concatenate, Layer
from keras.models import Model
import keras
import tensorflow as tf


# 自定义损失函数
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


class AttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        # 在 build 方法中创建变量，确保权重仅在第一次调用时创建
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], 1),  # [features, 1]
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        """
        输入:
        - inputs: [batch_size, time_steps, features]，即 LSTM 的输出。

        输出:
        - context: [batch_size, features]，加权后的上下文向量。
        """
        # 1. 计算注意力权重的分数 score
        score = tf.keras.backend.tanh(tf.matmul(inputs, self.attention_weights))  # [batch_size, time_steps, 1]

        # 2. 移除最后一个维度，得到 [batch_size, time_steps]
        score = tf.squeeze(score, axis=-1)

        # 3. 应用 softmax 得到注意力权重 alpha
        alpha = tf.nn.softmax(score, axis=-1)  # [batch_size, time_steps]

        # 4. 扩展 alpha 的最后一个维度，与 inputs 的形状匹配
        alpha = tf.expand_dims(alpha, axis=-1)  # [batch_size, time_steps, 1]

        # 5. 计算加权后的上下文向量
        context = tf.reduce_sum(inputs * alpha, axis=1)  # [batch_size, features]

        return context


# 构建双头 LSTM + Attention 模型
def build_bidirectional_lstm_with_attention(time_step,num_types,w):
    input_layer = Input(shape=(time_step, 6))  # 输入形状：时间步长 x 特征维度
    # 双向 LSTM 层
    lstm_out_1 = Bidirectional(LSTM(units=64, return_sequences=True))(input_layer)
    lstm_out_2 = Bidirectional(LSTM(units=128, return_sequences=True))(lstm_out_1)
    # 注意力层
    attention_out = AttentionLayer()(lstm_out_2)
    # 全连接层
    dense_out = Dense(units=64, activation='relu')(attention_out)
    output_layer = Dense(units=num_types, activation='softmax')(dense_out)

    # 模型编译
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, w),
                  optimizer=Adam(learning_rate=1e-5),
                  metrics=['accuracy'])
    return model