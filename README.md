# Fall Detection For LSTM  
## 环境配置  
- TensorFlow = 1.12
- python3
- 如果import无法识别则将keras.改为完整的tensorflow.keras.
## 数据集
  采用MobileFall的数据集合进行网络的训练和测试，考虑到算力因素仅识别是否跌倒，准确率为93.28%。
  数据集采样频率为200Hz，若实际传感器频率低于200Hz，建议将数据集截取降为50HZ再进行训练。
  ###采样与数据标签
  采用滑动时间窗采样，由于本数据集跌倒与非跌倒的比例差异很大，采用对四种跌倒加入高斯噪声的方法均匀的进行重采样。
  将跌倒超过一定比例的时间窗定义为跌倒，其余为非跌倒。
  ###滤波
  采用一阶Butterworth低通滤波进行去噪，截止频率为60
## 模型
本项目有两个模型文件，type `lstm.py`为两层的传统lstm模型；type`build_bidirectional_lstm_with_attention.py`为加入了attention机制的双头lstm模型。
后者准确率更高但是所需处理时间较长。
## 模型训练
type`fall-detection.py`为训练模型文件。
### 参数介绍
- 'file_label'：模型选用哪几个标签的数据进行训练
- 'num_label'：模型标签个数（数据集中一共有16类数据）
- `base_dir`：数据地址
- `batch_size `：数据管道一批输入模型多少条数据
- 'collect_samples_and_labels()':p=重叠时间窗/时间窗
### 损失函数
两个模型均采用自定义的带权交叉熵函数，权值由标签比例得到，具体计算公式见type`collect_samples_and_label()`函数中w的计算
