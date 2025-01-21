import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, SimpleRNN, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tcn import TCN
import matplotlib.pyplot as plt

# 设置 matplotlib 的默认字体为支持英文的字体
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用 Arial 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据预处理函数
def preprocess_data(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 保存原始特征数据
    original_features = df[
        ["Hop Latency (ms)", "e2e Delay (ms)", "Queue Occupancy", "IAT (us)", "Hop Jitter (us)", "e2e Jitter (us)"]]

    # 归一化特征
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(original_features)

    # 提取标签
    anomaly_labels = df["Is Anomaly"].values

    # 将数据转换为时序序列
    sequence_length = 50  # 使用前50个时间步预测当前状态
    X, y_anomaly, original_X = [], [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])
        y_anomaly.append(anomaly_labels[i])
        original_X.append(original_features.iloc[i - sequence_length:i].values)  # 保存原始特征数据

    X = np.array(X)
    y_anomaly = np.array(y_anomaly)
    original_X = np.array(original_X)  # 转换为NumPy数组

    # 划分训练集和测试集
    X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test = train_test_split(
        X, y_anomaly, original_X, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test

# 构建 LSTM 模型
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=False)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    outputs = Dense(1, activation='sigmoid')(lstm_out)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 构建 RNN 模型
def build_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    rnn_out = SimpleRNN(64, return_sequences=False)(inputs)
    rnn_out = Dropout(0.2)(rnn_out)
    outputs = Dense(1, activation='sigmoid')(rnn_out)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 构建 TCN 模型
def build_tcn_model(input_shape):
    inputs = Input(shape=input_shape)
    tcn_out = TCN(nb_filters=64, kernel_size=2, dilations=[1, 2, 4], return_sequences=False)(inputs)
    tcn_out = Dropout(0.2)(tcn_out)
    outputs = Dense(1, activation='sigmoid')(tcn_out)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 构建 Transformer 模型
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(x, x)
    x = Dropout(0.2)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)  # 残差连接
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练模型
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# 绘制所有模型的训练结果图表
def plot_all_training_results(histories):
    plt.figure(figsize=(14, 10))

    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    for model_type, history in histories.items():
        plt.plot(history.history['loss'], label=f'{model_type} Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for model_type, history in histories.items():
        plt.plot(history.history['val_loss'], label=f'{model_type} Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 2, 3)
    for model_type, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{model_type} Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    for model_type, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=f'{model_type} Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 数据文件路径
    filepath = os.path.join(os.pardir, "int-dataset", "int_data_bmv2_all.csv")

    # 数据预处理
    X_train, X_test, y_anomaly_train, y_anomaly_test, original_X_train, original_X_test = preprocess_data(filepath)

    # 定义模型类型列表
    model_types = ['LSTM', 'RNN', 'TCN', 'Transformer']

    # 记录每种模型的训练历史
    histories = {}

    # 遍历所有模型类型
    for model_type in model_types:
        print(f"Training {model_type} model...")

        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])  # [timesteps, features]
        if model_type == 'LSTM':
            model = build_lstm_model(input_shape)
        elif model_type == 'RNN':
            model = build_rnn_model(input_shape)
        elif model_type == 'TCN':
            model = build_tcn_model(input_shape)
        elif model_type == 'Transformer':
            model = build_transformer_model(input_shape)

        # 训练模型
        history = train_model(model, X_train, y_anomaly_train, X_test, y_anomaly_test)

        # 记录训练历史
        histories[model_type] = history

        # 保存模型
        model.save(f'{model_type.lower()}_anomaly_model.h5')
        print(f"{model_type} model saved as {model_type.lower()}_anomaly_model.h5")

    # 绘制所有模型的训练结果图表
    plot_all_training_results(histories)

# 运行主函数
if __name__ == "__main__":
    main()