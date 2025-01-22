import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
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

# 评估模型并保存结果
def evaluate_and_save_results(model, X_test, y_test, original_X_test, model_type):
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'{model_type} 测试准确率: {accuracy:.4f}')

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()  # 将概率转换为二分类标签
    y_true_labels = y_test

    # 计算精确率、召回率和F1分数
    precision = precision_score(y_true_labels, y_pred_labels)
    recall = recall_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels)

    print(f'{model_type} 测试精确率: {precision:.4f}')
    print(f'{model_type} 测试召回率: {recall:.4f}')
    print(f'{model_type} 测试F1分数: {f1:.4f}')

    # 保存结果到 CSV 文件
    results = pd.DataFrame({
        'Hop Latency (ms)': original_X_test[:, -1, 0],  # 原始特征数据
        'e2e Delay (ms)': original_X_test[:, -1, 1],
        'Queue Occupancy': original_X_test[:, -1, 2],
        'IAT (us)': original_X_test[:, -1, 3],
        'Hop Jitter (us)': original_X_test[:, -1, 4],
        'e2e Jitter (us)': original_X_test[:, -1, 5],
        'True Anomaly': y_true_labels,
        'Predicted Anomaly': y_pred_labels
    })
    output_file = f'{model_type.lower()}_anomaly_results.csv'
    results.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")

    # 返回评估指标
    return accuracy, precision, recall, f1

# 保存每轮的 val_accuracy 和 val_loss 到 CSV 文件
def save_training_history(history, model_type):
    """
    将每轮的 val_accuracy 和 val_loss 保存到 CSV 文件中。
    """
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['val_accuracy']) + 1),
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss']
    })
    output_file = f'{model_type.lower()}_training_history.csv'
    history_df.to_csv(output_file, index=False)
    print(f"{model_type} 训练历史已保存到 {output_file}")

# 绘制四种方法的测试准确率随 epoch 的变化折线图
def plot_val_accuracy_curves(histories, model_types):
    plt.figure(figsize=(10, 6))
    for history, model_type in zip(histories, model_types):
        plt.plot(history.history['val_accuracy'], label=f'{model_type} Val Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制四种方法的测试损失随 epoch 的变化折线图
def plot_val_loss_curves(histories, model_types):
    plt.figure(figsize=(10, 6))
    for history, model_type in zip(histories, model_types):
        plt.plot(history.history['val_loss'], label=f'{model_type} Val Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制最终四种方法的精确率、召回率、F1分数对比柱状图
def plot_metrics_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    precision = [metrics_dict[model]['precision'] for model in models]
    recall = [metrics_dict[model]['recall'] for model in models]
    f1 = [metrics_dict[model]['f1'] for model in models]

    x = np.arange(len(models))  # 模型标签位置
    width = 0.2  # 柱状图宽度

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1 Score')

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Comparison of Precision, Recall, and F1 Score for Four Models')
    plt.xticks(x, models)
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
    model_types = ['TCN', 'LSTM', 'RNN', 'Transformer']

    # 记录每种模型的评估指标
    metrics_dict = {}
    histories = []  # 用于保存每种模型的训练历史

    # 遍历所有模型类型
    for model_type in model_types:
        print(f"Training {model_type} model...")

        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])  # [timesteps, features]
        if model_type == 'LSTM':
            model = build_tcn_model(input_shape)
        elif model_type == 'Transformer':
            model = build_lstm_model(input_shape)
        elif model_type == 'TCN':
            model = build_rnn_model(input_shape)
        elif model_type == 'RNN':
            model = build_transformer_model(input_shape)

        # 训练模型
        history = train_model(model, X_train, y_anomaly_train, X_test, y_anomaly_test)
        histories.append(history)  # 保存训练历史

        # 保存每轮的 val_accuracy 和 val_loss 到 CSV 文件
        save_training_history(history, model_type)

        # 评估模型并保存结果
        accuracy, precision, recall, f1 = evaluate_and_save_results(
            model, X_test, y_anomaly_test, original_X_test, model_type
        )

        # 记录评估指标
        metrics_dict[model_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # 保存模型
        model.save(f'{model_type.lower()}_anomaly_model.h5')
        print(f"{model_type} model saved as {model_type.lower()}_anomaly_model.h5")

    # 绘制四种方法的测试准确率随 epoch 的变化折线图
    plot_val_accuracy_curves(histories, model_types)

    # 绘制四种方法的测试损失随 epoch 的变化折线图
    plot_val_loss_curves(histories, model_types)

    # 绘制最终四种方法的精确率、召回率、F1分数对比柱状图
    plot_metrics_comparison(metrics_dict)

# 运行主函数
if __name__ == "__main__":
    main()