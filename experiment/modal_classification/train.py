import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, SimpleRNN, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from tcn import TCN

# 设置 matplotlib 的默认字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据预处理函数
def preprocess_data(data):
    """
    数据预处理：将 Modal Type 转换为数值标签，标准化特征数据，并转换为模型所需的格式。
    """
    # 将 Modal Type 转换为数值标签
    label_encoder = LabelEncoder()
    data['Modal Type'] = label_encoder.fit_transform(data['Modal Type'])

    # 选择特征和目标变量
    features = data[['Hop Latency (ms)', 'e2e Delay (ms)', 'Queue Occupancy']]
    target = data['Modal Type']

    # 标准化特征数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 将目标变量转换为 one-hot 编码
    target_onehot = to_categorical(target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_onehot, test_size=0.2, random_state=42)

    # 将数据转换为 3D 格式 [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, label_encoder, scaler, features, target

# 2. 构建模型函数
def build_model(model_type, input_shape, num_classes):
    """
    根据模型类型构建模型。
    """
    if model_type == 'Transformer':
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_type == 'TCN':
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SimpleRNN(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_type == 'LSTM':
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(TCN(nb_filters=64, kernel_size=2, dilations=[1, 2, 4], return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_type == 'RNN':
        inputs = Input(shape=input_shape)
        x = inputs
        x = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(x, x)
        x = Dropout(0.2)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)  # 残差连接
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
    else:
        raise ValueError("未知的模型类型！")

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 3. 训练模型函数
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """
    训练模型，并返回训练历史记录。
    """
    # 使用 EarlyStopping 防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# 4. 保存每轮的 val_accuracy 和 val_loss 到 CSV 文件
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

# 5. 评估模型并保存结果函数
def evaluate_and_save_results(model, X_test, y_test, label_encoder, features_test, target_test, model_type):
    """
    评估模型，并计算准确率、精确率、召回率和F1分数，同时将结果保存到 CSV 文件中。
    """
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'{model_type} 测试准确率: {accuracy:.4f}')

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1).astype(int))  # 将预测结果转换为类别标签
    y_true_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1).astype(int))  # 将真实标签转换为类别标签

    # 计算精确率、召回率和F1分数
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')  # 加权平均
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')        # 加权平均
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')                # 加权平均

    print(f'{model_type} 测试精确率: {precision:.4f}')
    print(f'{model_type} 测试召回率: {recall:.4f}')
    print(f'{model_type} 测试F1分数: {f1:.4f}')

    # 保存结果到 CSV 文件
    features_test = features_test.to_numpy()
    results = pd.DataFrame({
        'Hop Latency (ms)': features_test[:, 0],  # 原始特征数据
        'e2e Delay (ms)': features_test[:, 1],
        'Queue Occupancy': features_test[:, 2],
        'True Modal Type': y_true_labels,
        'Predicted Modal Type': y_pred_labels
    })
    output_file = f'{model_type.lower()}_classification_results.csv'
    results.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")

    # 返回评估指标
    return accuracy, precision, recall, f1

# 6. 绘制四种方法的测试准确率随 epoch 的变化折线图
def plot_val_accuracy_curves(histories, model_types):
    plt.figure(figsize=(10, 6))
    for history, model_type in zip(histories, model_types):
        plt.plot(history.history['val_accuracy'], label=f'{model_type} Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# 7. 绘制四种方法的测试损失随 epoch 的变化折线图
def plot_val_loss_curves(histories, model_types):
    plt.figure(figsize=(10, 6))
    for history, model_type in zip(histories, model_types):
        plt.plot(history.history['val_loss'], label=f'{model_type} Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 8. 绘制最终四种方法的精确率、召回率、F1分数对比柱状图
def plot_metrics_comparison(metrics_dict):
    """
    绘制四种模型的精确率、召回率和F1分数对比图。
    """
    models = list(metrics_dict.keys())
    precision = [metrics_dict[model]['precision'] for model in models]
    recall = [metrics_dict[model]['recall'] for model in models]
    f1 = [metrics_dict[model]['f1'] for model in models]

    x = np.arange(len(models))  # 模型标签位置
    width = 0.2  # 柱状图宽度

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='精确率')
    plt.bar(x, recall, width, label='召回率')
    plt.bar(x + width, f1, width, label='F1分数')

    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.title('四种模型的精确率、召回率和F1分数对比')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 9. 主函数
def main():
    # 读取数据
    filename = os.path.join(os.pardir, "int-dataset", "int_data_bmv2_normal.csv")
    data = pd.read_csv(filename)

    # 数据预处理
    X_train, X_test, y_train, y_test, label_encoder, scaler, features, target = preprocess_data(data)

    # 获取测试集的原始特征数据
    _, X_test_original, _, y_test_original = train_test_split(features, target, test_size=0.2, random_state=42)

    # 定义模型类型列表
    model_types = ['Transformer', 'LSTM', 'RNN', 'TCN']

    # 记录每种模型的评估指标
    metrics_dict = {}
    histories = []  # 用于保存每种模型的训练历史

    # 遍历所有模型类型
    for model_type in model_types:
        print(f"正在训练 {model_type} 模型...")

        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])  # [timesteps, features]
        num_classes = y_train.shape[1]
        model = build_model(model_type, input_shape, num_classes)

        # 训练模型
        history = train_model(model, X_train, y_train, X_test, y_test)
        histories.append(history)  # 保存训练历史

        # 保存每轮的 val_accuracy 和 val_loss 到 CSV 文件
        save_training_history(history, model_type)

        # 评估模型并保存结果
        accuracy, precision, recall, f1 = evaluate_and_save_results(
            model, X_test, y_test, label_encoder, X_test_original, y_test_original, model_type
        )

        # 记录评估指标
        metrics_dict[model_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # 保存模型
        model.save(f'{model_type.lower()}_model.h5')
        print(f"{model_type} 模型已保存为 {model_type.lower()}_model.h5")

    # 绘制四种方法的测试准确率随 epoch 的变化折线图
    plot_val_accuracy_curves(histories, model_types)

    # 绘制四种方法的测试损失随 epoch 的变化折线图
    plot_val_loss_curves(histories, model_types)

    # 绘制最终四种方法的精确率、召回率、F1分数对比柱状图
    plot_metrics_comparison(metrics_dict)

# 运行主函数
if __name__ == "__main__":
    main()