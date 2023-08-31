import argparse
import tensorflow as tf
import numpy as np
import pandas as pd

def train_model(input_data, target_data, num_coefficients, epochs=500, batch_size=32):
    # 建立訓練資料
    X_train = []
    y_train = []

    for i in range(len(input_data) - num_coefficients):
        X_train.append(input_data[i:i + num_coefficients])
        y_train.append([target_data[i + num_coefficients]])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(num_coefficients,), activation=None)
    ])

    # 編譯模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # 獲得訓練後的係數
    trained_coefficients = model.get_weights()[0].flatten()
    return trained_coefficients

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FIR filter model')
    parser.add_argument('--coefficients', type=int, default=29, help='Number of FIR filter coefficients')
    args = parser.parse_args()

    # 從外部檔案讀取noisy_sin_wave和sin_wave數據
    noisy_sin_wave = np.genfromtxt("raw_Data.csv", delimiter=',')
    sin_wave = np.genfromtxt("ifft_Data.csv", delimiter=',')

    # 調用函式進行建立和訓練模型
    trained_coefficients = train_model(noisy_sin_wave, sin_wave, args.coefficients)

    print("訓練後的係數：", trained_coefficients)

    total_sum = sum(trained_coefficients)
    print("sum = ", total_sum)
    trained_coefficients_normalized = trained_coefficients / np.sum(trained_coefficients)
    total_sum_normalized = sum(trained_coefficients_normalized)
    print("sum = ", total_sum_normalized) # 可能丟到MCU上面跑的話需要正規化成1才能保證不會offset

    data = {'Coefficient': trained_coefficients_normalized}
    df = pd.DataFrame(data)
    df.to_csv('FIR_Coe.csv', index=False)
