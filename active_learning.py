import numpy as np
import pandas as pd
from data_utils import contains_sequence, prepare_data
from model_utils import build_model, train_model

def select_samples(pool_df, early_window, n_samples, criteria_list):
    """
    按照criteria_list的順序，逐步篩選，直到補滿 n_samples
    criteria_list: list of dicts, 每個dict有
        type: "element" 或 "sequence"
        value: 元素或子序列 (int or list)
    """
    selected_idx = set()
    samples = []
    
    for criterion in criteria_list:
        if len(selected_idx) >= n_samples:
            break
        # 濾出還沒被選過的 row
        available_df = pool_df.loc[~pool_df.index.isin(selected_idx)]

        if criterion["type"] == "element":
            filt = available_df['trajectory'].apply(lambda x: criterion["value"] in x[:early_window])
        elif criterion["type"] == "sequence":
            filt = available_df['trajectory'].apply(lambda x: contains_sequence(x[:early_window], criterion["value"]))
        else:
            raise ValueError("criteria type 必須是 'element' 或 'sequence'")
        
        new_idx = available_df[filt].index.tolist()
        # 只取還沒滿的部分
        needed = n_samples - len(selected_idx)
        chosen_idx = new_idx[:needed]
        selected_idx.update(chosen_idx)
        samples.extend(chosen_idx)
    # 按原始順序回傳
    return pool_df.loc[list(selected_idx)]

def active_learning_loop(X_train, y_train, X_test, y_test, pool_df, early_window=15, n_iterations=6, n_samples=1000, criteria_list=None):
    input_dim = 200
    output_dim = 40
    input_length = early_window

    for iteration in range(n_iterations):
        selected_samples = select_samples(pool_df, early_window, n_samples, criteria_list)
        if selected_samples.empty:
            print(f"[Iter {iteration+1}] 沒有找到符合條件的樣本。")
            continue

        # pool_df remove
        pool_df = pool_df.drop(selected_samples.index).reset_index(drop=True)

        # 新增樣本
        X_new, y_new = prepare_data(selected_samples, early_window)
        X_train = np.concatenate((X_train, X_new))
        y_train = np.concatenate((y_train, y_new))

        # 建新模型並訓練
        model = build_model(input_dim, output_dim, input_length)
        train_model(model, X_train, y_train)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"[Iter {iteration+1}] Test Accuracy: {accuracy:.4f}")

    print("Active learning 完成！")
