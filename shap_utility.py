import pandas as pd
import shap
import numpy as np

##########################
# 1. SHAP 計算
##########################

def compute_shap_values(model, X, nsamples=100):
    """
    使用 KernelExplainer 計算 SHAP 值。
    model: Keras/Tensorflow/Predict函數
    X: numpy array, 輸入特徵
    nsamples: 用多少樣本算SHAP
    回傳shap_values (list)
    """
    data_for_shap = X[:nsamples]
    explainer = shap.KernelExplainer(model.predict, data_for_shap)
    shap_values = explainer.shap_values(data_for_shap)
    return shap_values, data_for_shap

##########################
# 2. 合併 SHAP 和原始資料
##########################

def combine_features_shap(X, shap_values, prefix='F'):
    """
    將 X 和 shap_values 組合成同一個 DataFrame
    X: numpy array (n_samples, n_features)
    shap_values: list or array, shape同 X
    prefix: 欄位名稱前綴
    """
    n_samples, n_features = X.shape
    columns = [f'{prefix}{i+1}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=columns)

    # shap_values: 如果是list，取正類（通常index 0或1，根據你是二分類/多分類）
    if isinstance(shap_values, list):
        shap_val = shap_values[0] # 二分類可選 [0] or [1]
    else:
        shap_val = shap_values
    shap_columns = [f'SHAP {prefix}{i+1}' for i in range(n_features)]
    shap_df = pd.DataFrame(shap_val, columns=shap_columns)
    combined_df = pd.concat([data, shap_df], axis=1)
    return combined_df

##########################
# 3. transaction 轉換
##########################

def combined_df_to_transactions(combined_df, n_features=15, prefix='F'):
    """
    將合併好的 df 轉為 [(item, shap)] 結構
    """
    transactions = []
    for _, row in combined_df.iterrows():
        transaction = []
        for i in range(1, n_features + 1):
            feature = str(int(row[f'{prefix}{i}']))
            shap_value = row[f'SHAP {prefix}{i}']
            transaction.append((feature, shap_value))
        transactions.append(transaction)
    return transactions

##########################
# 4. Utility 序列計算
##########################

def utility_of_sequence_in_transaction(sequence, transaction):
    """計算序列在一筆交易中的 SHAP 總值"""
    utility = 0
    for item in sequence:
        for trans_item, item_utility in transaction:
            if item == trans_item:
                utility += item_utility
                break
    return utility

def is_sequence_in_transaction(sequence, transaction):
    """檢查序列是否有順序地出現在交易中"""
    if not sequence:
        return False
    seq_index = 0
    for trans_item, _ in transaction:
        if trans_item == sequence[seq_index]:
            seq_index += 1
            if seq_index == len(sequence):
                return True
    return False

def calculate_total_utility(sequence, transactions):
    """計算序列在所有 transactions 出現時的 utility 總和"""
    total_utility = 0
    for transaction in transactions:
        if is_sequence_in_transaction(sequence, transaction):
            total_utility += utility_of_sequence_in_transaction(sequence, transaction)
    return total_utility

##########################
# 5. 產生序列/搜尋高價值序列
##########################

def generate_sequences(items):
    """產生所有可能的項目序列"""
    if not items:
        return []
    sequences = [[item] for item in items]
    for i in range(len(items)):
        for subsequence in generate_sequences(items[i + 1:]):
            sequences.append([items[i]] + subsequence)
    return sequences

def find_top_k_utility_sequences(transactions, k=10):
    """找出 top-k utility sequence 與其分數"""
    items = set()
    for transaction in transactions:
        for item, _ in transaction:
            items.add(item)
    items = list(items)
    sequences = generate_sequences(items)
    sequence_utilities = []
    for sequence in sequences:
        utility = calculate_total_utility(sequence, transactions)
        sequence_utilities.append((sequence, utility))
    # 依utility排序
    top_k_sequences = sorted(sequence_utilities, key=lambda x: x[1], reverse=True)[:k]
    return top_k_sequences
