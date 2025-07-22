import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import load_data, prepare_data
from active_learning import active_learning_loop
import shap_utility as su

EARLY_WINDOW = 15
N_ITERATIONS = 6
N_SAMPLES = 1000
N_SHAP_SAMPLES = 100      # 計算 SHAP 的前多少筆樣本
TOP_K = 20                # Top K utility sequence

def main():
    # 1. 載入資料
    df = load_data('sampled_dataset.csv')
    pool_df = load_data('remaining_dataset.csv')

    # 2. 預處理
    X, y = prepare_data(df, early_window=EARLY_WINDOW)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Active learning (可根據需要調整篩選邏輯)
    criteria_list = [
        {"type": "element", "value": 2},
        {"type": "sequence", "value": [1, 2]},
        {"type": "element", "value": 5},
    ]
    print("=== Active Learning ===")
    active_learning_loop(
        X_train, y_train, X_test, y_test, pool_df,
        early_window=EARLY_WINDOW,
        n_iterations=N_ITERATIONS,
        n_samples=N_SAMPLES,
        criteria_list=criteria_list
    )

    # 4. SHAP 計算（取最新一輪的 X_train 及 model）
    # 這裡假設 active_learning_loop() 可回傳最新的 X_train, y_train, model
    # 如果沒回傳，請從 loop 裡直接取最新 model 物件
    # 這裡假設你會留一份最新 model
    # 範例：假設你有 model 變數
    # === 注意：請根據你的 active_learning_loop 回傳型態調整 ===

    print("=== SHAP 計算與 utility 分析 ===")
    # 例如：model = ...
    # 假設 active_learning_loop 結尾 model 物件你可以取得
    from model_utils import build_model, train_model

    # 這裡重新訓練一個 model 以演示 SHAP（你可以替換成剛才 AL loop 的最新 model）
    input_dim = 200
    output_dim = 40
    input_length = EARLY_WINDOW
    model = build_model(input_dim, output_dim, input_length)
    train_model(model, X_train, y_train, epochs=10)  # 用少一點 epoch 範例

    shap_values, data_for_shap = su.compute_shap_values(model, X_train, nsamples=N_SHAP_SAMPLES)
    combined_df = su.combine_features_shap(data_for_shap, shap_values, prefix='F')
    combined_df.to_csv('combined_data_shap.csv', index=False)

    transactions = su.combined_df_to_transactions(combined_df, n_features=EARLY_WINDOW, prefix='F')

    print(f"=== Top {TOP_K} utility sequences ===")
    top_k_sequences = su.find_top_k_utility_sequences(transactions, k=TOP_K)
    for idx, (seq, util) in enumerate(top_k_sequences, 1):
        print(f"{idx:2d}: 序列 {seq} ，總utility={util:.3f}")

if __name__ == "__main__":
    main()
