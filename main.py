import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import load_data, prepare_data
from active_learning import active_learning_loop

EARLY_WINDOW = 15
N_ITERATIONS = 6
N_SAMPLES = 1000

def main():
    # 讀資料
    df = load_data('sampled_dataset.csv')
    pool_df = load_data('remaining_dataset.csv')

    X, y = prepare_data(df, early_window=EARLY_WINDOW)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 指定篩選條件，依序篩選補滿
    criteria_list = [
        {"type": "element", "value": 2},
        {"type": "sequence", "value": [1, 2]},
        {"type": "element", "value": 5},
    ]

    # Active learning
    active_learning_loop(X_train, y_train, X_test, y_test, pool_df, 
                        early_window=EARLY_WINDOW, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES,
                        criteria_list=criteria_list)

if __name__ == "__main__":
    main()
