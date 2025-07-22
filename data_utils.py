import pandas as pd
import numpy as np
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['trajectory'] = df['trajectory'].apply(ast.literal_eval)
    return df

def prepare_data(df, early_window=15):
    X = pad_sequences(df['trajectory'], maxlen=early_window, padding='post')
    y = to_categorical(df['Churn'].values)
    return X, y

def contains_sequence(trajectory, sequence):
    """Check if a sequence is contained in the trajectory."""
    for i in range(len(trajectory) - len(sequence) + 1):
        if trajectory[i:i + len(sequence)] == sequence:
            return True
    return False
