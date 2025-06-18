import pandas as pd
import numpy as np


def load_strings_from_excel(file_path):
    """
    从Excel文件中读取字符串集合。
    """
    df = pd.read_excel(file_path, header=None)
    all_strings = df[0].tolist()
    string_to_index = {string: idx for idx, string in enumerate(all_strings)}
    return all_strings, string_to_index


def string_to_one_hot(input_string, string_to_index, all_strings_length):
    """
    将输入字符串转换为One-Hot编码向量。
    """
    idx = string_to_index.get(input_string, None)

    if idx is None:
        raise ValueError(f"输入字符串 '{input_string}' 不在所有可能的字符串集合中")

    one_hot_vector = np.zeros(all_strings_length)
    one_hot_vector[idx] = 1

    return one_hot_vector
