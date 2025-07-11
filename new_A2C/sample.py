from one_hot_encoder import load_strings_from_excel, string_to_one_hot

# 指定Excel文件路径
file_path = 'strings.xlsx'  # 将此路径替换为你的Excel文件路径

# 从Excel文件中读取字符串集合
all_strings, string_to_index = load_strings_from_excel(file_path)

# 示例使用
input_string = 'banana'
one_hot_vector = string_to_one_hot(input_string, string_to_index, len(all_strings))
print(f"输入字符串: {input_string}")
print(f"One-Hot编码向量: {one_hot_vector}")
