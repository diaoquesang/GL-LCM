import random
import os


def traverse_directory(directory, txt_name):
    # 创建一个空的列表用于存储文件名
    file_names = []
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    # 按照文件名排序（如果你希望的话）
    file_names.sort()
    # 创建一个新的txt文件，并将文件名写入该文件
    with open(txt_name, 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


def split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, former=None):
    # 读取数据集
    with open(file_path, 'r') as f:
        data = f.readlines()
    # 随机打乱数据集
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set = data[train_size + val_size:]
    with open(former + '_trainset.txt', 'w') as f:
        f.writelines(train_set)
    with open(former + '_valset.txt', 'w') as f:
        f.writelines(val_set)
    with open(former + '_testset.txt', 'w') as f:
        f.writelines(test_set)
    print(f"Finished.")


if __name__ == "__main__":
    traverse_directory('JSRTnew1024-241/CXR', "JSRT.txt")

    split_dataset('JSRT.txt',0.8,0.1, "JSRT")
