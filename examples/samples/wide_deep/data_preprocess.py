# @Time    : 2018/7/9 14:55
# @Author  : cap
# @FileName: data_preprocess.py
# @Software: PyCharm Community Edition
# @introduction:
def process(temp, filename):
    with open(temp, 'r') as tempfile:
        with open(filename, 'w') as targetfile:
            for line in tempfile:
                data = line.strip()
                data = data.replace(', ', ',')
                if not data or ',' not in data:
                    continue
                if data[-1] == '.':
                    data = data[:-1]
                data += '\n'
                targetfile.write(data)


def main():
    data_dir = 'D:\\softfiles\\workspace\\data\\tensorflow\\data\\wide_deep'
    train_file = data_dir + '\\adult.data'
    test_file = data_dir + '\\adult.test'

    train_temp = data_dir + '\\adult_data.txt'
    test_temp = data_dir + '\\adult_test2.txt'
    # process(train_temp, train_file)
    process(test_temp, test_file)


if __name__ == '__main__':
    main()
