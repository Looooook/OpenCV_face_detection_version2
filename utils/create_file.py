"""
用于生成用户照片文件夹
"""
import os


def create_user_file(higher_directory, user_name):
    # 用户文件的命名格式为  0_ZT_12547.jpg
    categories = 0
    users_idx_list = []

    for user_file in os.listdir(higher_directory):
        user_serial_number = user_file.split('_')[0]
        users_idx_list.append(int(user_serial_number))
        categories += 1

    users_idx_list.sort()


    if os.listdir(higher_directory) == []:
        os.mkdir(higher_directory + '0' + '_' + user_name + '/')
        user_file_name = '0' + '_' + user_name
    else:
        last_number = users_idx_list[-1]
        os.mkdir(higher_directory + str(int(last_number) + 1) + '_' + user_name + '/')
        user_file_name = str(int(last_number) + 1) + '_' + user_name

    return user_file_name


def remove(train_data_file_path):
    for i in os.listdir(train_data_file_path):
        os.remove(train_data_file_path + i)
