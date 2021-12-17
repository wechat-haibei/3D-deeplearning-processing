#-*- coding: UTF-8 -*- 
# 读取数据 bin 文件

import os
import struct

def read_data(file):
    file_path = file_dir+"/"+file
    final_text = open('final.txt', 'a')
    data_bin = open(file_path, 'rb')
    data_size = os.path.getsize(file_path)
    for i in range(data_size):
        for index in range(4):
            data_i = data_bin.read(4) # 每次输出4个字节 
            if len(data_i)== 4:
                num = struct.unpack('f', data_i)
                max_list[index].append(num[0]) #记录最大值
                min_list[index].append(num[0]) #记录最小值
    write = file +'\t'
    for index in range(4):
        max_list[index] = [max(max_list[index])] #最大列表中只保留最大值
        min_list[index] = [min(min_list[index])] #最小列表中只保留最小值
        write += str(max_list[index][0]) +'\t'+ str(min_list[index][0])+'\t' #输出目前的最大最小值
    print(write)
    final_text.write(write +'\n') #储存
    data_bin.close()
    final_text.close()

file_dir = '/root/pvrcnn/POINTCLOUD' #文件夹目录

files = os.listdir(file_dir) #得到文件夹下的所有文件名称

max_list = [[620.970458984375],[278.865478515625],[1.0],[1.0]]
min_list = [[2.3114852905273438],[-534.9176635742188],[-101.55160522460938],[1.0]] #004231.bin	
							
for file in files: #遍历文件夹
    read_data(file)
