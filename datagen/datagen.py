# coding=utf-8

import os
import random
import numpy as np
import pandas as pd 
from PIL import Image

# 将数据转化为图片，尺寸 time_span*time_span
# x - 时间
# y - 价
# g（灰度） - 量

# 生成蜡烛图（只有最高最低价）
def generate_data2(data_file, # 数据集文件 csv
    time_span=128, # 1个图片的时间跨度
    time_step=1, # 每个图片的间隔时间
    mask_num=5,
    output_dir='data',
    output_image=False): # 是否输出图片

    os.makedirs('%s/image'%output_dir, exist_ok=True)
    os.makedirs('%s/mask'%output_dir, exist_ok=True)

    training_set=pd.read_csv(data_file) 
    training_set_open=training_set.iloc[:,1:2].values # 开盘价
    training_set_close=training_set.iloc[:,4:5].values # 收盘价
    training_set_volume=training_set.iloc[:,5:6].values # 量

    # 纵向占 80%
    image_height = int(time_span * 0.8)
    image_y_padding = (time_span - image_height) // 2

    image_list = []
    mask_list = []

    time_span_test = time_span - mask_num

    for i in range((len(training_set_open)-time_span)//time_step):
        # 训练图片 RGB
        image = np.zeros([time_span, time_span, 3])
        # 预测图片 RGB
        mask = np.zeros([time_span, time_span, 3])

        start_x = i*time_step

        # 价范围
        price_max = max(training_set_open[start_x:start_x+time_span_test].max(),
            training_set_close[start_x:start_x+time_span_test].max())
        price_min = min(training_set_open[start_x:start_x+time_span_test].min(),
            training_set_close[start_x:start_x+time_span_test].min())
        price_span = price_max - price_min

        # 量范围
        volume_max = training_set_volume[start_x:start_x+time_span_test].max()
        volume_min = training_set_volume[start_x:start_x+time_span_test].min()
        volume_span = volume_max - volume_min

        # 填每列的点数据
        for x in range(time_span):
            price_open = training_set_open[start_x+x][0]
            price_close = training_set_close[start_x+x][0]
            volume = training_set_volume[start_x+x][0]
            #print(price, volume)
            yo = image_y_padding + image_height-int((price_open - price_min)/price_span*(image_height-1))-1
            yc = image_y_padding + image_height-int((price_close - price_min)/price_span*(image_height-1))-1
            g = (volume - volume_min)/volume_span*154.+1 # 取值 1～155
            #g = 255 # 不考虑 量
            #print(x, y, g)
            # 最后 mask_num 个点 作为预测点
            if yo>yc:
                y1=yc
                y2=yo
                channel = 0 # 绿色
            else:
                y1=yo
                y2=yc
                channel = 1 # 红色
            # y1, y2 不能超出边界
            y1 = max(y1, 0)
            y1 = min(y1, time_span-1) 
            y2 = max(y2+1, 0)
            y2 = min(y2+1, time_span-1)
            if x>=time_span_test:
                #mask[y][x] = 255 # 预测点 最大化
                mask[y1:y2+1,x,channel] = 255 # 预测点 最大化， 白色
            else:
                #image[y][x] = g + 100
                #mask[y][x] = g + 100
                image[y1:y2+1,x,channel] = g + 100
                mask[y1:y2+1,x,channel] = g + 100 

        image_list.append(image)
        mask_list.append(mask)
        #print(image)

        if output_image:
            im = Image.fromarray(image.astype(np.uint8))
            im = im.convert('RGB')
            im.save('%s/image/%d.png'%(output_dir, start_x))

            im = Image.fromarray(mask.astype(np.uint8))
            im = im.convert('RGB')
            im.save('%s/mask/%d.png'%(output_dir, start_x))

    return image_list, mask_list


# 为预测生存图片
def generate_data2_for_test(data_file, # 数据集文件 csv
    time_span=128, # 1个图片的时间跨度
    time_step=1, # 每个图片的间隔时间
    mask_num=5,
    last_y=None,
    output_dir='test',
    output_image=False): # 是否输出图片

    os.makedirs(output_dir, exist_ok=True)

    training_set=pd.read_csv(data_file) 
    training_set_open=training_set.iloc[:,1:2].values # 开盘价
    training_set_close=training_set.iloc[:,4:5].values # 收盘价
    training_set_volume=training_set.iloc[:,5:6].values

    #print("data len:", len(training_set_price))

    # 纵向占 80%
    image_height = int(time_span * 0.8)
    image_y_padding = (time_span - image_height) // 2

    image_list = []
    mask_list = []

    time_span_test = time_span - mask_num

    for i in range((len(training_set_open)-time_span_test)//time_step):
        # 训练图片
        image = np.zeros([time_span, time_span, 3])

        start_x = i*time_step

        # 价范围
        price_max = max(training_set_open[start_x:start_x+time_span_test].max(),
            training_set_close[start_x:start_x+time_span_test].max())
        price_min = min(training_set_open[start_x:start_x+time_span_test].min(),
            training_set_close[start_x:start_x+time_span_test].min())
        price_span = price_max - price_min

        # 量范围
        volume_max = training_set_volume[start_x:start_x+time_span_test].max()
        volume_min = training_set_volume[start_x:start_x+time_span_test].min()
        volume_span = volume_max - volume_min

        # 填每列的点数据
        for x in range(time_span_test):
            price_open = training_set_open[start_x+x][0]
            price_close = training_set_close[start_x+x][0]
            volume = training_set_volume[start_x+x][0]
            #print(price, volume)
            yo = image_y_padding + image_height-int((price_open - price_min)/price_span*(image_height-1))-1
            yc = image_y_padding + image_height-int((price_close - price_min)/price_span*(image_height-1))-1
            g = (volume - volume_min)/volume_span*154.+1 # 取值 1～255
            #g = 255 # 不考虑 量
            #print(x, y, g)
            if yo>yc:
                y1=yc
                y2=yo
                channel = 0 # 绿色
            else:
                y1=yo
                y2=yc
                channel = 1 # 红色
            # y1, y2 不能超出边界
            y1 = max(y1, 0)
            y1 = min(y1, time_span-1) 
            y2 = max(y2+1, 0)
            y2 = min(y2+1, time_span-1)
            # 最后 mask_num 个点 作为预测点
            image[y1:y2+1,x,channel] = g + 100

        # 重复预测时，添加最后nn列数据
        if last_y is not None:
            nn = last_y.shape[1]
            # 左移一列，最右mask_num清零
            image = np.roll(image, -nn, axis=1)
            image[:,-mask_num-nn:-mask_num] = last_y
            image[:,-mask_num:] = 0

        image_list.append(image)
        #print(image)

        if output_image:
            im = Image.fromarray(image.astype(np.uint8))
            im = im.convert('RGB')
            im.save('%s/%d.png'%(output_dir, start_x))

    return image_list


if __name__ == '__main__':
    # 生成图片
    generate_data2('../dataset/eth_history.csv', output_image=True)

    #generate_data2_for_test('../dataset/eth_now.csv', output_image=True)
