# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
import numpy as np
from datagen import datagen
from okapi import okapi
from unet import data as udata
from unet import model as unet

input_size = (128,128,3)
mask_num = 5
data_path = "data"
test_path = '%s/test'%data_path
results_path = '%s/results'%data_path
html_path = "data/predictor.html"
'''
服务端配置
crontab -l
2 * * * * python3 /opt/eth_predictor/predict_now.py > /tmp/predictor.log 2>&1
'''

HTML='''
<!DOCTYPE HTML>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="format-detection" content="telephone=no">
    <title>eth predictor</title>
</head>
<body>
<h3>%s</h3>
<div><center><img width="80%%" src="%s?%d"/><center></div>
</body>
</html>
'''

if __name__ == '__main__':
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # 从ok去最近数据
    X = okapi.get_recent('%s/eth_now.csv'%data_path, 'ETH-USDT', num=124) # 124个数据只会生成一个图片
    print('data from OKexi: ', len(X))

    # 加载模型
    model = unet.unet(input_size=input_size, pretrained_weights="%s/unet_test_candle.hdf5"%data_path)

    last_y = None

    for n in range(24): # 预测24小时的数据
        # 生成图片
        X_test = datagen.generate_data2_for_test('%s/eth_now.csv'%data_path, last_y=last_y,
            output_dir=test_path, output_image=True)

        # 预测结果
        testGene = udata.testGenerator(test_path, target_size=input_size[:2])
        file_list = os.listdir(test_path)
        results = model.predict_generator(testGene,len(file_list),verbose=1)

        # 保存中间结果，下次生成图片时使用
        last_col = results[0][:,-mask_num:-mask_num+1] * 255
        if last_y is None:
            last_y = last_col
        else:
            last_y = np.column_stack((last_y, last_col))

        # 保存每次测试图片
        #os.rename(os.path.join(test_path, '0.png'), os.path.join(results_path, '0_%d.png'%n))

        udata.saveResult(results_path, results, mask_num=mask_num)

        # 保存每次预测结果
        #os.rename(os.path.join(results_path, 'adjust_0.png'), os.path.join(results_path, 'adjust_0_%d.png'%n))

    # 调整，右移
    udata.saveResult(results_path, results, mask_num=mask_num, move=n)

    # 生成 html
    with open(html_path, "w") as f:
        f.write(HTML%(time.ctime(), 'results/adjust_0.png', time.time() ))