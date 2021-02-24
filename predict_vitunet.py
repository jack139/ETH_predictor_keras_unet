# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
import numpy as np
import skimage.io as io
import skimage.transform as trans
from datagen import datagen
from okapi import okapi
from vit_unet import vit_unet as unet

d_inner_hid=128
layers=4
n_head=4
d_model=512

input_size = (128,128,3)
mask_num = 5
time_span = input_size[0]
data_path = "data"
test_path = '%s/test'%data_path
results_path = '%s/results'%data_path
html_path = "%s/predictor.html"%data_path

'''
服务端配置

crontab -l
*/10 * * * * python3 /root/btc_predictor/predict_now.py > /tmp/predictor.log 2>&1
'''

HTML='''
<!DOCTYPE HTML>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="format-detection" content="telephone=no">
    <title>vit-unet - eth predictor</title>
</head>
<body>
<h2>ViT-UNet - %s</h2>
<h3>1小时</h3>
<div><center><img width="80%%" src="%s?%d"/><center></div>
</body>
</html>
'''

def load_image(filepath,target_size = (128,128),as_gray = False):
    img = io.imread(filepath, as_gray = as_gray)
    img = img / 255
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,)) if as_gray else img
    img = np.reshape(img,(1,)+img.shape)

    return img

def run_predict(model, img_test, results_path='results', out_img='out.png'):

    last_y = None
    predict_n = 8
    for n in range(predict_n):
        # 重复预测时，添加最后nn列数据
        if last_y is not None:
            nn = last_y.shape[1]
            # 左移一列，最右mask_num清零
            img_test[0] = np.roll(img_test[0], -nn, axis=1)
            img_test[0][:,-mask_num-nn:-mask_num] = last_y
            img_test[0][:,-mask_num:] = 0

        # 预测结果
        results = model.predict(img_test, verbose=1)

        # 保存中间结果，下次生成图片时使用
        last_col = results[0][:,-mask_num:-mask_num+1]
        last_y = last_col

    # 对比预测方向
    img_pred = results[0]

    # 右移
    img_pred = np.roll(img_pred, mask_num-1, axis=1)
    img_pred[:,:mask_num-1] = 0. 

    # 调整为大图，间隔空行
    new_img = np.zeros([time_span, time_span*2, 3])
    for x in range(time_span):
        if x==time_span-predict_n-1: # 画分割线
            for y in range(time_span):
                if (img_pred[y,x]<0.001).all():
                    img_pred[y,x] = 0.5

        new_img[:,x*2] = img_pred[:,x]

    io.imsave(os.path.join(results_path, out_img),(new_img*255).astype(np.uint8))


if __name__ == '__main__':
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # 从ok去最近数据
    X = okapi.get_recent('%s/eth_now_1h.csv'%data_path, 'ETH-USDT', num=123, gap=3600) # 123个数据只会生成一个图片
    print('data from OKexi: ', len(X))


    # 加载模型
    model = unet.unet(input_size=input_size, 
        d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, 
        pretrained_weights="%s/vit-unet_3_10000.weights"%data_path)

    # 生成图片
    X_test = datagen.generate_data2_for_test('%s/eth_now_1h.csv'%data_path,
        output_dir=test_path, output_image=True)

    img_test = load_image(os.path.join(test_path, '0.png'))

    run_predict(model, img_test, results_path=results_path, out_img='vit_1h.png')

    # 生成 html
    with open(html_path, "w") as f:
        f.write(HTML%(time.ctime(), 'results/vit_1h.png', time.time() ))