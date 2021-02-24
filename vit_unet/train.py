# coding=utf-8

# python3 -m vit_unet.train

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from .vit_unet import *
from .data import *


train = True
batch_size = 4
learning_rate = 1e-4
steps_per_epoch = 10000
epochs = 3
input_size = (128,128,3)

d_inner_hid=128
layers=4
n_head=4
d_model=512

train_path = 'datagen/data'
mfile = "vit_unet/vit-unet_%d_%d.weights"%(epochs,steps_per_epoch)

if train:
    # 新训练
    model = unet(input_size=input_size, d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, lr=learning_rate)
    # 继续训练
    #model = unet(input_size=input_size, d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, lr=learning_rate,
    #    pretrained_weights="vit_unet/vit-unet_3_10000_e3.weights")

    data_gen_args = dict(fill_mode='nearest')
    myGene = trainGenerator(batch_size,train_path,'image','mask',data_gen_args,
        target_size=(128,128),save_to_dir = None)

    model_checkpoint = ModelCheckpoint(mfile, monitor='loss',verbose=1, 
        save_best_only=True, save_weights_only=True)
    model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,
        callbacks=[model_checkpoint])
else:
    model = unet(input_size=input_size, pretrained_weights=mfile)

    test_path = "data/test"
    testGene = testGenerator(test_path, target_size=input_size[:2])
    file_list = os.listdir(test_path)
    results = model.predict_generator(testGene,len(file_list),verbose=1)
    saveResult("data/results",results,mask_num=5)
