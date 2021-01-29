# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from model import *
from data import *


train = True
steps_per_epoch = 10000
epochs = 3
input_size = (128,128,3)

if train:
    # 新训练
    model = unet(input_size=input_size)
    # 继续训练
    #model = unet(input_size=input_size, pretrained_weights="unet_3_10000.hdf5")
    data_gen_args = dict(#rotation_range=0.2,
                        #width_shift_range=0.05,
                        #height_shift_range=0.05,
                        #shear_range=0.05,
                        #zoom_range=0.05,
                        #horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(4,'../datagen/data','image','mask',data_gen_args,
        target_size=(128,128),save_to_dir = None)

    model_checkpoint = ModelCheckpoint("unet_%d_%d.hdf5"%(epochs,steps_per_epoch), 
        monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])
else:
    model = unet(input_size=input_size, pretrained_weights="unet_%d_%d.hdf5"%(epochs,steps_per_epoch))

    test_path = "data/test"
    testGene = testGenerator(test_path, target_size=input_size[:2])
    file_list = os.listdir(test_path)
    results = model.predict_generator(testGene,len(file_list),verbose=1)
    saveResult("data/results",results,mask_num=5)