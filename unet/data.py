# coding=utf-8

from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans


def adjustData(img,mask):
    img = img / 255
    mask = mask / 255
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)



def testGenerator(test_path,target_size = (256,256),as_gray = False):
    file_list = os.listdir(test_path)
    file_list = sorted(file_list)
    for i in file_list:
        img = io.imread(os.path.join(test_path,i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if as_gray else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def saveResult(save_path,npyfile,mask_num=5,move=0):
    for i,item in enumerate(npyfile):
        img = item
        #io.imsave(os.path.join(save_path,"predict_%d.png"%i),(img*255).astype(np.uint8))

        # 处理结果
        time_span = img.shape[0]
        for x in range(time_span):
            if move > 0: # move大于0，说明是最后一次结果，画分界线
                if x==time_span-mask_num-1-move: 
                    for y in range(time_span):
                        if (img[y,x]<0.001).all():
                            img[y,x] = 0.5

            if x>(time_span-mask_num):
                y_g = img[:,x,0] ## 绿色
                y_r = img[:,x,1] ## 红色

                # 取前max_n个最大值，的索引
                max_n = 5
                y_g_5 = y_g.argsort()[-max_n:][::-1]
                y_r_5 = y_r.argsort()[-max_n:][::-1]

                # 比较同一点，哪个颜色值大，就显示哪个颜色
                #for n in range(max_n):
                #    # 增强显示 数值比较大的颜色， 前几个最大值
                #    if y_g[y_g_5[n]]>y_r[y_r_5[n]]:
                #        y_g[y_g_5[n]]=min(1, y_g[y_g_5[n]]*1.5)
                #    else:
                #        y_r[y_r_5[n]]=min(1, y_r[y_r_5[n]]*1.5)

        # 右移 mask_num
        if move>0:
            img = np.roll(img, mask_num-1, axis=1)
            img[:,:mask_num-1] = 0. 

        io.imsave(os.path.join(save_path,"adjust_%d.png"%i),(img*255).astype(np.uint8))
