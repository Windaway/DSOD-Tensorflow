import os
import gc
import xml.etree.ElementTree as etxml
import math
import random
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorflow.python.ops import variables
import time
from imutils.object_detection import non_max_suppression
import imutils
import cv2
import matplotlib.pyplot as plt
batch_size = 16
running_count = 5000
file_name_list = os.listdir('./train_datasets/voc2012/JPEGImages/')
lable_arr = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
img_size = [300, 300]
# 分类总数量
classes_size = 21
# 背景分类的值
background_classes_val = 0
# 每个特征图单元的default box数量
default_box_size = [6, 6, 6, 6, 6, 6]
# default box 尺寸长宽比例
box_aspect_ratio = [
    [0.5, 1.0, 2.0, 3.0,1/3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0]
]
# 最小default box面积比例
min_box_scale = 0.1
# 最大default box面积比例
max_box_scale = 0.9
# 每个特征层的面积比例
# numpy生成等差数组，效果等同于论文中的s_k=s_min+(s_max-s_min)*(k-1)/(m-1)
default_box_scale = np.linspace(min_box_scale, max_box_scale, num=np.amax(default_box_size))
print('##   default_box_scale:' + str(default_box_scale))
# 卷积步长
conv_strides_1 = [1, 1, 1, 1]
conv_strides_2 = [1, 2, 2, 1]
conv_strides_3 = [1, 3, 3, 1]

tl_strides_1 = (1, 1)
tl_strides_2 = (2, 2)
tl_strides_3 = (3, 3)
# 池化窗口
pool_size = [1, 2, 2, 1]
tl_pool_size = (2, 2)
# 池化步长
pool_strides = [1, 2, 2, 1]
tl_pool_strides = (2, 2)
# Batch Normalization 算法的 decay 参数
conv_bn_decay = 0.9999
# Batch Normalization 算法的 variance_epsilon 参数
conv_bn_epsilon = 0.001
# Jaccard相似度判断阀值
jaccard_value = 0.55
feature_maps_shape=[]
all_default_boxs_len=0
all_default_boxs=[]

jitter = 0.2
def get_traindata_voc(batch_size):
    def get_actual_data_from_xml(xml_path):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_path).getroot()
            img_width = float(annotation_node.find('size').find('width').text.strip())
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')
            for obj_node in object_node_list:
                lable = lable_arr.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max) / 2 / img_width), ((y_min + y_max) / 2 / img_height),
                                    ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item
        except:
            return None

    train_data = []
    actual_data = []
    file_list = random.sample(file_name_list, batch_size)
    for f_name in file_list:
        img_path = './train_datasets/voc2012/JPEGImages/' + f_name
        xml_path = './train_datasets/voc2012/Annotations/' + f_name.replace('.jpg', '.xml')
        if os.path.splitext(img_path)[1].lower() == '.jpg':
            actual_item = get_actual_data_from_xml(xml_path)
            img = skimage.io.imread(img_path)
            if actual_item != None:
                countwhile=0
                while True:
                    clas=[]
                    coords=[]
                    for x in actual_item:
                        clas.append(x[4])
                        coords.append([x[0],x[1],x[2],x[3]])
                    tmp0 = random.randint(-30, 50)
                    tmp1 = random.randint(-30, 50)
                    imgr=img.copy()
                    scale = np.max((400 / float(img.shape[1]),
                                    400 / float(img.shape[0])))
                    im, coords = tl.prepro.obj_box_imresize(imgr, coords,
                                                            [int(img.shape[0] * scale) + tmp0, int(img.shape[1] * scale) + tmp1],
                                                            is_rescale=True, interp='bicubic')
                    # print(im.shape)
                    # print(coords)

                    for wi in range(7):
                        imt, clast, coordst = tl.prepro.obj_box_zoom(im, clas, coords, zoom_range=(1.0, 2.2),
                                                                  fill_mode='nearest',
                                                                  order=1, is_rescale=True, is_center=True,
                                                                  is_random=True,
                                                                  thresh_wh=0.04, thresh_wh2=8.0)
                        # print(im.shape)
                        if clast!=[]:
                            im=imt
                            clas= clast
                            coords =coordst
                            break
                        if wi>=6:
                            im, clas, coords = tl.prepro.obj_box_zoom(im, clas, coords, zoom_range=(0.7, 1.2),
                                                                         fill_mode='nearest',
                                                                         order=1, is_rescale=True, is_center=True,
                                                                         is_random=True,
                                                                         thresh_wh=0.05, thresh_wh2=8.0)

                    im, coords = tl.prepro.obj_box_left_right_flip(im,
                                                                   coords, is_rescale=True, is_center=True, is_random=True)
                    # print(coords)
                    for wi in range(8):
                        imt, clast, coordst = tl.prepro.obj_box_crop(im, clas, coords,
                                                                  wrg=300, hrg=300,
                                                                  is_rescale=True, is_center=True, is_random=True,
                        thresh_wh=0.07, thresh_wh2=7.0)
                        if clast!=[]:
                            im=imt
                            clas= clast
                            coords =coordst
                            break
                        if wi==7:
                            im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
                                                                         wrg=300, hrg=300,
                                                                         is_rescale=True, is_center=True,
                                                                         is_random=True,
                                                                         thresh_wh=0.07, thresh_wh2=8.0)


                    im = tl.prepro.illumination(im, gamma=(0.2, 1.2),
                                                contrast=(0.2, 1.2), saturation=(0.2, 1.2), is_random=True)
                    im = tl.prepro.adjust_hue(im, hout=0.1, is_offset=True,
                                              is_clip=True, is_random=True)
                    im = im / 127.5 - 1.
                    aitems = []
                    if clas!=[]:
                        for x in range(len(clas)):
                            aitem=[coords[x][0],coords[x][1],coords[x][2],coords[x][3],clas[x]]
                            aitems.append(aitem)
                        actual_data.append(aitems)
                        train_data.append(im)
                        break
                    countwhile+=1
                    if countwhile>=4:
                        clas = []
                        coords = []
                        for x in actual_item:
                            clas.append(x[4])
                            coords.append([x[0], x[1], x[2], x[3]])
                        tmp0 = random.randint(1, 30)
                        tmp1 = random.randint(1, 30)
                        imgr = img.copy()
                        im, coords = tl.prepro.obj_box_imresize(imgr, coords,
                                                                [300 + tmp0,
                                                                 300 + tmp1],
                                                                is_rescale=True, interp='bicubic')
                        im, coords = tl.prepro.obj_box_left_right_flip(im,
                                                                       coords, is_rescale=True, is_center=True,
                                                                       is_random=True)
                        im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
                                                                     wrg=300, hrg=300,
                                                                     is_rescale=True, is_center=True,
                                                                     is_random=True,
                                                                     thresh_wh=0.02, thresh_wh2=10.0)



                        im = tl.prepro.illumination(im, gamma=(0.8, 1.2),
                                                    contrast=(0.8, 1.2), saturation=(0.8, 1.2), is_random=True)
                        im = tl.prepro.pixel_value_scale(im, 0.1, [0, 255], is_random=True)
                        im = im / 127.5 - 1.

                        aitems = []
                        if len(clas) != 0:
                            for x in range(len(clas)):
                                aitem = [coords[x][0], coords[x][1], coords[x][2], coords[x][3], clas[x]]
                                aitems.append(aitem)
                            actual_data.append(aitems)
                            train_data.append(im)
                            break
            else:
                print('Error : ' + xml_path)
                continue
    return train_data, actual_data, file_list

def generate_groundtruth_data(input_actual_data):
    # 生成空数组，用于保存groundtruth
    input_actual_data_len = len(input_actual_data)
    gt_class = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_location = np.zeros((input_actual_data_len, all_default_boxs_len, 4))
    gt_positives_jacc = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_positives = np.zeros((input_actual_data_len, all_default_boxs_len))
    gt_negatives = np.zeros((input_actual_data_len, all_default_boxs_len))
    background_jacc = max(0, (jaccard_value - 0.2))
    # 初始化正例训练数据
    for img_index in range(input_actual_data_len):
        for pre_actual in input_actual_data[img_index]:
            gt_class_val = pre_actual[-1:][0]

            if gt_class_val>20 or gt_class_val<0:
                gt_class_val=0
            gt_box_val = pre_actual[:-1]
            for boxe_index in range(all_default_boxs_len):
                jacc,gt_box_val_loc = jaccard(gt_box_val, all_default_boxs[boxe_index])
                if jacc > jaccard_value or jacc == jaccard_value:
                    gt_class[img_index][boxe_index] = gt_class_val
                    gt_location[img_index][boxe_index] = gt_box_val_loc
                    gt_positives_jacc[img_index][boxe_index] = jacc
                    gt_positives[img_index][boxe_index] = 1
                    gt_negatives[img_index][boxe_index] = 0
        # 如果没有正例，则随机创建一个正例，预防nan
        if np.sum(gt_positives[img_index]) == 0:
            # print('【没有匹配jacc】:'+str(input_actual_data[img_index]))
            random_pos_index = np.random.randint(low=0, high=all_default_boxs_len, size=1)[0]
            gt_class[img_index][random_pos_index] = background_classes_val
            gt_location[img_index][random_pos_index] = [0.00001, 0.00001, 0.00001, 0.00001]
            gt_positives_jacc[img_index][random_pos_index] = jaccard_value
            gt_positives[img_index][random_pos_index] = 1
            gt_negatives[img_index][random_pos_index] = 0
        gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
        if (gt_neg_end_count + np.sum(gt_positives[img_index])) > all_default_boxs_len:
            gt_neg_end_count = all_default_boxs_len - np.sum(gt_positives[img_index])
        gt_neg_index = np.random.randint(low=0, high=all_default_boxs_len, size=gt_neg_end_count)
        for r_index in gt_neg_index:
            if gt_positives_jacc[img_index][r_index] < background_jacc and gt_positives[img_index][r_index] != 1:
                gt_class[img_index][r_index] = background_classes_val
                gt_positives[img_index][r_index] = 0
                gt_negatives[img_index][r_index] = 1
    gt_class = check_numerics(gt_class, 'gt_class')
    gt_location = check_numerics(gt_location, 'gt_class')
    gt_positives = check_numerics(gt_positives, 'gt_positives')
    gt_negatives = check_numerics(gt_negatives, 'gt_negatives')
    return gt_class, gt_location, gt_positives, gt_negatives

def jaccard(rect1, rect2):
    x_overlap = max(0, (min(rect1[0] + (rect1[2] / 2), rect2[0] + (rect2[2] / 2)) - max(rect1[0] - (rect1[2] / 2),
                                                                                        rect2[0] - (rect2[2] / 2))))
    y_overlap = max(0, (min(rect1[1] + (rect1[3] / 2), rect2[1] + (rect2[3] / 2)) - max(rect1[1] - (rect1[3] / 2),
                                                                                        rect2[1] - (rect2[3] / 2))))
    intersection = x_overlap * y_overlap
    # 删除超出图像大小的部分
    rect1_width_sub = 0
    rect1_height_sub = 0
    rect2_width_sub = 0
    rect2_height_sub = 0
    if (rect1[0] - rect1[2] / 2) < 0: rect1_width_sub += 0 - (rect1[0] - rect1[2] / 2)
    if (rect1[0] + rect1[2] / 2) > 1: rect1_width_sub += (rect1[0] + rect1[2] / 2) - 1
    if (rect1[1] - rect1[3] / 2) < 0: rect1_height_sub += 0 - (rect1[1] - rect1[3] / 2)
    if (rect1[1] + rect1[3] / 2) > 1: rect1_height_sub += (rect1[1] + rect1[3] / 2) - 1
    if (rect2[0] - rect2[2] / 2) < 0: rect2_width_sub += 0 - (rect2[0] - rect2[2] / 2)
    if (rect2[0] + rect2[2] / 2) > 1: rect2_width_sub += (rect2[0] + rect2[2] / 2) - 1
    if (rect2[1] - rect2[3] / 2) < 0: rect2_height_sub += 0 - (rect2[1] - rect2[3] / 2)
    if (rect2[1] + rect2[3] / 2) > 1: rect2_height_sub += (rect2[1] + rect2[3] / 2) - 1
    area_box_a = (rect1[2] - rect1_width_sub) * (rect1[3] - rect1_height_sub)
    area_box_b = (rect2[2] - rect2_width_sub) * (rect2[3] - rect2_height_sub)
    union = area_box_a + area_box_b - intersection
    if intersection > 0 and union > 0:
        return intersection / union,[(rect1[0]-(rect2[0]))/rect2[2],(rect1[1]-(rect2[1]))/rect2[3],math.log(rect1[2]/rect2[2]),math.log(rect1[3]/rect2[3])]

    else:
        return 0,[0.00001,0.00001,0.00001,0.00001]

def denseblock(input,blocknum=1,step=48,firstchannel=192,is_train=True,name='denseblock',reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nettemp=LambdaLayer(input, lambda x: tf.identity(x), name="INPUTS")
        for x in range(blocknum):
            netbn = BatchNormLayer(nettemp, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='bn/' + str(x))
            net=Conv2d(netbn, firstchannel, (1, 1), (1, 1), padding='SAME',name='neta/'+str(x))
            netbn = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name=name + 'bn2/' + str(x))
            net=Conv2d(netbn, step, (3, 3), (1, 1), padding='SAME',name='netb/'+str(x))
            nettemp= ConcatLayer([nettemp,net], -1,name='concattemp/'+str(x))
            net = nettemp
    return net

def denseblockpl(input,step=256,firstchannel=256,is_train=True,name='densepl',reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        input = LambdaLayer(input, lambda x: tf.identity(x), name="INPUTS")
        netbn2=MaxPool2d(input,(2,2),(2,2),padding='SAME', name='bnpool2')
        netbn2 = BatchNormLayer(netbn2, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name=name + 'bn2pl' )
        netbn2 = Conv2d(netbn2, firstchannel, (1, 1), (1, 1), padding='SAME', name='bnconv2' )
        netbn = BatchNormLayer(input, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name= 'bn' )
        net=Conv2d(netbn, firstchannel, (1, 1), (1, 1), padding='SAME',name='neta')
        netbn = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='bn2')
        net=Conv2d(netbn, step, (3, 3), (2, 2), padding='SAME',name='netb')
        nettemp = ConcatLayer([net,netbn2], -1,name='concat')
    return nettemp

def denseblockfin(input,step=256,firstchannel=256,is_train=True,name='densepl',reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        input = LambdaLayer(input, lambda x: tf.identity(x), name="INPUTS")
        netbn2=MaxPool2d(input,(3,3),(1,1),padding='VALID', name='bnpool2')
        netbn2 = BatchNormLayer(netbn2, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name=name + 'bn2pl' )
        netbn2 = Conv2d(netbn2, firstchannel, (1, 1), (1, 1), padding='SAME', name='bnconv2' )
        netbn = BatchNormLayer(input, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name= 'bn' )
        net=Conv2d(netbn, firstchannel, (1, 1), (1, 1), padding='SAME',name='neta')
        netbn = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='bn2')
        net=Conv2d(netbn, step, (3, 3), (1, 1), padding='VALID',name='netb')
        nettemp = ConcatLayer([net,netbn2], -1,name='concat')
    return nettemp

def inference(inputs, is_train, reuse):
    W_init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(inputs, name='input')
        net = Conv2d(net, 64, (3, 3), (2, 2), padding='SAME',
                     W_init=W_init, name='stem1')
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='stem1_bn')
        net = Conv2d(net, 64, (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='stem2')
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='stem2_bn')
        net = Conv2d(net, 128, (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='stem3')
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='stem3_bn')
        net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), name='stem3_pool')
        net = denseblock(net, blocknum=6, step=48, firstchannel=192, is_train=is_train, name='denseblock0', reuse=reuse)
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='denseblock0_bn')
        net = Conv2d(net, 416, (1, 1), (1, 1), padding='SAME',
                     W_init=W_init, name='denseblock0_cnn')
        net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), name='denseblock0_pool')
        net = denseblock(net, blocknum=8, step=48, firstchannel=192, is_train=is_train, name='denseblock1', reuse=reuse)
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='denseblock1_bn')
        net = Conv2d(net, 800, (1, 1), (1, 1), padding='SAME',
                     W_init=W_init, name='denseblock1_cnn')
        netfirst=BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='feature_first_bn')
        net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), name='denseblock2_pool1')
        net = denseblock(net, blocknum=8, step=48, firstchannel=192, is_train=is_train, name='denseblock2', reuse=reuse)
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='denseblock2_bn')
        net = Conv2d(net, 1184, (1, 1), (1, 1), padding='SAME',
                     W_init=W_init, name='denseblock2_cnn')
        net = denseblock(net, blocknum=8, step=48, firstchannel=192, is_train=is_train, name='denseblock3', reuse=reuse)
        net = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='denseblock3_bn')
        net = Conv2d(net, 256, (1, 1), (1, 1), padding='SAME',
                     W_init=W_init, name='denseblock2_cnna')
        netpl=MaxPool2d(netfirst, filter_size=(2, 2), strides=(2, 2), name='First_pool')
        netpl=BatchNormLayer(netpl, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='First_bn')
        netpl = Conv2d(netpl, 256, (1, 1), (1, 1), padding='SAME',
                     W_init=W_init, name='denseblock2_cnnb')
        net=ConcatLayer([net,netpl],-1,"Second_Cat")
        netsecond = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu, name='feature_second_bn')
        net = denseblockpl(net, step=256, firstchannel=256, is_train=is_train, name='denseplz1', reuse=reuse)
        netthird = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu,
                                   name='feature_third_bn')
        net = denseblockpl(net, step=128, firstchannel=128, is_train=is_train, name='denseplz2', reuse=reuse)
        netfourth = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu,
                                   name='feature_fourth_bn')
        net = denseblockpl(net, step=128, firstchannel=128, is_train=is_train, name='denseplz3', reuse=reuse)
        netfifth = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu,
                                   name='feature_fifth_bn')
        net = denseblockfin(net, step=128, firstchannel=128, is_train=is_train, name='denseplz4', reuse=reuse)
        netsixth = BatchNormLayer(net, is_train=is_train, decay=conv_bn_decay, act=tf.nn.relu,
                                   name='feature_sixth_bn')
        outfirst=Conv2d(netfirst, default_box_size[0] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='firstout')
        outsecond=Conv2d(netsecond, default_box_size[1] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='secondout')
        outthird=Conv2d(netthird, default_box_size[2] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='thirdout')
        outfourth=Conv2d(netfourth, default_box_size[3] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='fourthout')
        outfifth=Conv2d(netfifth, default_box_size[4] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='fifthout')
        outsixth=Conv2d(netsixth, default_box_size[5] * (classes_size + 4), (3, 3), (1, 1), padding='SAME',
                     W_init=W_init, name='sixthout')
        features1=outfirst.outputs
        features2=outsecond.outputs
        features3=outthird.outputs
        features4=outfourth.outputs
        features5=outfifth.outputs
        features6=outsixth.outputs
        feature_maps = [features1, features2, features3, features4, features5,features6]
        global feature_maps_shape
        feature_maps_shape = [m.get_shape().as_list() for m in feature_maps]
        tmp_all_feature = []
        for i, fmap in zip(range(len(feature_maps)), feature_maps):
            width = feature_maps_shape[i][1]
            height = feature_maps_shape[i][2]
            tmp_all_feature.append(
                tf.reshape(fmap, [-1, (width * height * default_box_size[i]), (classes_size + 4)]))
        tmp_all_feature = tf.concat(tmp_all_feature, axis=1)
        feature_class = tmp_all_feature[:, :, :classes_size]
        feature_location = tmp_all_feature[:, :, classes_size:]
        print('##   feature_class shape : ' + str(feature_class.get_shape().as_list()))
        print('##   feature_location shape : ' + str(feature_location.get_shape().as_list()))
        # 生成所有default boxs
        global all_default_boxs
        all_default_boxs = generate_all_default_boxs()
        # print(all_default_boxs)
        global all_default_boxs_len
        all_default_boxs_len = len(all_default_boxs)
        print('##   all default boxs : ' + str(all_default_boxs_len))
    return feature_class,feature_location,all_default_boxs,all_default_boxs_len

def smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

def elloss(feature_class,feature_location,groundtruth_class,groundtruth_location,groundtruth_positives,groundtruth_count):
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                           labels=groundtruth_class)
    loss_location = tf.div(tf.reduce_sum(tf.multiply(
        tf.reduce_sum(smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                      reduction_indices=2), groundtruth_positives), reduction_indices=1),
        tf.reduce_sum(groundtruth_positives, reduction_indices=1))
    loss_class = tf.div(
        tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
        tf.reduce_sum(groundtruth_count, reduction_indices=1))
    loss_all = tf.reduce_sum(tf.add(loss_class, loss_location*5))
    return loss_all,loss_class,loss_location

def generate_all_default_boxs():
    all_default_boxes = []
    for index, map_shape in zip(range(len(feature_maps_shape)), feature_maps_shape):
        width = int(map_shape[1])
        height = int(map_shape[2])
        cell_scale = default_box_scale[index]
        for x in range(width):
            for y in range(height):
                for ratio in box_aspect_ratio[index]:
                    center_x = (x / float(width)) + (0.5 / float(width))
                    center_y = (y / float(height)) + (0.5 / float(height))
                    box_width = cell_scale*np.sqrt(ratio)/1.2
                    box_height = cell_scale/np.sqrt(ratio)/1.2
                    all_default_boxes.append([center_x, center_y, box_width, box_height])
                all_default_boxes.append([(x / float(width)) + (0.5 / float(width)), (y / float(height)) + (0.5 / float(height)), cell_scale*1.5,cell_scale*1.4])
    all_default_boxes = np.array(all_default_boxes)
    all_default_boxes = check_numerics(all_default_boxes, 'all_default_boxes')
    return all_default_boxes

def check_numerics(input_dataset, message):
    if str(input_dataset).find('Tensor') == 0:
        input_dataset = tf.check_numerics(input_dataset, message)
    else:
        dataset = np.array(input_dataset)
        nan_count = np.count_nonzero(dataset != dataset)
        inf_count = len(dataset[dataset == float("inf")])
        n_inf_count = len(dataset[dataset == float("-inf")])
        if nan_count > 0 or inf_count > 0 or n_inf_count > 0:
            data_error = '【' + message + '】出现数据错误！【nan：' + str(nan_count) + '|inf：' + str(
                inf_count) + '|-inf：' + str(n_inf_count) + '】'
            raise Exception(data_error)
    return input_dataset

if __name__ == '__main__':
    imageinput=tf.placeholder(tf.float32,[None,300,300,3],"inputsimage")
    imageinputtest = tf.placeholder(tf.float32, [None, 300, 300, 3], "inputsimage")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    fc, fl, _, _ = inference(imageinput, True, None)

    fc2, fl2, _, _ = inference(imageinputtest, False, True)

    groundtruth_class = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.int32,
                                       name='groundtruth_class')
    groundtruth_location = tf.placeholder(shape=[None, all_default_boxs_len, 4], dtype=tf.float32,
                                          name='groundtruth_location')
    groundtruth_positives = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.float32,
                                           name='groundtruth_positives')
    groundtruth_negatives = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.float32,
                                           name='groundtruth_negatives')
    groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
    learning_rt=0.000001
    learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
    loss_allt, loss_classt, loss_locationt = elloss(fc, fl, groundtruth_class, groundtruth_location, groundtruth_positives, groundtruth_count)
    train = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss_allt)
    tf.summary.scalar('loss_all_train', loss_allt)
    tf.summary.scalar('loss_class_train', tf.reduce_sum(loss_classt) )
    tf.summary.scalar('loss_location_train', tf.reduce_sum(loss_locationt))
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        trainwrite = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        saver2 = tf.train.Saver(var_list=tf.trainable_variables())
        zzz = variables._all_saveable_objects().copy()
        print(zzz)
        saver = tf.train.Saver()
        if os.path.exists('./session_paramsdddaleasy/session2.ckpt.index') :
            print('\nStart Restore')
            saver2.restore(sess, './session_paramsdddaleasy/session2.ckpt')
            print('\nEnd Restore')
        print('\nStart Training')
        min_loss_location = 100000.
        min_loss_class = 100000.
        avg_loss=0
        avg_lossloc=0
        avg_losclass=0
        ptlos=0
        ptlosc=0
        ptlosl=0
        while((min_loss_location + min_loss_class) > 0.001 and running_count < 100000):
            running_count += 1
            train_data, actual_data, _ = get_traindata_voc(batch_size)
            starttime = time.time()
            gt_class, gt_location, gt_positives, gt_negatives=generate_groundtruth_data(actual_data)
            if len(train_data) > 0:
                loss_all,loss_class,loss_location,_,pred_class,pred_location = sess.run([loss_allt, loss_classt, loss_locationt,train,fc, fl],feed_dict={imageinput:train_data,groundtruth_class:gt_class,groundtruth_location:gt_location,groundtruth_positives:gt_positives,groundtruth_negatives:gt_negatives,learning_rate:learning_rt})
                l = np.sum(loss_location)
                c = np.sum(loss_class)
                avg_loss +=loss_all
                avg_lossloc += loss_class
                avg_losclass += loss_location
                if min_loss_location > l:
                    min_loss_location = l
                if min_loss_class > c:
                    min_loss_class = c
                print('Running:【' + str(running_count) + '】|Loss All:【' + str(
                    min_loss_location + min_loss_class) + '|' + str(loss_all) + '】|Location:【' + str(
                    np.sum(loss_location)) + '】|Class:【' + str(np.sum(loss_class)) + '】|pred_class:【' + str(
                    np.sum(pred_class)) + '|' + str(np.amax(pred_class)) + '|' + str(
                    np.min(pred_class)) + '】|pred_location:【' + str(np.sum(pred_location)) + '|' + str(
                    np.amax(pred_location)) + '|' + str(np.min(pred_location)) + '】TIME:'+str(time.time()-starttime))
                if running_count % 100 == 0:
                    print('---------')
                    print('avgloss')
                    print(avg_loss/100.)
                    print(np.sum(avg_lossloc/100.) )
                    print(np.sum(avg_losclass/100.) )
                    print(ptlos-avg_loss/100.)
                    print(ptlosc-np.sum(avg_lossloc/100.) )
                    print(ptlosl-np.sum(avg_losclass/100.) )
                    ptlos = avg_loss/100.
                    ptlosc = np.sum(avg_lossloc/100. )
                    ptlosl = np.sum(avg_losclass/100. )
                    print('---------')
                    avg_loss=0
                    avg_lossloc = 0
                    avg_losclass = 0
                if running_count % 100 == 0:
                    results = sess.run(merged,feed_dict={imageinput:train_data,groundtruth_class:gt_class,groundtruth_location:gt_location,groundtruth_positives:gt_positives,groundtruth_negatives:gt_negatives,learning_rate:learning_rt})
                    trainwrite.add_summary(results, running_count)
                if running_count % 500 == 0:
                    saver.save(sess, './session_paramsdddaleasy/session.ckpt')
                    print('session.ckpt has been saved.')
                    gc.collect()
            else:
                print('No Data Exists!')
                break