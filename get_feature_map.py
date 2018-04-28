#!/usr/bin/python
# -- coding: utf-8 --

import numpy as np
import skimage.io
import sys,os
import caffe

caffe_root = "/opt/caffe"
caffe_python_path = os.path.join(caffe_root, 'python')
sys.path.insert(0, caffe_python_path)


def init_params(input_n, input_c, input_h, input_w, input_dat_path, prototxt_path, caffemodel_path):
    input_shape_dict = {'n': input_n,\
                        'c': input_c,\
                        'h': input_h,\
                        'w': input_w}
    params_dict = {}
    params_dict['input_shape_dict'] = input_shape_dict
    params_dict['caffemodel_path'] = caffemodel_path
    params_dict['prototxt_path'] = prototxt_path
    params_dict['input_dat_path'] = input_dat_path
    return params_dict


def load_model(layer_name_str, input_shape_dict, prototxt_path, caffemodel_path):
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    net.blobs['data'].reshape(input_shape_dict['n'],\
                              input_shape_dict['c'],\
                              input_shape_dict['h'],\
                              input_shape_dict['w'])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    # transformer.set_raw_scale('data', 255)

    data_np = np.loadtxt(input_dat_path) #load data dir->numpy 
    img_np = data_np.reshape([input_shape_dict['h'],\
                              input_shape_dict['w'],\
                              input_shape_dict['c']])
    #img=caffe.io.load_image(os.path.join('./','yuanshuai/a.jpg'),False)  #load image -> numpy

    #numpy -> caffe-data
    img_np = caffe.io.resize_image(img_np, (input_shape_dict['h'], input_shape_dict['w']))

    #transeform the caffe-data
    net.blobs['data'].data[...] = transformer.preprocess('data', img_np)

    feature_map = net.blobs[layer_name_str].data.flatten()
    # for last feature map
    # out = net.forward()
    # output_feauture_map = out[layer_name_str].flatten()
    return feature_map


if __name__ == "__main__":
    # init
    output_layer_name_str = "conv3"
    output_feature_map_save_path = ".".join([output_layer_name_str, "dat"])
    input_n, input_c, input_h, input_w = 1, 3, 224, 224
    caffemodel_path = './ResNet-50-model.caffemodel'
    prototxt_path = './ResNet-50-deploy.prototxt'
    input_dat_path = './preproccessed_warship_224.dat'

    # pack params
    params_dict = init_params(input_n, input_c, input_h, input_w,\
                              input_dat_path,\
                              prototxt_path,\
                              caffemodel_path)

    # compute and save middle feature map
    feature_map = load_model(output_layer_name_str,\
                             params_dict['input_shape_dict'],\
                             params_dict['prototxt_path'],\
                             params_dict['caffemodel_path'])
    np.savetxt(output_feature_map_save_path,\
               feature_map,\
               fmt='%.6f',\
               delimiter='\n')
