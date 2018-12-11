#!/usr/bin/python

import os
import time
import numpy as np
import sys
caffe_root = "/home/houxin/caffe-ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe

DEBUG = True

def save_caffemodel(weight_list, save_prefix_dir=None):
    # create model directory used to save weights
    if save_prefix_dir == None:
        date = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        save_prefix_dir = "".join(["model", "-", date])
    if not os.path.exists(save_prefix_dir):
        os.mkdir(save_prefix_dir)
        print("[INFO] create directory {}".format(save_prefix_dir))

    # for-loop get weight of each layer
    for widx in xrange(len(weight_list)):
        weight_dict = weight_list[widx]

        blob_name    = weight_dict["blob_name"]
        blob_num     = weight_dict["blob_num"]
        filter_      = weight_dict["filter"]
        filter_shape = weight_dict["filter_shape"]
        filter_count = weight_dict["filter_count"]
        bias         = weight_dict["bias"]
        bias_shape   = weight_dict["bias_shape"]
        bias_count   = weight_dict["bias_count"]

        print("[INFO] ----- {} -----".format(widx))
        print("[INFO] blob_name:{}".format(blob_name))
        print("[INFO] blob_num:{}".format(blob_name))
        print("[INFO] filter_shape:{}".format(filter_shape))
        print("[INFO] filter_count:{}".format(filter_count))
        print("[INFO] bias_shape:{}".format(bias_shape))
        print("[INFO] bias_count:{}".format(bias_count))

        for bidx in xrange(weight_dict["blob_num"]):
            blob_data = []
            weight_save_file_path_str_list = [save_prefix_dir, "/", widx, "_", blob_name, "_", ]
            weight_save_file_path_str_list = map(str, weight_save_file_path_str_list)
            if bidx == 0: # filter
                blob_data = filter_
                weight_save_file_path_str_list.append("filter")
            elif bidx == 1: # bias
                blob_data = bias
                weight_save_file_path_str_list.append("bias")

            weight_save_file_path = "".join(weight_save_file_path_str_list)
            with open(weight_save_file_path, "w") as file_handle:
                line_list = map(lambda line: "{}\n".format(str(line)), blob_data)
                file_handle.writelines(line_list)
            print("[INFO] {} saved to {} successfully".format(blob_name, weight_save_file_path))
      

def get_caffemodel_weight_list(caffe_param):
    net = caffe.Net(caffe_param['prototxt'],
                    caffe_param['caffemodel'],
                    caffe.TEST)

    weight_list = []
    #(blob_name,blob_vec[0].data)
    for blob_name, blob_vec in net.params.items():
        if DEBUG:
            print("[DEBUG] ---- {} ----".format(blob_name))
            print("[DEBUG] type(blob_vec):{}".format(type(blob_vec)))
            print("[DEBUG] len(blob_vec):{}".format(len(blob_vec)))

        cur_layer_weight = {}
        cur_layer_weight["blob_name"]    = blob_name
        cur_layer_weight["blob_num"]     = len(blob_vec)
        cur_layer_weight["filter"]       = []
        cur_layer_weight["filter_shape"] = []
        cur_layer_weight["filter_count"] = []
 
        cur_layer_weight["bias"]         = []
        cur_layer_weight["bias_shape"]   = []
        cur_layer_weight["bias_count"]   = []

        # used by reduce func
        def mult(a, b): return a*b

        for bidx in xrange(len(blob_vec)):
            blob = blob_vec[bidx]
            #print(blob.data.shape[:])
            if bidx == 0:
                cur_layer_weight["filter"]       = blob.data[:]
                cur_layer_weight["filter_shape"] = blob.data[:].shape
                cur_layer_weight["filter_count"] = reduce(mult, cur_layer_weight["filter_shape"])
            if bidx == 1:
                cur_layer_weight["bias"]         = blob.data[:]
                cur_layer_weight["bias_shape"]   = blob.data[:].shape
                cur_layer_weight["bias_count"]   = reduce(mult, cur_layer_weight["bias_shape"])
        weight_list.append(cur_layer_weight)
    return weight_list

if __name__ == "__main__":
    caffe_param = {"prototxt": "./ship_detectionOutput.prototxt",
                   "caffemodel": "./ship_detectionOutput.caffemodel"}
    weight_list = get_caffemodel_weight_list(caffe_param)
    save_caffemodel(weight_list)
