#!/usr/bin/python

import numpy as np
import sys
caffe_root='/opt/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe

def save_input_data(transformed_image, input_feature_path="input.caffefeature"):
    import os
    if os.path.isfile(input_feature_path):
        print("[ERROR] feature file {} existed, please rename input_feature_path"\
              .format(input_feature_path))
        exit(1)
    print("[INFO] save input's feature data to {}".format(input_feature_path))
    print("[INFO] save input data ......")
    with open(input_feature_path, "w") as f:
        for i in transformed_image:
            for j in i:
                for k in j:
                    line = '%.8f\n' % k
                    f.write(line)
    print("[INFO] input data saved successfully in {}"\
          .format(input_feature_path))
    return

def save_feature_data(net, layer_name, save_feature_path):
    import os
    if os.path.isfile(save_feature_path):
        print("[ERROR] feature file {} existed, please rename save_feature_path"\
              .format(save_feature_path))
        exit(1)
    print("[INFO] save {}'s feature data to {}".format(layer_name, save_feature_path))
    print("[INFO] save {}'s feature data ......"\
          .format(layer_name))
    feature = net.blobs[layer_name].data[0].flatten()
    feature_list = map(lambda f: "".join([str(f), "\n"]), feature)
    with open(save_feature_path, "w") as f:
        f.writelines(feature_list)
    print("[INFO] {} data saved successfully in {}"\
          .format(layer_name, save_feature_path))

    return

def run_model(prototxt_file, caffemodel_path, image_path,\
              input_shape_dict, mean_npy_path, set_mean=True):
    # load model
    net = caffe.Net(prototxt_path,\
                    caffemodel_path,\
                    caffe.TEST)

    # set data transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    if input_shape_dict['c']==3:
        transformer.set_transpose('data', (2,0,1))

    if set_mean:
        mu = np.load(mean_npy_path)
        mu = mu.mean(1).mean(1)
        transformer.set_mean('data', mu)

    transformer.set_raw_scale('data', 255)
    if input_shape_dict['c']==3:
        transformer.set_channel_swap('data', (2,1,0))
    '''
    net.blobs['data'].reshape(input_shape_dict['c'],\
                              input_shape_dict['h'],\
                              input_shape_dict['w'])
    
    net.blobs['data'].reshape(input_shape_dict['n'],\
                              input_shape_dict['c'],\
                              input_shape_dict['h'],\
                              input_shape_dict['w'])
    '''
    # load image
    color = True if input_shape_dict['c']>1 else False
    image = caffe.io.load_image(image_path, color=color)
    transformed_image = transformer.preprocess('data', image)

    # feed transformed image to net
    print("type(transformed_image:{})".format(type(transformed_image)))
    print("transformed_image.shape:{})".format(transformed_image.shape))
    #net.blobs['data'].data[...] = transformed_image.reshape(1,1,1000,750)

    net.blobs['data'].data[...] = transformed_image
    net.forward()
    #output = net.forward()
    return transformed_image, net

if __name__ == "__main__":
    # init params 
    image_path = '/home/yuanshuai/cat.png'
    input_feature_path = "./cat_mobilenetv2_224x224.dat"
    layer_name_and_save_path_dict = {"pool6": "./mobilenetv2_result_pool6.dat",
                                     "fc7": "./mobilenetv2_result_fc7.dat",
                                     "prob": "./mobilenetv2_result_prob.dat"}
    prototxt_path='./mobilenetv2.prototxt'
    caffemodel_path='./mobilenetv2.caffemodel'
    input_shape_dict = {"n":1, "c":3, "h":224, "w":224}
    mean_npy_path = '/opt/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    set_mean = True

    # load model, feed model, run model
    transformed_image, net = run_model(prototxt_path,\
                                       caffemodel_path,\
                                       image_path,\
                                       input_shape_dict,\
                                       mean_npy_path,\
                                       set_mean=set_mean)

    # save data and feature
    save_input_data(transformed_image, input_feature_path)

    for layer_name in layer_name_and_save_path_dict.keys():
        save_path = layer_name_and_save_path_dict[layer_name]
        save_feature_data(net, layer_name, save_path)

#'''
   

#'''
#output_prob = output['prob'][0]
#output_prob = output['fc7'][0]

#print 'predicted class is:', output_prob.argmax()

#for layer_name, blob in net.blobs.iteritems():
#	print layer_name + '\t' + str(blob.data.shape)


#prob = net.blobs['prob'].data[0].flatten()
#prob = net.blobs['fc7'].data[0].flatten()
#prob = net.blobs['prob'].data[0].flatten()


#for i in range(len(prob)):
#    print(str(i) +"\t" + '%.8f'%prob[i])
#print len(prob)
