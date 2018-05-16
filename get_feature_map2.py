import numpy as np

import sys
caffe_root='/opt/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe

model_path='/home/yuanshuai/code/inferx_model/mobilenet/caffe/mobilenetv1.prototxt'
pretrained_path='/home/yuanshuai/code/inferx_model/mobilenet/caffe/mobilenetv1.caffemodel'
net = caffe.Net(model_path,pretrained_path,caffe.TEST)

mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
#image_path = '/home/yuanshuai/code/inferx_model/squeezenet/cat_perfxlab.jpg'
image_path = '/home/yuanshuai/cat.png'

# squeezenet 227
#net.blobs['data'].reshape(1, 3, 227, 227)
# mobilenet 224
net.blobs['data'].reshape(1, 3, 224, 224)

#image = caffe.io.load_image(caffe_root + '/cat.png')
image = caffe.io.load_image(image_path)
transformed_image = transformer.preprocess('data', image)

#'''
f = open("./cat_caffe_mobilenet224.dat", "w")
for i in transformed_image:
    for j in i:
        for k in j:
            f.write(str('%.8f'%k))
            f.write("\n")
f.close()
#'''
net.blobs['data'].data[...] = transformed_image
output = net.forward()
#output_prob = output['prob'][0]
#output_prob = output['fc7'][0]

#print 'predicted class is:', output_prob.argmax()

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


prob = net.blobs['prob'].data[0].flatten()
#prob = net.blobs['fc7'].data[0].flatten()
#prob = net.blobs['prob'].data[0].flatten()
#prob = net.blobs['conv1'].data[0].flatten()
#prob = net.blobs['conv2_1/dw'].data[0].flatten()
#prob = net.blobs['conv6/sep'].data[0].flatten()
#prob = net.blobs['pool6'].data[0].flatten()

print("==================")
for i in range(len(prob)):
    print(str(i) +"\t" + '%.8f'%prob[i])
#print len(prob)
