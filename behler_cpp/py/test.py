import caffe

net_path = './behler.prototxt'
net = caffe.Net(net_path, caffe.TRAIN)
print net.blobs['feature'].data
net.forward()
print net.blobs['feature'].data
