import caffe
import yaml
import numpy as np

class BehlerDataLayer(caffe.Layer):
    '''
    BehlerDataLayer
    '''
    def setup(self, bottom, top):
        assert len(top) == 2, 'requires two layer.top'
        params = yaml.load(self.param_str)
        self._feature_path = params['feature_path']
        self._label_path = params['label_path']
        self._batch_size = params['batch_size']
        
        self._features = np.loadtxt(self._feature_path, delimiter=',')
        self._labels = np.loadtxt(self._label_path, delimiter=',')
        
        self._feature_dim = self._features.shape[1]
        return

    def reshape(self, bottom, top):
        top[0].reshape(*(1, self._feature_dim))
        top[1].reshape(*(1, 2))
        return

    def forward(self, bottom, top):
        print 'iteration: '
        top[0].reshape(*(self._batch_size, self._feature_dim))
        top[0].data[...] = self._features[:self._batch_size, :]
        print top[0].data
        return

    def backward(self, bottom, top):
        pass

