#ifndef CAFFE_BEHLER_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_BEHLER_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	class BehlerEuclideanLossLayer : public LossLayer<Dtype> {
	public:
		explicit BehlerEuclideanLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		//virtual void layersetup(
		//	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BehlerEuclideanLoss"; }

	protected:
		/// @copydoc BehlerEuclideanLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> diff_;
		shared_ptr<vector<int> > data_label_map_;
		int num_data_rows_;
		int num_label_rows_;
	};

}  // namespace caffe

#endif  // CAFFE_BEHLER_EUCLIDEAN_LOSS_LAYER_HPP_
