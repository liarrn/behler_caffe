#include <vector>

#include "caffe/layers/behler_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	//template <typename Dtype>
	//void LossLayer<Dtype>::LayerSetUp(
	//	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	//{
	//	LossLayer<Dtype>::LayerSetUp(bottom, top);
	//}

	template <typename Dtype>
	void BehlerEuclideanLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
		diff_.Reshape(bottom[1]->shape(0), 1, 1, 1);
	}

	template <typename Dtype>
	void BehlerEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		num_data_rows_ = bottom[0]->shape(0);
		num_label_rows_ = bottom[1]->shape(0);
		CHECK_EQ(bottom[0]->shape(1), 1) << "data should have num_cols EQ 1";
		CHECK_EQ(bottom[1]->shape(1), 2) << "label should have num_cols EQ 2";

		// construct data_label_map_
		data_label_map_.reset(new vector<int>);
		data_label_map_->push_back(0);
		for (int i = 1; i <= num_label_rows_; i++)
			data_label_map_->push_back((int)bottom[1]->cpu_data()[(i - 1) * 2] + (*data_label_map_.get())[i - 1]);
		CHECK_EQ((*data_label_map_.get())[data_label_map_->size() - 1], num_data_rows_) << "data and label does not consist";

		Dtype *pred_epc = new Dtype[num_label_rows_]; // predicted energy per cluster 
		for (int i = 1; i <= num_label_rows_; i++)
		{
			pred_epc[i - 1] = 0.0;
			for (int j = (*data_label_map_.get())[i - 1]; j < (*data_label_map_.get())[i]; j++)
				pred_epc[i - 1] += bottom[0]->cpu_data()[j];
		}
		Dtype *gt_epc = new Dtype[num_label_rows_]; // ground energy per cluster 
		for (int i = 0; i < num_label_rows_; i++)
			gt_epc[i] = bottom[1]->cpu_data()[i*2+1];

		int count = num_label_rows_;
		caffe_sub(
			count,
			pred_epc,
			gt_epc,
			diff_.mutable_cpu_data());
		Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
		Dtype loss = dot / num_data_rows_ / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
		delete[] pred_epc;
		delete[] gt_epc;

	}

	template <typename Dtype>
	void BehlerEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
	{
		// only propagate through bottom[0]
		const Dtype alpha = 1 * top[0]->cpu_diff()[0] / num_data_rows_;
		Dtype *depc = new Dtype[num_label_rows_]; // de per cluster
		caffe_cpu_axpby(
			num_label_rows_,              // count
			alpha,                              // alpha
			diff_.cpu_data(),                   // a
			Dtype(0),                           // beta
			depc);  // b
		for (int i = 1; i <= num_label_rows_; i++)
			for (int j = (*data_label_map_.get())[i - 1]; j < (*data_label_map_.get())[i]; j++)
				bottom[0]->mutable_cpu_diff()[j] = depc[i-1];
		delete[] depc;
		
	}

#ifdef CPU_ONLY
	STUB_GPU(BehlerEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(BehlerEuclideanLossLayer);
REGISTER_LAYER_CLASS(BehlerEuclideanLoss);

}  // namespace caffe
