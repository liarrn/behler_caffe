#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/behler_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	void BehlerDataLayer<Dtype>::csv_parser(std::istream& is, Dtype *data, int num_row, int num_column)
	{
		string line;
		for (int i = 0; i < num_row; i++)
		{
			std::getline(is, line);
			std::stringstream lineStream(line);
			string cell;
			int j = 0;
			while (std::getline(lineStream, cell, ','))
			{
				data[i*num_column + j] = atof(cell.c_str());
				j++;
			}
		}
	}

	template <typename Dtype>
	int BehlerDataLayer<Dtype>::binary_search(int *data, int lo, int hi, int target)
	{
		int mid;
		while (lo <= hi)
		{
			mid = (lo + hi) / 2;
			if (data[mid] == target)
				return mid;
			else if (data[mid] > target)
				hi = mid - 1;
			else if (data[mid] < target)
				lo = mid + 1;
		}
		return hi;
	}

	template <typename Dtype>
	BehlerDataLayer<Dtype>::~BehlerDataLayer<Dtype>() {
		this->StopInternalThread();
		delete[] data_label_map_;
		delete[] data_;
		delete[] label_;
	}

	template <typename Dtype>
	void BehlerDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		data_id_ = 0;
		label_id_ = 0;

		const string& data_source = this->layer_param_.behler_data_param().data_source();
		const string& label_source = this->layer_param_.behler_data_param().label_source();

		num_data_rows_ = this->layer_param_.behler_data_param().num_data_rows();
		num_data_cols_ = this->layer_param_.behler_data_param().num_data_cols();
		num_label_rows_ = this->layer_param_.behler_data_param().num_label_rows();
		num_label_cols_ = this->layer_param_.behler_data_param().num_label_cols();
		data_ = new Dtype[num_data_rows_ * num_data_cols_];
		label_ = new Dtype[num_label_rows_ * num_label_cols_];

		// read data;
		LOG(INFO) << "Opening file " << data_source;
		std::ifstream data_infile(data_source.c_str());
		csv_parser(data_infile, data_, num_data_rows_, num_data_cols_);

		// read label;
		LOG(INFO) << "Opening file " << label_source;
		std::ifstream label_infile(label_source.c_str());
		csv_parser(label_infile, label_, num_label_rows_, num_label_cols_);

		// construct data_label_map_
		data_label_map_ = new int[num_label_rows_];
		data_label_map_[0] = (int)label_[0];
		for (int i = 1; i < num_label_rows_; i++)
			data_label_map_[i] = (int)label_[i*num_label_cols_] + data_label_map_[i - 1];
		CHECK_EQ(data_label_map_[num_label_rows_-1], num_data_rows_) << "data and label does not consist";


		vector<int> top_shape(4, 1);
		top_shape[1] = num_data_cols_;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		top[0]->Reshape(top_shape);

		top_shape[1] = num_label_cols_;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(top_shape);
		}
		top[1]->Reshape(top_shape);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void BehlerDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		const int propose_batch_size = this->layer_param_.behler_data_param().batch_size();
		int batch_size, label_batch_size;
		int next_data_id, next_label_id;
		int propose_next_data_id = data_id_ + propose_batch_size;
		if (propose_next_data_id <= num_data_rows_)
		{
			next_label_id = binary_search(data_label_map_, label_id_, num_label_rows_ - 1, propose_next_data_id);
			next_data_id = data_label_map_[next_label_id];
			batch_size = next_data_id - data_id_;
			next_label_id += 1;
			label_batch_size = next_label_id - label_id_;

			batch->data_.Reshape(batch_size, num_data_cols_, 1, 1);
			caffe_copy(batch->data_.count(), &data_[data_id_ * num_data_cols_], batch->data_.mutable_cpu_data());

			batch->label_.Reshape(label_batch_size, num_label_cols_, 1, 1);
			caffe_copy(batch->label_.count(), &label_[label_id_ * num_label_cols_], batch->label_.mutable_cpu_data());

			data_id_ = next_data_id;
			label_id_ = next_label_id;
		}
		else
		{
			propose_next_data_id %= num_data_rows_;
			next_label_id = binary_search(data_label_map_, 0, num_label_rows_ - 1, propose_next_data_id);
			if (next_label_id == -1)
				next_data_id = 0;
			else
				next_data_id = data_label_map_[next_label_id];
			next_label_id += 1;
			
			batch_size = next_data_id + num_data_rows_ - data_id_;
			label_batch_size = next_label_id + num_label_rows_ - label_id_ ;

			batch->data_.Reshape(batch_size, num_data_cols_, 1, 1);
			caffe_copy((num_data_rows_ - data_id_) * num_data_cols_,
				&data_[data_id_ * num_data_cols_], batch->data_.mutable_cpu_data());
			caffe_copy(next_data_id * num_data_cols_,
				data_, &batch->data_.mutable_cpu_data()[(num_data_rows_ - data_id_) * num_data_cols_]);

			batch->label_.Reshape(label_batch_size, num_label_cols_, 1, 1);
			caffe_copy((num_label_rows_ - label_id_) * num_label_cols_,
				&label_[label_id_ * num_label_cols_], batch->label_.mutable_cpu_data());
			if (next_label_id != 0)
				caffe_copy(next_label_id * num_label_cols_,
					label_, &batch->label_.mutable_cpu_data()[(num_label_rows_ - label_id_) * num_label_cols_]);

			data_id_ = next_data_id;
			label_id_ = next_label_id;
		}
	}

	INSTANTIATE_CLASS(BehlerDataLayer);
	REGISTER_LAYER_CLASS(BehlerData);

} 