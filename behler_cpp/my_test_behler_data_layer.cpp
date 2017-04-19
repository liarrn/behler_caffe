#include <map>
#include <string>
#include <vector>
#include<stdio.h>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/behler_data_layer.hpp"
//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/io.hpp"

using namespace caffe;
using namespace std;

template<typename Dtype>
float blob_l1(Blob<Dtype> *b)
{
	int count = b->count();
	float res = 0.0;
	for (int i = 0; i < count; i++)
		res += abs(b->cpu_data()[i]);
	return res;
}

template<typename Dtype>
void blob_print(Blob<Dtype> *b)
{
	int count = b->count();
	float res = 0.0;
	for (int i = 0; i < count; i++)
		cout << b->cpu_data()[i] << "  ";
}

int main()
{
	string data_source, label_source;
	Blob<float>* const blob_top_data_(new Blob<float>());
	Blob<float>* const blob_top_label_(new Blob<float>());
	vector<Blob<float>*> blob_bottom_vec_;
	vector<Blob<float>*> blob_top_vec_;
	blob_top_vec_.push_back(blob_top_data_);
	blob_top_vec_.push_back(blob_top_label_);

	data_source = "E:\\iris\\Belher_data\\test-features.dat";
	label_source = "E:\\iris\\Belher_data\\test-labels.dat";

	LayerParameter param;
	BehlerDataParameter* behler_data_param = param.mutable_behler_data_param();
	behler_data_param->set_data_source(data_source.c_str());
	behler_data_param->set_label_source(label_source.c_str());
	behler_data_param->set_batch_size(7);
	behler_data_param->set_num_data_rows(17);
	behler_data_param->set_num_data_cols(4);
	behler_data_param->set_num_label_rows(6);
	behler_data_param->set_num_label_cols(2);

	BehlerDataLayer<float> layer(param);
	layer.SetUp(blob_bottom_vec_, blob_top_vec_);
	
	vector<float> res;
	for (int i = 0; i < 10; i++)
	{
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		// res.push_back(blob_l1<float>(blob_top_data_));
		// res.push_back(blob_l1<float>(blob_top_label_));
		blob_print<float>(blob_top_data_);
		//blob_print<float>(blob_top_label_);
	}

	delete blob_top_data_;
	delete blob_top_label_;
	return 0;
}
