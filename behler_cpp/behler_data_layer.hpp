#ifndef CAFFE_BEHLER_DATA_LAYER_HPP_
#define CAFFE_BEHLER_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Provides data to the Net from csv file
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class BehlerDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit BehlerDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~BehlerDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BehlerData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual void load_batch(Batch<Dtype>* batch);
		void csv_parser(std::istream& is, Dtype *data, int num_row, int num_column);
		int binary_search(int *data, int lo, int hi, int target);
		
		int num_data_rows_, num_data_cols_, num_label_rows_, num_label_cols_;
		Dtype* data_;
		Dtype* label_;
		int data_id_, label_id_;
		int *data_label_map_;
	};


}  // namespace caffe

#endif  // CAFFE_BEHLER_DATA_LAYER_HPP_
