#ifndef CAFFE_STATISTICAL_LOSS_LAYER_HPP_
#define CAFFE_STATISTICAL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class StatisticalLossLayer : public LossLayer<Dtype> {
	public:
		explicit StatisticalLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline const char* type() const { return "StatisticalLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 1;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//   const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		//  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> ones_;
		Blob<Dtype> ones_class_;
		Blob<Dtype> dot_;
		Blob<Dtype> dist_sq_;  // cached for backward pass
		Blob<Dtype> dist_sq_center_;  // cached for backward pass
		//////////////////////////////////////////////////////
		Dtype inter_weight_, intra_weight_, lambda_; //���ڱ�ʾ��������������Ȩֵ
		int num_output_; //���ڴ洢�����Ŀ
		Blob<Dtype> num_label_; //���ڴ洢batch��ÿһ�������������Ŀ
		Blob<Dtype> class_center_; //�洢��ͬ������ĵ�
		//Blob<Dtype> class_center_1;
		Blob<Dtype> batch_center_; //batch���������������ĵ�
		Blob<Dtype> intra_cal_;//cached for intra loss
		Blob<Dtype> per_label_;
		//int M_;
		int N_;
		int K_;
		//////////////////////////////////////////////////////

		Blob<Dtype> class_non_zero_; //�洢����������Ϊ����������
		////////////////////////////////////////////////////////////
		//Blob<Dtype> class_center_; //�洢��ͬ������ĵ�
		Blob<Dtype> center_dot_;
		///////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		Blob<Dtype> whole_dot_; //����batch���ĵ�͸��������ĵ�֮��ĵ��
		Blob<Dtype> whole_inn_; //����batch���ĵ��ڻ�
		///////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		Blob<Dtype> loss_pos_; 
		Blob<Dtype> loss_neg_;
		Blob<Dtype> loss_div_;
		///////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////
		Blob<Dtype> scatter_class_;
		Blob<Dtype> dist_scatter_;
		Blob<Dtype> sample_each_;
		Blob<Dtype> center_each_;
		Blob<Dtype> dist_sample_center_;
		Blob<Dtype> dist_center_center_;
		Blob<Dtype> plus_scatter_scatter_;
		Blob<Dtype> mend_;
		Blob<Dtype> whole_inter_;
		Blob<Dtype> loss_inter_;
		Blob<Dtype> loss_class_inter_;
		Blob<Dtype> tend_inter_;
		Blob<Dtype> eyes_;
		Blob<Dtype> mend_loss_inter_;
		Blob<Dtype> save_scatter_matrix_;
		///////////////////////////////////////////////////////////
	};

}  // namespace caffe

#endif  // CAFFE_STATISTICAL_LOSS_LAYER_HPP_