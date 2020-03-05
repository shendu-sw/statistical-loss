#include <algorithm>
#include <vector>

#include "caffe/layers/statistical_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StatisticalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  N_ = bottom[0]->num(); 
  K_ = bottom[0]->channels(); 

  ///////////////////////////////////////////////////////////////////////////
  inter_weight_ = this->layer_param_.statistical_loss_param().inter_weight();
  intra_weight_ = this->layer_param_.statistical_loss_param().intra_weight();
  lambda_ = this->layer_param_.statistical_loss_param().lambda();
  num_output_ = this->layer_param_.statistical_loss_param().num_output();
  num_label_.Reshape(num_output_, 1, 1, 1);
  class_center_.Reshape(num_output_, bottom[0]->channels(), 1, 1);
  //class_center_1.Reshape(num_output_, bottom[0]->channels(), 1, 1);
  batch_center_.Reshape(bottom[0]->channels(), 1, 1, 1);

  dot_.Reshape(num_output_, bottom[0]->num(), 1, 1);
  ones_.Reshape(bottom[0]->num(), 1, 1, 1);  // n by 1 vector of ones.
  for (int i=0; i < bottom[0]->num(); ++i){
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
  ones_class_.Reshape(num_output_, 1, 1, 1);  // n by 1 vector of ones.
  for (int i = 0; i < num_output_; ++i){
	  ones_class_.mutable_cpu_data()[i] = Dtype(1);
  }
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_center_.Reshape(num_output_, 1, 1, 1);
  intra_cal_.Reshape(num_output_, 1, 1, 1);
  per_label_.Reshape(num_output_, bottom[0]->num(), 1, 1);
  center_dot_.Reshape(num_output_, num_output_, 1, 1);
  whole_dot_.Reshape(num_output_, 1, 1, 1);
  whole_inn_.Reshape(1, 1, 1, 1);
  loss_pos_.Reshape(K_, 1, 1, 1);
  loss_neg_.Reshape(num_output_, K_, 1, 1);
  loss_div_.Reshape(K_, 1, 1, 1);
  class_non_zero_.Reshape(1, 1, 1, 1);
  ///////////////////////////////////////////////////////////////////////////
  scatter_class_.Reshape(num_output_, K_, K_, 1);//scatter matrix
  dist_scatter_.Reshape(K_, K_, 1, 1);
  sample_each_.Reshape(K_, 1, 1, 1);
  center_each_.Reshape(K_, 1, 1, 1);
  dist_sample_center_.Reshape(K_, 1, 1, 1);
  dist_center_center_.Reshape(K_, 1, 1, 1);
  mend_.Reshape(K_, 1, 1, 1);
  plus_scatter_scatter_.Reshape(K_, K_, 1, 1);
  whole_inter_.Reshape(1, 1, 1, 1);
  mend_loss_inter_.Reshape(K_, 1, 1, 1);
  loss_inter_.Reshape(1, 1, 1, 1);
  loss_class_inter_.Reshape(K_, K_, 1, 1);
  tend_inter_.Reshape(K_, 1, 1, 1);
  eyes_.Reshape(K_, K_, 1, 1);
  ///////////////////////////////////////////////////////////////////////////
} 

template <typename Dtype>
void StatisticalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  ///////////////////////////////////////////////////////////////////////////////////////
  const Dtype* bottom_data = bottom[0]->cpu_data();

  //calculate sample number in each class
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* num_label = num_label_.mutable_cpu_data();
  caffe_set(num_output_, (Dtype)0., num_label_.mutable_cpu_data());
  for (int i = 0; i < N_; i++){
	  for (int j = 0; j < num_output_; j++){
		  int s1 = bottom[1]->cpu_data()[i];
		  if (s1 == j){
			  num_label[j] += 1;
		  }
	  }
  }

  //calculate class number which has samples in the batch
  Dtype* class_non_zero = class_non_zero_.mutable_cpu_data();
  caffe_set(1, (Dtype)0., class_non_zero);
  for (int i = 0; i < num_output_; i++){
	  if (num_label[i]>0.001){
		  class_non_zero[0] += 1;
	  }
  }

  //calculate the class mean value in each class
  caffe_set(num_output_ * K_, (Dtype)0., class_center_.mutable_cpu_data());
  for (int i = 0; i < N_; i++){
	  int s1 = bottom[1]->cpu_data()[i];
	  caffe_cpu_axpby(K_, (Dtype)1.0, bottom_data + i*K_, Dtype(1.0), class_center_.mutable_cpu_data() + s1*K_);
  }
  const Dtype* num_lab = num_label_.cpu_data();
  for (int i = 0; i < num_output_; i++){
	  const Dtype mend = static_cast<Dtype>(num_lab[i]);
	  if (mend > 0.001){
		  caffe_scal(K_, (Dtype)1 / mend, class_center_.mutable_cpu_data()+i*K_);
	  }
  }

  //calculate the scatter matrix of each class
  Dtype* scatter_class = scatter_class_.mutable_cpu_data();
  for (int i = 0; i < N_; i++){
	  caffe_copy(K_, bottom_data + i*K_, sample_each_.mutable_cpu_data());
	  int s1 = bottom[1]->cpu_data()[i];
	  caffe_copy(K_, class_center_.cpu_data() + s1*K_, center_each_.mutable_cpu_data());
	  caffe_sub(K_, sample_each_.cpu_data(), center_each_.cpu_data(), dist_sample_center_.mutable_cpu_data());
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, 1, Dtype(1.0), dist_sample_center_.cpu_data(), dist_sample_center_.cpu_data(), (Dtype)0., dist_scatter_.mutable_cpu_data());
	  caffe_cpu_axpby(K_*K_, Dtype(1.0), dist_scatter_.cpu_data(), Dtype(1.0), scatter_class + s1*K_*K_);
  }

  //Compute the optimization term
  Dtype margin = this->layer_param_.statistical_loss_param().margin();
  Dtype loss(0.0);
  const Dtype* bin = bottom[0]->cpu_data();

  const int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); i++){
	  dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (i*channels));
  }
  for (int i = 0; i < num_output_; i++){
	  dist_sq_center_.mutable_cpu_data()[i] = caffe_cpu_dot(channels, class_center_.cpu_data() + (i*channels), class_center_.cpu_data() + (i*channels));
  }
  Dtype dot_scaler(-2.0);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_output_, N_, K_, dot_scaler, class_center_.cpu_data(), bottom_data, (Dtype)0., dot_.mutable_cpu_data());
  // add ||c_i||^2 to all elements in row i
  for (int i = 0; i<num_output_; i++){
	  caffe_axpy(N_, dist_sq_center_.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }
  // add the norm vector to row i
  for (int i = 0; i<num_output_; i++){
	  caffe_axpy(N_, Dtype(1.0), dist_sq_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }
  caffe_set(num_output_, Dtype(0.0), intra_cal_.mutable_cpu_data());
  Dtype* intra_cal = intra_cal_.mutable_cpu_data();
  Dtype* dot = dot_.mutable_cpu_data();
  for (int i = 0; i < num_output_; i++){
	  caffe_copy(N_, dot_.cpu_data()+i*N_, per_label_.mutable_cpu_data());
	  for (int j = 0; j < N_; j++){
		  int s1 = bottom[1]->cpu_data()[j];
		  if (s1 == i){
			  intra_cal[i] += per_label_.mutable_cpu_data()[j];
		  }
	  }
	  if (num_label[i]>0.001){
		  intra_cal[i] = intra_cal[i] / num_label[i] / class_non_zero_.cpu_data()[0];
	  }
  }
  Dtype intra_loss = caffe_cpu_dot(num_output_, ones_class_.cpu_data(), intra_cal_.cpu_data());
  
  //computer the diversity term
  const Dtype* center_data1 = class_center_.cpu_data();
  const Dtype* center_data2 = class_center_.cpu_data();
  Dtype inter_loss(0.0);
  for (int i = 0; i < num_output_; i++){
	  for (int j = 0; j < num_output_; j++){
		  
		  if (num_label[i] >= 1 && num_label[j] >= 1 && (num_label[i] >= 2 || num_label[j] >= 2)){
			 caffe_sub(K_, center_data1 + i*K_, center_data2 + j*K_, dist_center_center_.mutable_cpu_data());
		     caffe_sub(K_, center_data1 + i*K_, center_data2 + j*K_, mend_.mutable_cpu_data());
		     caffe_add(K_*K_, scatter_class_.cpu_data() + i*K_*K_, scatter_class_.cpu_data() + j*K_*K_, plus_scatter_scatter_.mutable_cpu_data());
			 caffe_cpu_sgesv<Dtype>(K_, 1, plus_scatter_scatter_.cpu_data(), mend_.mutable_cpu_data());
		     //caffe_cpu_strsm<Dtype>(CblasRight, CblasUpper, CblasNoTrans, CblasUnit, 1, K_, (Dtype)1, plus_scatter_scatter_.cpu_data(), mend_.mutable_cpu_data());
			 //caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, K_, K_, Dtype(1.0), dist_center_center_.cpu_data(), eyes, (Dtype)0., mend_.mutable_cpu_data());
			 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 1, K_, Dtype(1.0), dist_center_center_.cpu_data(), mend_.cpu_data(), (Dtype)0., whole_inter_.mutable_cpu_data());
		     inter_loss += margin-(num_label[i] + num_label[j] - 2)*num_label[i] * num_label[j] / (num_label[i] + num_label[j]) * whole_inter_.mutable_cpu_data()[0];
		  }  
	  }
  }
  inter_loss = std::max(inter_loss, Dtype(0.0));

  ///////////////////////////////////////////////////////////////////////////////////////
  //computer the gradient
  Dtype* bout = bottom[0]->mutable_cpu_diff();
  // zero initialize bottom[0]->mutable_cpu_diff();
  for (int i = 0; i<N_; i++){
	  caffe_set(K_, Dtype(0.0), bout + i*K_);
  }
  Dtype scaler_pos(2.0);

  //computer the gradient of the optimization term 
  for (int i = 0; i < N_; i++){
      // update x_i
	  const int s1 = static_cast<int>(label[i]);
	  caffe_sub(K_, bottom[0]->cpu_data() + i*K_, class_center_.cpu_data() + s1*K_, loss_pos_.mutable_cpu_data());
	  caffe_axpy(K_, scaler_pos *intra_weight_ / num_label_.cpu_data()[s1] / class_non_zero_.cpu_data()[0], loss_pos_.cpu_data(), bout + i*K_);
  }

  //computer the gradient of the diversity term
  for (int i = 0; i < N_; i++){
	  // update x_i
	  const int s1 = static_cast<int>(label[i]);
	  for (int j = 0; j < num_output_; j++){
		  if (j != s1 && num_label[s1] >= 1 && num_label[j] >= 1 && (num_label[s1] >= 2 || num_label[j] >= 2)){
			  caffe_sub(K_, center_data1 + s1*K_, center_data2 + j*K_, mend_.mutable_cpu_data());
			  caffe_add(K_*K_, scatter_class_.cpu_data() + s1*K_*K_, scatter_class_.cpu_data() + j*K_*K_, plus_scatter_scatter_.mutable_cpu_data());
			  caffe_cpu_sgesv<Dtype>(K_, 1, plus_scatter_scatter_.cpu_data(), mend_.mutable_cpu_data());
			  //caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, 1, K_, Dtype(1.0), plus_scatter_scatter_.cpu_data(), mend_.cpu_data(), (Dtype)0., mend_loss_inter_.mutable_cpu_data());
			  //caffe_cpu_strsm<Dtype>(CblasLeft, CblasUpper, CblasNoTrans, CblasUnit, K_, 1, (Dtype)1, plus_scatter_scatter_.cpu_data(), mend_.mutable_cpu_data());
			  caffe_axpy(K_, Dtype(-2.0) * inter_weight_ *num_label_.cpu_data()[j] * (num_label_.cpu_data()[j] + num_label_.cpu_data()[s1] - 2) / (num_label_.cpu_data()[s1] + num_label_.cpu_data()[j]), mend_.cpu_data(), bout + i*K_);
			  /////////////////////////////////////////////////////////////////////////

			  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 1, K_, Dtype(1.0), mend_.cpu_data(), bottom[0]->cpu_data() + i*K_, (Dtype)0., loss_inter_.mutable_cpu_data());
			  caffe_axpy(K_, Dtype(1.0) * loss_inter_.cpu_data()[0] * inter_weight_ *num_label_.cpu_data()[j] * (num_label_.cpu_data()[s1] - 1) *(num_label_.cpu_data()[j] + num_label_.cpu_data()[s1] - 2) / (num_label_.cpu_data()[s1] + num_label_.cpu_data()[j]), mend_.cpu_data(), bout + i*K_);

			  /////////////////////////////////////////////////////////////////////////

			  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, 1, Dtype(1.0), mend_.cpu_data(), bottom[0]->cpu_data() + i*K_, (Dtype)0., loss_class_inter_.mutable_cpu_data());
			  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, 1, K_, Dtype(1.0), loss_class_inter_.cpu_data(), mend_.cpu_data(), (Dtype)0., tend_inter_.mutable_cpu_data());
			  caffe_axpy(K_, Dtype(1.0) * inter_weight_ *num_label_.cpu_data()[j] * (num_label_.cpu_data()[s1] - 1) *(num_label_.cpu_data()[j] + num_label_.cpu_data()[s1] - 2) / (num_label_.cpu_data()[s1] + num_label_.cpu_data()[j]), tend_inter_.cpu_data(), bout + i*K_);

			  /////////////////////////////////////////////////////////////////////////
		  }

	  }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  loss = intra_loss*intra_weight_ + inter_loss*inter_weight_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void StatisticalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype alpha = top[0]->cpu_diff()[0];
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for (int i = 0; i < num; i++){
    Dtype* bout = bottom[0]->mutable_cpu_diff();
    caffe_scal(channels, alpha, bout + (i*channels));
  }
}

#ifdef CPU_ONLY
STUB_GPU(StatisticalLossLayer);
#endif

INSTANTIATE_CLASS(StatisticalLossLayer);
REGISTER_LAYER_CLASS(StatisticalLoss);

}  // namespace caffe