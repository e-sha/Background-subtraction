#include "IGMMSubtractorImpl.h"

IGMMSubtractorImpl::IGMMSubtractorImpl()
{
	m_model_is_trained = true;
}

IGMMSubtractorImpl::~IGMMSubtractorImpl()
{
}

void IGMMSubtractorImpl::Train(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step)
{
	ReleaseModel();
	TrainUnsafe(in_img, in_num_rows, in_num_cols, in_img_step);
	m_model_is_trained = true;
}

void IGMMSubtractorImpl::Subtract(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step, unsigned char *out_mask,
	const unsigned int in_mask_step)
{
	if (m_model_is_trained)
	{
		SubtractUnsafe(in_img, in_num_rows, in_num_cols, in_img_step, out_mask,
			in_mask_step);
	}
	else
	{
		std::cerr << "Cannot subtract background. Model was not trained!" 
			<< std::endl;
		exit(-1);
	}
}

void IGMMSubtractorImpl::SetLearningRate(const float in_learning_rate)
{
	m_learning_rate = in_learning_rate;
}

void IGMMSubtractorImpl::SetBackgroundThreshold(const float in_threshold)
{
	m_threshold = in_threshold;
}

void IGMMSubtractorImpl::SetModelSize(const unsigned int in_num_gaussians)
{
	m_model_is_trained = false;
	ReleaseModel();
	m_num_gaussians = in_num_gaussians;
}

inline void IGMMSubtractorImpl::ReleaseModel()
{
	if ((m_mean_array != nullptr) || (m_covariation_array != nullptr))
		FreeModel();
	m_mean_array = nullptr;
	m_covariation_array = nullptr;
}
