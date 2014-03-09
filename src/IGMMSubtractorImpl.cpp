#include "IGMMSubtractorImpl.h"

IGMMSubtractorImpl::IGMMSubtractorImpl()
{
}

IGMMSubtractorImpl::~IGMMSubtractorImpl()
{
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
