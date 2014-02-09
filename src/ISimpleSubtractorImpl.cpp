#include "ISimpleSubtractorImpl.h"

ISimpleSubtractorImpl::ISimpleSubtractorImpl(float in_threshold) :
	m_model(nullptr), m_num_rows(0), m_num_cols(0), m_step(0), m_alpha(0.03)
{
	SetThreshold(in_threshold);
}

ISimpleSubtractorImpl::~ISimpleSubtractorImpl()
{
	FreeModel();
}

void ISimpleSubtractorImpl::Train(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
 	const unsigned int in_img_step)
{
	m_num_rows = in_num_rows;
	m_num_cols = in_num_cols;
	AllocateModel();
	InitializeModel(in_img, in_img_step);
}

void ISimpleSubtractorImpl::SetThreshold(const float in_threshold)
{
	m_threshold = in_threshold * in_threshold;
}

void ISimpleSubtractorImpl::FreeModel()
{
	if (m_model != nullptr)
		free(m_model);
}
