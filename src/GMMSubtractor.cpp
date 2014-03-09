#include "GMMSubtractor.h"

GMMSubtractor::GMMSubtractor(const unsigned int in_num_gaussians)
{

}

GMMSubtractor::~GMMSubtractor()
{

}

void GMMSubtractor::Train(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_num_channels, const unsigned int in_img_step)
{
	/*
	 * Construct implementation
	 */

  m_impl->Train(in_img, in_num_rows, in_num_cols, in_img_step);
}

void GMMSubtractor::Subtract(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_num_channels, const unsigned int in_img_step,
	unsigned char *out_mask, const unsigned int in_mask_step)
{
	m_impl->Subtract(in_img, in_num_rows, in_num_cols, in_img_step, out_mask,
		in_mask_step);
}

void GMMSubtractor::SetLearningRate(const float in_learning_rate)
{
	m_impl->SetLearningRate(in_learning_rate);
}

void GMMSubtractor::SetBackgroundThreshold(const float in_threshold)
{
	m_impl->SetBackgroundThreshold(in_threshold);
}
