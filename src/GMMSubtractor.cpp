#include "GMMSubtractor.h"
#include "GMMSubtractorImpl1.h"
#include "GMMSubtractorImpl3.h"
#include "GMMSubtractorImpl4.h"

#include <iostream>

GMMSubtractor::GMMSubtractor(const unsigned int in_num_gaussians)
	: m_impl(nullptr), m_num_gaussians(in_num_gaussians)
{
}

GMMSubtractor::~GMMSubtractor()
{
	ReleaseImplementation();
}

void GMMSubtractor::Train(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_num_channels, const unsigned int in_img_step)
{
	ReleaseImplementation();
	switch (in_num_channels)
	{
		case 1:
			m_impl = new GMMSubtractorImpl1;
			break;
		case 3:
			m_impl = new GMMSubtractorImpl3;
			break;
		case 4:
			m_impl = new GMMSubtractorImpl4;
			break;
		default:
			std::cerr << "Unsupported number of channels in the image!" << std::endl;
			exit(-1);
	}
	m_impl->SetModelSize(m_num_gaussians);
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

void GMMSubtractor::ReleaseImplementation()
{
	if (m_impl != nullptr)
	{
		delete m_impl;
		m_impl = nullptr;
	}
}
