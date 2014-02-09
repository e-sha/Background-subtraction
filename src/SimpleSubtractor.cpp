#include "SimpleSubtractor.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "SimpleSubtractorImpl1.h"
#include "SimpleSubtractorImpl3.h"
#include "SimpleSubtractorImpl4.h"

SimpleSubtractor::SimpleSubtractor(const float in_threshold):
  m_model(nullptr), m_num_rows(0), m_num_cols(0), m_step(0), m_num_channels(0),
	m_impl(nullptr)
{
  SetThreshold(in_threshold);
}

SimpleSubtractor::~SimpleSubtractor()
{
  if (m_model != nullptr)
    delete [] m_model;
	if (m_impl != nullptr)
		delete m_impl;
}
    
void SimpleSubtractor::Train(const unsigned char *in_img,
  const unsigned int in_num_rows, const unsigned int in_num_cols,
  const unsigned int in_num_channels, const unsigned int in_step)
{
  switch (in_num_channels)
  {
    case 1:
      m_impl = new SimpleSubtractorImpl1(m_threshold);
      break;
    case 3:
      m_impl = new SimpleSubtractorImpl3(m_threshold);
			break;
    case 4:
      m_impl = new SimpleSubtractorImpl4(m_threshold);
      break;
    default:
      std::cerr << "Wrong number of channels in the input image" << std::endl;
      exit(-1);
  }
  m_impl->Train(in_img, in_num_rows, in_num_cols, in_step);
}

void SimpleSubtractor::Subtract(const unsigned char *in_img,
  const unsigned int in_num_rows, const unsigned int in_num_cols,
  const unsigned int in_num_channels, const unsigned int in_img_step,
 	unsigned char *out_mask, const unsigned int in_mask_step)
{
	m_impl->Subtract(in_img, in_num_rows, in_num_cols, in_img_step, out_mask, in_mask_step);
}

void SimpleSubtractor::SetThreshold(const float in_threshold)
{
  m_threshold = in_threshold * in_threshold;
}
