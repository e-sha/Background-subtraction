#include "GMMSubtractorImpl1.h"

GMMSubtractorImpl1::GMMSubtractorImpl1()
{
}

GMMSubtractorImpl1::~GMMSubtractorImpl1()
{
}

void GMMSubtractorImpl1::TrainUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step)
{
	m_mean_step = in_num_cols;
	m_covariance_step = in_num_cols;
	unsigned int num_rows = m_num_rows * m_num_gaussians;
	size_t num_elems = m_mean_step * num_rows;

	m_mean_array = new float[num_elems];
	m_covariation_array = new float[num_elems];

	float *mean_row_ptr, *mean_elem_ptr;
	float *covar_row_ptr, *covar_elem_ptr;
	const unsigned char *img_row_ptr, *img_elem_ptr;
	for (auto row_idx = 0U; row_idx < in_num_rows; ++row_idx)
	{
		mean_elem_ptr = mean_row_ptr =
		 	(float*)((char*)m_mean_array + row_idx * m_mean_step);
		covar_elem_ptr = covar_row_ptr =
		 	(float*)((char*)m_covariation_array + row_idx * m_covar_step);
		img_elem_ptr = img_row_ptr = in_img + row_idx * in_img_step;
		for (auto col_idx = 0U; col_idx < in_num_cols; ++col_idx)
		{
			*mean_elem_ptr = *elem_ptr;
			*covar_elem_ptr = 30;
			
			++mean_elem_ptr;
			++covar_elem_ptr;
			++img_elem_ptr;
		}
	}
}

void GMMSubtractorImpl1::SubtractUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step, unsigned char *out_mask,
	const unsigned int in_mask_step)
{
}

void GMMSubtractorImpl1::FreeModel()
{
	delete [] m_mean_array;
	delete [] m_covariation_array;
}
