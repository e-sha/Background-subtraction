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
