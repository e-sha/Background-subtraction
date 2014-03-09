#include "GMMSubtractorImpl3.h"

GMMSubtractorImpl3::GMMSubtractorImpl3()
{
}

GMMSubtractorImpl3::~GMMSubtractorImpl3()
{
}

void GMMSubtractorImpl3::TrainUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step)
{

}

void GMMSubtractorImpl3::SubtractUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step, unsigned char *out_mask,
	const unsigned int in_mask_step)
{
}

void GMMSubtractorImpl3::FreeModel()
{
	delete [] m_mean_array;
	delete [] m_covariation_array;
}
