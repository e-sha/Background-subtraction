#include "GMMSubtractorImpl4.h"

GMMSubtractorImpl4::GMMSubtractorImpl4()
{
}

GMMSubtractorImpl4::~GMMSubtractorImpl4()
{
}

void GMMSubtractorImpl4::TrainUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step)
{

}

void GMMSubtractorImpl4::SubtractUnsafe(const unsigned char *in_img,
	const unsigned int in_num_rows, const unsigned int in_num_cols,
	const unsigned int in_img_step, unsigned char *out_mask,
	const unsigned int in_mask_step)
{
}

void GMMSubtractorImpl4::FreeModel()
{
	delete [] m_mean_array;
	delete [] m_covariation_array;
}
