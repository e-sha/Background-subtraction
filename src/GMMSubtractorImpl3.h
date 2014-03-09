#pragma once
#include "IGMMSubtractorImpl.h"

/**
@class GMMSubtractorImpl3
A class that realizes gmm subtractor for 3 channel images.
*/
class GMMSubtractorImpl3 : public IGMMSubtractorImpl
{
	public:
		/// Constructor
		GMMSubtractorImpl3();

		/// Destructor
		virtual ~GMMSubtractorImpl3();

		/// Method to train background model from only one image
		/// @param in_img is an array on pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		virtual void Train(const unsigned char *in_img,
			const unsigned int in_num_rows, const unsigned int in_num_cols,
			const unsigned int in_img_step);

		/// Method to compute foreground mask of the current image
		/// @param in_img is an array of pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columnst in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		/// @param out_mask is an array for mask
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		virtual void Subtract(const unsigned char *in_img,
		  const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_img_step, unsigned char *out_mask,
		 	const unsigned int in_mask_step);

	protected:
		/// Method to deallocate model. Do not checks its existence.
		virtual void FreeModel();

};
