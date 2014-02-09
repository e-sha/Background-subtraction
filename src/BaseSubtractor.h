#pragma once

/**
 @class BaseSubtractor
 An abstract class that defines interfaces for background subtraction classes
 */
class BaseSubtractor
{
	public:
		/// Method to train background model from only one image
		/// @param in_img is an array on pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		virtual void Train(const unsigned char *in_img,
			const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_num_channels, const unsigned int in_img_step) = 0;

		/// Method to compute foreground mask of the current image
		/// @param in_img is an array of pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columnst in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		/// @param out_mask is an array for mask
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		virtual void Subtract(const unsigned char *in_img,
		  const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_num_channels, const unsigned int in_img_step,
		 	unsigned char *out_mask, const unsigned int in_mask_step) = 0;

		/// Method to set background threshold
		/// @param in_threshold is a background threshold
	  virtual void SetThreshold(const float in_threshold) = 0;
};
