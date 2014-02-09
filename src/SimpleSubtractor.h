#pragma once

#include <BaseSubtractor.h>
#include "ISimpleSubtractorImpl.h"

/**
@class SimpleSubtractor
Class that subtracts background using simple model. The model consists of
single image.
*/
class SimpleSubtractor : public BaseSubtractor
{
  public:
		/// Constructor
		/// @param in_threshold is a background threshold
    SimpleSubtractor(const float in_threshold = 30);

    /// destructor
		~SimpleSubtractor();

		/// Method to train background model from only one image
		/// @param in_img is an array on pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		void Train(const unsigned char *in_img, const unsigned int in_num_rows,
				const unsigned int in_num_cols, const unsigned int in_num_channels,
				const unsigned int in_img_step);

		/// Method to compute foreground mask of the current image
		/// @param in_img is an array of pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columnst in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		/// @param out_mask is an array for mask
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		void Subtract(const unsigned char *in_img, const unsigned int in_num_rows,
				const unsigned int in_num_cols, const unsigned int in_num_channels,
				const unsigned int in_img_step, unsigned char *out_mask,
				const unsigned int in_mask_step);

		/// Method to set background threshold
		/// @param in_threshold is a background threshold
	  void SetThreshold(const float in_threshold);

  private:
    float *m_model;
		unsigned int m_num_channels;
    unsigned int m_num_rows;
    unsigned int m_num_cols;
    unsigned int m_step;

		float m_alpha;

		/// square background threshold
		float m_threshold;

		// pointer to the implementation of the subtractor
		ISimpleSubtractorImpl *m_impl;

    void rgb2gray(const unsigned char *in_rgb_img,
      const unsigned int in_num_rows, const unsigned int in_num_cols,
      const unsigned int in_rgb_step, const unsigned char *out_gray_img,
      const unsigned int in_gray_step);
};
