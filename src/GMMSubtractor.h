#pragma once
#include "BaseSubtractor.h"
#include "IGMMSubtractorImpl.h"

class GMMSubtractor : public BaseSubtractor
{
	public:
		/// Constructor
		/// @param in_num_gaussians is a number of gaussians in the model
		GMMSubtractor(const unsigned int in_num_gaussians = 5);

		/// Destructor
		~GMMSubtractor();

		/// Method to train background model from only one image
		/// @param in_img is an array on pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		void Train(const unsigned char *in_img,
			const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_num_channels, const unsigned int in_img_step);

		/// Method to compute foreground mask of the current image
		/// @param in_img is an array of pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columnst in the image
		/// @param in_num_channels is a number of channels per pixel in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		/// @param out_mask is an array for mask
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		void Subtract(const unsigned char *in_img,
		  const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_num_channels, const unsigned int in_img_step,
		 	unsigned char *out_mask, const unsigned int in_mask_step);

		/// Method to set learning rate of the algorithm
		/// @param in_learning_rate is a new learning rate
		void SetLearningRate(const float in_learning_rate);

		/// Method to set background threshold of the algorithm
		/// @param in_threshold is a new background threshold
		void SetBackgroundThreshold(const float in_threshold);

	private:
		/// Implementation of the background subtractor
		IGMMSubtractorImpl *m_impl;

		// Number of gaussians per pixel
		unsigned int m_num_gaussians;

		/// Method to release subtractor implementation
		void ReleaseImplementation();
};
