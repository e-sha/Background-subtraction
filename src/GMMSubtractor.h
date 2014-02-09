#pragma once
#include "BaseSubtractor.h"

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
		void SetLerningRate(const float in_learning_rate);

		/// Method to set background threshold of the algorithm
		/// @param in_threshold is a new background threshold
		void SetBackgroundThreshold(const float in_threshold);

	private:
		/// Mean values of gaussians stored in matrix format. Each row of the matrix
		/// is a mean value of the gaussian. Number of columns corresponds to number
		/// of color channels in the image (1 or 3)
		float *m_mean_array;

		/// Covariantion matrices of the gaussians stored in the matrix format. Each
		/// row of the matrix is a covariation matrix of the gaussian expanded to a
		/// row. At this moment all covariation matrices are diagonal. That's why
		/// we store only diagonal elements of the matrices
		float *m_covariaton_array;

		/// Learning rate of the algorithm
		float m_learning_rate;

		/// Background threshold
		float m_threshold;
};
