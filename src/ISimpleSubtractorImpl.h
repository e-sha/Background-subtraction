#pragma once
#include <stdlib.h>

/**
@class ISimpleSubtractorImpl
Abstract class that implies simple subtractor. Implementations differ in number
of channels per pixel.
*/
class ISimpleSubtractorImpl
{
	public:
		/// Constructor
		/// @param in_threshold is a background threshold
		ISimpleSubtractorImpl(float in_threshold = 30);

		// Destructor
		~ISimpleSubtractorImpl();

		/// Template method to train background model from only one image
		/// @param in_img is an array on pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		void Train(const unsigned char *in_img, const unsigned int in_num_rows,
			const unsigned int in_num_cols, const unsigned int in_img_step);

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
		 	const unsigned int in_mask_step) = 0;

		/// Method to set background threshold
		/// @param in_threshold is a background threshold
	  void SetThreshold(const float in_threshold);

	protected:
		/// background model	
    float *m_model;

		/// number of rows in the image
    unsigned int m_num_rows;

		/// number of columns in the image
    unsigned int m_num_cols;

		/// number of bytes between consecutive rows in the model
    unsigned int m_step;

		/// square background threshold
		float m_threshold;

		/// Method to allocate memory for background model. m_num_rows and m_num_cols
		/// update coefficient for the current image
		float m_alpha;

		/// properties have to be set.
		virtual void AllocateModel() = 0;

		/// Method to free memory for background model
		virtual void FreeModel();

		/// Method to initialize background model with the image. Memory for
		/// background model has to be allocated
		/// @param in_img is a background image stored row-by-row
		/// @param in_step is a number of bytes between consecutive rows of the image
		virtual void InitializeModel(const unsigned char *in_img,
			size_t in_step) = 0;
};
