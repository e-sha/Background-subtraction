#pragma once
#include "ISimpleSubtractorImpl.h"
#include <xmmintrin.h>

/**
@class SimpleSubtractorImpl1
Class that implies simple subtractor for four channels channel images
*/
class SimpleSubtractorImpl4 : public ISimpleSubtractorImpl
{
	public:
		/// Constructor
		/// @param in_threshold is a background threshold
		SimpleSubtractorImpl4(float in_threshold = 30);

		// Destructor
		~SimpleSubtractorImpl4();

		/// Method to compute foreground mask of the current image
		/// @param in_img is an array of pixels stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columnst in the image
		/// @param in_img_step is a shift between two consecutive rows in the image
		/// @param out_mask is an array for mask
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		void Subtract(const unsigned char *in_img,
			const unsigned int in_num_rows, const unsigned int in_num_cols,
		 	const unsigned int in_img_step, unsigned char *out_mask,
		 	const unsigned int in_mask_step);

	protected:
		/// Method to allocate memory for background model. m_num_rows and m_num_cols
		/// properties have to be set.
		void AllocateModel();

		/// Method to initialize background model with the image. Memory for
		/// background model has to be allocated
		/// @param in_img is a background image stored row-by-row
		/// @param in_step is a number of bytes between consecutive rows of the image
		void InitializeModel(const unsigned char *in_img, size_t in_step);

	private:
		/// number of bytes per block
		const unsigned int m_BYTES_PER_BLOCK;

		/// update coefficient for the current image
		__m128 m_alpha;
		/// update coefficient for the model
		__m128 m_inv_alpha;
		/// used background threshold
		__m128 m_threshold;

		/// Method to subtract background and update model for 16 elements
		/// @param @in_img is a pointer to four image pixels;
		/// @param @io_model is a pointer to four model pixel;
		/// @return foreground mask. If pixels is related to foreground than its
		/// value in the mask is 0xffffffff.
		inline __m128i SubtractBlock(const unsigned char *in_img, float *io_model);

		/// Method to update model for the current block
		/// @param io_model is a block of the model
		/// @param in_data is a block os the input image
		inline void UpdateBlock(__m128 &io_model, __m128 in_data);
};
