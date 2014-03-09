#pragma once
#include "ISimpleSubtractorImpl.h"
#include <xmmintrin.h>
#include "Configuration.h"

/**
@class SimpleSubtractorImpl1
Class that implies simple subtractor for one channel images
*/
class SimpleSubtractorImpl3 : public ISimpleSubtractorImpl
{
	public:
		/// Constructor
		/// @param in_threshold is a background threshold
		SimpleSubtractorImpl3(float in_threshold = 30);

		// Destructor
		~SimpleSubtractorImpl3();

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

#if DEBUG
	public:
#else
	private:
#endif
		/// number of bytes per block
		const unsigned int m_BYTES_PER_BLOCK;

		/// update coefficient for the current image
		__m128 m_alpha;
		/// update coefficient for the model
		__m128 m_inv_alpha;
		/// used background threshold
		__m128 m_threshold;

		/// mask of the results in the first array
		static const __m128i m_result_mask_0;
		/// mask of the results in the second array
		static const __m128i m_result_mask_1;
		/// mask of the results in the third array
		static const __m128i m_result_mask_2;
		/// mask of the results in the fourth array
		static const __m128i m_result_mask_3;

		/// Method to subtract background and update model for 4 elements
		/// @param @in_img is a pointer to the a of the image;
		/// @param @io_model is a pointer to the a of the model;
		/// @return byte-wise foreground mask.
		inline __m128i SubtractBlock(const unsigned char *in_img, float *io_model);

		/// Method to subtract background model color from the specified image value
		/// and update model
		/// @param io_model is a pointer to the background pixels
		/// @param io_img is a block of image pixels
		inline void SubtractAndUpdate(float *io_model, __m128 &io_img);

		/// Method to compute L2 norm and apply thresholding
		/// @param in_r is red values
		/// @param in_g is green values
		/// @param in_b is blue values
		/// return foreground mask. If pixel is associated to the foreground than
		/// corresponded mask value is 0xffffffff
		inline __m128i ApplyThreshold(__m128 in_r, __m128 in_g, __m128 in_b) const;

		/// Method to compute absolute value of the 4 input floating point values
		/// @param in_value is the 4 input floating point values;
		/// @return absolute values of the 4 input floating point values.
		static inline __m128 Abs_ps(__m128 in_value);
};
