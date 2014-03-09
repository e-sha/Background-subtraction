#include "SimpleSubtractorImpl4.h"
#include <stdlib.h>
#include <math.h>
#include "Configuration.h"
#include "MySSEFunctions.h"

using e_sha_SSELib::ConvertInt8ToFloat;
using e_sha_SSELib::Shuffle16Elems;

SimpleSubtractorImpl4::SimpleSubtractorImpl4(float in_threshold) :
	ISimpleSubtractorImpl(in_threshold), m_BYTES_PER_BLOCK(16)
{
}

SimpleSubtractorImpl4::~SimpleSubtractorImpl4()
{
}

void SimpleSubtractorImpl4::AllocateModel()
{
	size_t naive_bytes_per_row = 4 * m_num_cols * sizeof(float);
	bool last_block_is_fragmented = naive_bytes_per_row % m_BYTES_PER_BLOCK;
	unsigned int num_blocks = naive_bytes_per_row / m_BYTES_PER_BLOCK +
		(last_block_is_fragmented ? 1 : 0);
	m_step = num_blocks * m_BYTES_PER_BLOCK;
	unsigned int num_bytes = m_step * m_num_rows;
	posix_memalign((void**)&m_model, m_BYTES_PER_BLOCK, num_bytes); 
}

void SimpleSubtractorImpl4::InitializeModel(const unsigned char *in_img, size_t in_step)
{
	float *model_elem;
	const unsigned char *img_elem;
	auto num_elems_per_row = 4 * m_num_cols;
	auto num_blocks = num_elems_per_row / m_BYTES_PER_BLOCK;
	auto num_left_elems = num_elems_per_row % m_BYTES_PER_BLOCK;

//#pragma omp parallel for private(model_elem, img_elem)
	for (auto row_idx = 0U; row_idx < m_num_rows; ++row_idx)
	{
		model_elem = (float*)((char*)m_model + row_idx * m_step);
		img_elem = in_img + row_idx * in_step;
#if USE_SSE
		__m128i img_block_32, img_block_16, img_block_8;
		__m128 block[4];
		for (auto block_idx = 0U; block_idx < num_blocks; ++block_idx)
		{
			img_block_8 = _mm_loadu_si128((__m128i*)img_elem);
      ConvertInt8ToFloat(img_block_8, block);
			for (auto var_idx = 0U; var_idx < 4; ++var_idx)
				_mm_store_ps(model_elem + 4 * var_idx, block[var_idx]);

			model_elem += m_BYTES_PER_BLOCK;
			img_elem += m_BYTES_PER_BLOCK;
		}
		for (auto col_idx = 0U; col_idx < num_left_elems; ++col_idx)
		{
			model_elem[col_idx] = img_elem[col_idx];	
		}
#else
		for (auto col_idx = 0U; col_idx < m_num_cols; ++col_idx)
		{
			model_elem[col_idx] = img_elem[col_idx];
		}
#endif
	}
}


void SimpleSubtractorImpl4::Subtract(const unsigned char *in_img,
  const unsigned int in_num_rows, const unsigned int in_num_cols,
  const unsigned int in_img_step, unsigned char *out_mask,
  const unsigned int in_mask_step)
{
  float alpha = ISimpleSubtractorImpl::m_alpha;
  float inv_alpha = 1 - alpha;
  float threshold = ISimpleSubtractorImpl::m_threshold;
  m_alpha = _mm_set1_ps(alpha);
  m_inv_alpha = _mm_set1_ps(inv_alpha);
  m_threshold = _mm_set1_ps(threshold);
  
  auto block_size = m_BYTES_PER_BLOCK;
	auto num_blocks = m_num_cols / block_size;
	auto num_left_elems = m_num_cols % block_size;

  float *model_elem;
  const unsigned char *img_elem;
  unsigned char *mask_elem;

//#pragma omp parallel for private(model_elem, img_elem)
	for (auto row_idx = 0U; row_idx < m_num_rows; ++row_idx)
	{
		model_elem = (float*)((char*)m_model + row_idx * m_step);
		img_elem = in_img + row_idx * in_img_step;
    mask_elem = out_mask + row_idx * in_mask_step;
#if USE_SSE
		__m128i first_res, second_res, third_res, fourth_res;
		for (auto block_idx = 0U; block_idx < num_blocks; ++block_idx)
		{
      first_res = SubtractBlock(img_elem, model_elem);
      second_res = SubtractBlock(img_elem + m_BYTES_PER_BLOCK,
        model_elem + m_BYTES_PER_BLOCK);
      third_res = SubtractBlock(img_elem + 2 * m_BYTES_PER_BLOCK,
        model_elem + 2 * m_BYTES_PER_BLOCK);
      fourth_res = SubtractBlock(img_elem + 3 * m_BYTES_PER_BLOCK,
        model_elem + 3 * m_BYTES_PER_BLOCK);

      // postprocess foreground mask
      first_res = _mm_packs_epi32(first_res, second_res);
      third_res = _mm_packs_epi32(third_res, fourth_res);
      first_res = _mm_packs_epi16(first_res, third_res);

      // write foreground mask
      _mm_storeu_si128((__m128i*)mask_elem, first_res);

			model_elem += block_size;
			img_elem += block_size;
      mask_elem += m_BYTES_PER_BLOCK;
		}
		for (auto col_idx = 0U; col_idx < num_left_elems; ++col_idx)
		{
		  float dif = model_elem[col_idx] - img_elem[col_idx];
      float res = dif * dif;
      dif = model_elem[1] - img_elem[1];
      res += dif * dif;
      dif = model_elem[2] - img_elem[2];
      res += dif * dif;
      
      mask_elem[col_idx] = res > threshold ? 255 : 0;
      
			for (auto var_idx = 0U; var_idx < 4; ++var_idx)
			{
        model_elem[var_idx] =
				 	alpha * img_elem[var_idx] + inv_alpha * model_elem[var_idx];
			}

      model_elem += 4;
      img_elem += 4;
		}
#else
		for (auto col_idx = 0U; col_idx < m_num_cols; ++col_idx)
		{
		  float dif = model_elem[col_idx] - img_elem[col_idx];
      float res = dif * dif;
      dif = model_elem[1] - img_elem[1];
      res += dif * dif;
      dif = model_elem[2] - img_elem[2];
      res += dif * dif;
      
      mask_elem[col_idx] = res > threshold ? 255 : 0;
      
			for (auto var_idx = 0U; var_idx < 4; ++var_idx)
			{
        model_elem[var_idx] =
				 	alpha * img_elem[var_idx] + inv_alpha * model_elem[var_idx];
			}

      model_elem += 4;
      img_elem += 4;
		}
#endif
	}
  
}

inline __m128i SimpleSubtractorImpl4::SubtractBlock(
  const unsigned char *in_img, float *io_model)
{
  __m128 model[4];
  __m128 block[4];
  __m128 res[4];

  // load image
  __m128i mask = _mm_loadu_si128((__m128i*)in_img);
  ConvertInt8ToFloat(mask, block);

	for (auto var_idx = 0U; var_idx < 4; ++var_idx)
	{
    // load_model
		model[var_idx] = _mm_load_ps(io_model + 4 * var_idx);
    // subtract background
		res[var_idx] = _mm_sub_ps(block[var_idx], model[var_idx]);
	}

	Shuffle16Elems(res);
		
	for (auto var_idx = 0U; var_idx < 3; ++var_idx)	
		res[var_idx] = _mm_mul_ps(res[var_idx], res[var_idx]);
			
	for (auto var_idx = 1U; var_idx < 3; ++var_idx)
		res[0] = _mm_add_ps(res[0], res[var_idx]);

	mask = _mm_castps_si128(_mm_cmpgt_ps(res[0], m_threshold));

	for (auto var_idx = 0U; var_idx < 4U; ++var_idx)
	{
    // update model
	  UpdateBlock(model[var_idx], block[var_idx]);
		// store background model
		_mm_store_ps(io_model + 4 * var_idx, model[var_idx]);
	}

  return mask;
}

inline void SimpleSubtractorImpl4::UpdateBlock(__m128 &io_model,
  __m128 in_data)
{
	io_model = _mm_mul_ps(io_model, m_inv_alpha);
	in_data = _mm_mul_ps(in_data, m_alpha);
	io_model = _mm_add_ps(io_model, in_data);
}
