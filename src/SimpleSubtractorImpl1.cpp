#include "SimpleSubtractorImpl1.h"
#include <stdlib.h>
#include <math.h>
#include "Configuration.h"
#include "MySSEFunctions.h"

using e_sha_SSELib::Abs_ps;
using e_sha_SSELib::ConvertInt8ToFloat;

SimpleSubtractorImpl1::SimpleSubtractorImpl1(float in_threshold) :
	ISimpleSubtractorImpl(in_threshold), m_BYTES_PER_BLOCK(16)
{
}

SimpleSubtractorImpl1::~SimpleSubtractorImpl1()
{
}

void SimpleSubtractorImpl1::AllocateModel()
{
	size_t naive_bytes_per_row = m_num_cols * sizeof(float);
	bool last_block_is_fragmented = naive_bytes_per_row % m_BYTES_PER_BLOCK;
	unsigned int num_blocks = naive_bytes_per_row / m_BYTES_PER_BLOCK +
		(last_block_is_fragmented ? 1 : 0);
	m_step = num_blocks * m_BYTES_PER_BLOCK;
	unsigned int num_bytes = m_step * m_num_rows;
	posix_memalign((void**)&m_model, m_BYTES_PER_BLOCK, num_bytes); 
}

void SimpleSubtractorImpl1::InitializeModel(const unsigned char *in_img, size_t in_step)
{
	float *model_elem;
	const unsigned char *img_elem;
	auto num_blocks = m_num_cols / m_BYTES_PER_BLOCK;
	auto num_left_elems = m_num_cols % m_BYTES_PER_BLOCK;

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
      for (auto idx = 0U; idx < 4; ++idx)
        _mm_store_ps(model_elem + 4 * idx, block[idx]);

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


void SimpleSubtractorImpl1::Subtract(const unsigned char *in_img,
  const unsigned int in_num_rows, const unsigned int in_num_cols,
  const unsigned int in_img_step, unsigned char *out_mask,
  const unsigned int in_mask_step)
{
  float alpha = ISimpleSubtractorImpl::m_alpha;
  float inv_alpha = 1 - alpha;
  float threshold = sqrt(ISimpleSubtractorImpl::m_threshold);
  m_alpha = _mm_set1_ps(alpha);
  m_inv_alpha = _mm_set1_ps(inv_alpha);
  m_threshold = _mm_set1_ps(threshold);
  
	auto num_blocks = m_num_cols / m_BYTES_PER_BLOCK;
	auto num_left_elems = m_num_cols % m_BYTES_PER_BLOCK;

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
    __m128i res[4];
    __m128 block[4], model[4];
		for (auto block_idx = 0U; block_idx < num_blocks; ++block_idx)
		{
			res[0] = _mm_loadu_si128((__m128i*)img_elem);
      ConvertInt8ToFloat(res[0], block);

      for (auto idx = 0U; idx < 4; ++idx)
      {
        // load_model
        model[idx] = _mm_load_ps(model_elem + 4 * idx);
        // subtract background
        res[idx] = SubtractBlock(block[idx], model[idx]);
      }

      // postprocess foreground mask
      res[0] = _mm_packs_epi32(res[0], res[1]);
      res[2] = _mm_packs_epi32(res[2], res[3]);
      res[0] = _mm_packs_epi16(res[0], res[2]);

      // write foreground mask
      _mm_storeu_si128((__m128i*)mask_elem, res[0]);

      // write updated model
      for (auto idx = 0U; idx < 4; ++idx)
        _mm_store_ps(model_elem + 4 * idx, model[idx]);

			model_elem += m_BYTES_PER_BLOCK;
			img_elem += m_BYTES_PER_BLOCK;
      mask_elem += m_BYTES_PER_BLOCK;
		}
		for (auto col_idx = 0U; col_idx < num_left_elems; ++col_idx)
		{
		  float dif = fabs(model_elem[col_idx] - img_elem[col_idx]);
      mask_elem[col_idx] = dif > threshold ? 255 : 0;
      model_elem[col_idx] = alpha * img_elem[col_idx] +
        inv_alpha * model_elem[col_idx];
		}
#else
		for (auto col_idx = 0U; col_idx < m_num_cols; ++col_idx)
		{
		  float dif = fabs(model_elem[col_idx] - img_elem[col_idx]);
      mask_elem[col_idx] = dif > threshold ? 255 : 0;
      model_elem[col_idx] = alpha * img_elem[col_idx] +
        inv_alpha * model_elem[col_idx];
		}
#endif
	}
}

inline __m128i SimpleSubtractorImpl1::SubtractBlock(__m128 in_img,
  __m128 &io_model)
{
  // compute sad between the model and the image
  __m128 sad = _mm_sub_ps(in_img, io_model);
  sad = Abs_ps(sad);
  
  // update model
  in_img = _mm_mul_ps(in_img, m_alpha);
  io_model = _mm_mul_ps(io_model, m_inv_alpha);
  io_model = _mm_add_ps(io_model, in_img);

  return _mm_castps_si128(_mm_cmpgt_ps(sad, m_threshold));
}
