#include "SimpleSubtractorImpl3.h"
#include <stdlib.h>
#include <math.h>
#include "MySSEFunctions.h"
#include "Configuration.h"

const __m128i SimpleSubtractorImpl3::m_result_mask_0 = _mm_set1_epi32(0xff);
const __m128i SimpleSubtractorImpl3::m_result_mask_1 = _mm_set1_epi32(0xff00);
const __m128i SimpleSubtractorImpl3::m_result_mask_2 = _mm_set1_epi32(0xff0000);
const __m128i SimpleSubtractorImpl3::m_result_mask_3 = _mm_set1_epi32(0xff000000);

using e_sha_SSELib::ConvertInt8ToFloat;
using e_sha_SSELib::Shuffle48Elems;
using e_sha_SSELib::Abs_ps;

SimpleSubtractorImpl3::SimpleSubtractorImpl3(float in_threshold) :
	ISimpleSubtractorImpl(in_threshold), m_BYTES_PER_BLOCK(16)
{
}

SimpleSubtractorImpl3::~SimpleSubtractorImpl3()
{
}

void SimpleSubtractorImpl3::AllocateModel()
{
	size_t naive_bytes_per_row = 3 * m_num_cols * sizeof(float);
	bool last_block_is_fragmented = naive_bytes_per_row % m_BYTES_PER_BLOCK;
	unsigned int num_blocks = naive_bytes_per_row / m_BYTES_PER_BLOCK +
		(last_block_is_fragmented ? 1 : 0);
	m_step = num_blocks * m_BYTES_PER_BLOCK;
	unsigned int num_bytes = m_step * m_num_rows;
	posix_memalign((void**)&m_model, m_BYTES_PER_BLOCK, num_bytes);
}

void SimpleSubtractorImpl3::InitializeModel(const unsigned char *in_img, size_t in_step)
{
	float *model_elem;
	const unsigned char *img_elem;
	auto num_elems_per_row = 3 * m_num_cols;
	auto num_blocks = num_elems_per_row / m_BYTES_PER_BLOCK;
	auto num_left_elems = num_elems_per_row % m_BYTES_PER_BLOCK;

#pragma omp parallel for private(model_elem, img_elem)
	for (auto row_idx = 0U; row_idx < m_num_rows; ++row_idx)
	{
		model_elem = (float*)((char*)m_model + row_idx * m_step);
		img_elem = in_img + row_idx * in_step;
#if USE_SSE
		__m128i img_block_8;
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
			model_elem[col_idx] = img_elem[col_idx];	
#else
		auto num_elems_per_row = 3 * m_num_cols;
		for (auto col_idx = 0U; col_idx < num_elems_per_row; ++col_idx)
			model_elem[col_idx] = img_elem[col_idx];
#endif
	}
}

void SimpleSubtractorImpl3::Subtract(const unsigned char *in_img,
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
 
  auto block_size = m_BYTES_PER_BLOCK; 
	auto num_blocks = m_num_cols / block_size;
	auto num_left_elems = m_num_cols % block_size;

  float *model_elem;
  const unsigned char *img_elem;
  unsigned char *mask_elem;

#pragma omp parallel for private(model_elem, img_elem, mask_elem)
	for (auto row_idx = 0U; row_idx < m_num_rows; ++row_idx)
	{
		model_elem = (float*)((char*)m_model + row_idx * m_step);
		img_elem = in_img + row_idx * in_img_step;
    mask_elem = out_mask + row_idx * in_mask_step;
#if USE_SSE
		__m128i res;
		for (auto block_idx = 0U; block_idx < num_blocks; ++block_idx)
		{
      res = SubtractBlock(img_elem, model_elem);
      _mm_storeu_si128((__m128i*)mask_elem, res);

			model_elem += 3 * block_size;
			img_elem += 3 * block_size;
      mask_elem += block_size;
		}
		for (auto col_idx = 0U; col_idx < num_left_elems; ++col_idx)
		{
			float res = 0.f, dif;
			for (auto idx = 0U; idx < 3; ++idx)
			{
				dif = model_elem[idx] - img_elem[idx];
				res += res * res;
				model_elem[idx] = alpha * img_elem[idx] + inv_alpha * model_elem[idx];
			}

      mask_elem[col_idx] = res > threshold ? 255 : 0;

      model_elem += 3;
      img_elem += 3;
		}
#else
		for (auto col_idx = 0U; col_idx < m_num_cols; ++col_idx)
		{
			float res = 0.f, dif;
			for (auto idx = 0U; idx < 3; ++idx)
			{
				dif = model_elem[idx] - img_elem[idx];
				res += res * res;
				model_elem[idx] = alpha * img_elem[idx] + inv_alpha * model_elem[idx];
			}

      mask_elem[col_idx] = res > threshold ? 255 : 0;

      model_elem += 3;
      img_elem += 3;
		}
#endif
	}
}

inline __m128i SimpleSubtractorImpl3::SubtractBlock(
  const unsigned char *in_img, float *io_model)
{
	__m128i res[4];
	__m128 data[12];

	for (auto var_idx = 0U; var_idx < 3; ++var_idx)
	{
    // read image block
	  res[var_idx] =
		 	_mm_loadu_si128((__m128i*)(in_img + var_idx * m_BYTES_PER_BLOCK));
    // convert image block to float values
	  ConvertInt8ToFloat(res[var_idx], data + 4 * var_idx);
	}

  // subtract and update model
	for (auto var_idx = 0U; var_idx < 12; ++var_idx)
	  SubtractAndUpdate(io_model + 4 * var_idx, data[var_idx]);

  // Compute Eucledean distance
  Shuffle48Elems(data);

  // find foreground mask
	for (auto var_idx = 0U; var_idx < 4; ++var_idx)
	{
    res[var_idx] =
		 	ApplyThreshold(data[var_idx], data[4 + var_idx], data[8 + var_idx]);
	}

  // combine results
  res[0] = _mm_and_si128(res[0], m_result_mask_0);
  res[1] = _mm_and_si128(res[1], m_result_mask_1);
  res[2] = _mm_and_si128(res[2], m_result_mask_2);
  res[3] = _mm_and_si128(res[3], m_result_mask_3);

	for (auto var_idx = 1U; var_idx < 4; ++var_idx)
		res[0] = _mm_or_si128(res[0], res[var_idx]);
  
  return res[0];
}

inline void SimpleSubtractorImpl3::SubtractAndUpdate(float *io_model,
  __m128 &io_img)
{
  // load model
  __m128 model = _mm_load_ps(io_model);

  // update model
  __m128 updated_model = _mm_mul_ps(model, m_inv_alpha);
  __m128 tmp = _mm_mul_ps(io_img, m_alpha);
  updated_model = _mm_add_ps(updated_model, tmp);
  _mm_store_ps(io_model, updated_model);

  // subtract background model
  io_img = _mm_sub_ps(io_img, model);
}

inline __m128i SimpleSubtractorImpl3::ApplyThreshold(__m128 in_r, __m128 in_g,
  __m128 in_b) const
{
  in_r = _mm_mul_ps(in_r, in_r);
  in_g = _mm_mul_ps(in_g, in_g);
  in_b = _mm_mul_ps(in_b, in_b);

  in_r = _mm_add_ps(in_r, in_g);
  in_r = _mm_add_ps(in_r, in_b);

  return _mm_castps_si128(_mm_cmpgt_ps(in_r, m_threshold));
}

inline __m128 SimpleSubtractorImpl3::Abs_ps(__m128 in_value)
{
  static const __m128 sign_mask = _mm_set1_ps(1 << 31);
  return _mm_andnot_ps(sign_mask, in_value);
}
