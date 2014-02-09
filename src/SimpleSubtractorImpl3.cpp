#include "SimpleSubtractorImpl3.h"
#include <stdlib.h>
#include <math.h>
#include "Configuration.h"

const __m128i SimpleSubtractorImpl3::m_result_mask_0 = _mm_set1_epi32(0xff);
const __m128i SimpleSubtractorImpl3::m_result_mask_1 = _mm_set1_epi32(0xff00);
const __m128i SimpleSubtractorImpl3::m_result_mask_2 = _mm_set1_epi32(0xff0000);
const __m128i SimpleSubtractorImpl3::m_result_mask_3 = _mm_set1_epi32(0xff000000);

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

//#pragma omp parallel for private(model_elem, img_elem)
	for (auto row_idx = 0U; row_idx < m_num_rows; ++row_idx)
	{
		model_elem = (float*)((char*)m_model + row_idx * m_step);
		img_elem = in_img + row_idx * in_step;
#if USE_SSE
		__m128i img_block_8;
		__m128 first_block, second_block, third_block, fourth_block;
		for (auto block_idx = 0U; block_idx < num_blocks; ++block_idx)
		{
			img_block_8 = _mm_loadu_si128((__m128i*)img_elem);
      ConvertInt8ToFloat(img_block_8, first_block, second_block, third_block,
        fourth_block);
			_mm_store_ps(model_elem, first_block);
			_mm_store_ps(model_elem + 4, second_block);
			_mm_store_ps(model_elem + 8, third_block);
			_mm_store_ps(model_elem + 12, fourth_block);

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

//#pragma omp parallel for private(model_elem, img_elem)
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
		  float res = model_elem[0] - img_elem[0];
      res *= res;
      float dif = model_elem[1] - img_elem[1];
      res += dif * dif;
      dif = model_elem[2] - img_elem[2];
      res += dif * dif;

      mask_elem[col_idx] = res > threshold ? 255 : 0;

      model_elem[0] = alpha * img_elem[0] + inv_alpha * model_elem[0];
      model_elem[1] = alpha * img_elem[1] + inv_alpha * model_elem[1];
      model_elem[2] = alpha * img_elem[2] + inv_alpha * model_elem[2];

      model_elem += 3;
      img_elem += 3;
		}
#else
		for (auto col_idx = 0U; col_idx < m_num_cols; ++col_idx)
		{
		  float res = model_elem[0] - img_elem[0];
      res *= res;
      float dif = model_elem[1] - img_elem[1];
      res += dif * dif;
      dif = model_elem[2] - img_elem[2];
      res += dif * dif;

      mask_elem[col_idx] = res > threshold ? 255 : 0;

      model_elem[0] = alpha * img_elem[0] + inv_alpha * model_elem[0];
      model_elem[1] = alpha * img_elem[1] + inv_alpha * model_elem[1];
      model_elem[2] = alpha * img_elem[2] + inv_alpha * model_elem[2];

      model_elem += 3;
      img_elem += 3;
		}
#endif
	}
  
}

#if DEBUG == 0
inline
#endif
void SimpleSubtractorImpl3::ShufflePixels(__m128 &io_r1, __m128 &io_r2, __m128 &io_r3,
  __m128 &io_r4, __m128 &io_g1, __m128 &io_g2, __m128 &io_g3, __m128 &io_g4,
  __m128 &io_b1, __m128 &io_b2, __m128 &io_b3, __m128 &io_b4)
{
  ShufflePixels(io_r1, io_r4, io_g3, io_b2);
  ShufflePixels(io_r2, io_g1, io_g4, io_b3);
  ShufflePixels(io_r3, io_g2, io_b1, io_b4);

  __m128 r1 = io_r1;
  __m128 r2 = io_b2;
  __m128 r3 = io_g4;
  __m128 r4 = io_g2;

  __m128 g1 = io_r4;
  __m128 g2 = io_r2;
  __m128 g3 = io_b3;
  __m128 g4 = io_b1;

  __m128 b1 = io_g3;
  __m128 b2 = io_g1;
  __m128 b3 = io_r3;
  __m128 b4 = io_b4;

  // store final results
  io_r1 = r1;
  io_r2 = r2;
  io_r3 = r3;
  io_r4 = r4;

  io_g1 = g1;
  io_g2 = g2;
  io_g3 = g3;
  io_g4 = g4;

  io_b1 = b1;
  io_b2 = b2;
  io_b3 = b3;
  io_b4 = b4;
}

inline void SimpleSubtractorImpl3::ShufflePixels(__m128 &io_a,
	__m128 &io_b, __m128 &io_c, __m128 &io_d)
{
	__m128 ccdd1 = _mm_unpackhi_ps(io_a, io_b);
	__m128 ccdd2 = _mm_unpackhi_ps(io_c, io_d);
	__m128 aabb1 = _mm_unpacklo_ps(io_a, io_b);
	__m128 aabb2 = _mm_unpacklo_ps(io_c, io_d);

	io_a = 
		_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(aabb1), _mm_castps_pd(aabb2)));
	io_b =
	 	_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(aabb1), _mm_castps_pd(aabb2)));
	io_c =
		_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ccdd1), _mm_castps_pd(ccdd2)));
	io_d = 
		_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ccdd1), _mm_castps_pd(ccdd2)));
}

void SimpleSubtractorImpl3::ConvertInt8ToFloat(__m128i in_value,
  __m128 &out_first, __m128 &out_second, __m128 &out_third,
  __m128 &out_fourth)
{
  __m128i input_16, input_32;
	const __m128i ZERO = _mm_setzero_si128();

  // convert first half to 16bit integer
  input_16 = _mm_unpacklo_epi8(in_value, ZERO);
  // convert first fourth to 32bit interger
  input_32 = _mm_unpacklo_epi16(input_16, ZERO);
  // convert first fourth to 32bit floating point value
  out_first = _mm_cvtepi32_ps(input_32);

  // convert second fourth to 32bit integer
  input_32 = _mm_unpackhi_epi16(input_16, ZERO);
  // convert second fourth to 32bit floating point value
  out_second = _mm_cvtepi32_ps(input_32);

  // convert second half to 16bit integer
  input_16 = _mm_unpackhi_epi8(in_value, ZERO);
  // convert third fourth to 32bit interger
  input_32 = _mm_unpacklo_epi16(input_16, ZERO);
  // convert third fourth to 32bit floating point value
  out_third = _mm_cvtepi32_ps(input_32);

  // convert fourth fourth to 32bit integer
  input_32 = _mm_unpackhi_epi16(input_16, ZERO);
  // convert fourth fourth to 32bit floating point value
  out_fourth = _mm_cvtepi32_ps(input_32);
}

inline __m128i SimpleSubtractorImpl3::SubtractBlock(
  const unsigned char *in_img, float *io_model)
{
  __m128i res0, res1, res2, res3;
  __m128 r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3, model;

  // read image block
  res0 = _mm_loadu_si128((__m128i*)in_img);
  res1 = _mm_loadu_si128((__m128i*)(in_img + m_BYTES_PER_BLOCK));
  res2 = _mm_loadu_si128((__m128i*)(in_img + 2 * m_BYTES_PER_BLOCK));

  // convert image block to float values
  ConvertInt8ToFloat(res0, r0, r1, r2, r3);
  ConvertInt8ToFloat(res1, g0, g1, g2, g3);
  ConvertInt8ToFloat(res2, b0, b1, b2, b3);

  // subtract and update model
  SubtractAndUpdate(io_model, r0);
  SubtractAndUpdate(io_model + 4, r1);
  SubtractAndUpdate(io_model + 8, r2);
  SubtractAndUpdate(io_model + 12, r3);
  SubtractAndUpdate(io_model + 16, g0);
  SubtractAndUpdate(io_model + 20, g1);
  SubtractAndUpdate(io_model + 24, g2);
  SubtractAndUpdate(io_model + 28, g3);
  SubtractAndUpdate(io_model + 32, b0);
  SubtractAndUpdate(io_model + 36, b1);
  SubtractAndUpdate(io_model + 40, b2);
  SubtractAndUpdate(io_model + 44, b3);

  // Compute Eucledean distance
  ShufflePixels(r0, r1, r2, r3, g0, g1, g2, g3, b0, b1, b2, b3);

  // find foreground mask
  res0 = ApplyThreshold(r0, g0, b0);
  res1 = ApplyThreshold(r1, g1, b1);
  res2 = ApplyThreshold(r2, g2, b2);
  res3 = ApplyThreshold(r3, g3, b3); 

  // combine results
  res0 = _mm_and_si128(res0, m_result_mask_0);
  res1 = _mm_and_si128(res1, m_result_mask_1);
  res2 = _mm_and_si128(res2, m_result_mask_2);
  res3 = _mm_and_si128(res3, m_result_mask_3);

  res0 = _mm_or_si128(res0, res1);
  res0 = _mm_or_si128(res0, res2);
  res0 = _mm_or_si128(res0, res3);
  
  return res0;
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
