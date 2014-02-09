#include "SimpleSubtractorImpl4.h"
#include <stdlib.h>
#include <math.h>
#include "Configuration.h"

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
      
      model_elem[0] = alpha * img_elem[0] + inv_alpha * model_elem[0];
      model_elem[1] = alpha * img_elem[1] + inv_alpha * model_elem[1];
      model_elem[2] = alpha * img_elem[2] + inv_alpha * model_elem[2];
      model_elem[3] = alpha * img_elem[3] + inv_alpha * model_elem[3];

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
      
      model_elem[0] = alpha * img_elem[0] + inv_alpha * model_elem[0];
      model_elem[1] = alpha * img_elem[1] + inv_alpha * model_elem[1];
      model_elem[2] = alpha * img_elem[2] + inv_alpha * model_elem[2];
      model_elem[3] = alpha * img_elem[3] + inv_alpha * model_elem[3];

      model_elem += 4;
      img_elem += 4;
		}
#endif
	}
  
}

void SimpleSubtractorImpl4::ConvertInt8ToFloat(__m128i in_value,
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

inline void SimpleSubtractorImpl4::ShufflePixels(__m128 &io_a,
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

inline __m128i SimpleSubtractorImpl4::SubtractBlock(
  const unsigned char *in_img, float *io_model)
{
  __m128 first_model, second_model, third_model, fourth_model;
  __m128 first_block, second_block, third_block, fourth_block;
  __m128 first_res, second_res, third_res, fourth_res;

  // load image
  __m128i res = _mm_loadu_si128((__m128i*)in_img);
  ConvertInt8ToFloat(res, first_block, second_block, third_block,
    fourth_block);

  // load_model
  first_model = _mm_load_ps(io_model);
  second_model = _mm_load_ps(io_model + 4);
  third_model = _mm_load_ps(io_model + 8);
  fourth_model = _mm_load_ps(io_model + 12);

  // subtract background
	first_res = _mm_sub_ps(first_block, first_model);
	second_res = _mm_sub_ps(second_block, second_model);
	third_res = _mm_sub_ps(third_block, third_model);
	fourth_res = _mm_sub_ps(fourth_block, fourth_model);

	ShufflePixels(first_res, second_res, third_res, fourth_res);
			
	first_res = _mm_mul_ps(first_res, first_res);
	second_res = _mm_mul_ps(second_res, second_res);
	third_res = _mm_mul_ps(third_res, third_res);
			
	first_res = _mm_add_ps(first_res, second_res);
	first_res = _mm_add_ps(first_res, third_res);

	res = _mm_castps_si128(_mm_cmpgt_ps(first_res, m_threshold));

	// update model
	UpdateBlock(first_model, first_block);
	UpdateBlock(second_model, second_block);
	UpdateBlock(third_model, third_block);
	UpdateBlock(fourth_model, fourth_block);

  // store background model
  _mm_store_ps(io_model, first_model);
  _mm_store_ps(io_model + 4, second_model);
  _mm_store_ps(io_model + 8, third_model);
  _mm_store_ps(io_model + 12, fourth_model);

  return res;
}

inline __m128 SimpleSubtractorImpl4::Abs_ps(__m128 in_value)
{
  static const __m128 sign_mask = _mm_set1_ps(1 << 31);
  return _mm_andnot_ps(sign_mask, in_value);
}
		
inline void SimpleSubtractorImpl4::UpdateBlock(__m128 &io_model,
  __m128 in_data)
{
	io_model = _mm_mul_ps(io_model, m_inv_alpha);
	in_data = _mm_mul_ps(in_data, m_alpha);
	io_model = _mm_add_ps(io_model, in_data);
}
