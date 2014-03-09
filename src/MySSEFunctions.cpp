#include "MySSEFunctions.h"

namespace e_sha_SSELib
{
	static const __m128i ZERO = _mm_setzero_si128();

	void Shuffle48Elems(__m128 *io_data)
	{
		Shuffle16Elems(io_data[0], io_data[3], io_data[6], io_data[9]);
		Shuffle16Elems(io_data[1], io_data[4], io_data[7], io_data[10]);
		Shuffle16Elems(io_data[2], io_data[5], io_data[8], io_data[11]);

		__m128 r1 = io_data[0];
		__m128 r2 = io_data[9];
		__m128 r3 = io_data[7];
		__m128 r4 = io_data[5];

		__m128 g1 = io_data[3];
		__m128 g2 = io_data[1];
		__m128 g3 = io_data[10];
		__m128 g4 = io_data[8];

		__m128 b1 = io_data[6];
		__m128 b2 = io_data[4];
		__m128 b3 = io_data[2];
		__m128 b4 = io_data[11];

		// store final results
		io_data[0] = r1;
		io_data[1] = r2;
		io_data[2] = r3;
		io_data[3] = r4;

		io_data[4] = g1;
		io_data[5] = g2;
		io_data[6] = g3;
		io_data[7] = g4;

		io_data[8] = b1;
		io_data[9] = b2;
		io_data[10] = b3;
		io_data[11] = b4;
	}

	void Shuffle16Elems(__m128 &io_data0, __m128 &io_data1, __m128 &io_data2,
		__m128 &io_data3)
	{
		__m128 ccdd1 = _mm_unpackhi_ps(io_data0, io_data1);
		__m128 ccdd2 = _mm_unpackhi_ps(io_data2, io_data3);
		__m128 aabb1 = _mm_unpacklo_ps(io_data0, io_data1);
		__m128 aabb2 = _mm_unpacklo_ps(io_data2, io_data3);

		io_data0 = 
			_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(aabb1), _mm_castps_pd(aabb2)));
		io_data1 =
			_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(aabb1), _mm_castps_pd(aabb2)));
		io_data2 =
			_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ccdd1), _mm_castps_pd(ccdd2)));
		io_data3 = 
			_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ccdd1), _mm_castps_pd(ccdd2)));
	}

	void Shuffle16Elems(__m128 *io_data)
	{
		Shuffle16Elems(io_data[0], io_data[1], io_data[2], io_data[3]);
	}


	void ConvertInt8ToFloat(__m128i in_input,
		__m128 *out_output)
	{
		__m128i input_16, input_32;

		// convert first half to 16bit integer
		input_16 = _mm_unpacklo_epi8(in_input, ZERO);
		// convert first fourth to 32bit interger
		input_32 = _mm_unpacklo_epi16(input_16, ZERO);
		// convert first fourth to 32bit floating point value
		out_output[0] = _mm_cvtepi32_ps(input_32);

		// convert second fourth to 32bit integer
		input_32 = _mm_unpackhi_epi16(input_16, ZERO);
		// convert second fourth to 32bit floating point value
		out_output[1] = _mm_cvtepi32_ps(input_32);

		// convert second half to 16bit integer
		input_16 = _mm_unpackhi_epi8(in_input, ZERO);
		// convert third fourth to 32bit interger
		input_32 = _mm_unpacklo_epi16(input_16, ZERO);
		// convert third fourth to 32bit floating point value
		out_output[2] = _mm_cvtepi32_ps(input_32);

		// convert fourth fourth to 32bit integer
		input_32 = _mm_unpackhi_epi16(input_16, ZERO);
		// convert fourth fourth to 32bit floating point value
		out_output[3] = _mm_cvtepi32_ps(input_32);
	}

  __m128 Abs_ps(__m128 in_value)
  {
	  static const __m128 SIGN_MASK = _mm_set1_ps(1 << 31);
    return _mm_andnot_ps(SIGN_MASK, in_value);
  }
}
