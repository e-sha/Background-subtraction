#pragma once
#include <xmmintrin.h>
#include <emmintrin.h>

namespace e_sha_SSELib
{

	/// Method to convert 48 floating point values from 16 tuples of 3 elements to
	/// 3 planes of 16 elements. The result elements are stored in the following
	/// order [048C] [159D] [26AE] [37BF].
	/// @param io_data is a pointer to the 12 128 bit variables with input/output
	/// data.
	void Shuffle48Elems(__m128 *io_data);

	/// Method to convert 16 floating point values from 4 tuples of 4 elements to
	/// 4 planes of 4 planes. The result elements are stored in the following
	/// order [0123].
	/// @param io_data0 is a variable with first 4 input/output floating point
	/// values.
	/// @param io_data1 is a variable with second 4 input/output floating point
	/// values.
	/// @param io_data2 is a variable with third 4 input/output floating point
	/// values.
	/// @param io_data3 is a variable with fourth 4 input/output floating point
	/// values.
	void Shuffle16Elems(__m128 &io_data0, __m128 &io_data1, __m128 &io_data2,
		__m128 &io_data3);

	/// Method to convert 16 floating point values from 4 tuples of 4 elements to
	/// 4 planes of 4 planes. The result elements are stored in the following
	/// order [0123].
	/// @param io_data is a pointer to the 4 128 bit variables with input/output
	/// data.
	void Shuffle16Elems(__m128 *io_data);

	/// Method to convert 16 8 bit unsigned integer values to the 16 floating
	/// point values.
	/// @param in_input is a 16 8 bit unsigned integer values stored in the 128
	/// bit variable;
	/// @param out_output is a pointer to 4 128 bit variables with 4 floating
	/// point variables for result.
	void ConvertInt8ToFloat(__m128i in_input, __m128 *out_output);

	/// Method to compute absolute values of the fourth stored floating point
	/// elements
	/// @param in_value is an input values;
	/// @return the absolut values of the input.
	__m128 Abs_ps(__m128 in_value);
}
