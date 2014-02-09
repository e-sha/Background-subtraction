#pragma once
#include <stdlib.h>
#include "BaseSubtractorTester.h"
#include <xmmintrin.h>

class SubtractorTester : public BaseSubtractorTester
{
	public:
		/// Default constructor
		SubtractorTester();

		/// Destructor
		~SubtractorTester();

		/// Method to test results of the algithm
		/// @param out_precision_array is an array of precision values
		/// @param out_recall_array is an array of recall array
		void Test(std::vector<float> &out_precision_array,
			 	std::vector<float> &out_recall_array);

		/// Method to set a template of the input image file name
		/// @param in_template is a template of the input image file name
		void SetInputTemplate(const char *in_template);

		/// Method to set start index of the input sequence
		/// @param in_idx is a start index of the input sequence
		void SetStartIndex(unsigned int in_idx);

		/// Method to set last index of the input sequence
		/// @param in_idx is a last index of the input sequence
		void SetStopIndex(unsigned int in_idx);

		/// Method to set template of the groundtruth image file name
		/// @param in_template is a template of the groundtruth image file name
		void SetGTTemplate(const char *in_template);

		/// Method to set start index of the testing part of the sequence
		/// @param in_idx is a start index of the testing part of the sequence
		void SetGTStartIndex(unsigned int in_idx);

		/// Method to set stop index of the testing part of the sequence
		/// @param in_idx is a stop index of the testing part of the sequence
		void SetGTStopIndex(unsigned int in_idx);

		/// Method to set an array of threshold values for testing
		/// @param in_threshold_array is an array of threshold values for testing
		void SetThresholds(std::vector<float> &in_threshold_array);

	private:
		/// Template of the input image file names
		char *m_input_template;

		/// Start index of the sequence
		unsigned int m_seq_start_idx;

		/// Last index of the sequence
		unsigned int m_seq_stop_idx;

		/// Template of the groundtruth image file names
		char *m_gt_template;
		
		/// Start index of the testing part of the sequence
		unsigned int m_gt_start_idx;
		
		/// Last index of the testing part of the sequence
		unsigned int m_gt_stop_idx;
		
		/// Array of threshold values for testing
		std::vector<float> m_threshold_array;

		/// Name of the file with image
		char *m_file_name;

		/// Constant that contains 16 8-bit elements each of which is equal to 1
		static const __m128i m_ONES_U8;

		/// Method to reset buffer for name of files
		void ResetFileNameBuffer();

		/// Method to compute subtraction statistics
		/// @param in_mask is an array of computed foreground indicators for each
		/// pixel stored row-by-row
		/// @param in_num_rows is a number of rows in the image
		/// @param in_num_cols is a number of columns in the image
		/// @param in_mask_step is a shift between two consecutive rows in the mask
		/// @param in_gt is an array of ground truth foreground indicators for each
		/// pixel stored row-by-row
		/// @param in_gt_step is a shift between two consecutive rows in the ground
		/// truth mask
		/// @param out_tp is a number of true positives
		/// @param out_tn is a number of true negative
		/// @param out_fp is a number of false positives
		/// @param out_fn is a number of false negative
		static void ComputeStatistics(const unsigned char *in_mask,
			unsigned int in_num_rows, unsigned int in_num_cols, size_t in_mask_step,
			const unsigned char *in_gt, size_t in_gt_step, unsigned int &out_tp,
			unsigned int &out_tn, unsigned int &out_fp, unsigned int &out_fn);

		/// Method to compute subtraction statistics for one row of the image
		/// @param in_mask_row is an array of computed foreground indicators for each
		/// pixel in a row
		/// @param in_num_cols is a number of columns in the image
		/// @param in_gt_row is an array of ground truth foreground indicators for each
		/// pixel in a row
		/// @param out_tp is a number of true positives
		/// @param out_tn is a number of true negative
		/// @param out_fp is a number of false positives
		/// @param out_fn is a number of false negative
		static inline void ComputeStatisticsRowSSE(const unsigned char *in_mask_row,
			unsigned int in_num_cols, const unsigned char *in_gt_row,
		 	unsigned int &out_tp, unsigned int &out_tn, unsigned int &out_fp,
		 	unsigned int &out_fn);
		
		/// Method to compute subtraction statistics for one array of blocks.
		/// @param in_mask_ptr is a pointer to the first element of computed
		/// foreground indicators to process;
		/// @param in_gt_ptr is a pointer to the first element of ground truth
		/// foreground indicators to process;
		/// @param in_num_blocks is a number of blocks. It must to be less than 255;
		/// @param out_tp is a number of true positives stored in __m128i as an array
		/// of 16 8-bit elements;
		/// @param out_tn is a number of true negative stored in __m128i as an array
		/// of 16 8-bit elements;
		/// @param out_fp is a number of false positives stored in __m128i as an
		/// array of 16 8-bit elements;
		/// @param out_fn is a number of false negative stored in __m128i as an array
		/// of 16 8-bit elements.
		static inline void ComputeStatisticsCellSSE(const unsigned char *in_mask_ptr,
			const unsigned char *in_gt_ptr, const unsigned char in_num_blocks,
		 	__m128i &out_tp, __m128i &out_tn, __m128i &out_fp, __m128i &out_fn);

		/// Method to add from 16 element array of uint8_t to 4 element array
		/// of uint32_t
		/// @param io_result is a result 4 element array of uint32_t;
		/// @param in_input is a input 16 element array of uint8_t.
		static inline void SumSSE8to32bit(__m128i &io_result, __m128i in_input);
};
