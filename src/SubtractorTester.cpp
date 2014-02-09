#include "SubtractorTester.h"
#include <string.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <xmmintrin.h>

#define NUM_CHANNELS 4

const unsigned int NUM_ELEMS_PER_BLOCK = 16;
const unsigned int NUM_BLOCKS_PER_CELL = 255;
const __m128i SubtractorTester::m_ONES_U8 = _mm_set1_epi8(1);

SubtractorTester::SubtractorTester()
 	: m_seq_start_idx(0), m_seq_stop_idx(0), m_gt_start_idx(0),
	m_gt_stop_idx(0), m_input_template(nullptr), m_gt_template(nullptr),
  m_file_name(nullptr)
{
}

SubtractorTester::~SubtractorTester()
{
	if (m_input_template != nullptr)
		delete [] m_input_template;
	if (m_gt_template != nullptr)
		delete [] m_gt_template;
  if (m_file_name != nullptr)
    delete [] m_file_name;
}

void SubtractorTester::Test(std::vector<float> &out_precision_array,
		std::vector<float> &out_recall_array)
{
	if (m_subtractor == nullptr)
	{
		std::cerr << "Subtractor was not set!" << std::endl;
		return;
	}
	if (m_seq_start_idx >= m_seq_stop_idx)
	{
		std::cerr << "Last index of the input sequence must be greater than start "
		  "index!" << std::endl;
		return;
	}
	if (m_gt_start_idx >= m_gt_stop_idx)
	{
		std::cerr << "Last index of the testing part of the sequence must be greater"
			" than start index!" << std::endl;
		return;
	}
	if (m_gt_start_idx < m_seq_start_idx)
	{
		std::cerr << "Start index of the input sequence must be smaller or equal "
			"than start index of the testing part!" << std::endl;
		return;
	}
	if (m_gt_stop_idx > m_seq_stop_idx)
	{
		std::cerr << "Last index of the input sequence must be greater or equal than "
			"stop index of the testing part!" << std::endl;
	}
	if (m_input_template == nullptr)
	{
		std::cerr << "Template of the input sequence file names is unset!"
		 	<< std::endl;
		return;
	}
	if (m_gt_template == nullptr)
	{
		std::cerr << "Template of the groundtruth sequence file names is unset!"
			<< std::endl;
		return;
	}
  if (m_threshold_array.empty())
  {
    std::cerr << "Thresholds of the algorithm were not set!" << std::endl;
    return;
  }

  ResetFileNameBuffer();
  unsigned long tp, tn, fp, fn;
  unsigned int local_tp, local_tn, local_fp, local_fn;

  sprintf(m_file_name, m_input_template, m_seq_start_idx);
  cv::Mat img;
#if NUM_CHANNELS == 1
  img = cv::imread(m_file_name, CV_LOAD_IMAGE_GRAYSCALE);
#elif NUM_CHANNELS == 3
  img = cv::imread(m_file_name);
#else
  cv::cvtColor(cv::imread(m_file_name), img, CV_RGB2RGBA);
#endif
  cv::Mat mask(img.rows, img.cols, CV_8UC1);
  cv::Mat gt;

  auto num_thresholds = m_threshold_array.size();
  out_precision_array.resize(num_thresholds);
  out_recall_array.resize(num_thresholds);

  for (auto threshold_idx = 0; threshold_idx < num_thresholds; ++threshold_idx)
  {
    auto threshold = m_threshold_array[threshold_idx];
    tp = tn = fp = fn = 0;

    // train background model
#if NUM_CHANNELS == 1
    img = cv::imread(m_file_name, CV_LOAD_IMAGE_GRAYSCALE);
#elif NUM_CHANNELS == 3
    img = cv::imread(m_file_name);
#else
    cv::cvtColor(cv::imread(m_file_name), img, CV_RGB2RGBA);
#endif
    m_subtractor->Train(img.data, img.rows, img.cols, img.channels(), img.step);

    m_subtractor->SetThreshold(threshold);

    for (auto frame_idx = m_seq_start_idx; frame_idx < m_gt_start_idx;
        ++frame_idx)
    {
      sprintf(m_file_name, m_input_template, frame_idx);
#if NUM_CHANNELS == 1
    img = cv::imread(m_file_name, CV_LOAD_IMAGE_GRAYSCALE);
#elif NUM_CHANNELS == 3
    img = cv::imread(m_file_name);
#else
    cv::cvtColor(cv::imread(m_file_name), img, CV_RGB2RGBA);
#endif
      m_subtractor->Subtract(img.data, img.rows, img.cols, img.channels(),
        img.step, mask.data, mask.step);
    }

    // test on real images
    for (auto frame_idx = m_gt_start_idx; frame_idx <= m_gt_stop_idx;
        ++frame_idx)
    {
      sprintf(m_file_name, m_input_template, frame_idx);
      img = cv::imread(m_file_name);
      m_subtractor->Subtract(img.data, img.rows, img.cols, img.channels(),
        img.step, mask.data, mask.step);
      sprintf(m_file_name, m_gt_template, frame_idx);
      gt = cv::imread(m_file_name);
      ComputeStatistics(mask.data, mask.rows, mask.cols, mask.step, gt.data,
        gt.step, local_tp, local_tn, local_fp, local_fn);
      tp += local_tp;
      tn += local_tn;
      fp += local_fp;
      fn += local_fn;
    }
    if (tp != 0)
    {
      out_precision_array[threshold_idx] = (float)tp / (tp + fp);
      out_recall_array[threshold_idx] = (float)tp / (tp + fn);
    }
    else
    {
      out_precision_array[threshold_idx] = out_recall_array[threshold_idx] = 0;
    }
  }
}

void SubtractorTester::SetInputTemplate(const char *in_template)
{
	auto num_elems = strlen(in_template) + 1;
	m_input_template = new char[num_elems];
	if (m_input_template == nullptr)
	{
		std::cerr << "Cannot allocate memory for input template!" << std::endl;
		return;
	}
	strcpy(m_input_template, in_template);
}

void SubtractorTester::SetStartIndex(unsigned int in_idx)
{
	m_seq_start_idx = in_idx;
}

void SubtractorTester::SetStopIndex(unsigned int in_idx)
{
	m_seq_stop_idx = in_idx;
}

void SubtractorTester::SetGTTemplate(const char *in_template)
{
	auto num_elems = strlen(in_template) + 1;
	m_gt_template = new char[num_elems];
	if (m_gt_template == nullptr)
	{
		std::cerr << "Cannot allocate memory for gt template!" << std::endl;
		return;
	}
	strcpy(m_gt_template, in_template);
}

void SubtractorTester::SetGTStartIndex(unsigned int in_idx)
{
	m_gt_start_idx = in_idx;
}

void SubtractorTester::SetGTStopIndex(unsigned int in_idx)
{
	m_gt_stop_idx = in_idx;
}

void SubtractorTester::SetThresholds(std::vector<float> &in_threshold_array)
{
	m_threshold_array = in_threshold_array;
}

void SubtractorTester::ResetFileNameBuffer()
{
  if (m_file_name != nullptr)
    delete [] m_file_name;
  size_t num_elems = strlen(m_gt_template);
  size_t tmp_num_elems = strlen(m_input_template);
  num_elems = (num_elems > tmp_num_elems ? num_elems : tmp_num_elems) + 11;
  m_file_name = new char[num_elems];
  if (m_file_name == nullptr)
  {
    std::cerr << "Cannot allocate memory for the file name buffer!"
      << std::endl;
    return;
  }
}

void SubtractorTester::ComputeStatistics(const unsigned char *in_mask,
	unsigned int in_num_rows, unsigned int in_num_cols, size_t in_mask_step,
	const unsigned char *in_gt, size_t in_gt_step, unsigned int &out_tp,
	unsigned int &out_tn, unsigned int &out_fp, unsigned int &out_fn)
{
  unsigned int row_tp, row_tn, row_fp, row_fn, tp, tn, fp, fn;
	tp = tn = fp = fn = 0;
	const unsigned char *mask_row, *gt_row, *mask_elem, *gt_elem;

//#pragma omp parallel for private(mask_row,gt_row,mask_elem,gt_elem,row_tp,\
	row_tn,row_fp,row_fn) reduction(+:tp,tn,fp,fn)
	for (auto row_idx = 0U; row_idx < in_num_rows; ++row_idx)
	{
		mask_row = in_mask + in_mask_step * row_idx;
    gt_row = in_gt + in_gt_step * row_idx;
#if 1
		ComputeStatisticsRowSSE(mask_row, in_num_cols, gt_row, row_tp, row_tn,
			row_fp, row_fn);
		tp += row_tp;
		tn += row_tn;
		fp += row_fp;
		fn += row_fn;
#else
    for (auto col_idx = 0U; col_idx < in_num_cols; ++col_idx)
    {
      unsigned int gt_res = gt_row[col_idx] ? 1 : 0;
      unsigned int mask_res = mask_row[col_idx] ? 1 : 0;
      if (mask_res)
      {
        tp += gt_res;
        fp += 1 - gt_res;
      }
      else
      {
        fn += gt_res;
        tn += 1 - gt_res;
      }
    }
#endif
	}
  out_tp = tp;
  out_tn = tn;
  out_fp = fp;
  out_fn = fn;
}


inline void SubtractorTester::ComputeStatisticsRowSSE(
    const unsigned char *in_mask_row, unsigned int in_num_cols,
    const unsigned char *in_gt_row, unsigned int &out_tp,
    unsigned int &out_tn, unsigned int &out_fp, unsigned int &out_fn)
{ 
  const unsigned int NUM_LEFT_ELEMS = in_num_cols % NUM_ELEMS_PER_BLOCK;
  const unsigned int NUM_BLOCKS = in_num_cols / NUM_ELEMS_PER_BLOCK;
  const unsigned int NUM_LEFT_BLOCKS = NUM_BLOCKS % NUM_BLOCKS_PER_CELL;
  const bool last_is_not_full = NUM_LEFT_BLOCKS > 0;
  const unsigned int NUM_FULL_CELLS = NUM_BLOCKS / NUM_BLOCKS_PER_CELL;

  // pointer to elems to process
  auto mask_elem = in_mask_row;
  auto gt_elem = in_gt_row;
  // statistics in an array of uint32_t format
  __m128i tp = _mm_setzero_si128();
  __m128i tn = _mm_setzero_si128();
  __m128i fp = _mm_setzero_si128();
  __m128i fn = _mm_setzero_si128();
  // statistics for a cell in an array of uint8_t format
  __m128i cell_tp = _mm_setzero_si128();
  __m128i cell_tn = _mm_setzero_si128();
  __m128i cell_fp = _mm_setzero_si128();
  __m128i cell_fn = _mm_setzero_si128();
  // variable for convertions
  __m128i local_statistics;

  // step between pointers for two cells
  size_t step = sizeof(unsigned char) * NUM_ELEMS_PER_BLOCK *
    NUM_BLOCKS_PER_CELL;

  // process full cells
  for (auto cell_idx = 0U; cell_idx < NUM_FULL_CELLS; ++cell_idx)
  {
    ComputeStatisticsCellSSE(mask_elem, gt_elem, NUM_BLOCKS_PER_CELL,
      cell_tp, cell_tn, cell_fp, cell_fn);
    SumSSE8to32bit(tp, cell_tp);
    SumSSE8to32bit(tn, cell_tn);
    SumSSE8to32bit(fp, cell_fp);
    SumSSE8to32bit(fn, cell_fn);
    mask_elem += step;
    gt_elem += step;
  }

  // process left blocks
  if (last_is_not_full)
  {
    step = sizeof(unsigned char) * NUM_ELEMS_PER_BLOCK * NUM_LEFT_BLOCKS;
    ComputeStatisticsCellSSE(mask_elem, gt_elem, NUM_LEFT_BLOCKS, cell_tp,
      cell_tn, cell_fp, cell_fn);
    SumSSE8to32bit(tp, cell_tp);
    SumSSE8to32bit(tn, cell_tn);
    SumSSE8to32bit(fp, cell_fp);
    SumSSE8to32bit(fn, cell_fn);
    mask_elem += step;
    gt_elem += step;
  }

	out_tp = out_tn = out_fp = out_fn = 0;

  unsigned char tmp_array[16];
  _mm_storeu_si128((__m128i*)tmp_array, tp);
  for (auto array_idx = 0U; array_idx < NUM_ELEMS_PER_BLOCK; ++array_idx)
    out_tp += tmp_array[array_idx];
  _mm_store_si128((__m128i*)tmp_array, tn);
  for (auto array_idx = 0U; array_idx < NUM_ELEMS_PER_BLOCK; ++array_idx)
    out_tn += tmp_array[array_idx];
  _mm_store_si128((__m128i*)tmp_array, fp);
  for (auto array_idx = 0U; array_idx < NUM_ELEMS_PER_BLOCK; ++array_idx)
    out_fp += tmp_array[array_idx];
  _mm_store_si128((__m128i*)tmp_array, fn);
  for (auto array_idx = 0U; array_idx < NUM_ELEMS_PER_BLOCK; ++array_idx)
    out_fn += tmp_array[array_idx];
  // left elements processing
  for (auto col_idx = 0; col_idx < NUM_LEFT_ELEMS; ++col_idx)
  {
    unsigned int gt_res = *gt_elem ? 1 : 0;
    unsigned int mask_res = *mask_elem ? 1 : 0;
    if (mask_res)
    {
      out_tp += gt_res;
      out_fp += 1 - gt_res;
    }
    else
    {
      out_fn += gt_res;
      out_tn += 1 - gt_res;
    }
    ++mask_elem;
    ++gt_elem;
  }
}

inline void SubtractorTester::ComputeStatisticsCellSSE(
    const unsigned char *in_mask_ptr, const unsigned char *in_gt_ptr,
    const unsigned char in_num_blocks, __m128i &out_tp, __m128i &out_tn,
    __m128i &out_fp, __m128i &out_fn)
{
  out_tp = _mm_setzero_si128();
  out_tn = _mm_setzero_si128();
  out_fp = _mm_setzero_si128();
  out_fn = _mm_setzero_si128();

  auto mask_elem = in_mask_ptr;
  auto gt_elem = in_gt_ptr;

	__m128i mask_res_sse, gt_res_sse, local_statistic;

  for (auto block_idx = 0; block_idx < in_num_blocks; ++block_idx)
  {
    // load results
    mask_res_sse = _mm_loadu_si128((__m128i*)mask_elem);
    gt_res_sse = _mm_loadu_si128((__m128i*)gt_elem);

    // convert to 0-1 format
    mask_res_sse = _mm_min_epu8(mask_res_sse, m_ONES_U8);
    gt_res_sse = _mm_min_epu8(gt_res_sse, m_ONES_U8);

    // true positive
    local_statistic = _mm_and_si128(mask_res_sse, gt_res_sse);
    out_tp = _mm_add_epi8(local_statistic, out_tp);

    // true negative
    local_statistic = _mm_or_si128(mask_res_sse, gt_res_sse);
    local_statistic = _mm_sub_epi8(m_ONES_U8, local_statistic);
    out_tn = _mm_add_epi8(local_statistic, out_tn);
    
    // false positive
    local_statistic = _mm_andnot_si128(gt_res_sse, mask_res_sse);
    out_fp = _mm_add_epi8(local_statistic, out_fp);

    // false negative
    local_statistic = _mm_andnot_si128(mask_res_sse, gt_res_sse);
    out_fn = _mm_add_epi8(local_statistic, out_fn);

    mask_elem += NUM_ELEMS_PER_BLOCK;
    gt_elem += NUM_ELEMS_PER_BLOCK;
  }
}

inline void SubtractorTester::SumSSE8to32bit(__m128i &io_result,
  __m128i in_input)
{
  __m128i local;
  // convert to 16-bit
  local = _mm_unpacklo_epi8(in_input, _mm_setzero_si128());
  in_input = _mm_unpackhi_epi8(in_input, _mm_setzero_si128());
  in_input = _mm_add_epi16(in_input, local);
  // convert to 32-bit
  local = _mm_unpacklo_epi16(in_input, _mm_setzero_si128());
  in_input = _mm_unpackhi_epi16(in_input, _mm_setzero_si128());
  in_input = _mm_add_epi32(in_input, local);
  // add to result vector
  io_result = _mm_add_epi32(io_result, in_input);
}
