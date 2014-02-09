#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "SimpleSubtractor.h"
#include "SubtractorTester.h"
#include "Configuration.h"

#include "SimpleSubtractorImpl3.h"

#define MAIN_NUM_CHANNELS 3

#if DEBUG
void test3()
{
  char a;
  const unsigned int NUM_ROWS = 16;
  const unsigned int NUM_COLS = 16;

  cv::Mat bg(NUM_ROWS, NUM_COLS, CV_8UC3);
  cv::Mat img(NUM_ROWS, NUM_COLS, CV_8UC3);
  cv::Mat mask(NUM_ROWS, NUM_COLS, CV_8UC1);
  unsigned char *bg_row, *img_row, *mask_row;
  unsigned char base;
  for (auto row_idx = 0U; row_idx < NUM_ROWS; ++row_idx)
  {
    bg_row = bg.data + row_idx * bg.step;
    img_row = img.data + row_idx * img.step;
    base = row_idx * bg.cols * 3;
    for (auto col_idx = 0U; col_idx < 3 * NUM_COLS; ++col_idx)
    {
      bg_row[col_idx] = 0;
      img_row[col_idx] = (col_idx / 3 == row_idx) ? 255 : bg_row[col_idx]; 
    }
  }

  SimpleSubtractorImpl3 s;
  s.Train(bg.data, bg.rows, bg.cols, bg.step);
  s.Subtract(img.data, img.rows, img.cols, img.step, mask.data, mask.step);
  for (auto row_idx = 0U; row_idx < NUM_ROWS; ++row_idx)
  {
    mask_row = mask.data + row_idx * mask.step;
    for (auto col_idx = 0U; col_idx < NUM_COLS; ++col_idx)
    {
      auto value_is_incorrect =
        mask_row[col_idx] != (row_idx == col_idx ? 255 : 0);
      if (value_is_incorrect)
        std::cout << row_idx << ',' << col_idx << std::endl;
    }
  }
  std::cin >> a;
}

void map(unsigned int in_value)
{
  auto channel_idx = in_value % 3;
  auto pixel_idx = in_value / 3;
  switch (channel_idx)
  {
    case 0:
      std::cout << 'r';
      break;
    case 1:
      std::cout << 'g';
      break;
    case 2:
      std::cout << 'b';
      break;
  }
  std::cout << pixel_idx;
}

void test3Mapping()
{
  float data[48];
  float res[48];
  for (auto idx = 0; idx < 48; ++idx)
    data[idx] = idx;

  for (auto pixel_idx = 0; pixel_idx < 16; ++pixel_idx)
  {
    auto base = pixel_idx * 3;
    for (auto channel_idx = 0; channel_idx < 3; ++channel_idx)
    {
      map(data[base + channel_idx]);
      std::cout << ' ';
    }
    std::cout << std::endl;
  }

  __m128 data_sse[12];
  for (auto idx = 0; idx < 12; ++idx)
    data_sse[idx] = _mm_loadu_ps(data + idx * 4);

  SimpleSubtractorImpl3::ShufflePixels(data_sse[0], data_sse[1], data_sse[2],
    data_sse[3], data_sse[4], data_sse[5], data_sse[6], data_sse[7],
    data_sse[8], data_sse[9], data_sse[10], data_sse[11]);

  for (auto idx = 0; idx < 12; ++idx)
    _mm_storeu_ps(res + idx * 4, data_sse[idx]);

  for (auto idx = 0; idx < 12; ++idx)
  {
    auto base = idx * 4;
    for (auto elem_idx = 0; elem_idx < 4; ++elem_idx)
    {
      map(res[base + elem_idx]);
      std::cout << ' ';
    }
    std::cout << std::endl;
  }
}

int testOnCam();

void computeSobel(const unsigned char *in_img, const unsigned int in_num_rows,
  const unsigned int in_num_cols, const unsigned int in_step,
  unsigned char *out_edges, const unsigned int in_edges_step)
{
  const unsigned char *img_row;
  const unsigned char *prev_row;
  const unsigned char *next_row;
  unsigned char *edges_row;
  const unsigned char *img_elem;
  const unsigned char *prev_elem;
  const unsigned char *next_elem;
  for (auto col_idx = 0; col_idx < in_num_cols; ++col_idx)
    out_edges[col_idx] = 0;
  for (auto row_idx = 1; row_idx < in_num_rows - 1; ++row_idx)
  {
    img_row = in_img + row_idx * in_step;
    prev_row = img_row - in_step;
    next_row = img_row + in_step;
    edges_row = out_edges + in_edges_step * row_idx;
    prev_elem = img_row;
    img_elem = prev_elem + 3;
    next_elem = img_elem + 3;
    for (auto col_idx = 1; col_idx < in_num_cols - 1; ++col_idx)
    {
      short dx = 2 * (next_elem[0] - prev_elem[0]) + 
        *(next_elem - in_step) + *(next_elem + in_step) -
        (*(prev_elem - in_step) + *(prev_elem + in_step));
      short dy = 2 * (next_row[0] - prev_row[0]) + 
        *(next_row - 3) + *(next_row + 3) -
        (*(prev_row - 3) + *(prev_row + 3));
      edges_row[col_idx] = sqrt((dx * dx + dy * dy) / 2) / 4;
      edges_row[0] = 0;
      edges_row[in_num_cols - 1] = 0;

      prev_elem = img_elem;
      img_elem = next_elem;
      next_elem += 3;
    }
  }
  edges_row = out_edges + (in_num_rows - 1) * in_step;
  for (auto col_idx = 0; col_idx < in_num_cols; ++col_idx)
    edges_row[col_idx] = 0;
}

#endif

int main(int argc, const char* argv[])
{
#if DEBUG == 1
  test3Mapping();
  test3();
  return 0;
	return testOnCam();
#endif
  BaseSubtractor *subtractor = new SimpleSubtractor;
  std::vector<float> precision_array, recall_array;
	int num_thresholds = 2;
	std::vector<float> threshold_array(num_thresholds);
	for (auto threshold_idx = 0; threshold_idx < num_thresholds; ++threshold_idx)
	{
		threshold_array[threshold_idx] =
      255 * (float)threshold_idx / (num_thresholds - 1);
	}
	SubtractorTester *tester = new SubtractorTester;
	tester->SetInputTemplate("/home/e_sha/Dropbox/Documents/programs/background_subtraction/data/office/input/in%06d.jpg");
	tester->SetStartIndex(1);
	tester->SetStopIndex(2050);
	tester->SetGTTemplate("/home/e_sha/Dropbox/Documents/programs/background_subtraction/data/office/groundtruth/gt%06d.png");
	tester->SetGTStartIndex(570);
	tester->SetGTStopIndex(2050);
	tester->SetThresholds(threshold_array);
  tester->SetSubtractor(subtractor);
  tester->Test(precision_array, recall_array);
  delete tester;
  delete subtractor;
  for (auto threshold_idx = 0U; threshold_idx < num_thresholds; ++threshold_idx)
  {
    std::cout << precision_array[threshold_idx] << "% "
      << recall_array[threshold_idx] << "%" << std::endl;
  }
  return 0;
}

int testOnCam()
{
  cv::VideoCapture cap(0);
  if (!cap.isOpened())
    return -1;

  cv::Mat frame, gray_frame;
  cv::namedWindow("Frame");
  cv::namedWindow("Mask");

  cap >> frame;
#if MAIN_NUM_CHANNELS == 1
  cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
  frame = gray_frame;
#elif MAIN_NUM_CHANNELS == 3
#elif MAIN_NUM_CHANNELS == 4
  cv::cvtColor(frame, gray_frame, CV_RGB2RGBA);
  frame = gray_frame;
#endif
  BaseSubtractor *subtractor = new SimpleSubtractor;
  subtractor->Train(frame.data, frame.rows, frame.cols, frame.channels(),
    frame.step);
  cv::Mat mask(frame.rows, frame.cols, CV_8UC1);

  for (;;)
  {
    cap >> frame;
#if MAIN_NUM_CHANNELS == 1
    cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
    frame = gray_frame;
#elif MAIN_NUM_CHANNELS == 3
#elif MAIN_NUM_CHANNELS == 4
    cv::cvtColor(frame, gray_frame, CV_RGB2RGBA);
    frame = gray_frame;
#endif
    subtractor->Subtract(frame.data, frame.rows, frame.cols, frame.channels(),
      frame.step, mask.data, mask.step);
//    computeSobel(frame.data, frame.rows, frame.cols, frame.step, mask.data, mask.step);
    cv::imshow("Frame", frame);
    cv::imshow("Mask", mask);
    if (cv::waitKey(30) >= 0)
      break;

  }
  delete subtractor;
	return 0;
}
