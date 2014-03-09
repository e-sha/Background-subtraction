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
	return testOnCam();
#endif
  BaseSubtractor *subtractor = new SimpleSubtractor;
  std::vector<float> precision_array, recall_array;
	int num_thresholds = 10;
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
    computeSobel(frame.data, frame.rows, frame.cols, frame.step, mask.data, mask.step);
    cv::imshow("Frame", frame);
    cv::imshow("Mask", mask);
    if (cv::waitKey(30) >= 0)
      break;

  }
  delete subtractor;
	return 0;
}
