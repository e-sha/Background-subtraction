#include "colorModels.h"

void rgb2hsv(const unsigned char *in_rgb, float *out_hsv)
{
	float rgb[3];
	rgb[0] = in_rgb[0] / 255.f;
	rgb[1] = in_rgb[1] / 255.f;
	rgb[2] = in_rgb[2] / 255.f;

	float max, min;
	unsigned char max_idx;
	unsigned char min_idx;

	if (rgb[0] > rgb[1])
	{
		max_idx = 0;
		min_idx = 1;
	}
	else
	{
		max_idx = 1;
		min_idx = 0;
	}

	max_idx = rgb[2] > rgb[max_idx] ? 2 : max_idx;
	min_idx = rgb[2] > rgb[min_idx] ? min_idx : 2;

	max = rgb[max_idx];
	min = rgb[min_idx];

	float dif = max - min;

	if (dif == 0)
		out_hsv[0] = 0;
	else if (max_idx == 0)
	{
		out_hsv[0] = 60 * (rgb[1] - rgb[2]) / dif;
		if (out_hsv[0] < 0)
			out_hsv[0] += 360;
	}
	else if (max_idx == 1)
		out_hsv[0] = 60 * (rgb[2] - rgb[0]) / dif + 120;
	else
		out_hsv[0] = 60 * (rgb[0] - rgb[2]) / dif + 240;

	out_hsv[1] = max > 0 ? 1 - min / max : 0;
	out_hsv[2] = max;
}
