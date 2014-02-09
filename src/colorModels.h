#pragma once

/// Converts RGB color model to HSV color model
/// @param in_rgb is an array of red, green and blue channels.
/// @param out_hsv is an array of hue, saturation and value
void rgb2hsv(const unsigned char *in_rgb, float *out_hsv);
