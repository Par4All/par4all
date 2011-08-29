#include <stdio.h>
#include "freia.h"

int main(int argc, char *argv[])
{
  freia_dataio fdin, fdout;

  freia_data2d *in, *immir, *imopen, *imclose, *imopenth, *imcloseth;
  freia_data2d *imand, *imfilt, *imout, *out;

  const  int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
  const  int32_t kernel3x1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
  freia_data2d *imtmp_0, *imtmp_1, *imtmp_2, *imtmp_3;
  freia_status ret;

  freia_initialize(argc, argv);

  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);

  in = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  immir = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imopen = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imclose = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imopenth = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imcloseth = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imand = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imfilt = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imout = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  out = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

  freia_common_rx_image(in, &fdin);

  imtmp_0 = freia_common_create_data(imopen->bpp, imopen->widthWa, imopen->heightWa);

  ret = freia_aipo_copy(imtmp_0, in);

  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);

  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);
  ret |= freia_aipo_dilate_8c(imopen, imtmp_0, kernel1x3);
  ret |= freia_aipo_copy(imtmp_0, imopen);

  freia_common_destruct_data(imtmp_0);

  imtmp_1 = freia_common_create_data(imclose->bpp, imclose->widthWa, imclose->heightWa);

  ret |= freia_aipo_copy(imtmp_1, in);

  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);

  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);
  ret |= freia_aipo_erode_8c(imclose, imtmp_1, kernel1x3);
  ret |= freia_aipo_copy(imtmp_1, imclose);

  freia_common_destruct_data(imtmp_1);

  ret |= freia_aipo_threshold(imopenth, imopen, 1, 50, 1);
  ret |= freia_aipo_threshold(imcloseth, imclose, 150, 255, 1);
  ret |= freia_aipo_and(imand, imopenth, imcloseth);

  imtmp_2 = freia_common_create_data(imfilt->bpp, imfilt->widthWa, imfilt->heightWa);

  ret |= freia_aipo_copy(imtmp_2, imand);

  ret |= freia_aipo_erode_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_erode_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_erode_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_erode_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);

  ret |= freia_aipo_dilate_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_dilate_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_dilate_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);
  ret |= freia_aipo_dilate_8c(imfilt, imtmp_2, kernel3x1);
  ret |= freia_aipo_copy(imtmp_2, imfilt);

  freia_common_destruct_data(imtmp_2);

  imtmp_3 = freia_common_create_data(imout->bpp, imout->widthWa, imout->heightWa);

  ret |= freia_aipo_copy(imtmp_3, imfilt);

  ret |= freia_aipo_erode_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_erode_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_erode_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_erode_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);

  ret |= freia_aipo_dilate_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_dilate_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_dilate_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);
  ret |= freia_aipo_dilate_8c(imout, imtmp_3, kernel1x3);
  ret |= freia_aipo_copy(imtmp_3, imout);

  freia_common_destruct_data(imtmp_3);

  ret |= freia_aipo_dilate_8c(out, imout, freia_morpho_kernel_8c);
  ret |= freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
  ret |= freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
  ret |= freia_aipo_and(out, out, in);

  freia_common_tx_image(out, &fdout);

  freia_common_destruct_data(in);
  freia_common_destruct_data(immir);
  freia_common_destruct_data(imopen);
  freia_common_destruct_data(imclose);
  freia_common_destruct_data(imopenth);
  freia_common_destruct_data(imcloseth);
  freia_common_destruct_data(imand);
  freia_common_destruct_data(imfilt);
  freia_common_destruct_data(imout);
  freia_common_destruct_data(out);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  freia_shutdown();
  return ret;
}
