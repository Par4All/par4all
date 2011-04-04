#include <stdio.h>
#include "freia.h"

int license_plate(void)
{
   freia_dataio fdin;
   freia_dataio fdout;

   freia_data2d *in;
   freia_data2d *immir;
   freia_data2d *imopen;
   freia_data2d *imclose;
   freia_data2d *imopenth;
   freia_data2d *imcloseth;
   freia_data2d *imand;
   freia_data2d *imfilt;
   freia_data2d *imout;
   freia_data2d *out;
   
   const int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
   const int32_t kernel3x1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};

   // PIPS generated variable
   int _return0, _return1, _return2, _return3, _return4;
   freia_status ret_0;
   int i_0;
   freia_status ret_1;
   int i_1;
   freia_status ret_2;
   int i_2;
   freia_status ret_3;
   int i_3, i_4;
   
   /* open videos flow */
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);
   
   /* images creation */
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
   
   /* processing */
   freia_common_rx_image(in, &fdin);
   
   ret_0 = freia_aipo_erode_8c(imopen, in, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_erode_8c(imopen, imopen, kernel1x3);
   i_0 = 15;

   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   ret_0 |= freia_aipo_dilate_8c(imopen, imopen, kernel1x3);
   i_0 = 15;
   
   _return1 = ret_0;

   ret_1 = freia_aipo_dilate_8c(imclose, in, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_dilate_8c(imclose, imclose, kernel1x3);
   i_1 = 8;

   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   ret_1 |= freia_aipo_erode_8c(imclose, imclose, kernel1x3);
   i_1 = 8;
   
   
   _return0 = ret_1;

   freia_aipo_threshold(imopenth, imopen, 1, 50, 1);
   freia_aipo_threshold(imcloseth, imclose, 150, 255, 1);

   freia_aipo_and(imand, imopenth, imcloseth);
   
   ret_2 = freia_aipo_erode_8c(imfilt, imand, kernel3x1);
   ret_2 |= freia_aipo_erode_8c(imfilt, imfilt, kernel3x1);
   ret_2 |= freia_aipo_erode_8c(imfilt, imfilt, kernel3x1);
   ret_2 |= freia_aipo_erode_8c(imfilt, imfilt, kernel3x1);
   i_2 = 4;

   ret_2 |= freia_aipo_dilate_8c(imfilt, imfilt, kernel3x1);
   ret_2 |= freia_aipo_dilate_8c(imfilt, imfilt, kernel3x1);
   ret_2 |= freia_aipo_dilate_8c(imfilt, imfilt, kernel3x1);
   ret_2 |= freia_aipo_dilate_8c(imfilt, imfilt, kernel3x1);
   i_2 = 4;
   
   _return2 = ret_2;
   
   ret_3 = freia_aipo_erode_8c(imout, imfilt, kernel1x3);
   ret_3 |= freia_aipo_erode_8c(imout, imout, kernel1x3);
   ret_3 |= freia_aipo_erode_8c(imout, imout, kernel1x3);
   ret_3 |= freia_aipo_erode_8c(imout, imout, kernel1x3);
   i_3 = 4;

   ret_3 |= freia_aipo_dilate_8c(imout, imout, kernel1x3);
   ret_3 |= freia_aipo_dilate_8c(imout, imout, kernel1x3);
   ret_3 |= freia_aipo_dilate_8c(imout, imout, kernel1x3);
   ret_3 |= freia_aipo_dilate_8c(imout, imout, kernel1x3);
   i_3 = 4;
   
   _return3 = ret_3;
   
   freia_aipo_dilate_8c(out, imout, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   i_4 = 3;
   
   _return4 = 0;

   freia_aipo_and(out, out, in);
   freia_common_tx_image(out, &fdout);
   
   /* images destruction */
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
   
   /* close videos flow */
   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
