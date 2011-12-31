#include <stdio.h>
#include "freia.h"
int main(int argc, char **argv)
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *out, *immap, *cst;
   freia_data2d *filter1, *filter2, *filter3, *filter4;
   freia_data2d *filter5, *filter6, *filter7, *filter8;
   freia_data2d *implan1, *implan2, *implan3, *implan4;
   freia_data2d *implan5, *implan6, *implan7, *implan8;
   int32_t max;
   int32_t min;
   int32_t kernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
   //PIPS generated variable
   register freia_status _return0, _return1, _return2;
   //PIPS generated variable
   register freia_status ret;
   //PIPS generated variable
   freia_data2d *imtmp;
   //PIPS generated variable
   register int i_0, i_1;

   freia_initialize(argc, argv);
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   in = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   immap = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   out = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   cst = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

   filter1 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter2 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter3 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter4 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter5 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter6 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter7 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   filter8 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

   implan1 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan2 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan3 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan4 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan5 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan6 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan7 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   implan8 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

   freia_common_rx_image(in, &fdin);
   
   // const elaboration
   freia_aipo_xor(cst, cst, cst);
   freia_aipo_add_const(cst, cst, 7);
   
   // map elaboration
   
   imtmp = freia_common_create_data(immap->bpp, immap->widthWa, immap->heightWa);
   
   
   freia_aipo_dilate_8c(imtmp, in, freia_morpho_kernel_8c);
   i_0 = 1;
   
   
   _return1 = 0;
   ret = 0;
   
   
   freia_aipo_erode_8c(immap, in, freia_morpho_kernel_8c);
   i_1 = 1;
   
   
   _return2 = 0;
   ret |= freia_aipo_sub(immap, imtmp, immap);

   freia_common_destruct_data(imtmp);
   _return0 = 0;
   
   //freia_aipo_not(immap, immap);
   freia_aipo_global_max(immap, &max);
   freia_aipo_global_min(immap, &min);
   freia_aipo_sub_const(immap, immap, min);
   freia_aipo_mul_const(immap, immap, 32767/(max-min));
   freia_aipo_div_const(immap, immap, 128);
   freia_aipo_log2(immap, immap);
   freia_aipo_sub(immap, cst, immap);

   freia_aipo_copy(filter1, in);
   freia_aipo_convolution(filter2, filter1, kernel, 3, 3);
   freia_aipo_convolution(filter3, filter2, kernel, 3, 3);
   freia_aipo_convolution(filter4, filter3, kernel, 3, 3);
   freia_aipo_convolution(filter5, filter4, kernel, 3, 3);
   freia_aipo_convolution(filter6, filter5, kernel, 3, 3);
   freia_aipo_convolution(filter7, filter6, kernel, 3, 3);
   freia_aipo_convolution(filter8, filter7, kernel, 3, 3);
   
   // select filters
   freia_aipo_replace_const(implan1, immap, filter1, 0);
   freia_aipo_replace_const(implan2, immap, filter2, 1);
   freia_aipo_replace_const(implan3, immap, filter3, 2);
   freia_aipo_replace_const(implan4, immap, filter4, 3);
   freia_aipo_replace_const(implan5, immap, filter5, 4);
   freia_aipo_replace_const(implan6, immap, filter6, 5);
   freia_aipo_replace_const(implan7, immap, filter7, 6);
   freia_aipo_replace_const(implan8, immap, filter8, 7);
   
   // merge results
   freia_aipo_copy(out, implan1);
   freia_aipo_or(out, out, implan2);
   freia_aipo_or(out, out, implan3);
   freia_aipo_or(out, out, implan4);
   freia_aipo_or(out, out, implan5);
   freia_aipo_or(out, out, implan6);
   freia_aipo_or(out, out, implan7);
   freia_aipo_or(out, out, implan8);

   freia_common_tx_image(in, &fdout);
   freia_common_tx_image(out, &fdout);
   
   
   
   
   
   freia_common_destruct_data(in);
   freia_common_destruct_data(immap);
   freia_common_destruct_data(out);
   freia_common_destruct_data(cst);
   freia_common_destruct_data(filter1);
   freia_common_destruct_data(filter2);
   freia_common_destruct_data(filter3);
   freia_common_destruct_data(filter4);
   freia_common_destruct_data(filter5);
   freia_common_destruct_data(filter6);
   freia_common_destruct_data(filter7);
   freia_common_destruct_data(filter8);
   freia_common_destruct_data(implan1);
   freia_common_destruct_data(implan2);
   freia_common_destruct_data(implan3);
   freia_common_destruct_data(implan4);
   freia_common_destruct_data(implan5);
   freia_common_destruct_data(implan6);
   freia_common_destruct_data(implan7);
   freia_common_destruct_data(implan8);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   freia_shutdown();
   return 0;
}
