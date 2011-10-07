#include <stdio.h>
#include "freia.h"

// shorter image names
#define imin in
#define imout out
#define imconst cst

int main(int argc, char ** argv)
{
  freia_dataio fdin, fdout;
  freia_data2d *imin, *imout, *immap,*imconst;
  freia_data2d *filter1, *filter2, *filter3, *filter4;
  freia_data2d *filter5, *filter6, *filter7, *filter8;
  freia_data2d *imout_plan1, *imout_plan2, *imout_plan3, *imout_plan4;
  freia_data2d *imout_plan5, *imout_plan6, *imout_plan7, *imout_plan8;
  int32_t max;
  int32_t min;
  int32_t kernel[] = {1,1,1,1,1,1,1,1,1};

  freia_initialize(argc, argv);
  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  imin = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  immap = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imconst = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);

  filter1 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter2 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter3 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter4 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter5 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter6 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter7 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  filter8 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);

  imout_plan1 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan2 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan3 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan4 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan5 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan6 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan7 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);
  imout_plan8 = freia_common_create_data(16,  fdin.framewidth, fdin.frameheight);

  freia_common_rx_image(imin, &fdin);

  // const elaboration
  freia_aipo_xor(imconst,imconst, imconst);
  freia_aipo_add_const(imconst, imconst, 7);
 
  // map elaboration
  freia_cipo_gradient(immap,imin,8,1);

  //freia_aipo_not(immap, immap);
  freia_aipo_global_max(immap,&max);
  freia_aipo_global_min(immap,&min);
  freia_aipo_sub_const(immap,immap,min);
  freia_aipo_mul_const(immap,immap,32767/(max-min));
  freia_aipo_div_const(immap,immap,128);
  freia_aipo_log2(immap,immap);
  freia_aipo_sub(immap,imconst,immap);

  freia_aipo_copy(filter1, imin);
  freia_aipo_convolution(filter2, filter1, kernel, 3, 3);
  freia_aipo_convolution(filter3, filter2, kernel, 3, 3);
  freia_aipo_convolution(filter4, filter3, kernel, 3, 3);
  freia_aipo_convolution(filter5, filter4, kernel, 3, 3);
  freia_aipo_convolution(filter6, filter5, kernel, 3, 3);
  freia_aipo_convolution(filter7, filter6, kernel, 3, 3);
  freia_aipo_convolution(filter8, filter7, kernel, 3, 3);

  // select filters
  freia_aipo_replace_const(imout_plan1, immap, filter1, 0);
  freia_aipo_replace_const(imout_plan2, immap, filter2, 1);
  freia_aipo_replace_const(imout_plan3, immap, filter3, 2);
  freia_aipo_replace_const(imout_plan4, immap, filter4, 3);
  freia_aipo_replace_const(imout_plan5, immap, filter5, 4);
  freia_aipo_replace_const(imout_plan6, immap, filter6, 5);
  freia_aipo_replace_const(imout_plan7, immap, filter7, 6);
  freia_aipo_replace_const(imout_plan8, immap, filter8, 7);

  // merge results
  freia_aipo_copy(imout, imout_plan1);
  freia_aipo_or(imout, imout, imout_plan2);
  freia_aipo_or(imout, imout, imout_plan3);
  freia_aipo_or(imout, imout, imout_plan4);
  freia_aipo_or(imout, imout, imout_plan5);
  freia_aipo_or(imout, imout, imout_plan6);
  freia_aipo_or(imout, imout, imout_plan7);
  freia_aipo_or(imout, imout, imout_plan8);
 
  freia_common_tx_image(imin, &fdout);
  freia_common_tx_image(imout, &fdout);
#ifdef DEBUG
  freia_common_tx_image(immap, &fdout);
  freia_common_tx_image(filter8, &fdout);
#endif // DEBUG

  freia_common_destruct_data(imin);
  freia_common_destruct_data(immap);
  freia_common_destruct_data(imout);
  freia_common_destruct_data(imconst);
  freia_common_destruct_data(filter1);
  freia_common_destruct_data(filter2);
  freia_common_destruct_data(filter3);
  freia_common_destruct_data(filter4);
  freia_common_destruct_data(filter5);
  freia_common_destruct_data(filter6);
  freia_common_destruct_data(filter7);
  freia_common_destruct_data(filter8);
  freia_common_destruct_data(imout_plan1);
  freia_common_destruct_data(imout_plan2);
  freia_common_destruct_data(imout_plan3);
  freia_common_destruct_data(imout_plan4);
  freia_common_destruct_data(imout_plan5);
  freia_common_destruct_data(imout_plan6);
  freia_common_destruct_data(imout_plan7);
  freia_common_destruct_data(imout_plan8);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  freia_shutdown();
  return 0;
}
