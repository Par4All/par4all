#include <stdio.h>
#include "freia.h"

void transposeNeighbor(int32_t neighborOut[9], const int32_t neighborIn[9])
{
  int i;
  for(i=0; i < 9; i++) {
    neighborOut[8-i] = neighborIn[i];
  }
}

static const int32_t freia_morpho_k8_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k8_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
static const int32_t freia_morpho_k8_2[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

int main(int argc, char *argv[])
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *top, *lin, *rec, *out, *imWork2;
   int32_t neighborTransposed_0[9];
   int32_t neighborTransposed_1[9];
   int32_t neighborTransposed_2[9];
   freia_data2d *w2_0, *w1_0;
   int32_t neighborTransposed_3[9];
   int32_t neighborTransposed_4[9];
   int32_t neighborTransposed_5[9];
   freia_data2d *w2_1, *w1_1;
   register int32_t volprevious;
   int32_t volcurrent;

   freia_initialize(argc, argv);
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   // there should be a freia_initialize()?
   // for opencl freia_op_init_opencl(0xffffffff, 2, false, fdin.framewidth, fdin.frameheight);
   in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   top = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   rec = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   lin = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   out = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imWork2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   // input
   freia_common_rx_image(in, &fdin);

   freia_aipo_erode_8c(out, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);

   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);

   freia_aipo_erode_8c(top, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(top, top, freia_morpho_kernel_8c);

   freia_aipo_dilate_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(top, top, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(top, top, freia_morpho_kernel_8c);

   freia_aipo_sub(top, in, top);
   freia_aipo_mul_const(top, top, 2);
   freia_aipo_not(out, in);

   w1_0 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);
   w2_0 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);

   freia_aipo_set_constant(w2_0, 255);

   transposeNeighbor(neighborTransposed_2, freia_morpho_k8_0);

   freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);

   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);

   freia_aipo_inf(w2_0, w2_0, w1_0);

   transposeNeighbor(neighborTransposed_1, freia_morpho_k8_1);

   freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);

   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);

   freia_aipo_inf(w2_0, w2_0, w1_0);

   transposeNeighbor(neighborTransposed_0, freia_morpho_k8_2);

   freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);

   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);

   freia_aipo_inf(w2_0, w2_0, w1_0);

   freia_aipo_copy(out, w2_0);

   freia_common_destruct_data(w1_0);
   freia_common_destruct_data(w2_0);

   freia_aipo_not(out, out);

   freia_aipo_sub(lin, in, out);
   freia_aipo_mul_const(lin, lin, 2);

   freia_aipo_not(out, in);

   // temporary images
   w1_1 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);
   w2_1 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);

   freia_aipo_set_constant(w2_1, 255);

   transposeNeighbor(neighborTransposed_5, freia_morpho_k8_0);

   freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);

   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_5);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_5);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_5);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_5);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_5);

   freia_aipo_inf(w2_1, w2_1, w1_1);

   transposeNeighbor(neighborTransposed_4, freia_morpho_k8_1);

   freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);

   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_4);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_4);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_4);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_4);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_4);

   freia_aipo_inf(w2_1, w2_1, w1_1);

   transposeNeighbor(neighborTransposed_3, freia_morpho_k8_2);

   freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);

   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_3);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_3);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_3);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_3);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_3);

   freia_aipo_inf(w2_1, w2_1, w1_1);

   freia_aipo_copy(out, w2_1);

   freia_common_destruct_data(w1_1);
   freia_common_destruct_data(w2_1);

   freia_aipo_not(out, out);

   freia_aipo_global_vol(out, &volcurrent);
   do {
      volprevious = volcurrent;
      freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
      freia_aipo_inf(out, out, in);
      freia_aipo_global_vol(out, &volcurrent);
   }
   while (volcurrent!=volprevious);

   freia_aipo_sub(rec, in, out);
   freia_aipo_mul_const(rec, rec, 2);
   freia_aipo_threshold(out, rec, 90, 255, 1);
   freia_aipo_sup(imWork2, out, in);

   // output
   freia_common_tx_image(in, &fdout);
   freia_common_tx_image(out, &fdout);

   // cleanup
   freia_common_destruct_data(in);
   freia_common_destruct_data(top);
   freia_common_destruct_data(lin);
   freia_common_destruct_data(rec);
   freia_common_destruct_data(out);
   freia_common_destruct_data(imWork2);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   freia_shutdown();
   return 0;
}
