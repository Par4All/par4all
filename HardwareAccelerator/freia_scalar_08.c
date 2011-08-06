#include <stdio.h>
#include "freia.h"

void transposeNeighbor(int32_t neighborOut[9], const int32_t neighborIn[9])
{
  int i;
  for(i=0; i < 9; i++) {
    neighborOut[8-i] = neighborIn[i];
  }
}

int freia_scalar_08(int argc, char *argv[])
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *imClose, *imWork1, *out;

   static const int32_t k8_ce[9] = {0, 0, 0, 0, 1, 1, 0, 0, 0};
   static const int32_t k8_cn[9] = {0, 1, 0, 0, 1, 0, 0, 0, 0};
   //PIPS generated variable
   int32_t neighborTransposed_0[9];
   int32_t neighborTransposed_1[9];

   freia_initialize(argc, argv);
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imClose = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imWork1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   out = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   freia_common_rx_image(in, &fdin);

   transposeNeighbor(neighborTransposed_0, k8_ce);
   freia_aipo_dilate_8c(imWork1, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork1, imWork1, neighborTransposed_0);
   transposeNeighbor(neighborTransposed_1, k8_cn);
   freia_aipo_dilate_8c(out, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(out, out, neighborTransposed_1);
   freia_aipo_inf(imClose, imWork1, out);

   freia_common_tx_image(imClose, &fdout);

   freia_common_destruct_data(in);
   freia_common_destruct_data(imClose);
   freia_common_destruct_data(imWork1);
   freia_common_destruct_data(out);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   freia_shutdown();
   return 0;
}
