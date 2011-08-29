#include <stdio.h>
#include "freia.h"

int main(int argc, char *argv[])
{
  freia_dataio fdin, fdout;
  freia_data2d *in, *imWork1, *imWork2, *out;
  register int i, j;
  int32_t measure_vol;
  freia_data2d *mask, *work2, *work1;

  freia_initialize(argc, argv);

  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imWork1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imWork2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  out = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   freia_common_rx_image(in, &fdin);

   freia_aipo_dilate_8c(imWork1, in, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);

   freia_aipo_erode_8c(imWork2, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork2, imWork2, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork2, imWork2, freia_morpho_kernel_8c);

   work1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   work2 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   mask = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

   freia_aipo_sub(work1, imWork1, in);
   freia_aipo_sub(work2, in, imWork2);

   freia_aipo_sup(mask, work1, work2);
   freia_aipo_sub(mask, work1, mask);
   freia_aipo_threshold(mask, mask, 1, 255, 1);

   freia_aipo_replace_const(work1, mask, imWork1, 255);
   freia_aipo_replace_const(work2, mask, imWork2, 0);
   freia_aipo_sup(out, work1, work2);

   freia_common_destruct_data(work1);
   freia_common_destruct_data(work2);
   freia_common_destruct_data(mask);

   freia_aipo_global_vol(out, &measure_vol);

   printf("volume %d area %g\n", measure_vol,
          measure_vol/(fdin.framewidth*fdin.frameheight*1.0));

   freia_common_tx_image(out, &fdout);

   freia_common_destruct_data(in);
   freia_common_destruct_data(imWork1);
   freia_common_destruct_data(imWork2);
   freia_common_destruct_data(out);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);
   freia_shutdown();

   return 0;
}
