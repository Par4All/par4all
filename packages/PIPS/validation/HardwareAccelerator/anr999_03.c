#include "freia.h"
#include <stdio.h>

// shorten image names to ease the graph output.
#define imin in
#define imout_dilate od
#define imout_gradient og
#define imtmp_0 t

int anr999_03(void)
{
   freia_dataio fdin;
   freia_dataio fdout;
   freia_data2d *imin;
   freia_data2d *imout_gradient;
   freia_data2d *imout_dilate;
   freia_data2d *imtmp_0;

   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   imin = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imout_dilate = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imout_gradient = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imtmp_0 = freia_common_create_data(imout_gradient->bpp, imout_gradient->widthWa, imout_gradient->heightWa);

   freia_common_rx_image(imin, &fdin);

   freia_aipo_dilate_8c(imout_dilate, imin, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imout_dilate, imout_dilate, freia_morpho_kernel_8c);

   freia_aipo_dilate_8c(imtmp_0, imin, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imout_gradient, imin, freia_morpho_kernel_8c);
   freia_aipo_sub(imout_gradient, imtmp_0, imout_gradient);

   freia_common_tx_image(imin, &fdout);
   freia_common_tx_image(imout_dilate, &fdout);
   freia_common_tx_image(imout_gradient, &fdout);

   freia_common_destruct_data(imtmp_0);
   freia_common_destruct_data(imin);
   freia_common_destruct_data(imout_dilate);
   freia_common_destruct_data(imout_gradient);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
