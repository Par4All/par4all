#include "freia.h"
#include <stdio.h>

int freia_dup_15(void)
{
   freia_dataio fdin;
   freia_dataio fdout;
   freia_data2d *in;
   freia_data2d *og;
   freia_data2d *od;
   freia_data2d *t;

   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   od = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   og = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   t = freia_common_create_data(og->bpp, og->widthWa, og->heightWa);

   freia_common_rx_image(in, &fdin);

   freia_aipo_dilate_8c(od, in, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(t, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, in, freia_morpho_kernel_8c);
   freia_aipo_sub(og, t, og);

   freia_common_tx_image(od, &fdout);
   freia_common_tx_image(og, &fdout);

   freia_common_destruct_data(in);
   freia_common_destruct_data(od);
   freia_common_destruct_data(og);
   freia_common_destruct_data(t);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
