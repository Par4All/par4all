#include <stdio.h>
#include "freia.h"

int freia_42(void)
{
   freia_dataio fdin;
   freia_dataio fdout;

   freia_data2d *i0;
   freia_data2d *i1;
   freia_data2d *i2;
   freia_data2d *i3;
   freia_data2d *i4;
   freia_data2d *i6;
   freia_data2d *i7;
   freia_data2d *i5;

   freia_common_open_input(&fdin, 0);

   i0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i4 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i5 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i6 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   i7 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);

   freia_common_rx_image(i0, &fdin);
   freia_common_rx_image(i2, &fdin);
   freia_common_rx_image(i4, &fdin);

   freia_aipo_dilate_8c(i3, i0, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(i1, i0, freia_morpho_kernel_8c);
   freia_aipo_sub(i1, i3, i1);
   freia_aipo_absdiff(i1, i2, i1);
   freia_aipo_dilate_8c(i5, i0, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(i2, i0, freia_morpho_kernel_8c);
   freia_aipo_sub(i2, i5, i2);
   freia_aipo_erode_8c(i6, i4, freia_morpho_kernel_8c);
   freia_aipo_sub(i6, i5, i6);
   freia_aipo_absdiff(i6, i2, i6);
   freia_aipo_inf(i7, i6, i1);

   freia_common_tx_image(i7, &fdout);

   freia_common_destruct_data(i0);
   freia_common_destruct_data(i1);
   freia_common_destruct_data(i2);
   freia_common_destruct_data(i3);
   freia_common_destruct_data(i4);
   freia_common_destruct_data(i5);
   freia_common_destruct_data(i6);
   freia_common_destruct_data(i7);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);
   return 0;
}
