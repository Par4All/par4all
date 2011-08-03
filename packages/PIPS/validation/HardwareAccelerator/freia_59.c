#include <stdio.h>
#include "freia.h"

int freia_59(void)
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *map, *tmp;
   freia_data2d *f2, *f3, *f4;
   int32_t kernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   in = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   map = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   f2 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   f3 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   f4 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   tmp = freia_common_create_data(map->bpp, map->widthWa, map->heightWa);

   freia_common_rx_image(in, &fdin);

   freia_aipo_dilate_8c(tmp, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(map, in, freia_morpho_kernel_8c);
   freia_aipo_sub(map, tmp, map);

   freia_aipo_convolution(f2, in, kernel, 3, 3);
   freia_aipo_convolution(f3, f2, kernel, 3, 3);
   freia_aipo_convolution(f4, f3, kernel, 3, 3);

   freia_common_tx_image(in, &fdout);
   freia_common_tx_image(f2, &fdout);
   freia_common_tx_image(f3, &fdout);
   freia_common_tx_image(f4, &fdout);
   freia_common_tx_image(map, &fdout);

   freia_common_destruct_data(in);
   freia_common_destruct_data(map);
   freia_common_destruct_data(f2);
   freia_common_destruct_data(f3);
   freia_common_destruct_data(f4);
   freia_common_destruct_data(tmp);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
