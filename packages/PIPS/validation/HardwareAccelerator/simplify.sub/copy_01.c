#include <stdio.h>
#include "freia.h"

int copy_01(void)
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *open, *tmp, *out;
   int32_t ret_0;
   
   const int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};

   /* open videos flow */
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);
   
   /* images creation */
   in = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   open = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   tmp = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   out = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   
   /* processing */
   freia_common_rx_image(in, &fdin);

   tmp = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   
   ret_0 = freia_aipo_copy(tmp, in);
   ret_0 |= freia_aipo_erode_8c(open, tmp, kernel1x3);
   ret_0 |= freia_aipo_copy(tmp, open);
   ret_0 |= freia_aipo_erode_8c(open, tmp, kernel1x3);
   ret_0 |= freia_aipo_copy(tmp, open);

   ret_0 |= freia_aipo_erode_8c(out, open, kernel1x3);

   freia_common_destruct_data(tmp);
   
   freia_common_tx_image(out, &fdout);
   
   /* images destruction */
   freia_common_destruct_data(in);
   freia_common_destruct_data(open);
   freia_common_destruct_data(tmp);
   freia_common_destruct_data(out);
   
   /* close videos flow */
   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
