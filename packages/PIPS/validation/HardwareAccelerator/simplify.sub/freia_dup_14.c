#include <stdio.h>
#include "freia.h"

int freia_dup_14(void)
{
   freia_dataio fdin;
   freia_dataio fdout;

   freia_data2d *imin;
   freia_data2d *imopen;
   freia_data2d *imclose;
   freia_data2d *imout;
   
   const  int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
   const  int32_t kernel3x1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};

   freia_status _return0, _return1, _return2, _return3, _return4;
   freia_data2d *imtmp_0;
   freia_status ret_0;
   freia_data2d *imtmp_1;
   freia_status ret_1;
   
   /* open videos flow */
   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);
   
   /* images creation */
   imin = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   imopen = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   imclose = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   imout = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   
   freia_common_rx_image(imin, &fdin);

   imtmp_0 = freia_common_create_data(imopen->bpp, imopen->widthWa, imopen->heightWa);

   ret_0 = freia_aipo_copy(imtmp_0, imin);
   ret_0 |= freia_aipo_erode_8c(imopen, imtmp_0, kernel1x3);

   freia_common_destruct_data(imtmp_0);
   
   imtmp_1 = freia_common_create_data(imclose->bpp, imclose->widthWa, imclose->heightWa);

   ret_1 = freia_aipo_copy(imtmp_1, imin);
   ret_1 |= freia_aipo_dilate_8c(imclose, imtmp_1, kernel1x3);

   freia_common_destruct_data(imtmp_1);

   freia_aipo_and(imout, imopen, imclose);

   freia_common_tx_image(imout, &fdout);
   
   /* images destruction */
   freia_common_destruct_data(imin);
   freia_common_destruct_data(imopen);
   freia_common_destruct_data(imclose);
   freia_common_destruct_data(imout);
   
   /* close videos flow */
   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
