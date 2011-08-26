#include <stdio.h>
#include "freia.h"
int main(int argc, char *argv[])
{
   freia_dataio fdin, fdout;
   freia_data2d *in, *og, *od;
   int32_t measure_max, measure_vol;
   freia_data2d *imtmp;

   freia_initialize(argc, argv);

   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

   in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   od = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   og = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   freia_common_rx_image(in, &fdin);
   
   freia_aipo_global_max(in, &measure_max);
   freia_aipo_global_vol(in, &measure_vol);
   
   freia_aipo_dilate_8c(od, in, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(od, od, freia_morpho_kernel_8c);

   imtmp = freia_common_create_data(og->bpp, og->widthWa, og->heightWa);
   
   freia_aipo_dilate_8c(imtmp, in, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp, imtmp, freia_morpho_kernel_8c);
   
   freia_aipo_erode_8c(og, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(og, og, freia_morpho_kernel_8c);
   
   freia_aipo_sub(og, imtmp, og);

   freia_common_destruct_data(imtmp);

   printf("input global max = %d\n", measure_max);
   printf("input global volume = %d\n", measure_vol);

   freia_common_tx_image(in, &fdout);
   freia_common_tx_image(od, &fdout);
   freia_common_tx_image(og, &fdout);

   freia_common_destruct_data(in);
   freia_common_destruct_data(od);
   freia_common_destruct_data(og);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   freia_shutdown();

   return 0;
}
