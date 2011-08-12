#include <stdio.h>
#include "freia.h"

int vs_core_loop(void)
{
   freia_dataio fdin;
   freia_dataio fdmotion;

   freia_data2d *imcurrent;
   freia_data2d *imprevious;
   freia_data2d *imstab;
   freia_data2d *imtmp1;
   freia_data2d *imtmp2;
   freia_data2d *imehn;
   freia_data2d *imtmp3;
   freia_data2d *imbg16;
   freia_data2d *imbg;
   freia_data2d *immotion;
   freia_data2d *imtmp;

   int32_t maxmotion, minmotion;
   register int32_t binvalue = 128;

   register int32_t xshiftacc, yshiftacc;
   register freia_status end = 0;
   
   register uint32_t horizon = 10;
   register int32_t maximal_shape = 21;
   register int32_t minimal_contrast = 50;
   register int32_t motion_a = 10;
   register int32_t motion_b = 90;
   register int32_t motion_th = 30;
   register int32_t motion_trig = 75;

   freia_initialize(0, NULL);

   freia_common_open_input(&fdin, 0);

   imcurrent = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   imprevious = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
   imtmp1 = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

   imstab = freia_common_create_data(16, fdin.framewidth-20, fdin.frameheight-20);
   imtmp2 = freia_common_create_data(16, imstab->width, imstab->height);

   imtmp3 = freia_common_create_data(16, imstab->width, imstab->height);
   imehn = freia_common_create_data(16, imstab->width, imstab->height);
   imbg16 = freia_common_create_data(16, imstab->width, imstab->height);
   imbg = freia_common_create_data(16, imstab->width, imstab->height);
   immotion = freia_common_create_data(16, imstab->width, imstab->height);

   freia_common_open_output(&fdmotion, 2, imstab->width, imstab->height, imstab->bpp);

   freia_aipo_set_constant(imbg16, 0);
   freia_aipo_set_constant(imbg, 0);
   freia_aipo_xor(imtmp1, imtmp1, imtmp1);
   
   // application start
   xshiftacc = 0;
   yshiftacc = 0;
   
   end = freia_common_rx_image(imcurrent, &fdin);

   // video loop
l99979:   ;

   if (end==0) {
   }
   else {
      goto l99980;
   }
   
   // stabilization
   freia_aipo_copy(imprevious, imcurrent);
   end = freia_common_rx_image(imcurrent, &fdin);

   if (end!=0) goto l99980;

   freia_common_set_wa(imcurrent, xshiftacc+10, yshiftacc+10, fdin.framewidth-20, fdin.frameheight-20);
   freia_aipo_copy(imstab, imcurrent);
   freia_common_reset_wa(imcurrent);
   
   // constrast enhance
   freia_aipo_copy(imehn, imstab);
   
   // absolute difference background and current frame
   freia_aipo_absdiff(immotion, imehn, imbg);
   
   // background update
   freia_aipo_cast(imtmp3, imehn);
   freia_aipo_mul_const(imtmp3, imtmp3, 10);
   freia_aipo_mul_const(imbg16, imbg16, 90);
   freia_aipo_add(imbg16, imbg16, imtmp3);
   freia_aipo_div_const(imbg16, imbg16, 100);
   freia_aipo_cast(imbg, imbg16);
   
   // measures
   freia_aipo_global_max(immotion, &maxmotion);
   freia_aipo_global_min(immotion, &minmotion);

   if (maxmotion-minmotion>75)
      binvalue = 30*maxmotion/100;
   
   // threshold
   freia_aipo_threshold(immotion, immotion, binvalue, 255, 1);
   freia_aipo_erode_8c(imtmp2, immotion, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imtmp2, imtmp2, freia_morpho_kernel_8c);

   imtmp = freia_common_create_data(immotion->bpp, immotion->widthWa, immotion->heightWa);
   
   freia_aipo_dilate_8c(imtmp, imtmp2, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(immotion, imtmp2, freia_morpho_kernel_8c);
   freia_aipo_sub(immotion, imtmp, immotion);

   freia_common_destruct_data(imtmp);
   
   // save contrast motion detection
   freia_aipo_sub_const(immotion, immotion, 1);
   freia_aipo_and_const(immotion, immotion, 1);
   freia_aipo_mul(immotion, imstab, immotion);

   // output image
   freia_common_tx_image(immotion, &fdmotion);

   fprintf(stdout, "INFO: %s: ""frame %d processed\n", __FUNCTION__, fdin.frameindex);
   goto l99979;
l99980:   ;

   // cleanup at end of application
   freia_common_destruct_data(imcurrent);
   freia_common_destruct_data(imprevious);
   freia_common_destruct_data(imtmp1);
   freia_common_destruct_data(imstab);
   freia_common_destruct_data(imtmp2);
   freia_common_destruct_data(imtmp3);
   freia_common_destruct_data(imehn);
   freia_common_destruct_data(imbg16);
   freia_common_destruct_data(imbg);
   freia_common_destruct_data(immotion);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdmotion);

   freia_shutdown();
   return 0;
}
