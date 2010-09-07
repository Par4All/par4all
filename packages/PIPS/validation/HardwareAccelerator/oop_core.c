#include <stdio.h>
#include "freia.h"

int oop_core(void)
{
   freia_dataio fdin;
   freia_dataio fdout;
   freia_data2d *img0;
   freia_data2d *img1;
   freia_data2d *img2;
   freia_data2d *img3;

   // five last images
   freia_data2d *imgtt0;
   freia_data2d *imgtt1;
   freia_data2d *imgtt2;
   freia_data2d *imgtt3;
   freia_data2d *imgtt4;

   freia_data2d *imgtmp;
   freia_data2d *imgg1;
   freia_data2d *imgg2;
   freia_data2d *imgsav;
   freia_data2d *imtmp_4;
   freia_data2d *imtmp_5;
   freia_data2d *imtmp_6;
   freia_data2d *imtmp_7;

   freia_status end;

   // Input/output stream and image creations
   freia_common_open_input(&fdin, 0);

   img0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);

   img1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   img2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   img3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   imgtt0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imgtt1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imgtt2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imgtt3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imgtt4 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   imgg1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imgg2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

   imgtmp = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);


   end = freia_common_rx_image(imgtt4, &fdin);
   end &= freia_common_rx_image(imgtt3, &fdin);
   end &= freia_common_rx_image(imgtt2, &fdin);
   end &= freia_common_rx_image(imgtt1, &fdin);
   end &= freia_common_rx_image(imgtt0, &fdin);
   end &= freia_common_rx_image(img0, &fdin);

   freia_aipo_copy(img1, img0);
   freia_aipo_copy(imgtt0, img1);

   imtmp_4 = freia_common_create_data
        (imgtmp->bpp, imgtmp->widthWa, imgtmp->heightWa);

   freia_aipo_dilate_8c(imtmp_4, imgtt0, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imgtmp, imgtt0, freia_morpho_kernel_8c);
   freia_aipo_sub(imgtmp, imtmp_4, imgtmp);

   freia_common_destruct_data(imtmp_4);

   imtmp_5 = freia_common_create_data
        (img1->bpp, img1->widthWa, img1->heightWa);

   freia_aipo_dilate_8c(imtmp_5, imgtt2, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(img1, imgtt2, freia_morpho_kernel_8c);
   freia_aipo_sub(img1, imtmp_5, img1);
   freia_common_destruct_data(imtmp_5);

   freia_aipo_absdiff(img1, imgtmp, img1);
   freia_aipo_erode_6c(img1, img1, freia_morpho_kernel_6c);
   freia_aipo_dilate_6c(img1, img1, freia_morpho_kernel_6c);

   imtmp_6 = freia_common_create_data
        (imgtmp->bpp, imgtmp->widthWa, imgtmp->heightWa);

   freia_aipo_dilate_8c(imtmp_6, imgtt2, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imgtmp, imgtt2, freia_morpho_kernel_8c);
   freia_aipo_sub(imgtmp, imtmp_6, imgtmp);
   freia_common_destruct_data(imtmp_6);

   imtmp_7 = freia_common_create_data
     (img2->bpp, img2->widthWa, img2->heightWa);
   freia_aipo_dilate_8c(imtmp_7, imgtt4, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(img2, imgtt4, freia_morpho_kernel_8c);
   freia_aipo_sub(img2, imtmp_7, img2);
   freia_common_destruct_data(imtmp_7);

   freia_aipo_absdiff(img2, imgtmp, img2);

   freia_aipo_erode_6c(img2, img2, freia_morpho_kernel_6c);
   freia_aipo_dilate_6c(img2, img2, freia_morpho_kernel_6c);
   freia_aipo_inf(img3, img2, img1);
   freia_aipo_threshold(img3, img3, 15, 255, 1);

   // skipped stuff... drawing on img0...

   freia_common_tx_image(img3, &fdout);

   /* images destruction */
   freia_common_destruct_data(img0);
   freia_common_destruct_data(img1);
   freia_common_destruct_data(img2);
   freia_common_destruct_data(img3);
   freia_common_destruct_data(imgtt0);
   freia_common_destruct_data(imgtt1);
   freia_common_destruct_data(imgtt2);
   freia_common_destruct_data(imgtt3);

   freia_common_destruct_data(imgtt4);
   freia_common_destruct_data(imgg1);
   freia_common_destruct_data(imgg2);

   freia_common_destruct_data(imgtmp);

   /* close videos flow */
   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);
   return 0;
}
