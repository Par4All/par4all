/**
   \file toggle.c
   \author Michel Bilodeau
   \version $Id$
   \date August 2011

   \brief exercise toggle mapping




*/

#include <stdio.h>
#include "freia.h"

// better graph display
#define imIn in
#define imOut out

#define DEPTH 3
#define CONNEX 8


/**
   toggleMapping is a contrast enhancement function that works with
   two pre-computed images and a reference image.  When the difference
   between the reference image and the 1st image is lower than between
   the reference image and 2nd one, the output takes the value of the
   first image. The pre-computed image is usually a dilation and an
   erosion of the reference image.  More generally, the pre-computed
   iamges must be an extensive one and an anti-extensive one.


\param[out] imOut output image \param[in] imIn reference image
\param[in] imCompare1 1st comparison image 
\param[in] imCompare2 2nd
comparison image 
\param[in] "bigger" if (imCompare1-imIn) > (imCompare2-imIn) is true take the value of imCompare1. Otherwise take
the imCompare2 value

*/


void toggleMapping(freia_data2d *imOut, freia_data2d *imIn, freia_data2d *imCompareOver, freia_data2d *imCompareUnder)
{
    freia_data2d *work1, *work2, *mask;
    uint32_t sx, sy, nx, ny;

    work1 = freia_common_create_data(imIn->bpp, imIn->widthWa, imIn->heightWa);
    work2 = freia_common_create_data(imIn->bpp, imIn->widthWa, imIn->heightWa);
    mask = freia_common_create_data(imIn->bpp, imIn->widthWa, imIn->heightWa);
  
  
    freia_aipo_sub( work1, imCompareOver, imIn); 
    freia_aipo_sub(work2 , imIn, imCompareUnder);

    /*  take the biggest difference  and create a mask*/
    freia_aipo_sup(mask, work1, work2); 
    freia_aipo_sub(mask, work1, mask); /*  mask pixels at 0 have biggest difference between imIn and imCompareOver */


    freia_aipo_threshold(mask, mask, 1,255, true); /*  binarize */
    /* now select image pixels according to the mask */
    freia_aipo_replace_const(work1, mask, imCompareOver, 255);

    freia_aipo_replace_const(work2, mask, imCompareUnder, 0);
    freia_aipo_sup(imOut, work1, work2);

    freia_common_destruct_data(work1);
    freia_common_destruct_data(work2);
    freia_common_destruct_data(mask);
}

int main(int argc, char *argv[])
{
  freia_dataio fdin, fdout;
  freia_data2d *imIn, *imWork1, *imWork2, *imOut;
  int i, j;
  int32_t measure_vol;

  //time_t nowtime;
  //struct tm *nowtm;
  char tmbuf[64], buf[64];

  freia_initialize(argc, argv);

  freia_common_open_input(&fdin, 0); 
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  imIn = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imWork1 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imWork2 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imOut= freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);



  // input
  freia_common_rx_image(imIn, &fdin);
  // make a border


 
  freia_cipo_dilate(imWork1, imIn, 8, DEPTH);
  freia_cipo_erode(imWork2, imIn, 8, DEPTH);


  
  toggleMapping(imOut, imIn, imWork1, imWork2);
 

  freia_aipo_global_vol(imOut, &measure_vol);

  /* print volume for verification purpose */
  printf("volume %d area %g\n", measure_vol, measure_vol/(fdin.framewidth*fdin.frameheight*1.0));

  freia_common_tx_image(imOut, &fdout);


  // cleanup
  freia_common_destruct_data(imIn);

  freia_common_destruct_data(imWork1);
  freia_common_destruct_data(imWork2);
  freia_common_destruct_data(imOut);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);
  freia_shutdown();

  return 0;
}
