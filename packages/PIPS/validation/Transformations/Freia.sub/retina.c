/**
   \file retina.c
   \author Michel Bilodeau
   \version $Id$
   \date July 2011

   \brief retina demo

   The objective is to extract anevrism, which are small spots around vessels.

   To extract them, we will compare the efficiency of 3 tophats:

     1) regular
     2) sup of linear ones
     3) sup of linear one with reconstruction

     This is taken from a Micromorh example
*/

#include <stdio.h>
#include "freia.h"
#include "freiaExtendedOpenMorpho.h"

#define DEPTH 5
#define CONNEX 8
#define MUL 2

// nicer show
#define imin in
#define imWork1 out
#define imout_tophat top
#define imout_tophatLin lin
#define imout_tophatLinRecon rec

int main(int argc, char * argv[])
{
  freia_dataio fdin, fdout;
  freia_data2d *imin, *imout_tophat ,*imout_tophatLin, *imout_tophatLinRecon, *imWork1, *imWork2;

  freia_initialize(argc, argv);
  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  // there should be a freia_initialize()?
  // for opencl freia_op_init_opencl(0xffffffff, 2, false, fdin.framewidth, fdin.frameheight);
  imin = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imout_tophat = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imout_tophatLinRecon = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imout_tophatLin = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imWork1 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imWork2 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);

  // input
  freia_common_rx_image(imin, &fdin);

  // 1st tophat
  freia_cipo_open(imWork1, imin, CONNEX, DEPTH);
  // freia_common_tx_image(imWork1, &fdout);
  freia_cipo_open_tophat(imout_tophat, imin, CONNEX, DEPTH);
  freia_aipo_mul_const(imout_tophat, imout_tophat, MUL); // for display

  // 2nd tophat
  freia_ecipo_sup_open(imWork1, imin, CONNEX, DEPTH);
  // freia_common_tx_image(imWork1, &fdout);
  freia_aipo_sub(imout_tophatLin,  imin, imWork1);
  freia_aipo_mul_const(imout_tophatLin, imout_tophatLin, MUL);// for display

  // 3nd tophat
  freia_ecipo_sup_open(imWork1, imin, CONNEX, DEPTH);

  freia_cipo_geodesic_reconstruct_dilate (imWork1, imin, CONNEX);
  freia_aipo_sub(imout_tophatLinRecon, imin, imWork1);
  freia_aipo_mul_const(imout_tophatLinRecon, imout_tophatLinRecon, MUL);// for display
  // threshold one these tophat
  freia_aipo_threshold(imWork1, imout_tophatLinRecon, 90, 255, true);
  // the threshold over the source image
  freia_aipo_sup(imWork2, imWork1, imin);

  // output
  freia_common_tx_image(imin, &fdout);
  freia_common_tx_image(imWork1, &fdout);
#ifdef DEBUG
  // some more output for debugging
  freia_common_tx_image(imout_tophat, &fdout);
  freia_common_tx_image(imout_tophatLin, &fdout);
  freia_common_tx_image(imout_tophatLinRecon, &fdout);
  freia_common_tx_image(imWork2, &fdout);
#endif // DEBUG

  // cleanup
  freia_common_destruct_data(imin);
  freia_common_destruct_data(imout_tophat);
  freia_common_destruct_data(imout_tophatLin);
  freia_common_destruct_data(imout_tophatLinRecon);
  freia_common_destruct_data(imWork1);
  freia_common_destruct_data(imWork2);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  freia_shutdown();
  return 0;
}
