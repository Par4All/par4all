/**
   \file burner.c
   \author Michel Bilodeau
   \version $Id$
   \date July 2011

   \brief burner demo

   The objective is to segmentate a burner chamber in few zones.

     This is taken from a Micromorh example
*/

#include <stdio.h>
#include "freia.h"
#include "freiaExtendedOpenMorpho.h"

// improved demo graph display
#define iminBorder in
#define imWork2 out

#define DEPTH 100
#define DEPTH_BUILD 2
#define CONNEX 8
#define MUL 2

/**
   draw a border around the image

   \param[out] output image
   \param[in]  border value

*/
void drawBorder(freia_data2d *imOut, freia_data2d *imIn, int borderValue)
{
  freia_aipo_copy(imOut, imIn);
  freia_common_draw_rect(imOut, imOut->xStartWa, imOut->yStartWa,
			 imOut->xStartWa+imOut->widthWa-1, imOut->yStartWa+imOut->heightWa-1,
			 borderValue);
}

int main(int argc, char * argv[])
{
  freia_dataio fdin, fdout;
  freia_data2d *imin, *iminBorder, *imTophat ,*imClose, *imZone1, *imZone2, *imZone3, *imWork1, *imWork2;

  static const int32_t freia_morpho_k8_center_east[9] = {0, 0, 0, 0, 1, 1, 0, 0, 0};
  static const int32_t freia_morpho_k8_center_north[9] = {0, 1, 0, 0, 1, 0, 0, 0, 0};

  // there should be a freia_initialize()? yup we should have one for opencl
  // 1st arg: 1=GPU, 2=CPU, 2nd arg sub-device
  freia_initialize(argc, argv);
  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

#if 0
  // hmmm... this looks like post initialization?
  if(  freia_op_init_opencl(2, 0, false, fdin.framewidth, fdin.frameheight))
    exit(0);
#endif

  imin = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  iminBorder = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imClose= freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imTophat= freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imZone1 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imZone2 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imZone3 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);

  imWork1 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imWork2 = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);

  // input
  freia_common_rx_image(imin, &fdin);
  drawBorder(iminBorder, imin, 255);

  // Closing
  freia_ecipo_close(imWork1, iminBorder, freia_morpho_k8_center_east, CONNEX, DEPTH);
  freia_ecipo_close(imWork2, iminBorder, freia_morpho_k8_center_north, CONNEX, DEPTH);
  freia_aipo_inf(imClose, imWork1, imWork2);

  // Tophat
  freia_aipo_sub(imTophat, imClose, imin);

  // Thresholding in different zones
  freia_aipo_threshold(imZone1, imTophat, 105, 255, true);
  freia_aipo_threshold(imZone2, imTophat, 60, 105, true);
  freia_aipo_threshold(imZone3, imClose, 100, 150, true);

  // Put all zones in one and filter it
  freia_aipo_sup(imWork1, imZone1, imZone2);
  freia_aipo_sup(imWork1, imWork1, imZone3);
  freia_aipo_not(imWork1, imWork1);
  freia_cipo_geodesic_reconstruct_close(imWork2, imWork1, CONNEX, DEPTH_BUILD);

  // Change  pixel values for the display
  freia_aipo_inf_const(imZone1, imZone1, 40);
  freia_aipo_inf_const(imZone2, imZone2, 100);
  freia_aipo_inf_const(imZone3, imZone3, 170);
  freia_aipo_inf_const(imWork2, imWork2, 250);

  freia_aipo_sup(imWork1, imZone2, imZone1);
  freia_aipo_sup(imWork1, imZone3, imWork1);
  freia_aipo_sup(imWork2, imWork2, imWork1);

  // output
  freia_common_tx_image(imin, &fdout);
  freia_common_tx_image(imWork2, &fdout);

#ifdef DEBUG
  // not shown by default, optional, didactic purpose
  freia_common_tx_image(imClose, &fdout);
  freia_common_tx_image(imZone1, &fdout);
  freia_common_tx_image(imZone2, &fdout);
  freia_common_tx_image(imZone3, &fdout);
  freia_common_tx_image(imTophat, &fdout);
#endif // DEBUG

  // cleanup
  freia_common_destruct_data(imin);
  freia_common_destruct_data(iminBorder);
  freia_common_destruct_data(imTophat);
  freia_common_destruct_data(imClose);
  freia_common_destruct_data(imZone1);
  freia_common_destruct_data(imZone2);
  freia_common_destruct_data(imZone3);
  freia_common_destruct_data(imWork1);
  freia_common_destruct_data(imWork2);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  freia_shutdown();
  return 0;
}
