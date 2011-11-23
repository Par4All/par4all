/**
   \file antibio.c
   \author Michel Bilodeau
   \version $Id$
   \date November 2011

   Computes the diameter of the halos on a antibiogram test. The bigger is the halo, the stronger
   is the reaction of the antibiotic on the sample.




*/

#include <stdio.h>
#include "freia.h"
#include "freiaExtendedOpenMorpho.h"


#define CONNEX 8

/**
   Dection of the diameter of each spot of an antibiogram
   
   \param[out] output image. each spot of the antibiogram has a value proportional to the diameter of the reaction
   \param[in] input image of an antibiogram
   \param[in] threshold about the same value then the dynamic of spots
   \param[in] second threshold about of the value of the background.
*/
void antibio(freia_data2d *imOut, freia_data2d *imIn, const int32_t par1, const int32_t par2){

	freia_data2d * w1 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
	freia_data2d * w2 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
	freia_data2d * w3 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
	freia_data2d * w4 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);

	// 1st step detect centers of halos
	freia_cipo_dilate( w1, imIn, CONNEX , 2);
	freia_cipo_erode(w1, w1, CONNEX, 5);
	freia_cipo_erode(w2, w1, CONNEX, 4);
	freia_cipo_geodesic_reconstruct_dilate(w1, imIn, CONNEX);

	freia_cipo_geodesic_reconstruct_dilate(w2, imIn, CONNEX);

	freia_aipo_sub(w1, w1, w2);

	freia_aipo_threshold(w1, w1, par1, w1->bpp==16?32767: 255, true);

	freia_cipo_dilate(w1, w1, 8, 1);
	freia_aipo_copy(w2, w1);

	// 2nd detect halos
	freia_aipo_threshold(w3,imIn, par2, imIn->bpp==16?32767: 255, true);
	freia_aipo_not(w3, w3);

	freia_aipo_sup(w4, w1, w3);

	freia_cipo_close(w4, w4, 8, 1);
	freia_ecipo_distance(w3, w4, 8);

	freia_aipo_inf(w4, w3, w1);

	freia_cipo_geodesic_dilate(imOut, w4, w1, 8, 10);


	freia_common_destruct_data(w1);
	freia_common_destruct_data(w2);
	freia_common_destruct_data(w3);
	freia_common_destruct_data(w4);
}
	
int main(int argc, char *argv[])
{
        freia_dataio fdin, fdout;
	freia_data2d *imIn,  *imOut;
	int i, j;

	freia_initialize(argc, argv);

	freia_common_open_input(&fdin, 0); 
	freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

	imIn = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
	imOut = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);

	// input
	freia_common_rx_image(imIn, &fdin);

	// do the computation
	antibio(imOut, imIn, 30, 30);

	// rearrange for display
	freia_aipo_mul_const(imOut, imOut, 4); // For display

	// Write images
	freia_common_tx_image(imIn, &fdout);
	freia_common_tx_image(imOut, &fdout);

	// cleanup
	freia_common_destruct_data(imIn);

	freia_common_destruct_data(imOut);
	freia_common_close_input(&fdin);
	freia_common_close_output(&fdout);
	freia_shutdown();

	return 0;
}
