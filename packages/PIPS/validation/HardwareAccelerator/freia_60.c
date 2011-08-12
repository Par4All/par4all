#include <stdio.h>
#include "freia.h"

int freia_60(freia_dataio * in, freia_dataio * out)
{
  freia_data2d *im1, *im2, *im3, *im4;

  // allocate
  im1 = freia_common_create_data(in->framebpp, in->framewidth, in->frameheight);
  im2 = freia_common_create_data(in->framebpp, in->framewidth, in->frameheight);
  im3 = freia_common_create_data(in->framebpp, in->framewidth, in->frameheight);
  im4 = freia_common_create_data(in->framebpp, in->framewidth, in->frameheight);

  // read images
  freia_common_rx_image(im1, in);
  freia_common_rx_image(im2, in);
  freia_common_rx_image(im3, in);

  // computation
  freia_aipo_inf_const(im1, im1, 150);
  freia_aipo_inf_const(im2, im2, 200);
  freia_aipo_inf_const(im3, im3, 250);
  freia_aipo_sup(im4, im2, im1);
  freia_aipo_sup(im3, im3, im4);

  // output images
  freia_common_tx_image(im1, out);
  freia_common_tx_image(im2, out);
  freia_common_tx_image(im3, out);
  freia_common_tx_image(im4, out);

  // cleanup
  freia_common_destruct_data(im1);
  freia_common_destruct_data(im2);
  freia_common_destruct_data(im3);
  freia_common_destruct_data(im4);

  return 0;
}
