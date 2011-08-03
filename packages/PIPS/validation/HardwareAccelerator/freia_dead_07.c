#include <stdio.h>
#include "freia.h"

int freia_dead_07(void)
{
  freia_dataio fdin, fdout;
  freia_data2d *in, *out;

  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  in = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  out = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  freia_common_rx_image(in, &fdin);

  freia_aipo_erode_8c(out, in, freia_morpho_kernel_8c);
  freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
  freia_aipo_not(out, in);

  freia_common_tx_image(out, &fdout);

  // cleanup
  freia_common_destruct_data(in);
  freia_common_destruct_data(out);
  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  return 0;
}
