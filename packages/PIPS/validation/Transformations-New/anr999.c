#include <stdio.h>
#include <freiaCommon.h>
#include <freiaAtomicOp.h>
#include <freiaComplexOp.h>

int main(void) {
  freia_dataio fdin, fdout;
  freia_data2d *imin, *imout_gradient, *imout_dilate;
  int32_t measure_min, measure_vol;

  freia_common_open_input(&fdin, 0);
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);

  imin = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imout_dilate = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);
  imout_gradient = freia_common_create_data(fdin.framebpp,  fdin.framewidth, fdin.frameheight);

  freia_common_rx_image(imin, &fdin);

  // 33 "elementary" operation calls (2 + 10 + 2*10+1)
  // manually optimized library for spoc hardware: 6 calls
  freia_aipo_global_min(imin,&measure_min);
  freia_aipo_global_vol(imin,&measure_vol);
  freia_cipo_dilate(imout_dilate,imin,8,10);
  freia_cipo_gradient(imout_gradient,imin,8,10);

  printf("input global min = %d\n",measure_min);
  printf("input global volume = %d\n",measure_vol);

  freia_common_tx_image(imin, &fdout);
  freia_common_tx_image(imout_dilate, &fdout);
  freia_common_tx_image(imout_gradient, &fdout);

  freia_common_destruct_data(imin);
  freia_common_destruct_data(imout_dilate);
  freia_common_destruct_data(imout_gradient);

  freia_common_close_input(&fdin);
  freia_common_close_output(&fdout);

  return 0;
}
