#include <stdio.h>
#include "freia.h"

int freia_51(int32_t * k)
{
  freia_dataio fdin, fdout;
  freia_data2d *in, *out, *t0, *t1, *t2, *t3, *t4, *t5;
  in = freia_common_create_data(16, 1024, 1024);
  t0 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t2 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t3 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  // t4 is later
  t5 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  out = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  int32_t volcurrent, volprevious;

  freia_common_rx_image(in, &fdin);

  freia_aipo_erode_8c(out, in, k);
  freia_aipo_erode_8c(t1, in, k);
  freia_aipo_sub(t1, in, t1);
  freia_aipo_not(out, in);

  t4 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);

  freia_aipo_set_constant(t3, 255);
  freia_aipo_dilate_8c(t2, out, k);
  freia_aipo_inf(t3, t3, t2);
  freia_aipo_copy(out, t3);
  freia_aipo_sub(t0, in, out);
  freia_aipo_set_constant(t5, 255);
  freia_aipo_dilate_8c(t4, out, k);
  freia_aipo_inf(t5, t5, t4);
  freia_aipo_dilate_8c(t4, out, k);
  freia_aipo_inf(out, t5, t4);

  freia_aipo_global_vol(out, &volcurrent);
  do {
    volprevious = volcurrent;
    freia_aipo_dilate_8c(out, out, k);
    freia_aipo_inf(out, out, in);
    freia_aipo_global_vol(out, &volcurrent);
  }
  while (volcurrent!=volprevious);

  freia_aipo_sub(out, in, out);
  freia_aipo_mul_const(out, out, 2);

  freia_common_tx_image(out, &fdout);

  freia_common_destruct_data(in);
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);
  freia_common_destruct_data(out);

  return 0;
}
