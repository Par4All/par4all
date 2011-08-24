#include "freia.h"

freia_status median(freia_data2d *o, freia_data2d *i)
{
  int32_t c = 8;
  freia_status ret;
  freia_data2d * t = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);
  ret =  freia_cipo_close(t, i, c, 1);
  ret |= freia_cipo_open(t, t, c, 1);
  ret |= freia_cipo_close(t, t, c, 1);

  ret |= freia_aipo_inf(o, t, i);

  ret |= freia_cipo_open(t, i, c, 1);
  ret |= freia_cipo_close(t, t, c, 1);
  ret |= freia_cipo_open(t, t, c, 1);

  ret |= freia_aipo_sup(o, o, t);
  ret |= freia_common_destruct_data(t);
  return ret;
}

int main(void)
{
  freia_dataio fdin, fdout;
  freia_data2d * out, *in;
  in = freia_common_create_data(16, 1024, 720);
  out = freia_common_create_data(16, 1024, 720);
  freia_common_rx_image(in, &fdin);
  median(out, in);
  freia_common_tx_image(out, &fdout);
  freia_commen_destruct_data(out);
  freia_commen_destruct_data(in);
  return 0;
}
