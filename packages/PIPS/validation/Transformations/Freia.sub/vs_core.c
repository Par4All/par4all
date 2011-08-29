#include "freia.h"

int main(void)
{
  freia_dataio fdin, fdout;
  freia_data2d
    *in = freia_common_create_data(16, 128, 128),
    *mv = freia_common_create_data(16, 128, 128),
    *bg = freia_common_create_data(16, 128, 128),
    *st = freia_common_create_data(16, 128, 128),
    *t0 = freia_common_create_data(16, 128, 128),
    *t1 = freia_common_create_data(16, 128, 128);
  int binvalue, motion_a, motion_b, motion_trig, motion_th;
  int maxmotion, minmotion;

  freia_common_open_input(&fdin, 0); 
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);

  freia_common_rx_image(bg, &fdin);
  freia_common_rx_image(st, &fdin);
  freia_common_rx_image(in, &fdin);

  freia_aipo_absdiff(mv, in, bg);

  // bg update
  freia_aipo_copy(t0, in);
  freia_aipo_mul_const(t0, t0, motion_a);
  freia_aipo_mul_const(bg, bg, motion_b);
  freia_aipo_add(bg, bg, t0);
  freia_aipo_div_const(bg, bg, motion_a+motion_b);

  // measure
  freia_aipo_global_max(mv, &maxmotion);
  freia_aipo_global_min(mv, &minmotion);

  if ((maxmotion-minmotion) > motion_trig)
    binvalue = (maxmotion*motion_th)/100;

  freia_aipo_threshold(mv, mv, binvalue, 255, true);
  freia_cipo_open(t1, mv, 8, 10);
  freia_cipo_gradient(mv, t1, 8, 10);

  freia_aipo_sub_const(mv, mv, 1);
  freia_aipo_and_const(mv, mv, 1);
  freia_aipo_mul(mv, st, mv);

  freia_common_tx_image(mv, &fdout);

  freia_common_destruct_data(in);
  freia_common_destruct_data(mv);
  freia_common_destruct_data(bg);
  freia_common_destruct_data(st);
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);

  freia_common_close_input(&fdin); 
  freia_common_close_output(&fdout); 

  return 0;
}
