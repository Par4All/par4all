#include <stdio.h>
#include "freia.h"

freia_status vs_core(
  freia_data2d *bg, freia_data2d *mv, freia_data2d *in, freia_data2d *stab,
  int motion_a, int motion_b, int motion_trig, int motion_th)
{
  freia_data2d
    *t0 = freia_common_create_data(16, 128, 128),
    *t1 = freia_common_create_data(16, 128, 128),
    *t2 = freia_common_create_data(16, 128, 128);
  int binvalue, maxmotion, minmotion;

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

  // open
  freia_aipo_erode_8c(t1, mv, freia_morpho_kernel_8c);
  freia_aipo_dilate_8c(t1, t1, freia_morpho_kernel_8c);

  // gradient
  freia_aipo_dilate_8c(t2, t1, freia_morpho_kernel_8c);
  freia_aipo_erode_8c(mv, t1, freia_morpho_kernel_8c);
  freia_aipo_sub(mv, t2, mv);

  freia_aipo_sub_const(mv, mv, 1);
  freia_aipo_and_const(mv, mv, 1);
  freia_aipo_mul(mv, stab, mv);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  return FREIA_OK;
}
