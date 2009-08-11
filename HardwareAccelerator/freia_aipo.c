#include "freia.h"

// Arith
freia_status my_inf
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_inf(o, i0, i1);
}

freia_status my_inf_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_inf_const(o, i0, p0);
}

freia_status my_sup
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_sup(o, i0, i1);
}

freia_status my_sup_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_sup_const(o, i0, p0);
}

freia_status my_sub
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_sub(o, i0, i1);
}

freia_status my_sub_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_sub_const(o, i0, p0);
}

freia_status my_subsat
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_subsat(o, i0, i1);
}

freia_status my_subsat_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_subsat_const(o, i0, p0);
}

freia_status my_add
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_add(o, i0, i1);
}

freia_status my_add_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_add_const(o, i0, p0);
}

freia_status my_addsat
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_addsat(o, i0, i1);
}

freia_status my_addsat_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_addsat_const(o, i0, p0);
}

freia_status my_absdiff
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_absdiff(o, i0, i1);
}

freia_status my_absdiff_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_absdiff_const(o, i0, p0);
}

freia_status my_mul
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_mul(o, i0, i1);
}

freia_status my_mul_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_mul_const(o, i0, p0);
}

freia_status my_div
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_div(o, i0, i1);
}

freia_status my_div_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_div_const(o, i0, p0);
}

freia_status my_and
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_and(o, i0, i1);
}

freia_status my_and_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_and_const(o, i0, p0);
}

freia_status my_or
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_or(o, i0, i1);
}

freia_status my_or_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_and_const(o, i0, p0);
}

freia_status my_xor
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_xor(o, i0, i1);
}

freia_status my_xor_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_xor_const(o, i0, p0);
}

freia_status my_not
(freia_data2d *o, freia_data2d *i0)
{
  return freia_aipo_not(o, i0);
}

// Linear
freia_status my_convolution
(freia_data2d *o, freia_data2d *i0, int32_t *p0, uint32_t p1, uint32_t p2)
{
  return freia_aipo_convolution(o, i0, p0, p1, p2);
}

freia_status my_fast_correlation
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1, uint32_t p0)
{
  return freia_aipo_fast_correlation(o, i0, i1, p0);
}

// Measure
freia_status my_global_min(freia_data2d * i0, int32_t * p0)
{
  return freia_aipo_global_min(i0, p0);
}

freia_status my_global_max(freia_data2d * i0, int32_t * p0)
{
  return freia_aipo_global_max(i0, p0);
}

freia_status my_global_min_coord
(freia_data2d * i0, int32_t * p0, int32_t * p1, int32_t * p2)
{
  return freia_aipo_global_min_coord(i0, p0, p1, p2);
}

freia_status my_global_max_coord
(freia_data2d * i0, int32_t * p0, int32_t * p1, int32_t * p2)
{
  return freia_aipo_global_max_coord(i0, p0, p1, p2);
}

freia_status my_global_vol(freia_data2d *image, int32_t *vol)
{
  return freia_aipo_global_vol(image, vol);
}

// Misc
freia_status my_copy(freia_data2d *o, freia_data2d *i0)
{
  return freia_aipo_copy(o, i0);
}

freia_status my_cast(freia_data2d *o, freia_data2d *i0)
{
  return freia_aipo_cast(o, i0);
}

freia_status my_set_constant(freia_data2d *o, int32_t p0)
{
  return freia_aipo_set_constant(o, p0);
}

freia_status my_threshold
(freia_data2d *o, freia_data2d * i0, int32_t p0, int32_t p1, int32_t p2)
{
  return freia_aipo_threshold(o, i0, p0, p1, p2);
}

// Morpho
freia_status my_erode_8c
(freia_data2d *o, freia_data2d *i0, int32_t *p0)
{
  return freia_aipo_erode_8c(o, i0, p0);
}

freia_status my_dilate_8c
(freia_data2d *o, freia_data2d *i0, int32_t *p0)
{
  return freia_aipo_dilate_8c(o, i0, p0);
}

freia_status my_erode_6c
(freia_data2d *o, freia_data2d *i0, int32_t *p0)
{
  return freia_aipo_erode_6c(o, i0, p0);
}

freia_status my_dilate_6c
(freia_data2d *o, freia_data2d *i0, int32_t *p0)
{
  return freia_aipo_dilate_6c(o, i0, p0);
}
