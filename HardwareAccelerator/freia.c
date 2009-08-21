// fake implementations for effects

#include <stdlib.h>
#include "freia.h"

// COMMON
freia_data2d * freia_common_create_data(uint32_t x , uint32_t y , uint32_t z)
{
  freia_data2d * img = (freia_data2d *) malloc(sizeof(freia_data2d));
  img->stuff = (int32_t) (x & y & z);
  return img;
}

freia_status freia_common_destruct_data(freia_data2d * img)
{
  free(img);
  return FREIA_OK;
}

freia_status freia_common_open_input(freia_dataio * in, uint32_t n)
{
  return FREIA_OK;
}

freia_status freia_common_open_output(freia_dataio *out,
   uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
  return FREIA_OK;
}

freia_status freia_common_rx_image(freia_data2d * img, freia_dataio * in)
{
  img->stuff = in;
  return FREIA_OK;
}

freia_status freia_common_tx_image(freia_data2d * img, freia_dataio * out)
{
  out = img->stuff;
  return FREIA_OK;
}

freia_status freia_common_close_input(freia_data2d * n)
{
  return FREIA_OK;
}

freia_status freia_common_close_output(freia_data2d * n)
{
  return FREIA_OK;
}

#define Fbin(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)	\
  {								\
    o->stuff = i0->stuff | i1->stuff;				\
    return FREIA_OK;						\
  }

#define FbinP(name)						\
  freia_status							\
  name(freia_data2d * o,					\
       freia_data2d * i0, freia_data2d * i1, uint32_t c)	\
  {								\
    o->stuff = i0->stuff | i1->stuff | (int32_t) c;		\
    return FREIA_OK;						\
  }

#define Fun(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i)			\
  {								\
    o->stuff = i->stuff;					\
    return FREIA_OK;						\
  }

#define FunP(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i, int32_t c)		\
  {								\
    o->stuff = i->stuff | c;					\
    return FREIA_OK;						\
  }

#define FunK(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i, int32_t * k)		\
  {								\
    o->stuff = i->stuff | *k;					\
    return FREIA_OK;						\
  }

#define FunK2P(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i,			\
       int32_t * k, uint32_t c0, uint32_t c1)			\
  {								\
    o->stuff = i->stuff | *k | (c0 & c1);			\
    return FREIA_OK;						\
  }

#define Fun2P(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i,			\
       int32_t c1, int32_t c2)					\
  {								\
    o->stuff = i->stuff | (c1+c2);				\
    return FREIA_OK;						\
  }

#define Fun3P(name)						\
  freia_status							\
  name(freia_data2d * o, freia_data2d * i,			\
       int32_t c1, int32_t c2, bool b)				\
  {								\
    o->stuff = i->stuff | b? c1: c2;				\
    return FREIA_OK;						\
  }

#define FnuP(name)						\
  freia_status							\
  name(freia_data2d * o, int32_t c)				\
  {								\
    o->stuff = c;						\
    return FREIA_OK;						\
  }

#define Fred1(name)					\
  freia_status						\
  name(freia_data2d * i, int32_t * r)			\
  {							\
    *r = i->stuff;					\
    return FREIA_OK;					\
  }

#define Fred3(name)							\
  freia_status								\
  name(freia_data2d * i, int32_t * r0, int32_t * r1, int32_t * r2)	\
  {									\
    *r0 = i->stuff;							\
    *r1 = i->stuff;							\
    *r2 = i->stuff;							\
    return FREIA_OK;							\
  }

// AIPO function definitions
// Arithmetic
Fbin(freia_aipo_inf);
Fbin(freia_aipo_sup);
Fbin(freia_aipo_sub);
Fbin(freia_aipo_subsat);
Fbin(freia_aipo_add);
Fbin(freia_aipo_addsat);
Fbin(freia_aipo_absdiff);
Fbin(freia_aipo_mul);
Fbin(freia_aipo_div);
Fbin(freia_aipo_and);
Fbin(freia_aipo_or);
Fbin(freia_aipo_xor);

FunP(freia_aipo_inf_const);
FunP(freia_aipo_sup_const);
FunP(freia_aipo_sub_const);
FunP(freia_aipo_subsat_const);
FunP(freia_aipo_add_const);
FunP(freia_aipo_addsat_const);
FunP(freia_aipo_absdiff_const);
FunP(freia_aipo_mul_const);
FunP(freia_aipo_div_const);
FunP(freia_aipo_and_const);
FunP(freia_aipo_or_const);
FunP(freia_aipo_xor_const);

Fun(freia_aipo_not);

// Linear
FunK2P(freia_aipo_convolution);
FbinP(freia_aipo_fast_correlation);

// Measure
Fred1(freia_aipo_global_min);
Fred1(freia_aipo_global_max);
Fred1(freia_aipo_global_vol);
Fred3(freia_aipo_global_min_coord);
Fred3(freia_aipo_global_max_coord);

// Misc
Fun(freia_aipo_copy);
Fun(freia_aipo_cast);
FnuP(freia_aipo_set_constant);
Fun3P(freia_aipo_threshold);

// Morpho
FunK(freia_aipo_erode_8c);
FunK(freia_aipo_dilate_8c);
FunK(freia_aipo_erode_6c);
FunK(freia_aipo_dilate_6c);

// cipo
Fun2P(freia_cipo_gradient);
Fun2P(freia_cipo_inner_gradient);
