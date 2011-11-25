// fake implementations, should be right for effects

// #include <stdlib.h>
extern void * malloc(unsigned int);
extern void free(void *);

#include "freia.h"

static int global_io_effect = 0;

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

int32_t freia_common_get(const freia_data2d * img, int32_t i, int32_t j)
{
  return img->stuff + i + j;
}

void freia_common_set(freia_data2d * img, int32_t i, int32_t j, int32_t v)
{
  img->stuff += (i+j+v);
}

freia_ptr freia_common_alloc(uint32_t s)
{
  return (freia_ptr) malloc(s);
}

void freia_common_free(freia_ptr p)
{
  free(p);
}

freia_status freia_common_open_input(freia_dataio * in, uint32_t n)
{
  in->framebpp = 16;
  in->frameheight = 256;
  in->framewidth = 512;
  in->frameindex = 0;
  in->stuff = n;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_open_output(freia_dataio *out,
   uint32_t n, uint32_t w, uint32_t h, uint32_t b)
{
  out->framebpp = b;
  out->frameheight = h;
  out->framewidth = w;
  out->frameindex = 0;
  out->stuff = n;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_rx_image(freia_data2d * img, freia_dataio * in)
{
  img->stuff = in->stuff;
  in->frameindex++;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_tx_image(const freia_data2d * img, freia_dataio * out)
{
  out->stuff += img->stuff;
  out->frameindex++;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_close_input(freia_dataio * n)
{
  n->stuff = 0;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_close_output(freia_dataio * n)
{
  n->stuff = 0;
  global_io_effect++;
  return FREIA_OK;
}

freia_status freia_common_draw_line(freia_data2d *image,
   int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t color)
{
  image->stuff += color+x1+y1+x2+y2;
  return FREIA_OK;
}

freia_status freia_common_draw_rect(freia_data2d *image,
   int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t color)
{
  image->stuff += color+x1+y1+x2+y2;
  return FREIA_OK;
}

freia_status freia_common_draw_disc(freia_data2d *image,
   int32_t x1, int32_t y1, int32_t rad, int32_t color)
{
  image->stuff += color+x1+y1+rad;
  return FREIA_OK;
}

freia_status freia_common_set_wa(freia_data2d *image,
   int32_t x1, int32_t y1, int32_t w, int32_t h)
{
  image->stuff += x1+y1+w+h;
  return FREIA_OK;
}

freia_status freia_common_get_wa(const freia_data2d *image,
   int32_t * x, int32_t * y, int32_t * w, int32_t * h)
{
  *x = image->xStartWa;
  *y = image->yStartWa;
  *w = image->widthWa;
  *h = image->heightWa;
  return FREIA_OK;
}

freia_status freia_common_reset_wa(freia_data2d *image)
{
  image->stuff += 1;
  return FREIA_OK;
}

void freia_initialize(int argc, char * argv[])
{
  global_io_effect++;
}

void freia_shutdown(void)
{
  global_io_effect++;
}

#define Fbin(name)                                        \
  freia_status                                            \
  name(freia_data2d * o,                                  \
       const freia_data2d * i0, const freia_data2d * i1)  \
  {                                                       \
    o->stuff = i0->stuff | i1->stuff;                     \
    return FREIA_OK;                                      \
  }

#define FbinP(name,ctype)                                 \
  freia_status                                            \
  name(freia_data2d * o,                                  \
       const freia_data2d * i0, const freia_data2d * i1,  \
       ctype c)                                        \
  {                                                       \
    o->stuff = i0->stuff | i1->stuff | (int32_t) c;       \
    return FREIA_OK;                                      \
  }

#define Fun(name)                                   \
  freia_status                                      \
  name(freia_data2d * o, const freia_data2d * i)    \
  {                                                 \
    o->stuff = i->stuff;                            \
    return FREIA_OK;                                \
  }

#define FunP(name)                                          \
  freia_status                                              \
  name(freia_data2d * o, const freia_data2d * i, int32_t c) \
  {                                                         \
    o->stuff = i->stuff | c;                                \
    return FREIA_OK;                                        \
  }

#define FunK(name)                                \
  freia_status                                    \
  name(freia_data2d * o,                          \
       const freia_data2d * i, const int32_t * k) \
  {                                               \
    o->stuff = i->stuff | *k;                     \
    return FREIA_OK;                              \
  }

#define FunK2P(name)                                \
  freia_status                                      \
  name(freia_data2d * o, const freia_data2d * i,    \
       const int32_t * k, uint32_t c0, uint32_t c1) \
  {                                                 \
    o->stuff = i->stuff | *k | (c0 & c1);           \
    return FREIA_OK;                                \
  }

#define Fun2P(name)                               \
  freia_status                                    \
  name(freia_data2d * o, const freia_data2d * i,  \
       int32_t c1, uint32_t c2)                   \
  {                                               \
    o->stuff = i->stuff | (c1+c2);                \
    return FREIA_OK;                              \
  }

#define Fun3P(name)                               \
  freia_status                                    \
  name(freia_data2d * o, const freia_data2d * i,  \
       int32_t c1, int32_t c2, bool b)            \
  {                                               \
    o->stuff = i->stuff | b? c1: c2;              \
    return FREIA_OK;                              \
  }

#define FnuP(name)                      \
  freia_status                          \
  name(freia_data2d * o, int32_t c)     \
  {                                     \
    o->stuff = c;                       \
    return FREIA_OK;                    \
  }

#define Fred1(name)                           \
  freia_status                                \
  name(const freia_data2d * i, int32_t * r)   \
  {                                           \
    *r = i->stuff;                            \
    return FREIA_OK;                          \
  }

#define Fred3(name)                                 \
  freia_status                                      \
  name(const freia_data2d * i,                      \
       int32_t * r0, uint32_t * r1, uint32_t * r2)  \
  {                                                 \
    *r0 = i->stuff;                                 \
    *r1 = (uint32_t) i->stuff;                      \
    *r2 = (uint32_t) i->stuff;                      \
    return FREIA_OK;                                \
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
FunP(freia_aipo_const_sub);
FunP(freia_aipo_subsat_const);
FunP(freia_aipo_add_const);
FunP(freia_aipo_addsat_const);
FunP(freia_aipo_absdiff_const);
FunP(freia_aipo_mul_const);
FunP(freia_aipo_div_const);
FunP(freia_aipo_const_div);
FunP(freia_aipo_and_const);
FunP(freia_aipo_or_const);
FunP(freia_aipo_xor_const);

Fun(freia_aipo_not);
Fun(freia_aipo_log2);

// Linear
FunK2P(freia_aipo_convolution);
FbinP(freia_aipo_fast_correlation, uint32_t);
FbinP(freia_aipo_replace_const, int32_t);

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
Fun2P(freia_cipo_open);
Fun2P(freia_cipo_close);
Fun2P(freia_cipo_erode);
FbinP(freia_cipo_fast_correlation, uint32_t);
