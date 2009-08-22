// $Id$
// Minimal self contained types for these headers

#ifndef _freia_h_included_
#define _freia_h_included_
#ifndef NULL
#define NULL ((void*)0)
#endif // NULL
typedef enum { false, true } bool;
typedef int int32_t;
typedef unsigned int uint32_t; // ??? for convolution & correlation
typedef struct {
  int bpp, widthWa, heightWa;
  int stuff;
} freia_data2d;
typedef enum { FREIA_OK, FREIA_ERROR } freia_status;
typedef struct {
  int framebpp, framewidth, frameheight;
} freia_dataio;

static const int32_t freia_morpho_kernel_8c[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int32_t freia_morpho_kernel_6c[9] = {0, 1, 1, 1, 1, 1, 0, 1, 1};
static const int32_t freia_morpho_kernel_4c[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};

// FREIA image allocation & deallocation
extern freia_data2d * freia_common_create_data(uint32_t, uint32_t, uint32_t);
extern freia_status freia_common_destruct_data(freia_data2d *);

// IO
extern freia_status freia_common_open_input(freia_dataio *, uint32_t);
extern freia_status freia_common_open_output(freia_dataio *, uint32_t, uint32_t, uint32_t, uint32_t);
extern freia_status freia_common_rx_image(const freia_data2d *, freia_dataio *);
extern freia_status freia_common_tx_image(freia_data2d *, freia_dataio *);
extern freia_status freia_common_close_input(freia_dataio *);
extern freia_status freia_common_close_output(freia_dataio *);

// 2 CIPO functions
extern freia_status freia_cipo_gradient(freia_data2d *, const freia_data2d *, int32_t, uint32_t);
extern freia_status freia_cipo_inner_gradient(freia_data2d *, const freia_data2d *, int32_t, uint32_t);

// AIPO function definitions
// Arithmetic
extern freia_status freia_aipo_inf(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_inf_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_sup(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_sup_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_sub(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_sub_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_subsat(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_subsat_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_add(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_add_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_addsat(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_addsat_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_absdiff(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_absdiff_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_mul(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_mul_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_div(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_div_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_and(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_and_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_or(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_or_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_xor(freia_data2d *, const freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_xor_const(freia_data2d *, const freia_data2d *, int32_t);
extern freia_status freia_aipo_not(freia_data2d *, const freia_data2d *);
// Linear
extern freia_status freia_aipo_convolution(freia_data2d *, const freia_data2d *, int32_t *, uint32_t, uint32_t);
extern freia_status freia_aipo_fast_correlation(freia_data2d *, const freia_data2d *, const freia_data2d *, uint32_t);
// Measure
extern freia_status freia_aipo_global_min(const freia_data2d *, int32_t *);
extern freia_status freia_aipo_global_max(const freia_data2d *, int32_t *);
extern freia_status freia_aipo_global_min_coord(const freia_data2d *, int32_t *, int32_t *, int32_t *);
extern freia_status freia_aipo_global_max_coord(const freia_data2d *, int32_t *, int32_t *, int32_t *);
extern freia_status freia_aipo_global_vol(const freia_data2d *, int32_t *);
// Misc
extern freia_status freia_aipo_copy(freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_cast(freia_data2d *, const freia_data2d *);
extern freia_status freia_aipo_set_constant(freia_data2d *, int32_t);
extern freia_status freia_aipo_threshold(freia_data2d *, const freia_data2d *, int32_t, int32_t, bool);
// Morpho
extern freia_status freia_aipo_erode_8c(freia_data2d *, const freia_data2d *, const int32_t *);
extern freia_status freia_aipo_dilate_8c(freia_data2d *, const freia_data2d *, const int32_t *);
extern freia_status freia_aipo_erode_6c(freia_data2d *, const freia_data2d *, const int32_t *);
extern freia_status freia_aipo_dilate_6c(freia_data2d *, const freia_data2d *, const int32_t *);
#endif // _freia_h_included_
