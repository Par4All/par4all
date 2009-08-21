// Minimal self contained types for these headers
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

// FREIA image allocation & deallocation
extern freia_data2d * freia_common_create_data(uint32_t, uint32_t, uint32_t);
extern freia_status freia_common_destruct_data(freia_data2d *);

// 2 CIPO functions
extern freia_status freia_cipo_gradient(freia_data2d *, freia_data2d *, int32_t, uint32_t);
extern freia_status freia_cipo_inner_gradient(freia_data2d *, freia_data2d *, int32_t, uint32_t);

// AIPO function definitions
// Arithmetic
extern freia_status freia_aipo_inf(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_inf_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_sup(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_sup_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_sub(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_sub_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_subsat(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_subsat_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_add(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_add_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_addsat(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_addsat_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_absdiff(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_absdiff_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_mul(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_mul_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_div(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_div_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_and(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_and_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_or(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_or_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_xor(freia_data2d *, freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_xor_const(freia_data2d *, freia_data2d *, int32_t);
extern freia_status freia_aipo_not(freia_data2d *, freia_data2d *);
// Linear
extern freia_status freia_aipo_convolution(freia_data2d *, freia_data2d *, int32_t *, uint32_t, uint32_t);
extern freia_status freia_aipo_fast_correlation(freia_data2d *, freia_data2d *, freia_data2d *, uint32_t);
// Measure
extern freia_status freia_aipo_global_min(freia_data2d *, int32_t *);
extern freia_status freia_aipo_global_max(freia_data2d *, int32_t *);
extern freia_status freia_aipo_global_min_coord(freia_data2d *, int32_t *, int32_t *, int32_t *);
extern freia_status freia_aipo_global_max_coord(freia_data2d *, int32_t *, int32_t *, int32_t *);
extern freia_status freia_aipo_global_vol(freia_data2d *, int32_t *);
// Misc
extern freia_status freia_aipo_copy(freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_cast(freia_data2d *, freia_data2d *);
extern freia_status freia_aipo_set_constant(freia_data2d *, int32_t);
extern freia_status freia_aipo_threshold(freia_data2d *, freia_data2d *, int32_t, int32_t, bool);
// Morpho
extern freia_status freia_aipo_erode_8c(freia_data2d *, freia_data2d *, int32_t *);
extern freia_status freia_aipo_dilate_8c(freia_data2d *, freia_data2d *, int32_t *);
extern freia_status freia_aipo_erode_6c(freia_data2d *, freia_data2d *, int32_t *);
extern freia_status freia_aipo_dilate_6c(freia_data2d *, freia_data2d *, int32_t *);
