/* Check inlining of unstructured code due to multiple return
   statements. Same as inlining11.c, but without enum */

#include <stdio.h>

#define FREIA_OK 0
#define FREIA_SIZE_ERROR 1
#define FREIA_INVALID_PARAM 2

typedef int freia_error;

typedef int freia_data2d;

typedef int int32_t;
typedef unsigned int uint32_t;

extern int freia_common_check_image_bpp_compat(freia_data2d *, freia_data2d *, void *);
extern int freia_common_print_backtrace();
extern int freia_aipo_copy(freia_data2d *, freia_data2d *);
extern int freia_aipo_dilate_8c(freia_data2d *, freia_data2d *, int *);
extern int freia_aipo_dilate_6c(freia_data2d *, freia_data2d *, int *);

freia_error freia_cipo_dilate(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  int kernel_8c[9] = {1,1,1,
        1,1,1,
        1,1,1};

  int kernel_6c[9] = {0, 1,1,
          1,1,1,
        0, 1,1};

  int kernel_4c[9] = {0,1,0,
        1,1,1,
        0,1,0};

  int i;

  if(freia_common_check_image_bpp_compat(imout,imin,((void *)0)) != 1) {
    fprintf(stderr,"ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "./freia.src/cipo/src/freiaComplexOpMorpho.c",90,__FUNCTION__);freia_common_print_backtrace();
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imout,imin, ((void *)0)) != 1) {
    fprintf(stderr,"ERROR: file %s, line %d, function %s: ""bpp of images are not compatibles\n", "./freia.src/cipo/src/freiaComplexOpMorpho.c",95,__FUNCTION__);freia_common_print_backtrace();
    return FREIA_SIZE_ERROR;
  }

  if(size==0) {
    freia_aipo_copy(imout,imin);
    return FREIA_OK;
  }

  switch(connexity) {
  case 4:
    freia_aipo_dilate_8c(imout,imin,kernel_4c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,kernel_4c); break;

  case 6:
    freia_aipo_dilate_6c(imout,imin,kernel_6c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_6c(imout,imout,kernel_6c); break;

  case 8:
    freia_aipo_dilate_8c(imout,imin,kernel_8c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,kernel_8c); break;

  default:
    return FREIA_INVALID_PARAM;
  }

  return FREIA_OK;
}

freia_error freia_cipo_outer_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_error ret;

  ret = freia_cipo_dilate(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imout,imin);

  return ret;
}
