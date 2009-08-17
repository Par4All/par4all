/*  Same as inlining16.c, but for a different tpips */

#include <stdio.h>

#define FREIA_OK 0
#define FREIA_SIZE_ERROR 1
#define FREIA_INVALID_PARAM 2

typedef int freia_error;

typedef int freia_data2d;

typedef int int32_t;
typedef unsigned int uint32_t;

freia_error freia_cipo_dilate(freia_data2d *imout,
			      freia_data2d *imin,
			      int32_t connexity,
			      uint32_t size)
{
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
  freia_error ret;

  if(freia_common_check_image_bpp_compat(imout,imin,((void *)0)) != 1) {
    fprintf(stderr,"ERROR: file %s, line %d, function %s: "
	    "bpp of images are not compatibles\n",
	    "./freia.src/cipo/src/freiaComplexOpMorpho.c",90,"freia_cipo_dilate");
    freia_common_print_backtrace();
    ret = FREIA_SIZE_ERROR;
  }

  else if(freia_common_check_image_bpp_compat(imout,imin, ((void *)0)) != 1) {
    fprintf(stderr,"ERROR: file %s, line %d, function %s: "
	    "bpp of images are not compatibles\n",
	    "./freia.src/cipo/src/freiaComplexOpMorpho.c",95,"freia_cipo_dilate");
    freia_common_print_backtrace();
    ret = FREIA_SIZE_ERROR;
  }

  else if(size==0) {
    freia_aipo_copy(imout,imin);
    ret = FREIA_OK;
  }

  else {
    if(connexity==4) {
      freia_aipo_dilate_8c(imout,imin,kernel_4c);
    l4:      for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,kernel_4c);
      ret = FREIA_OK;
    }
    else if(connexity == 6) {
      freia_aipo_dilate_6c(imout,imin,kernel_6c);
    l6:      for(i=1 ; i<size ; i++) freia_aipo_dilate_6c(imout,imout,kernel_6c);
      ret = FREIA_OK;
    }
    else if(connexity == 8) {
      freia_aipo_dilate_8c(imout,imin,kernel_8c);
    l8:      for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,kernel_8c);
      ret = FREIA_OK;
    }
    else
      ret = FREIA_INVALID_PARAM;
  }

  return ret;
}

freia_error freia_cipo_outer_gradient(freia_data2d *imout,
				      freia_data2d *imin,
				      int32_t connexity,
				      uint32_t size) {
  freia_error ret;

  ret = freia_cipo_dilate(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imout,imin);

  return ret;
}

int main()
{
  freia_data2d *imout;
  freia_data2d *imin;
  int32_t connexity = 8;
  uint32_t size = 4;
  (void) freia_cipo_outer_gradient(imout,
				   imin,
				   connexity,
				   size);
  (void) freia_cipo_outer_gradient(imout,
				   imin,
				   connexity,
				   size);
}
