#include <stdio.h>
#include "freia.h"
#include "freiaExtendedOpenMorpho.h"

static const int32_t freia_morpho_k8_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k8_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
static const int32_t freia_morpho_k8_2[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

static const int32_t freia_morpho_k6_0[9] = {0, 0, 1, 1, 1, 0, 0, 0, 0};
static const int32_t freia_morpho_k6_1[9] = {0, 1, 0, 1, 0, 1, 0, 0, 0};
static const int32_t freia_morpho_k6_2[9] = {1, 0, 0, 1, 0, 0, 1, 0, 0};

static const int32_t freia_morpho_k4_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k4_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};


/**
   Transpose the neighbor
*/
void transposeNeighbor(int32_t neighborOut[9], const int32_t neighborIn[9])
{
  int i;
  for(i=0; i < 9; i++) {
    neighborOut[8-i] = neighborIn[i];
  }
}



freia_status freia_ecipo_inf_close(
  freia_data2d *imOut, freia_data2d *imIn, int32_t connexity, int32_t size)
{
  freia_status err = FREIA_OK;

  if(size < 0)
    return FREIA_INVALID_PARAM;

  // temporary images
  freia_data2d * w1 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
  freia_data2d * w2 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);

  freia_aipo_set_constant (w2, 255);
  switch (connexity)
  {
  case 6:
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k6_0, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k6_1, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k6_2, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    break;
  case 4:
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k4_0, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k4_1, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    break;
  case 8:
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k8_0, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k8_1, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    err |= freia_ecipo_close(w1, imIn, freia_morpho_k8_2, connexity, size);
    err |= freia_aipo_inf(w2, w2, w1);
    break;
  default:
    return FREIA_INVALID_PARAM;
  }

  err |= freia_aipo_copy(imOut, w2);

  freia_common_destruct_data(w1);
  freia_common_destruct_data(w2);

  return err;
}

freia_status freia_ecipo_sup_open(
  freia_data2d *imOut, freia_data2d *imIn, int32_t connexity, int32_t size)
{
  freia_status err = FREIA_OK;
  err |= freia_aipo_not(imOut, imIn);
  err |= freia_ecipo_inf_close(imOut, imOut, connexity, size);
  err |= freia_aipo_not(imOut, imOut);
  return err;
}

freia_status freia_ecipo_sup_open2(
  freia_data2d *imOut, freia_data2d *imIn, int32_t connexity, int32_t size)
{
  freia_status err = FREIA_OK;

  if(size < 0)
    return FREIA_SIZE_ERROR;

  freia_data2d * w1 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
  freia_data2d * w2 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);

  freia_aipo_set_constant (w2, 0);
  switch (connexity){
  case 6:
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k6_0, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k6_1, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k6_2, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    break;
  case 4:
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k4_0, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k4_1, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    break;
  case 8:
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k8_0, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k8_1, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    err |= freia_ecipo_open(w1, imIn, freia_morpho_k8_2, connexity, size);
    err |= freia_aipo_sup(w2, w2, w1);
    break;
  default:
    return FREIA_INVALID_PARAM;
  }

  err |= freia_aipo_copy(imOut, w2);
  freia_common_destruct_data(w1);
  freia_common_destruct_data(w2);
  return err;
}

freia_status freia_ecipo_erode(
  freia_data2d *imout, freia_data2d *imin,
  const int32_t *neighbor, int32_t connexity, uint32_t size)
{
  uint32_t i;

  if(freia_common_check_image_window_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(size==0) {
    freia_aipo_copy(imout,imin);
    return FREIA_OK;
  }

  if (connexity==4) {
  freia_aipo_erode_8c(imout,imin,neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_erode_8c(imout,imout,neighbor);
  }
  else if (connexity==6) {
    freia_aipo_erode_6c(imout,imin,neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_erode_6c(imout,imout,neighbor);
  }
  else if (connexity==8) {
    freia_aipo_erode_8c(imout,imin,neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_erode_8c(imout,imout,neighbor);
  }
  else
    return FREIA_INVALID_PARAM;

  return FREIA_OK;
}

freia_status freia_ecipo_dilate(
  freia_data2d *imout, freia_data2d *imin,
  const int32_t *neighbor, int32_t connexity, uint32_t size)
{
  uint32_t i;

  if(freia_common_check_image_window_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;

  }

  if(size==0) {
    freia_aipo_copy(imout,imin);
    return FREIA_OK;
  }

  if(connexity==4) {
    freia_aipo_dilate_8c(imout,imin,neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,neighbor);
  }
  else if (connexity==6) {
    freia_aipo_dilate_6c(imout,imin, neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_6c(imout,imout,neighbor);
  }
  else if (connexity==8) {
    freia_aipo_dilate_8c(imout,imin, neighbor);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,neighbor);
  }
  else
    return FREIA_INVALID_PARAM;

  return FREIA_OK;
}

freia_status freia_ecipo_open(
  freia_data2d *imout, freia_data2d *imin,
  const int32_t *neighbor, int32_t connexity, uint32_t size)
{
  freia_status ret;
  int32_t neighborTransposed[9];
  transposeNeighbor(neighborTransposed, neighbor);
  ret = freia_ecipo_erode(imout, imin, neighbor, connexity, size);
  ret |= freia_ecipo_dilate(imout, imout, neighborTransposed, connexity, size);

  return ret;
}

freia_status freia_ecipo_close(
  freia_data2d *imout, freia_data2d *imin,
  const int32_t *neighbor, int32_t connexity, uint32_t size)
{
  freia_status ret;
  int32_t neighborTransposed[9];
  transposeNeighbor(neighborTransposed, neighbor);
  ret = freia_ecipo_dilate(imout, imin, neighbor, connexity, size);
  ret |= freia_ecipo_erode(imout, imout, neighborTransposed, connexity, size);

  return ret;
}

void freia_ecipo_distance(freia_data2d *imOut, freia_data2d *imIn, const int32_t connexity){

	freia_data2d * w1 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
	freia_data2d * w2 = freia_common_create_data(imIn->bpp,  imIn->widthWa, imIn->heightWa);
	int32_t measure_oldVol, measure_vol;

	freia_aipo_threshold(w1, imIn, 1, imIn->bpp==16?32767:255, true); // binarize

	freia_aipo_global_vol(w1, &measure_vol);
	freia_aipo_set_constant(w2, w1->bpp==16?32767:255);
	freia_aipo_subsat_const(w1, w1, w1->bpp==16?32766:254);

	freia_aipo_copy(imOut, w1);
	measure_oldVol = 0;

	while (measure_vol != measure_oldVol) {
		freia_cipo_erode(w1, w1, connexity, 1);
		freia_aipo_add(imOut, imOut, w1);
		measure_oldVol = measure_vol;
		freia_aipo_global_vol(w1, &measure_vol);
	}

	freia_common_destruct_data(w1);
	freia_common_destruct_data(w2);
}		                                     
