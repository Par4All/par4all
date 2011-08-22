#include <stdio.h>
#include "freia.h"

static const int32_t freia_morpho_k8_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k8_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
static const int32_t freia_morpho_k8_2[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

void transposeNeighbor(int32_t neighborOut[9], const int32_t neighborIn[9])
{
  int i;
  for(i=0; i < 9; i++) {
    neighborOut[8-i] = neighborIn[i];
  }
}

void freia_70(freia_data2d * out, const freia_data2d * in)
{
  freia_data2d *w1_0, *w2_0, *w1_1, *w2_1;
  int32_t neighborTransposed_0[9];
  int32_t neighborTransposed_1[9];
  int32_t neighborTransposed_2[9];

  w1_0 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);
  w2_0 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);
  w1_1 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);
  w2_1 = freia_common_create_data(out->bpp, out->widthWa, out->heightWa);

  transposeNeighbor(neighborTransposed_0, freia_morpho_k8_0);
  transposeNeighbor(neighborTransposed_1, freia_morpho_k8_1);
  transposeNeighbor(neighborTransposed_2, freia_morpho_k8_2);

  freia_aipo_not(out, in);
  freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_0);
  freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
  freia_aipo_inf_const(w2_0, w1_0, 255);
  freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_1);
  freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
  freia_aipo_inf(w2_0, w2_0, w1_0);
  freia_aipo_dilate_8c(w1_0, out, freia_morpho_k8_2);
  freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
  freia_aipo_inf(w2_0, w2_0, w1_0);

  freia_aipo_not(out, in);
  freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_0);
  freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_0);
  freia_aipo_inf_const(w2_1, w1_1, 255);
  freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_1);
  freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_1);
  freia_aipo_inf(w2_1, w2_1, w1_1);
  freia_aipo_dilate_8c(w1_1, out, freia_morpho_k8_2);
  freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_2);
  freia_aipo_inf(out, w2_1, w1_1);

  freia_common_destruct_data(w1_0);
  freia_common_destruct_data(w2_0);
  freia_common_destruct_data(w1_1);
  freia_common_destruct_data(w2_1);
}
