#include <stdio.h>
#include "freia.h"

static const int32_t freia_morpho_k8_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k8_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
static const int32_t freia_morpho_k8_2[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
static const int32_t freia_morpho_k6_0[9] = {0, 0, 1, 1, 1, 0, 0, 0, 0};
static const int32_t freia_morpho_k6_1[9] = {0, 1, 0, 1, 0, 1, 0, 0, 0};
static const int32_t freia_morpho_k6_2[9] = {1, 0, 0, 1, 0, 0, 1, 0, 0};
static const int32_t freia_morpho_k4_0[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
static const int32_t freia_morpho_k4_1[9] = {0, 1, 0, 0, 1, 0, 0, 1, 0};

int freia_52(void)
{

   freia_dataio fdin, fdout;
   freia_data2d *imin, *imout_tophat, *imout_tophatLin, *imout_tophatLinRecon, *imWork1, *imWork2;
   //PIPS generated variable
   freia_status _return0, _return1, _return2, _return3, _return4, _return5, _return6, _return7, _return8, _return9, _return10, _return11, _return12, _return13, _return14, _return15, _return16, _return17, _return18, _return19, _return20, _return21, _return22, _return23, _return24, _return25, _return26, _return27, _return28, _return29, _return30, _return31, _return32, _return33, _return34, _return35, _return36, _return37, _return38, _return39, _return40, _return41, _return42, _return43, _return44, _return45, _return46, _return47, _return48, _return49, _return50, _return51, _return52, _return53, _return54, _return55, _return56, _return57, _return58, _return59, _return60, _return61;
   //PIPS generated variable
   freia_status ret_0;
   //PIPS generated variable
   int i_0, i_1;
   //PIPS generated variable
   freia_status ret_1, ret_2;
   //PIPS generated variable
   int i_2, i_3;
   //PIPS generated variable
   freia_status err_0 = 0;
   //PIPS generated variable
   freia_data2d *w1_0, *w2_0;
   //PIPS generated variable
   freia_status err_1 = 0, ret_3;
   //PIPS generated variable
   int32_t neighborTransposed_0[9];
   //PIPS generated variable
   int i_4;
   //PIPS generated variable
   uint32_t i_5, i_6;
   //PIPS generated variable
   freia_status ret_4;
   //PIPS generated variable
   int32_t neighborTransposed_1[9];
   //PIPS generated variable
   int i_7;
   //PIPS generated variable
   uint32_t i_8, i_9;
   //PIPS generated variable
   freia_status ret_5;
   //PIPS generated variable
   int32_t neighborTransposed_2[9];
   //PIPS generated variable
   int i_10;
   //PIPS generated variable
   uint32_t i_11, i_12;
   //PIPS generated variable
   freia_status ret_6;
   //PIPS generated variable
   int32_t neighborTransposed_3[9];
   //PIPS generated variable
   int i_13;
   //PIPS generated variable
   uint32_t i_14, i_15;
   //PIPS generated variable
   freia_status ret_7;
   //PIPS generated variable
   int32_t neighborTransposed_4[9];
   //PIPS generated variable
   int i_16;
   //PIPS generated variable
   uint32_t i_17, i_18;
   //PIPS generated variable
   freia_status ret_8;
   //PIPS generated variable
   int32_t neighborTransposed_5[9];
   //PIPS generated variable
   int i_19;
   //PIPS generated variable
   uint32_t i_20, i_21;
   //PIPS generated variable
   freia_status ret_9;
   //PIPS generated variable
   int32_t neighborTransposed_6[9];
   //PIPS generated variable
   int i_22;
   //PIPS generated variable
   uint32_t i_23, i_24;
   //PIPS generated variable
   freia_status ret_10;
   //PIPS generated variable
   int32_t neighborTransposed_7[9];
   //PIPS generated variable
   int i_25;
   //PIPS generated variable
   uint32_t i_26, i_27;
   //PIPS generated variable
   freia_status err_2 = 0;
   //PIPS generated variable
   freia_data2d *w1_1, *w2_1;
   //PIPS generated variable
   freia_status err_3 = 0, ret_11;
   //PIPS generated variable
   int32_t neighborTransposed_8[9];
   //PIPS generated variable
   int i_28;
   //PIPS generated variable
   uint32_t i_29, i_30;
   //PIPS generated variable
   freia_status ret_12;
   //PIPS generated variable
   int32_t neighborTransposed_9[9];
   //PIPS generated variable
   int i_31;
   //PIPS generated variable
   uint32_t i_32, i_33;
   //PIPS generated variable
   freia_status ret_13;
   //PIPS generated variable
   int32_t neighborTransposed_10[9];
   //PIPS generated variable
   int i_34;
   //PIPS generated variable
   uint32_t i_35, i_36;
   //PIPS generated variable
   freia_status ret_14;
   //PIPS generated variable
   int32_t neighborTransposed_11[9];
   //PIPS generated variable
   int i_37;
   //PIPS generated variable
   uint32_t i_38, i_39;
   //PIPS generated variable
   freia_status ret_15;
   //PIPS generated variable
   int32_t neighborTransposed_12[9];
   //PIPS generated variable
   int i_40;
   //PIPS generated variable
   uint32_t i_41, i_42;
   //PIPS generated variable
   freia_status ret_16;
   //PIPS generated variable
   int32_t neighborTransposed_13[9];
   //PIPS generated variable
   int i_43;
   //PIPS generated variable
   uint32_t i_44, i_45;
   //PIPS generated variable
   freia_status ret_17;
   //PIPS generated variable
   int32_t neighborTransposed_14[9];
   //PIPS generated variable
   int i_46;
   //PIPS generated variable
   uint32_t i_47, i_48;
   //PIPS generated variable
   freia_status ret_18;
   //PIPS generated variable
   int32_t neighborTransposed_15[9];
   //PIPS generated variable
   int i_49;
   //PIPS generated variable
   uint32_t i_50, i_51;
   //PIPS generated variable
   freia_status ret_19 = 0;
   //PIPS generated variable
   int32_t volcurrent, volprevious;
   //PIPS generated variable
   freia_status ret_20;
   //PIPS generated variable
   int i_52;

   freia_common_open_input(&fdin, 0);
   freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, fdin.framebpp);
   
   // there should be a freia_initialize()?
   // for opencl freia_op_init_opencl(0xffffffff, 2, false, fdin.framewidth, fdin.frameheight);
   imin = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imout_tophat = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imout_tophatLinRecon = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imout_tophatLin = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imWork1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   imWork2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
   
   // input
   freia_common_rx_image(imin, &fdin);
   
   // 1st tophat
   
   
   freia_aipo_erode_8c(imWork1, imin, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   
   
   _return6 = 0;
   ret_0 = 0;
   
   
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
   
   
   _return5 = 0;
   
   
   _return1 = ret_0;
   // freia_common_tx_image(imWork1, &fdout);
   
   
   freia_aipo_erode_8c(imout_tophat, imin, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   
   
   _return13 = 0;
   ret_2 = 0;
   
   
   freia_aipo_dilate_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(imout_tophat, imout_tophat, freia_morpho_kernel_8c);
   
   
   _return11 = 0;
   
   
   _return8 = ret_2;
   ret_1 = _return8;
   ret_1 |= freia_aipo_sub(imout_tophat, imin, imout_tophat);
   
   
   _return2 = ret_1;
   freia_aipo_mul_const(imout_tophat, imout_tophat, 2);
   // for display
   
   // 2nd tophat
   err_0 |= freia_aipo_not(imWork1, imin);
   
   
   w1_0 = freia_common_create_data(imWork1->bpp, imWork1->widthWa, imWork1->heightWa);
   w2_0 = freia_common_create_data(imWork1->bpp, imWork1->widthWa, imWork1->heightWa);

   freia_aipo_set_constant(w2_0, 255);

   neighborTransposed_0[-0+8] = freia_morpho_k8_0[0];
   neighborTransposed_0[-1+8] = freia_morpho_k8_0[1];
   neighborTransposed_0[-2+8] = freia_morpho_k8_0[2];
   neighborTransposed_0[-3+8] = freia_morpho_k8_0[3];
   neighborTransposed_0[-4+8] = freia_morpho_k8_0[4];
   neighborTransposed_0[-5+8] = freia_morpho_k8_0[5];
   neighborTransposed_0[-6+8] = freia_morpho_k8_0[6];
   neighborTransposed_0[-7+8] = freia_morpho_k8_0[7];
   neighborTransposed_0[-8+8] = freia_morpho_k8_0[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_0);
   
   
   _return35 = 0;
   ret_3 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_0);
   
   
   _return51 = 0;
   
   
   _return19 = ret_3;
   err_1 |= _return19;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_1[-0+8] = freia_morpho_k8_1[0];
   neighborTransposed_1[-1+8] = freia_morpho_k8_1[1];
   neighborTransposed_1[-2+8] = freia_morpho_k8_1[2];
   neighborTransposed_1[-3+8] = freia_morpho_k8_1[3];
   neighborTransposed_1[-4+8] = freia_morpho_k8_1[4];
   neighborTransposed_1[-5+8] = freia_morpho_k8_1[5];
   neighborTransposed_1[-6+8] = freia_morpho_k8_1[6];
   neighborTransposed_1[-7+8] = freia_morpho_k8_1[7];
   neighborTransposed_1[-8+8] = freia_morpho_k8_1[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_1);
   
   
   _return36 = 0;
   ret_4 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_1);
   
   
   _return52 = 0;
   
   
   _return20 = ret_4;
   err_1 |= _return20;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_2[-0+8] = freia_morpho_k8_2[0];
   neighborTransposed_2[-1+8] = freia_morpho_k8_2[1];
   neighborTransposed_2[-2+8] = freia_morpho_k8_2[2];
   neighborTransposed_2[-3+8] = freia_morpho_k8_2[3];
   neighborTransposed_2[-4+8] = freia_morpho_k8_2[4];
   neighborTransposed_2[-5+8] = freia_morpho_k8_2[5];
   neighborTransposed_2[-6+8] = freia_morpho_k8_2[6];
   neighborTransposed_2[-7+8] = freia_morpho_k8_2[7];
   neighborTransposed_2[-8+8] = freia_morpho_k8_2[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k8_2);
   
   
   _return37 = 0;
   ret_5 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_2);
   
   
   _return53 = 0;
   
   
   _return21 = ret_5;
   err_1 |= _return21;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_3[-0+8] = freia_morpho_k4_0[0];
   neighborTransposed_3[-1+8] = freia_morpho_k4_0[1];
   neighborTransposed_3[-2+8] = freia_morpho_k4_0[2];
   neighborTransposed_3[-3+8] = freia_morpho_k4_0[3];
   neighborTransposed_3[-4+8] = freia_morpho_k4_0[4];
   neighborTransposed_3[-5+8] = freia_morpho_k4_0[5];
   neighborTransposed_3[-6+8] = freia_morpho_k4_0[6];
   neighborTransposed_3[-7+8] = freia_morpho_k4_0[7];
   neighborTransposed_3[-8+8] = freia_morpho_k4_0[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_0);
   
   
   _return33 = 0;
   ret_6 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_3);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_3);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_3);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_3);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_3);
   
   
   _return49 = 0;
   
   
   _return17 = ret_6;
   err_1 |= _return17;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_4[-0+8] = freia_morpho_k4_1[0];
   neighborTransposed_4[-1+8] = freia_morpho_k4_1[1];
   neighborTransposed_4[-2+8] = freia_morpho_k4_1[2];
   neighborTransposed_4[-3+8] = freia_morpho_k4_1[3];
   neighborTransposed_4[-4+8] = freia_morpho_k4_1[4];
   neighborTransposed_4[-5+8] = freia_morpho_k4_1[5];
   neighborTransposed_4[-6+8] = freia_morpho_k4_1[6];
   neighborTransposed_4[-7+8] = freia_morpho_k4_1[7];
   neighborTransposed_4[-8+8] = freia_morpho_k4_1[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k4_1);
   
   
   _return34 = 0;
   ret_7 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_4);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_4);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_4);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_4);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_4);
   
   
   _return50 = 0;
   
   
   _return18 = ret_7;
   err_1 |= _return18;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_5[-0+8] = freia_morpho_k6_0[0];
   neighborTransposed_5[-1+8] = freia_morpho_k6_0[1];
   neighborTransposed_5[-2+8] = freia_morpho_k6_0[2];
   neighborTransposed_5[-3+8] = freia_morpho_k6_0[3];
   neighborTransposed_5[-4+8] = freia_morpho_k6_0[4];
   neighborTransposed_5[-5+8] = freia_morpho_k6_0[5];
   neighborTransposed_5[-6+8] = freia_morpho_k6_0[6];
   neighborTransposed_5[-7+8] = freia_morpho_k6_0[7];
   neighborTransposed_5[-8+8] = freia_morpho_k6_0[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_0);
   
   
   _return30 = 0;
   ret_8 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_5);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_5);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_5);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_5);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_5);
   
   
   _return46 = 0;
   
   
   _return14 = ret_8;
   err_1 |= _return14;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_6[-0+8] = freia_morpho_k6_1[0];
   neighborTransposed_6[-1+8] = freia_morpho_k6_1[1];
   neighborTransposed_6[-2+8] = freia_morpho_k6_1[2];
   neighborTransposed_6[-3+8] = freia_morpho_k6_1[3];
   neighborTransposed_6[-4+8] = freia_morpho_k6_1[4];
   neighborTransposed_6[-5+8] = freia_morpho_k6_1[5];
   neighborTransposed_6[-6+8] = freia_morpho_k6_1[6];
   neighborTransposed_6[-7+8] = freia_morpho_k6_1[7];
   neighborTransposed_6[-8+8] = freia_morpho_k6_1[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_1);
   
   
   _return31 = 0;
   ret_9 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_6);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_6);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_6);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_6);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_6);
   
   
   _return47 = 0;
   
   
   _return15 = ret_9;
   err_1 |= _return15;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   neighborTransposed_7[-0+8] = freia_morpho_k6_2[0];
   neighborTransposed_7[-1+8] = freia_morpho_k6_2[1];
   neighborTransposed_7[-2+8] = freia_morpho_k6_2[2];
   neighborTransposed_7[-3+8] = freia_morpho_k6_2[3];
   neighborTransposed_7[-4+8] = freia_morpho_k6_2[4];
   neighborTransposed_7[-5+8] = freia_morpho_k6_2[5];
   neighborTransposed_7[-6+8] = freia_morpho_k6_2[6];
   neighborTransposed_7[-7+8] = freia_morpho_k6_2[7];
   neighborTransposed_7[-8+8] = freia_morpho_k6_2[8];
   
   
   freia_aipo_dilate_8c(w1_0, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_0, w1_0, freia_morpho_k6_2);
   
   
   _return32 = 0;
   ret_10 = 0;
   
   
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_7);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_7);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_7);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_7);
   freia_aipo_erode_8c(w1_0, w1_0, neighborTransposed_7);
   
   
   _return48 = 0;
   
   
   _return16 = ret_10;
   err_1 |= _return16;
   err_1 |= freia_aipo_inf(w2_0, w2_0, w1_0);

   err_1 |= freia_aipo_copy(imWork1, w2_0);
   freia_common_destruct_data(w1_0);
   freia_common_destruct_data(w2_0);
   
   
   _return9 = err_1;
   err_0 |= _return9;
   err_0 |= freia_aipo_not(imWork1, imWork1);
   _return3 = err_0;
   // freia_common_tx_image(imWork1, &fdout);
   freia_aipo_sub(imout_tophatLin, imin, imWork1);
   freia_aipo_mul_const(imout_tophatLin, imout_tophatLin, 2);
   // for display
   
   // 3nd tophat
   err_2 |= freia_aipo_not(imWork1, imin);
   
   
   w1_1 = freia_common_create_data(imWork1->bpp, imWork1->widthWa, imWork1->heightWa);
   w2_1 = freia_common_create_data(imWork1->bpp, imWork1->widthWa, imWork1->heightWa);

   freia_aipo_set_constant(w2_1, 255);

   neighborTransposed_8[-0+8] = freia_morpho_k8_0[0];
   neighborTransposed_8[-1+8] = freia_morpho_k8_0[1];
   neighborTransposed_8[-2+8] = freia_morpho_k8_0[2];
   neighborTransposed_8[-3+8] = freia_morpho_k8_0[3];
   neighborTransposed_8[-4+8] = freia_morpho_k8_0[4];
   neighborTransposed_8[-5+8] = freia_morpho_k8_0[5];
   neighborTransposed_8[-6+8] = freia_morpho_k8_0[6];
   neighborTransposed_8[-7+8] = freia_morpho_k8_0[7];
   neighborTransposed_8[-8+8] = freia_morpho_k8_0[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_0);
   
   
   _return43 = 0;
   ret_11 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_8);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_8);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_8);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_8);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_8);
   
   
   _return59 = 0;
   
   
   _return27 = ret_11;
   err_3 |= _return27;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_9[-0+8] = freia_morpho_k8_1[0];
   neighborTransposed_9[-1+8] = freia_morpho_k8_1[1];
   neighborTransposed_9[-2+8] = freia_morpho_k8_1[2];
   neighborTransposed_9[-3+8] = freia_morpho_k8_1[3];
   neighborTransposed_9[-4+8] = freia_morpho_k8_1[4];
   neighborTransposed_9[-5+8] = freia_morpho_k8_1[5];
   neighborTransposed_9[-6+8] = freia_morpho_k8_1[6];
   neighborTransposed_9[-7+8] = freia_morpho_k8_1[7];
   neighborTransposed_9[-8+8] = freia_morpho_k8_1[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_1);
   
   
   _return44 = 0;
   ret_12 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_9);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_9);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_9);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_9);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_9);
   
   
   _return60 = 0;
   
   
   _return28 = ret_12;
   err_3 |= _return28;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_10[-0+8] = freia_morpho_k8_2[0];
   neighborTransposed_10[-1+8] = freia_morpho_k8_2[1];
   neighborTransposed_10[-2+8] = freia_morpho_k8_2[2];
   neighborTransposed_10[-3+8] = freia_morpho_k8_2[3];
   neighborTransposed_10[-4+8] = freia_morpho_k8_2[4];
   neighborTransposed_10[-5+8] = freia_morpho_k8_2[5];
   neighborTransposed_10[-6+8] = freia_morpho_k8_2[6];
   neighborTransposed_10[-7+8] = freia_morpho_k8_2[7];
   neighborTransposed_10[-8+8] = freia_morpho_k8_2[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k8_2);
   
   
   _return45 = 0;
   ret_13 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_10);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_10);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_10);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_10);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_10);
   
   
   _return61 = 0;
   
   
   _return29 = ret_13;
   err_3 |= _return29;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_11[-0+8] = freia_morpho_k4_0[0];
   neighborTransposed_11[-1+8] = freia_morpho_k4_0[1];
   neighborTransposed_11[-2+8] = freia_morpho_k4_0[2];
   neighborTransposed_11[-3+8] = freia_morpho_k4_0[3];
   neighborTransposed_11[-4+8] = freia_morpho_k4_0[4];
   neighborTransposed_11[-5+8] = freia_morpho_k4_0[5];
   neighborTransposed_11[-6+8] = freia_morpho_k4_0[6];
   neighborTransposed_11[-7+8] = freia_morpho_k4_0[7];
   neighborTransposed_11[-8+8] = freia_morpho_k4_0[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_0);
   
   
   _return41 = 0;
   ret_14 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_11);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_11);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_11);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_11);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_11);
   
   
   _return57 = 0;
   
   
   _return25 = ret_14;
   err_3 |= _return25;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_12[-0+8] = freia_morpho_k4_1[0];
   neighborTransposed_12[-1+8] = freia_morpho_k4_1[1];
   neighborTransposed_12[-2+8] = freia_morpho_k4_1[2];
   neighborTransposed_12[-3+8] = freia_morpho_k4_1[3];
   neighborTransposed_12[-4+8] = freia_morpho_k4_1[4];
   neighborTransposed_12[-5+8] = freia_morpho_k4_1[5];
   neighborTransposed_12[-6+8] = freia_morpho_k4_1[6];
   neighborTransposed_12[-7+8] = freia_morpho_k4_1[7];
   neighborTransposed_12[-8+8] = freia_morpho_k4_1[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k4_1);
   
   
   _return42 = 0;
   ret_15 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_12);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_12);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_12);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_12);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_12);
   
   
   _return58 = 0;
   
   
   _return26 = ret_15;
   err_3 |= _return26;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_13[-0+8] = freia_morpho_k6_0[0];
   neighborTransposed_13[-1+8] = freia_morpho_k6_0[1];
   neighborTransposed_13[-2+8] = freia_morpho_k6_0[2];
   neighborTransposed_13[-3+8] = freia_morpho_k6_0[3];
   neighborTransposed_13[-4+8] = freia_morpho_k6_0[4];
   neighborTransposed_13[-5+8] = freia_morpho_k6_0[5];
   neighborTransposed_13[-6+8] = freia_morpho_k6_0[6];
   neighborTransposed_13[-7+8] = freia_morpho_k6_0[7];
   neighborTransposed_13[-8+8] = freia_morpho_k6_0[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_0);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_0);
   
   
   _return38 = 0;
   ret_16 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_13);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_13);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_13);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_13);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_13);
   
   
   _return54 = 0;
   
   
   _return22 = ret_16;
   err_3 |= _return22;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_14[-0+8] = freia_morpho_k6_1[0];
   neighborTransposed_14[-1+8] = freia_morpho_k6_1[1];
   neighborTransposed_14[-2+8] = freia_morpho_k6_1[2];
   neighborTransposed_14[-3+8] = freia_morpho_k6_1[3];
   neighborTransposed_14[-4+8] = freia_morpho_k6_1[4];
   neighborTransposed_14[-5+8] = freia_morpho_k6_1[5];
   neighborTransposed_14[-6+8] = freia_morpho_k6_1[6];
   neighborTransposed_14[-7+8] = freia_morpho_k6_1[7];
   neighborTransposed_14[-8+8] = freia_morpho_k6_1[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_1);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_1);
   
   
   _return39 = 0;
   ret_17 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_14);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_14);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_14);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_14);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_14);
   
   
   _return55 = 0;
   
   
   _return23 = ret_17;
   err_3 |= _return23;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   neighborTransposed_15[-0+8] = freia_morpho_k6_2[0];
   neighborTransposed_15[-1+8] = freia_morpho_k6_2[1];
   neighborTransposed_15[-2+8] = freia_morpho_k6_2[2];
   neighborTransposed_15[-3+8] = freia_morpho_k6_2[3];
   neighborTransposed_15[-4+8] = freia_morpho_k6_2[4];
   neighborTransposed_15[-5+8] = freia_morpho_k6_2[5];
   neighborTransposed_15[-6+8] = freia_morpho_k6_2[6];
   neighborTransposed_15[-7+8] = freia_morpho_k6_2[7];
   neighborTransposed_15[-8+8] = freia_morpho_k6_2[8];
   
   
   freia_aipo_dilate_8c(w1_1, imWork1, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_2);
   freia_aipo_dilate_8c(w1_1, w1_1, freia_morpho_k6_2);
   
   
   _return40 = 0;
   ret_18 = 0;
   
   
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_15);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_15);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_15);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_15);
   freia_aipo_erode_8c(w1_1, w1_1, neighborTransposed_15);
   
   
   _return56 = 0;
   
   
   _return24 = ret_18;
   err_3 |= _return24;
   err_3 |= freia_aipo_inf(w2_1, w2_1, w1_1);

   err_3 |= freia_aipo_copy(imWork1, w2_1);
   freia_common_destruct_data(w1_1);
   freia_common_destruct_data(w2_1);
   
   
   _return10 = err_3;
   err_2 |= _return10;
   err_2 |= freia_aipo_not(imWork1, imWork1);
   _return4 = err_2;

   ret_19 = freia_aipo_global_vol(imWork1, &volcurrent);
   do {
      
      
      freia_aipo_dilate_8c(imWork1, imWork1, freia_morpho_kernel_8c);
      i_52 = 1;
      
      
      _return12 = 0;
      ret_20 = 0;
      ret_20 |= freia_aipo_inf(imWork1, imWork1, imin);
      
      
      _return7 = ret_20;
      ret_19 |= _return7;
      volprevious = volcurrent;
      ret_19 |= freia_aipo_global_vol(imWork1, &volcurrent);
   }
   while (volcurrent!=volprevious);
   
   
   _return0 = ret_19;
   freia_aipo_sub(imout_tophatLinRecon, imin, imWork1);
   freia_aipo_mul_const(imout_tophatLinRecon, imout_tophatLinRecon, 2);
   // for display
   // threshold one these tophat
   freia_aipo_threshold(imWork1, imout_tophatLinRecon, 90, 255, 1);
   // the threshold over the source image
   freia_aipo_sup(imWork2, imWork1, imin);
   
   // output
   freia_common_tx_image(imin, &fdout);
   freia_common_tx_image(imWork1, &fdout);
   
   
   
   
   
   
   
   // cleanup
   freia_common_destruct_data(imin);
   freia_common_destruct_data(imout_tophat);
   freia_common_destruct_data(imout_tophatLin);
   freia_common_destruct_data(imout_tophatLinRecon);
   freia_common_destruct_data(imWork1);
   freia_common_destruct_data(imWork2);

   freia_common_close_input(&fdin);
   freia_common_close_output(&fdout);

   return 0;
}
