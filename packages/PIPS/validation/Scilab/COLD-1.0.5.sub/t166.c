/*
 * (c) HPC Project - 2010-2011 - All rights reserved
 *
 */

#include "scilab_rt.h"


int __lv0;
int __lv1;
int __lv2;
int __lv3;

/*----------------------------------------------------*/


/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t166.sce: function sum */
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  scilab_rt_display_s0i2_("a",3,3,_u_a);
  int _u_aSum;
  scilab_rt_sum_i2_i0(3,3,_u_a,&_u_aSum);
  scilab_rt_display_s0i0_("aSum",_u_aSum);
  int _u_aSumC[3][1];
  scilab_rt_sum_i2s0_i2(3,3,_u_a,"c",3,1,_u_aSumC);
  scilab_rt_display_s0i2_("aSumC",3,1,_u_aSumC);
  int _u_aSum2[3][1];
  scilab_rt_sum_i2i0_i2(3,3,_u_a,2,3,1,_u_aSum2);
  scilab_rt_display_s0i2_("aSum2",3,1,_u_aSum2);
  int _u_aSumR[1][3];
  scilab_rt_sum_i2s0_i2(3,3,_u_a,"r",1,3,_u_aSumR);
  scilab_rt_display_s0i2_("aSumR",1,3,_u_aSumR);
  int _u_aSum1[1][3];
  scilab_rt_sum_i2i0_i2(3,3,_u_a,1,1,3,_u_aSum1);
  scilab_rt_display_s0i2_("aSum1",1,3,_u_aSum1);
  int _u_aSumM[1][3];
  scilab_rt_sum_i2s0_i2(3,3,_u_a,"m",1,3,_u_aSumM);
  scilab_rt_display_s0i2_("aSumM",1,3,_u_aSumM);
  int _tmpxx0;
  scilab_rt_sum_i2s0_i0(1,3,_u_aSumM,"m",&_tmpxx0);
  int _u_aSumM2 = (_tmpxx0+1000);
  scilab_rt_display_s0i0_("aSumM2",_u_aSumM2);
  int _u_b[1][3];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  scilab_rt_display_s0i2_("b",1,3,_u_b);
  int _u_bSum;
  scilab_rt_sum_i2_i0(1,3,_u_b,&_u_bSum);
  scilab_rt_display_s0i0_("bSum",_u_bSum);
  int _u_bSumC;
  scilab_rt_sum_i2s0_i0(1,3,_u_b,"c",&_u_bSumC);
  scilab_rt_display_s0i0_("bSumC",_u_bSumC);
  int _u_bSum2;
  scilab_rt_sum_i2i0_i0(1,3,_u_b,2,&_u_bSum2);
  scilab_rt_display_s0i0_("bSum2",_u_bSum2);
  int _u_bSumR[1][3];
  scilab_rt_sum_i2s0_i2(1,3,_u_b,"r",1,3,_u_bSumR);
  scilab_rt_display_s0i2_("bSumR",1,3,_u_bSumR);
  int _u_bSum1[1][3];
  scilab_rt_sum_i2i0_i2(1,3,_u_b,1,1,3,_u_bSum1);
  scilab_rt_display_s0i2_("bSum1",1,3,_u_bSum1);
  int _u_bSumM;
  scilab_rt_sum_i2s0_i0(1,3,_u_b,"m",&_u_bSumM);
  scilab_rt_display_s0i0_("bSumM",_u_bSumM);
  int _tmpxx1 = scilab_rt_sum_i0s0_(_u_bSumM,"m");
  int _u_bSumM2 = (_tmpxx1+1000);
  scilab_rt_display_s0i0_("bSumM2",_u_bSumM2);
  int _u_c[3][1];
  _u_c[0][0]=1;
  _u_c[1][0]=2;
  _u_c[2][0]=3;
  scilab_rt_display_s0i2_("c",3,1,_u_c);
  int _u_cSum;
  scilab_rt_sum_i2_i0(3,1,_u_c,&_u_cSum);
  scilab_rt_display_s0i0_("cSum",_u_cSum);
  int _u_cSumC[3][1];
  scilab_rt_sum_i2s0_i2(3,1,_u_c,"c",3,1,_u_cSumC);
  scilab_rt_display_s0i2_("cSumC",3,1,_u_cSumC);
  int _u_cSum2[3][1];
  scilab_rt_sum_i2i0_i2(3,1,_u_c,2,3,1,_u_cSum2);
  scilab_rt_display_s0i2_("cSum2",3,1,_u_cSum2);
  int _u_cSumR;
  scilab_rt_sum_i2s0_i0(3,1,_u_c,"r",&_u_cSumR);
  scilab_rt_display_s0i0_("cSumR",_u_cSumR);
  int _u_cSum1;
  scilab_rt_sum_i2i0_i0(3,1,_u_c,1,&_u_cSum1);
  scilab_rt_display_s0i0_("cSum1",_u_cSum1);
  int _u_cSumM;
  scilab_rt_sum_i2s0_i0(3,1,_u_c,"m",&_u_cSumM);
  scilab_rt_display_s0i0_("cSumM",_u_cSumM);
  int _tmpxx2 = scilab_rt_sum_i0s0_(_u_cSumM,"m");
  int _u_cSumM2 = (_tmpxx2+1000);
  scilab_rt_display_s0i0_("cSumM2",_u_cSumM2);
  int _u_e[3][4];
  _u_e[0][0]=1;
  _u_e[0][1]=2;
  _u_e[0][2]=3;
  _u_e[0][3]=3;
  _u_e[1][0]=4;
  _u_e[1][1]=5;
  _u_e[1][2]=6;
  _u_e[1][3]=6;
  _u_e[2][0]=7;
  _u_e[2][1]=8;
  _u_e[2][2]=9;
  _u_e[2][3]=9;
  scilab_rt_display_s0i2_("e",3,4,_u_e);
  int _u_eSum;
  scilab_rt_sum_i2_i0(3,4,_u_e,&_u_eSum);
  scilab_rt_display_s0i0_("eSum",_u_eSum);
  int _u_eSume[3][1];
  scilab_rt_sum_i2s0_i2(3,4,_u_e,"c",3,1,_u_eSume);
  scilab_rt_display_s0i2_("eSume",3,1,_u_eSume);
  int _u_eSum2[3][1];
  scilab_rt_sum_i2i0_i2(3,4,_u_e,2,3,1,_u_eSum2);
  scilab_rt_display_s0i2_("eSum2",3,1,_u_eSum2);
  int _u_eSumR[1][4];
  scilab_rt_sum_i2s0_i2(3,4,_u_e,"r",1,4,_u_eSumR);
  scilab_rt_display_s0i2_("eSumR",1,4,_u_eSumR);
  int _u_eSum1[1][4];
  scilab_rt_sum_i2i0_i2(3,4,_u_e,1,1,4,_u_eSum1);
  scilab_rt_display_s0i2_("eSum1",1,4,_u_eSum1);
  int _u_eSumM[1][4];
  scilab_rt_sum_i2s0_i2(3,4,_u_e,"m",1,4,_u_eSumM);
  scilab_rt_display_s0i2_("eSumM",1,4,_u_eSumM);
  int _tmpxx3;
  scilab_rt_sum_i2s0_i0(1,4,_u_eSumM,"m",&_tmpxx3);
  int _u_eSumM2 = (_tmpxx3+1000);
  scilab_rt_display_s0i0_("eSumM2",_u_eSumM2);

  scilab_rt_terminate();
}

