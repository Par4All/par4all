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

  int _u_a[1][10];
  for(int __tri0=0;__tri0 < 10;__tri0++) {
    _u_a[0][__tri0] = 1+__tri0*1;
  }
  scilab_rt_display_s0i2_("a",1,10,_u_a);
  int _u_b[1][2];
  scilab_rt_size_i2_i2(1,10,_u_a,1,2,_u_b);
  scilab_rt_display_s0i2_("b",1,2,_u_b);
  int _u_c[1][5];
  for(int __tri1=0;__tri1 < 5;__tri1++) {
    _u_c[0][__tri1] = 1+__tri1*2;
  }
  scilab_rt_display_s0i2_("c",1,5,_u_c);
  int _u_d[1][2];
  scilab_rt_size_i2_i2(1,5,_u_c,1,2,_u_d);
  scilab_rt_display_s0i2_("d",1,2,_u_d);
  double _u_e[1][991];
  for(int __tri2=0;__tri2 < 991;__tri2++) {
    _u_e[0][__tri2] = 1+__tri2*0.1;
  }
  int _u_f[1][2];
  scilab_rt_size_d2_i2(1,991,_u_e,1,2,_u_f);
  scilab_rt_display_s0i2_("f",1,2,_u_f);

  scilab_rt_terminate();
}

