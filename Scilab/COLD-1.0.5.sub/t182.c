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

  /*  t182.sce: write_to_scilab function */
  int _u_s_int = 42;
  scilab_rt_display_s0i0_("s_int",_u_s_int);
  double _u_s_real = 42.24;
  scilab_rt_display_s0d0_("s_real",_u_s_real);
  double complex _tmpxx0 = (3*I);
  double complex _u_s_complex = (2+_tmpxx0);
  scilab_rt_display_s0z0_("s_complex",_u_s_complex);
  char* _u_s_string = (char *) malloc(5);
  strcpy(_u_s_string, "titi");
  scilab_rt_display_s0s0_("s_string",_u_s_string);
  scilab_rt_write_to_scilab_s0i0_("s_int",_u_s_int);
  scilab_rt_write_to_scilab_s0d0_("s_real",_u_s_real);
  scilab_rt_write_to_scilab_s0z0_("s_complex",_u_s_complex);
  scilab_rt_write_to_scilab_s0s0_("s_string",_u_s_string);
  /*  scilab */
  char* __cmd0[4]={
    "  disp(s_int);",
    "  disp(s_real);",
    "  disp(s_complex);",
    "  disp(s_string);"};
      scilab_rt_send_to_scilab_s1_(4,__cmd0);
  /*  endscilab */
  int _u_m_int[2][4];
  _u_m_int[0][0]=1;
  _u_m_int[0][1]=2;
  _u_m_int[0][2]=3;
  _u_m_int[0][3]=4;
  _u_m_int[1][0]=5;
  _u_m_int[1][1]=6;
  _u_m_int[1][2]=7;
  _u_m_int[1][3]=8;
  scilab_rt_display_s0i2_("m_int",2,4,_u_m_int);
  double _u_m_real[2][4];
  _u_m_real[0][0]=1.1;
  _u_m_real[0][1]=2.2;
  _u_m_real[0][2]=3.3;
  _u_m_real[0][3]=4.4;
  _u_m_real[1][0]=5.5;
  _u_m_real[1][1]=6.6;
  _u_m_real[1][2]=7.7;
  _u_m_real[1][3]=8.8;
  scilab_rt_display_s0d2_("m_real",2,4,_u_m_real);
  double complex _tmpxx1 = (2*I);
  double complex _tmpxx2 = (4*I);
  double complex _tmpxx3 = (6*I);
  double complex _tmpxx4 = (8*I);
  double complex _u_m_complex[2][2];
  _u_m_complex[0][0]=(1+_tmpxx1);
  _u_m_complex[0][1]=(3+_tmpxx2);
  _u_m_complex[1][0]=(5+_tmpxx3);
  _u_m_complex[1][1]=(7+_tmpxx4);
  scilab_rt_display_s0z2_("m_complex",2,2,_u_m_complex);
  char* _u_m_string[2][2];
  _u_m_string[0][0]="foo";
  _u_m_string[0][1]="bar";
  _u_m_string[1][0]="quux";
  _u_m_string[1][1]="titi";
  scilab_rt_display_s0s2_("m_string",2,2,_u_m_string);
  scilab_rt_write_to_scilab_s0i2_("m_int",2,4,_u_m_int);
  scilab_rt_write_to_scilab_s0d2_("m_real",2,4,_u_m_real);
  scilab_rt_write_to_scilab_s0z2_("m_complex",2,2,_u_m_complex);
  scilab_rt_write_to_scilab_s0s2_("m_string",2,2,_u_m_string);
  /*  scilab */
  char* __cmd1[4]={
    "  disp(m_int);",
    "  disp(m_real);",
    "  disp(m_complex);",
    "  disp(m_string);"};
      scilab_rt_send_to_scilab_s1_(4,__cmd1);
  /*  endscilab */

  scilab_rt_terminate();
}

