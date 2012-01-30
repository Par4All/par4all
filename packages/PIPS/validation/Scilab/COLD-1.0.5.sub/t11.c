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

  int _u_a = 1;
  int _u_b = 2;
  int _u_c = 3;
  int _u_d = 4;
  (_u_a<_u_b);
  /*  a < b */
  (1<10);
  /*  1 < 10 */
  int _tmpxx0 = (_u_a<_u_b);
  int _tmpxx1 = (_u_c>_u_d);
  (_tmpxx0 || _tmpxx1);
  /*  (a < b) | (c > d) */
  int _tmpxx2 = (_u_b && _u_c);
  (_u_a || _tmpxx2);
  /*  a | (b & c) */
  int _tmpxx3 = (_u_a && _u_b);
  (_tmpxx3 || _u_c);
  /*  (a & b) | c */
  int _tmpxx4 = (_u_a==_u_b);
  int _tmpxx5 = (_u_c==_u_d);
  (_tmpxx4 && _tmpxx5);
  /*  (a == b) & (c == d) */
  /*  ~ not supported */

  scilab_rt_terminate();
}

