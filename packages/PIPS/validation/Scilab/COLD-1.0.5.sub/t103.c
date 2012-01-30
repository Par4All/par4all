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

  double _u_sub[1][3];
  _u_sub[0][0]=1547.0;
  _u_sub[0][1]=5478;
  _u_sub[0][2]=3652;
  double _u_super[1][24];
  _u_super[0][0]=895.0;
  _u_super[0][1]=254;
  _u_super[0][2]=6652;
  _u_super[0][3]=874;
  _u_super[0][4]=1547;
  _u_super[0][5]=5477;
  _u_super[0][6]=9852;
  _u_super[0][7]=1549;
  _u_super[0][8]=3652;
  _u_super[0][9]=1458;
  _u_super[0][10]=9745;
  _u_super[0][11]=214;
  _u_super[0][12]=3650;
  _u_super[0][13]=3651;
  _u_super[0][14]=1549;
  _u_super[0][15]=5475;
  _u_super[0][16]=2145;
  _u_super[0][17]=87;
  _u_super[0][18]=4589;
  _u_super[0][19]=6521;
  _u_super[0][20]=5861;
  _u_super[0][21]=1544;
  _u_super[0][22]=3649;
  _u_super[0][23]=5481;
  double _tmp0[1][24];
  scilab_rt_datefind_d2d2_d2(1,3,_u_sub,1,24,_u_super,1,24,_tmp0);
  scilab_rt_display_s0d2_("ans",1,24,_tmp0);

  scilab_rt_terminate();
}

