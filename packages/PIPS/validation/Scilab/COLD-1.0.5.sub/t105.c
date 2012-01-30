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

  double _u_ad[5][4];
  _u_ad[0][0]=2.0;
  _u_ad[0][1]=3;
  _u_ad[0][2]=4.6;
  _u_ad[0][3]=5;
  _u_ad[1][0]=8;
  _u_ad[1][1]=5;
  _u_ad[1][2]=4;
  _u_ad[1][3]=7;
  _u_ad[2][0]=9;
  _u_ad[2][1]=8;
  _u_ad[2][2]=5;
  _u_ad[2][3]=2;
  _u_ad[3][0]=7;
  _u_ad[3][1]=8;
  _u_ad[3][2]=5;
  _u_ad[3][3]=4;
  _u_ad[4][0]=9;
  _u_ad[4][1]=2;
  _u_ad[4][2]=1;
  _u_ad[4][3]=4;
  scilab_rt_display_s0d2_("ad",5,4,_u_ad);
  scilab_rt_disp_s0_("r i");
  double _u_bd[5][4];
  scilab_rt_gsort_d2s0s0_d2(5,4,_u_ad,"r","i",5,4,_u_bd);
  scilab_rt_display_s0d2_("bd",5,4,_u_bd);
  scilab_rt_disp_s0_("c d");
  double _u_cd[5][4];
  scilab_rt_gsort_d2s0s0_d2(5,4,_u_ad,"c","d",5,4,_u_cd);
  scilab_rt_display_s0d2_("cd",5,4,_u_cd);
  scilab_rt_disp_s0_("r");
  double _u_dd[5][4];
  scilab_rt_gsort_d2s0_d2(5,4,_u_ad,"r",5,4,_u_dd);
  scilab_rt_display_s0d2_("dd",5,4,_u_dd);
  scilab_rt_disp_s0_("no options");
  double _u_ed[5][4];
  scilab_rt_gsort_d2_d2(5,4,_u_ad,5,4,_u_ed);
  scilab_rt_display_s0d2_("ed",5,4,_u_ed);
  scilab_rt_disp_s0_("Two return values");
  scilab_rt_disp_s0_("g d");
  double _u_fd[5][4];
  int _u_id[5][4];
  scilab_rt_gsort_d2s0s0_d2i2(5,4,_u_ad,"g","d",5,4,_u_fd,5,4,_u_id);
  scilab_rt_display_s0i2_("id",5,4,_u_id);
  scilab_rt_display_s0d2_("fd",5,4,_u_fd);
  scilab_rt_disp_s0_("r");
  double _u_fd2[5][4];
  int _u_id2[5][4];
  scilab_rt_gsort_d2s0_d2i2(5,4,_u_ad,"r",5,4,_u_fd2,5,4,_u_id2);
  scilab_rt_display_s0i2_("id2",5,4,_u_id2);
  scilab_rt_display_s0d2_("fd2",5,4,_u_fd2);
  scilab_rt_disp_s0_("c i");
  double _u_fd3[5][4];
  int _u_id3[5][4];
  scilab_rt_gsort_d2s0s0_d2i2(5,4,_u_ad,"c","i",5,4,_u_fd3,5,4,_u_id3);
  scilab_rt_display_s0i2_("id3",5,4,_u_id3);
  scilab_rt_display_s0d2_("fd3",5,4,_u_fd3);
  /* ai=[20,3,4,5;8,5,4,7;9,8,5,2;7,8,5,4;9,2,1,4]  */
  /* disp("r i"); */
  /* bi=gsort(ai,'r','i') */
  /* disp("c d"); */
  /* ci=gsort(ai,'c','d') */
  /* disp("r"); */
  /* di=gsort(ai,'r') */
  /* disp("no args"); */
  /* ei=gsort(ai) */

  scilab_rt_terminate();
}

