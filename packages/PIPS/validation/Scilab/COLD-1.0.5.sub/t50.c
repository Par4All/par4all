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

  /*  Compute the image */
  int _u_nmax = 300;
  double _u_xmin = 0.2675;
  double _u_xmax = 0.2685;
  double _u_ymin = 0.591;
  double _u_ymax = 0.592;
  int _u_xsize = 400;
  int _u_ysize = 400;
  double _tmpxx0 = (_u_xmax-_u_xmin);
  int _tmpxx1 = (_u_xsize-1);
  double _u_xstep = (_tmpxx0 / _tmpxx1);
  double _tmpxx2 = (_u_ymax-_u_ymin);
  int _tmpxx3 = (_u_ysize-1);
  double _u_ystep = (_tmpxx2 / _tmpxx3);
  double _u_result[400][400];
  scilab_rt_zeros_i0i0_d2(_u_xsize,_u_ysize,400,400,_u_result);
  for (int _u_i=1; _u_i<=_u_xsize; _u_i++) {
    for (int _u_j=1; _u_j<=_u_ysize; _u_j++) {
      double _tmpxx4 = (_u_i*_u_xstep);
      double _u_x = (_u_xmin+_tmpxx4);
      double _tmpxx5 = (_u_j*_u_ystep);
      double _u_y = (_u_ymin+_tmpxx5);
      double _u_x0 = _u_x;
      double _u_y0 = _u_y;
      int _u_k = 0;
      while ( ((((_u_x*_u_x)+(_u_y*_u_y))<4) && (_u_k<_u_nmax)) ) {
        double _tmpxx6 = (_u_x*_u_x);
        double _tmpxx7 = (_u_y*_u_y);
        double _tmpxx8 = (_tmpxx6-_tmpxx7);
        double _u_xtemp = (_tmpxx8+_u_x0);
        double _tmpxx9 = (2*_u_x);
        double _tmpxx10 = (_tmpxx9*_u_y);
        _u_y = (_tmpxx10+_u_y0);
        _u_x = _u_xtemp;
        _u_k = (_u_k+1);
      }
      /*  result(i,j)= k + 1 - (log(log((sqrt(x*x + y*y)))))/(log(2)); */
      /* result(i,j)= k + 1 - (log(log((sqrt(x*x + y*y)))))/(log(2.)); */
      _u_result[_u_i-1][_u_j-1] = _u_k;
    }
  }
  /*  scilab */
  char* __cmd0[11]={
    "",
    "map = jetcolormap(512);",
    "cmap = [map(:,3), map(:,2), map(:,1)];",
    "f=get(\"current_figure\");",
    "f.color_map = cmap;",
    "f.auto_resize = \"off\";",
    "f.axes_size = [1000,1000];",
    "f.figure_size = [800,800];",
    "clf();",
    "a=get(\"current_axes\");",
    "a.margins = [0,0,0,0];"};
      scilab_rt_send_to_scilab_s1_(11,__cmd0);
  /*  endscilab */
  double _tmpxx11;
  scilab_rt_max_d2_d0(400,400,_u_result,&_tmpxx11);
  double _u_fact = (_tmpxx11 / 512.);
  scilab_rt_display_s0d0_("fact",_u_fact);
  double _tmpxx12[400][400];
  scilab_rt_eltdiv_d2d0_d2(400,400,_u_result,_u_fact,400,400,_tmpxx12);
  
  scilab_rt_assign_d2_d2(400,400,_tmpxx12,400,400,_u_result);
  scilab_rt_Matplot_d2s0_(400,400,_u_result,"080");
  scilab_rt_sleep_i0_(5000);

  scilab_rt_terminate();
}

