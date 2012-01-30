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

  /*  scilab */
  char* __cmd0[10]={
    "map = jetcolormap(512);",
    "cmap = [map(:,3), map(:,2), map(:,1)];",
    "f=get(\"current_figure\");",
    "f.color_map=cmap;",
    "f.auto_resize = \"off\";",
    "f.axes_size= [1000, 1000];",
    "c.figure_size= [800,800];",
    "clf();",
    "a=get(\"current_axes\");",
    "a.margins = [0,0,0,0];"};
      scilab_rt_send_to_scilab_s1_(10,__cmd0);
  /*  endscilab */
  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);
  scilab_rt_Matplot_d2s0_(10,10,_u_a,"080");
  scilab_rt_sleep_i0_(5000);

  scilab_rt_terminate();
}

