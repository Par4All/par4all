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

  /*  t238.sce - from mcgill/nb1d.sce */
  /* This function M-file simulates the gravitational movement */
  /* of a set of objects */
  scilab_rt_tic__();
  double _u_seed = 1;
  int _u_scale = 1;
  double _tmpxx0 = pow(_u_scale,0.4);
  int _u_n = scilab_rt_round_d0_((_tmpxx0*30));
  _u_n = 30;
  double _u_dT = (0.5*0.0833);
  double _tmpxx1 = (0.5*32.4362);
  double _tmpxx2 = scilab_rt_sqrt_i0_(_u_scale);
  double _u_T = (_tmpxx1*_tmpxx2);
  /* m = n   n = 1  seed = 0.1  M = Rx */
  _u_seed = 0.1;
  int _tmpxx3 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx3);
  double _u_Rx[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Rx);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx4 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_Rx[_u_i-1][_u_j-1] = _tmpxx4;
      double _tmpxx5 = _u_Rx[_u_i-1][_u_j-1];
      double _tmpxx6 = scilab_rt_sqrt_i0_(100);
      double _tmpxx7 = (_tmpxx5*_tmpxx6);
      double _tmpxx8 = (_u_seed+_tmpxx7);
      double _tmpxx9 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx8+_tmpxx9);
    }
  }
  double _tmpxx10[30][1];
  scilab_rt_mul_d2d0_d2(30,1,_u_Rx,1000.23,30,1,_tmpxx10);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx10,30,1,_u_Rx);
  /* m = n   n = 1  seed = 0.4  M = Ry */
  _u_seed = 0.4;
  int _tmpxx11 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx11);
  double _u_Ry[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Ry);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx12 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_Ry[_u_i-1][_u_j-1] = _tmpxx12;
      double _tmpxx13 = _u_Ry[_u_i-1][_u_j-1];
      double _tmpxx14 = scilab_rt_sqrt_i0_(100);
      double _tmpxx15 = (_tmpxx13*_tmpxx14);
      double _tmpxx16 = (_u_seed+_tmpxx15);
      double _tmpxx17 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx16+_tmpxx17);
    }
  }
  double _tmpxx18[30][1];
  scilab_rt_mul_d2d0_d2(30,1,_u_Ry,1000.23,30,1,_tmpxx18);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx18,30,1,_u_Ry);
  /* m = n   n = 1  seed = 0.9  M = Rz */
  _u_seed = 0.9;
  int _tmpxx19 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx19);
  double _u_Rz[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Rz);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx20 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_Rz[_u_i-1][_u_j-1] = _tmpxx20;
      double _tmpxx21 = _u_Rz[_u_i-1][_u_j-1];
      double _tmpxx22 = scilab_rt_sqrt_i0_(100);
      double _tmpxx23 = (_tmpxx21*_tmpxx22);
      double _tmpxx24 = (_u_seed+_tmpxx23);
      double _tmpxx25 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx24+_tmpxx25);
    }
  }
  double _tmpxx26[30][1];
  scilab_rt_mul_d2d0_d2(30,1,_u_Rz,1000.23,30,1,_tmpxx26);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx26,30,1,_u_Rz);
  /* m = n   n = 1  seed = -0.4  M = m */
  _u_seed = (-0.4);
  int _tmpxx27 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx27);
  double _u_m[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_m);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx28 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_m[_u_i-1][_u_j-1] = _tmpxx28;
      double _tmpxx29 = _u_m[_u_i-1][_u_j-1];
      double _tmpxx30 = scilab_rt_sqrt_i0_(100);
      double _tmpxx31 = (_tmpxx29*_tmpxx30);
      double _tmpxx32 = (_u_seed+_tmpxx31);
      double _tmpxx33 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx32+_tmpxx33);
    }
  }
  double _tmpxx34[30][1];
  scilab_rt_mul_d2i0_d2(30,1,_u_m,345,30,1,_tmpxx34);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx34,30,1,_u_m);
  double _u_Fx[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Fx);
  double _u_Fy[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Fy);
  double _u_Fz[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Fz);
  double _u_Vx[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Vx);
  double _u_Vy[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Vy);
  double _u_Vz[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_Vz);
  double _u_G = 1E-11;
  for (double _u_t=1; _u_t<=_u_T; _u_t+=_u_dT) {
    for (int _u_k=1; _u_k<=_u_n; _u_k++) {
      double _tmpxx35 = _u_Rx[(_u_k-1)][0];
      double _u_drx[30][1];
      scilab_rt_sub_d2d0_d2(30,1,_u_Rx,_tmpxx35,30,1,_u_drx);
      double _tmpxx36 = _u_Ry[(_u_k-1)][0];
      double _u_dry[30][1];
      scilab_rt_sub_d2d0_d2(30,1,_u_Ry,_tmpxx36,30,1,_u_dry);
      double _tmpxx37 = _u_Rz[(_u_k-1)][0];
      double _u_drz[30][1];
      scilab_rt_sub_d2d0_d2(30,1,_u_Rz,_tmpxx37,30,1,_u_drz);
      double _tmpxx38[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_drx,30,1,_u_drx,30,1,_tmpxx38);
      double _tmpxx39[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_dry,30,1,_u_dry,30,1,_tmpxx39);
      double _tmpxx40[30][1];
      scilab_rt_add_d2d2_d2(30,1,_tmpxx38,30,1,_tmpxx39,30,1,_tmpxx40);
      double _tmpxx41[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_drz,30,1,_u_drz,30,1,_tmpxx41);
      double _u_r[30][1];
      scilab_rt_add_d2d2_d2(30,1,_tmpxx40,30,1,_tmpxx41,30,1,_u_r);
      _u_r[(_u_k-1)][0] = 1.0;
      double _tmpxx42 = _u_m[(_u_k-1)][0];
      double _u_M[30][1];
      scilab_rt_mul_d2d0_d2(30,1,_u_m,_tmpxx42,30,1,_u_M);
      _u_M[(_u_k-1)][0] = 0.0;
      double _tmpxx43[30][1];
      scilab_rt_eltdiv_d2d2_d2(30,1,_u_M,30,1,_u_r,30,1,_tmpxx43);
      double _u_f[30][1];
      scilab_rt_mul_d0d2_d2(_u_G,30,1,_tmpxx43,30,1,_u_f);
      double _tmpxx44[30][1];
      scilab_rt_sqrt_d2_d2(30,1,_u_r,30,1,_tmpxx44);
      
      scilab_rt_assign_d2_d2(30,1,_tmpxx44,30,1,_u_r);
      double _tmpxx45[30][1];
      scilab_rt_eltdiv_d2d2_d2(30,1,_u_drx,30,1,_u_r,30,1,_tmpxx45);
      
      scilab_rt_assign_d2_d2(30,1,_tmpxx45,30,1,_u_drx);
      double _tmpxx46[30][1];
      scilab_rt_eltdiv_d2d2_d2(30,1,_u_dry,30,1,_u_r,30,1,_tmpxx46);
      
      scilab_rt_assign_d2_d2(30,1,_tmpxx46,30,1,_u_dry);
      double _tmpxx47[30][1];
      scilab_rt_eltdiv_d2d2_d2(30,1,_u_drz,30,1,_u_r,30,1,_tmpxx47);
      
      scilab_rt_assign_d2_d2(30,1,_tmpxx47,30,1,_u_drz);
      double _u_frx[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_f,30,1,_u_drx,30,1,_u_frx);
      double _u_fry[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_f,30,1,_u_dry,30,1,_u_fry);
      double _u_frz[30][1];
      scilab_rt_eltmul_d2d2_d2(30,1,_u_f,30,1,_u_drz,30,1,_u_frz);
      double _tmpxx48 = (scilab_rt_mean_d2_(30,1,_u_frx)*_u_n);
      _u_Fx[(_u_k-1)][0] = _tmpxx48;
      double _tmpxx49 = (scilab_rt_mean_d2_(30,1,_u_fry)*_u_n);
      _u_Fy[(_u_k-1)][0] = _tmpxx49;
      double _tmpxx50 = (scilab_rt_mean_d2_(30,1,_u_frz)*_u_n);
      _u_Fz[(_u_k-1)][0] = _tmpxx50;
    }
    double _u_ax[30][1];
    scilab_rt_eltdiv_d2d2_d2(30,1,_u_Fx,30,1,_u_m,30,1,_u_ax);
    double _u_ay[30][1];
    scilab_rt_eltdiv_d2d2_d2(30,1,_u_Fy,30,1,_u_m,30,1,_u_ay);
    double _u_az[30][1];
    scilab_rt_eltdiv_d2d2_d2(30,1,_u_Fz,30,1,_u_m,30,1,_u_az);
    double _tmpxx51[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_ax,_u_dT,30,1,_tmpxx51);
    double _tmpxx52[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Vx,30,1,_tmpxx51,30,1,_tmpxx52);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx52,30,1,_u_Vx);
    double _tmpxx53[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_ay,_u_dT,30,1,_tmpxx53);
    double _tmpxx54[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Vy,30,1,_tmpxx53,30,1,_tmpxx54);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx54,30,1,_u_Vy);
    double _tmpxx55[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_az,_u_dT,30,1,_tmpxx55);
    double _tmpxx56[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Vz,30,1,_tmpxx55,30,1,_tmpxx56);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx56,30,1,_u_Vz);
    double _tmpxx57[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_Vx,_u_dT,30,1,_tmpxx57);
    double _tmpxx58[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Rx,30,1,_tmpxx57,30,1,_tmpxx58);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx58,30,1,_u_Rx);
    double _tmpxx59[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_Vy,_u_dT,30,1,_tmpxx59);
    double _tmpxx60[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Ry,30,1,_tmpxx59,30,1,_tmpxx60);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx60,30,1,_u_Ry);
    double _tmpxx61[30][1];
    scilab_rt_mul_d2d0_d2(30,1,_u_Vz,_u_dT,30,1,_tmpxx61);
    double _tmpxx62[30][1];
    scilab_rt_add_d2d2_d2(30,1,_u_Rz,30,1,_tmpxx61,30,1,_tmpxx62);
    
    scilab_rt_assign_d2_d2(30,1,_tmpxx62,30,1,_u_Rz);
  }
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean value of matrix Vx");
  double _tmpxx63;
  scilab_rt_mean_d2_d0(30,1,_u_Vx,&_tmpxx63);
  scilab_rt_disp_d0_(_tmpxx63);
  scilab_rt_disp_s0_("Mean value of matrix Vy");
  double _tmpxx64;
  scilab_rt_mean_d2_d0(30,1,_u_Vy,&_tmpxx64);
  scilab_rt_disp_d0_(_tmpxx64);
  scilab_rt_disp_s0_("Mean value of matrix Vz");
  double _tmpxx65;
  scilab_rt_mean_d2_d0(30,1,_u_Vz,&_tmpxx65);
  scilab_rt_disp_d0_(_tmpxx65);
  scilab_rt_disp_s0_("Mean value of matrix Rx");
  double _tmpxx66;
  scilab_rt_mean_d2_d0(30,1,_u_Rx,&_tmpxx66);
  scilab_rt_disp_d0_(_tmpxx66);
  scilab_rt_disp_s0_("Mean value of matrix Ry");
  double _tmpxx67;
  scilab_rt_mean_d2_d0(30,1,_u_Ry,&_tmpxx67);
  scilab_rt_disp_d0_(_tmpxx67);
  scilab_rt_disp_s0_("Mean value of matrix Rz");
  double _tmpxx68;
  scilab_rt_mean_d2_d0(30,1,_u_Rz,&_tmpxx68);
  scilab_rt_disp_d0_(_tmpxx68);

  scilab_rt_terminate();
}

