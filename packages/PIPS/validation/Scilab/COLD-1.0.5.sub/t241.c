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

void nb3d__d2d2(int _u_V_n0,int _u_V_n1,double _u_V[_u_V_n0][_u_V_n1], int _u_R_n0,int _u_R_n1,double _u_R[_u_R_n0][_u_R_n1])
{
  int _u_scale = 1;
  double _tmpxx2 = pow(_u_scale,0.4);
  int _u_n = scilab_rt_round_d0_((_tmpxx2*30));
  _u_n = 30;
  double _u_dT = (0.5*0.0833);
  double _tmpxx3 = (0.5*32.4362);
  double _tmpxx4 = scilab_rt_sqrt_i0_(_u_scale);
  double _u_T = (_tmpxx3*_tmpxx4);
  double _u_seed = 0.1;
  int _tmpxx5 = (_u_n*3);
  _u_seed = (_u_seed+_tmpxx5);
  
  scilab_rt_zeros_i0i0_d2(_u_n,3,_u_R_n0,_u_R_n1,_u_R);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=3; _u_j++) {
      double _tmpxx6 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_R[_u_i-1][_u_j-1] = _tmpxx6;
      double _tmpxx7 = _u_R[_u_i-1][_u_j-1];
      double _tmpxx8 = scilab_rt_sqrt_i0_(100);
      double _tmpxx9 = (_tmpxx7*_tmpxx8);
      double _tmpxx10 = (_u_seed+_tmpxx9);
      double _tmpxx11 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx10+_tmpxx11);
    }
  }
  double _tmpxx12[30][3];
  scilab_rt_mul_d2d0_d2(_u_R_n0,_u_R_n1,_u_R,1000.23,30,3,_tmpxx12);
  
  scilab_rt_assign_d2_d2(30,3,_tmpxx12,_u_R_n0,_u_R_n1,_u_R);
  _u_seed = 0.9;
  int _tmpxx13 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx13);
  double _u_m[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_m);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx14 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_m[_u_i-1][_u_j-1] = _tmpxx14;
      double _tmpxx15 = _u_m[_u_i-1][_u_j-1];
      double _tmpxx16 = scilab_rt_sqrt_i0_(100);
      double _tmpxx17 = (_tmpxx15*_tmpxx16);
      double _tmpxx18 = (_u_seed+_tmpxx17);
      double _tmpxx19 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx18+_tmpxx19);
    }
  }
  double _tmpxx20[30][1];
  scilab_rt_mul_d2i0_d2(30,1,_u_m,345,30,1,_tmpxx20);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx20,30,1,_u_m);
  double _u_F[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_F);
  
  scilab_rt_zeros_i0i0_d2(_u_n,3,_u_V_n0,_u_V_n1,_u_V);
  double _u_G = 1E-11;
  int _u_vno[1][30];
  for(int __tri0=0;__tri0 < 30;__tri0++) {
    _u_vno[0][__tri0] = 1+__tri0*1;
  }
  int _u_vnt[30][1];
  scilab_rt_transposeConjugate_i2_i2(1,30,_u_vno,30,1,_u_vnt);
  double _tmpxx21[30][30];
  scilab_rt_ones_i0i0_d2(30,30,30,30,_tmpxx21);
  int _u_ii[30][30];
  scilab_rt_round_d2_i2(30,30,_tmpxx21,30,30,_u_ii);
  for (int _u_i=1; _u_i<=30; _u_i++) {
    for (int _u_j=1; _u_j<=30; _u_j++) {
      _u_ii[_u_i-1][_u_j-1] = _u_j;
    }
  }
  double _tmpxx22[30][30];
  scilab_rt_ones_i0i0_d2(30,30,30,30,_tmpxx22);
  int _u_jj[30][30];
  scilab_rt_round_d2_i2(30,30,_tmpxx22,30,30,_u_jj);
  for (int _u_i=1; _u_i<=30; _u_i++) {
    for (int _u_j=1; _u_j<=30; _u_j++) {
      _u_jj[_u_i-1][_u_j-1] = _u_i;
    }
  }
  int _tmpxx23[1][30];
  for(int __tri1=0;__tri1 < 30;__tri1++) {
    _tmpxx23[0][__tri1] = 1+__tri1*30;
  }
  int _tmpxx24[1][30];
  for(int __tri2=0;__tri2 < 30;__tri2++) {
    _tmpxx24[0][__tri2] = 0+__tri2*1;
  }
  int _u_kk[1][30];
  scilab_rt_add_i2i2_i2(1,30,_tmpxx23,1,30,_tmpxx24,1,30,_u_kk);
  double _u_dr[30][30][3];
  scilab_rt_zeros_i0i0i0_d3(_u_n,_u_n,3,30,30,3,_u_dr);
  double _u_fr[30][30][3];
  scilab_rt_zeros_i0i0i0_d3(_u_n,_u_n,3,30,30,3,_u_fr);
  double _u_a[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_a);
  for (double _u_t=1; _u_t<=_u_T; _u_t+=_u_dT) {
    for (int _u_z=1; _u_z<=3; _u_z++) {
      for (int _u_i=1; _u_i<=30; _u_i++) {
        for (int _u_j=1; _u_j<=30; _u_j++) {
          double _tmpxx25 = (_u_R[_u_jj[_u_i-1][_u_j-1]-1][_u_z-1]-_u_R[_u_ii[_u_i-1][_u_j-1]-1][_u_z-1]);
          _u_dr[_u_i-1][_u_j-1][_u_z-1] = _tmpxx25;
        }
      }
    }
    double _u_r[30][30][3];
    scilab_rt_eltmul_d3d3_d3(30,30,3,_u_dr,30,30,3,_u_dr,30,30,3,_u_r);
    double _u_rt[30][30];
    scilab_rt_ones_i0i0_d2(_u_n,_u_n,30,30,_u_rt);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx26 = ((_u_r[_u_i-1][_u_j-1][0]+_u_r[_u_i-1][_u_j-1][1])+_u_r[_u_i-1][_u_j-1][2]);
        _u_rt[_u_i-1][_u_j-1] = _tmpxx26;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      _u_rt[((_u_kk[0][(_u_i-1)]-1) % 30)][((_u_kk[0][(_u_i-1)]-1) / 30)] = 1.0;
    }
    double _tmpxx27[1][30];
    scilab_rt_transposeConjugate_d2_d2(30,1,_u_m,1,30,_tmpxx27);
    double _u_MM[30][30];
    scilab_rt_mul_d2d2_d2(30,1,_u_m,1,30,_tmpxx27,30,30,_u_MM);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      _u_MM[((_u_kk[0][(_u_i-1)]-1) % 30)][((_u_kk[0][(_u_i-1)]-1) / 30)] = 0.0;
    }
    double _tmpxx28[30][30];
    scilab_rt_eltdiv_d2d2_d2(30,30,_u_MM,30,30,_u_rt,30,30,_tmpxx28);
    double _u_f[30][30];
    scilab_rt_mul_d0d2_d2(_u_G,30,30,_tmpxx28,30,30,_u_f);
    double _tmpxx29[30][30];
    scilab_rt_sqrt_d2_d2(30,30,_u_rt,30,30,_tmpxx29);
    
    scilab_rt_assign_d2_d2(30,30,_tmpxx29,30,30,_u_rt);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx30 = (_u_dr[_u_i-1][_u_j-1][0] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][0] = _tmpxx30;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx31 = (_u_dr[_u_i-1][_u_j-1][1] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][1] = _tmpxx31;
      }
    }
    /* dr(:, :, 3) = dr(:, :, 3) ./ r; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx32 = (_u_dr[_u_i-1][_u_j-1][2] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][2] = _tmpxx32;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx33 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][0]);
        _u_fr[_u_i-1][_u_j-1][0] = _tmpxx33;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx34 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][1]);
        _u_fr[_u_i-1][_u_j-1][1] = _tmpxx34;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx35 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][2]);
        _u_fr[_u_i-1][_u_j-1][2] = _tmpxx35;
      }
    }
    double _u_mn[1][30][3];
    scilab_rt_zeros_i0i0i0_d3(1,_u_n,3,1,30,3,_u_mn);
    
    scilab_rt_mean_d3s0_d3(30,30,3,_u_fr,"m",1,30,3,_u_mn);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=3; _u_j++) {
        double _tmpxx36 = (_u_mn[0][_u_i-1][_u_j-1]*_u_n);
        _u_F[_u_i-1][_u_j-1] = _tmpxx36;
      }
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx37 = (_u_F[_u_i-1][0] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][0] = _tmpxx37;
    }
    /*      a(:, 2) = F(:, 2) ./ m; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx38 = (_u_F[_u_i-1][1] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][1] = _tmpxx38;
    }
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx39 = (_u_F[_u_i-1][2] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][2] = _tmpxx39;
    }
    double _tmpxx40[30][3];
    scilab_rt_mul_d2d0_d2(30,3,_u_a,_u_dT,30,3,_tmpxx40);
    double _tmpxx41[30][3];
    scilab_rt_add_d2d2_d2(_u_V_n0,_u_V_n1,_u_V,30,3,_tmpxx40,30,3,_tmpxx41);
    
    scilab_rt_assign_d2_d2(30,3,_tmpxx41,_u_V_n0,_u_V_n1,_u_V);
    double _tmpxx42[30][3];
    scilab_rt_mul_d2d0_d2(_u_V_n0,_u_V_n1,_u_V,_u_dT,30,3,_tmpxx42);
    double _tmpxx43[30][3];
    scilab_rt_add_d2d2_d2(_u_R_n0,_u_R_n1,_u_R,30,3,_tmpxx42,30,3,_tmpxx43);
    
    scilab_rt_assign_d2_d2(30,3,_tmpxx43,_u_R_n0,_u_R_n1,_u_R);
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t241.sce - from mcgill/nb3d_function.sce */
  scilab_rt_tic__();
  double _u_V[30][3];
  double _u_R[30][3];
  nb3d__d2d2(30,3,_u_V,30,3,_u_R);
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean value of matrix V: ");
  double _tmpxx0;
  scilab_rt_mean_d2_d0(30,3,_u_V,&_tmpxx0);
  scilab_rt_disp_d0_(_tmpxx0);
  scilab_rt_disp_s0_("Mean value of matrix R: ");
  double _tmpxx1;
  scilab_rt_mean_d2_d0(30,3,_u_R,&_tmpxx1);
  scilab_rt_disp_d0_(_tmpxx1);

  scilab_rt_terminate();
}

