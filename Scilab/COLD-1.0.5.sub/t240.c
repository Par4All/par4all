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

  /*  t240.sce - from mcgill/nb3d.sce */
  scilab_rt_tic__();
  int _u_scale = 1;
  double _tmpxx0 = pow(_u_scale,0.4);
  int _u_n = scilab_rt_round_d0_((_tmpxx0*30));
  _u_n = 30;
  double _u_dT = (0.5*0.0833);
  double _tmpxx1 = (0.5*32.4362);
  double _tmpxx2 = scilab_rt_sqrt_i0_(_u_scale);
  double _u_T = (_tmpxx1*_tmpxx2);
  /* m = n   n = 3  seed = 0.1  M = R */
  double _u_seed = 0.1;
  int _tmpxx3 = (_u_n*3);
  _u_seed = (_u_seed+_tmpxx3);
  double _u_R[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_R);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=3; _u_j++) {
      double _tmpxx4 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_R[_u_i-1][_u_j-1] = _tmpxx4;
      double _tmpxx5 = _u_R[_u_i-1][_u_j-1];
      double _tmpxx6 = scilab_rt_sqrt_i0_(100);
      double _tmpxx7 = (_tmpxx5*_tmpxx6);
      double _tmpxx8 = (_u_seed+_tmpxx7);
      double _tmpxx9 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx8+_tmpxx9);
    }
  }
  double _tmpxx10[30][3];
  scilab_rt_mul_d2d0_d2(30,3,_u_R,1000.23,30,3,_tmpxx10);
  
  scilab_rt_assign_d2_d2(30,3,_tmpxx10,30,3,_u_R);
  /* m = n   n = 1  seed = 0.9  M = m */
  _u_seed = 0.9;
  int _tmpxx11 = (_u_n*1);
  _u_seed = (_u_seed+_tmpxx11);
  double _u_m[30][1];
  scilab_rt_zeros_i0i0_d2(_u_n,1,30,1,_u_m);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=1; _u_j++) {
      double _tmpxx12 = scilab_rt_modulo_d0i0_(_u_seed,1);
      _u_m[_u_i-1][_u_j-1] = _tmpxx12;
      double _tmpxx13 = _u_m[_u_i-1][_u_j-1];
      double _tmpxx14 = scilab_rt_sqrt_i0_(100);
      double _tmpxx15 = (_tmpxx13*_tmpxx14);
      double _tmpxx16 = (_u_seed+_tmpxx15);
      double _tmpxx17 = scilab_rt_sqrt_i0_(2);
      _u_seed = (_tmpxx16+_tmpxx17);
    }
  }
  double _tmpxx18[30][1];
  scilab_rt_mul_d2i0_d2(30,1,_u_m,345,30,1,_tmpxx18);
  
  scilab_rt_assign_d2_d2(30,1,_tmpxx18,30,1,_u_m);
  double _u_F[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_F);
  double _u_V[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_V);
  double _u_G = 1E-11;
  int _u_vno[1][30];
  for(int __tri0=0;__tri0 < 30;__tri0++) {
    _u_vno[0][__tri0] = 1+__tri0*1;
  }
  int _u_vnt[30][1];
  scilab_rt_transposeConjugate_i2_i2(1,30,_u_vno,30,1,_u_vnt);
  /* ii = vno(ones(n, 1), :);  */
  double _tmpxx19[30][30];
  scilab_rt_ones_i0i0_d2(30,30,30,30,_tmpxx19);
  int _u_ii[30][30];
  scilab_rt_round_d2_i2(30,30,_tmpxx19,30,30,_u_ii);
  for (int _u_i=1; _u_i<=30; _u_i++) {
    for (int _u_j=1; _u_j<=30; _u_j++) {
      _u_ii[_u_i-1][_u_j-1] = _u_j;
    }
  }
  /* jj = vnt(:, ones(n, 1));  */
  double _tmpxx20[30][30];
  scilab_rt_ones_i0i0_d2(30,30,30,30,_tmpxx20);
  int _u_jj[30][30];
  scilab_rt_round_d2_i2(30,30,_tmpxx20,30,30,_u_jj);
  for (int _u_i=1; _u_i<=30; _u_i++) {
    for (int _u_j=1; _u_j<=30; _u_j++) {
      _u_jj[_u_i-1][_u_j-1] = _u_i;
    }
  }
  int _tmpxx21[1][30];
  for(int __tri1=0;__tri1 < 30;__tri1++) {
    _tmpxx21[0][__tri1] = 1+__tri1*30;
  }
  int _tmpxx22[1][30];
  for(int __tri2=0;__tri2 < 30;__tri2++) {
    _tmpxx22[0][__tri2] = 0+__tri2*1;
  }
  int _u_kk[1][30];
  scilab_rt_add_i2i2_i2(1,30,_tmpxx21,1,30,_tmpxx22,1,30,_u_kk);
  double _u_dr[30][30][3];
  scilab_rt_zeros_i0i0i0_d3(_u_n,_u_n,3,30,30,3,_u_dr);
  double _u_fr[30][30][3];
  scilab_rt_zeros_i0i0i0_d3(_u_n,_u_n,3,30,30,3,_u_fr);
  double _u_a[30][3];
  scilab_rt_zeros_i0i0_d2(_u_n,3,30,3,_u_a);
  for (double _u_t=1; _u_t<=_u_T; _u_t+=_u_dT) {
    /* dr(:) = R(jj, :) - R(ii, :); */
    /*  acc = 0; */
    for (int _u_z=1; _u_z<=3; _u_z++) {
      for (int _u_i=1; _u_i<=30; _u_i++) {
        for (int _u_j=1; _u_j<=30; _u_j++) {
          double _tmpxx23 = (_u_R[_u_jj[_u_i-1][_u_j-1]-1][_u_z-1]-_u_R[_u_ii[_u_i-1][_u_j-1]-1][_u_z-1]);
          _u_dr[_u_i-1][_u_j-1][_u_z-1] = _tmpxx23;
          /*        acc = acc + dr(i,j,z); */
        }
      }
    }
    /*  acc */
    double _u_r[30][30][3];
    scilab_rt_eltmul_d3d3_d3(30,30,3,_u_dr,30,30,3,_u_dr,30,30,3,_u_r);
    /*  r = r(:, :, 1) + r(:, :, 2) + r(:, :, 3);  */
    double _u_rt[30][30];
    scilab_rt_ones_i0i0_d2(_u_n,_u_n,30,30,_u_rt);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx24 = ((_u_r[_u_i-1][_u_j-1][0]+_u_r[_u_i-1][_u_j-1][1])+_u_r[_u_i-1][_u_j-1][2]);
        _u_rt[_u_i-1][_u_j-1] = _tmpxx24;
      }
    }
    /* r(kk) = 1.0; */
    /* r(kk(1,i)) = 1.0; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      _u_rt[((_u_kk[0][(_u_i-1)]-1) % 30)][((_u_kk[0][(_u_i-1)]-1) / 30)] = 1.0;
    }
    double _tmpxx25[1][30];
    scilab_rt_transposeConjugate_d2_d2(30,1,_u_m,1,30,_tmpxx25);
    double _u_MM[30][30];
    scilab_rt_mul_d2d2_d2(30,1,_u_m,1,30,_tmpxx25,30,30,_u_MM);
    /* MM(kk) = 0.0; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      _u_MM[((_u_kk[0][(_u_i-1)]-1) % 30)][((_u_kk[0][(_u_i-1)]-1) / 30)] = 0.0;
    }
    double _tmpxx26[30][30];
    scilab_rt_eltdiv_d2d2_d2(30,30,_u_MM,30,30,_u_rt,30,30,_tmpxx26);
    double _u_f[30][30];
    scilab_rt_mul_d0d2_d2(_u_G,30,30,_tmpxx26,30,30,_u_f);
    double _tmpxx27[30][30];
    scilab_rt_sqrt_d2_d2(30,30,_u_rt,30,30,_tmpxx27);
    
    scilab_rt_assign_d2_d2(30,30,_tmpxx27,30,30,_u_rt);
    /*  dr(:, :, 1) = dr(:, :, 1) ./ r; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx28 = (_u_dr[_u_i-1][_u_j-1][0] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][0] = _tmpxx28;
      }
    }
    /* dr(:, :, 2) = dr(:, :, 2) ./ r; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx29 = (_u_dr[_u_i-1][_u_j-1][1] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][1] = _tmpxx29;
      }
    }
    /* dr(:, :, 3) = dr(:, :, 3) ./ r; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx30 = (_u_dr[_u_i-1][_u_j-1][2] / _u_rt[_u_i-1][_u_j-1]);
        _u_dr[_u_i-1][_u_j-1][2] = _tmpxx30;
      }
    }
    /*   */
    /* fr(:, :, 1) = f .* dr(:, :, 1); */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx31 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][0]);
        _u_fr[_u_i-1][_u_j-1][0] = _tmpxx31;
      }
    }
    /* fr(:, :, 2) = f .* dr(:, :, 2); */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx32 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][1]);
        _u_fr[_u_i-1][_u_j-1][1] = _tmpxx32;
      }
    }
    /* fr(:, :, 3) = f .* dr(:, :, 3); */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=30; _u_j++) {
        double _tmpxx33 = (_u_f[_u_i-1][_u_j-1]*_u_dr[_u_i-1][_u_j-1][2]);
        _u_fr[_u_i-1][_u_j-1][2] = _tmpxx33;
      }
    }
    /* F(:) = mean(fr) * n; */
    double _u_mn[1][30][3];
    scilab_rt_zeros_i0i0i0_d3(1,_u_n,3,1,30,3,_u_mn);
    
    scilab_rt_mean_d3s0_d3(30,30,3,_u_fr,"m",1,30,3,_u_mn);
    for (int _u_i=1; _u_i<=30; _u_i++) {
      for (int _u_j=1; _u_j<=3; _u_j++) {
        double _tmpxx34 = (_u_mn[0][_u_i-1][_u_j-1]*_u_n);
        _u_F[_u_i-1][_u_j-1] = _tmpxx34;
      }
    }
    /*      a(:, 1) = F(:, 1) ./ m; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx35 = (_u_F[_u_i-1][0] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][0] = _tmpxx35;
    }
    /*      a(:, 2) = F(:, 2) ./ m; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx36 = (_u_F[_u_i-1][1] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][1] = _tmpxx36;
    }
    /*      a(:, 3) = F(:, 3) ./ m; */
    for (int _u_i=1; _u_i<=30; _u_i++) {
      double _tmpxx37 = (_u_F[_u_i-1][2] / _u_m[_u_i-1][0]);
      _u_a[_u_i-1][2] = _tmpxx37;
    }
    double _tmpxx38[30][3];
    scilab_rt_mul_d2d0_d2(30,3,_u_a,_u_dT,30,3,_tmpxx38);
    double _tmpxx39[30][3];
    scilab_rt_add_d2d2_d2(30,3,_u_V,30,3,_tmpxx38,30,3,_tmpxx39);
    
    scilab_rt_assign_d2_d2(30,3,_tmpxx39,30,3,_u_V);
    double _tmpxx40[30][3];
    scilab_rt_mul_d2d0_d2(30,3,_u_V,_u_dT,30,3,_tmpxx40);
    double _tmpxx41[30][3];
    scilab_rt_add_d2d2_d2(30,3,_u_R,30,3,_tmpxx40,30,3,_tmpxx41);
    
    scilab_rt_assign_d2_d2(30,3,_tmpxx41,30,3,_u_R);
  }
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time"); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean value of matrix V");
  double _tmpxx42;
  scilab_rt_mean_d2_d0(30,3,_u_V,&_tmpxx42);
  scilab_rt_disp_d0_(_tmpxx42);
  scilab_rt_disp_s0_("Mean value of matrix R");
  double _tmpxx43;
  scilab_rt_mean_d2_d0(30,3,_u_R,&_tmpxx43);
  scilab_rt_disp_d0_(_tmpxx43);

  scilab_rt_terminate();
}

