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

  /*  t230.sce - from mcgill/fdtd.sce */
  /* This function applies the Finite Difference */
  /* Time Domain (FDTD) technique on a hexahedral cavity */
  /* with conducting walls. FDTD is a powerful tool for */
  /* transient electromagnetic analysis.  */
  scilab_rt_tic__();
  int _u_scale = 1;
  double _u_Lx = 0.05;
  double _u_Ly = 0.04;
  double _u_Lz = 0.03;
  int _u_Nx = 25;
  int _u_Ny = 20;
  int _u_Nz = 15;
  double _u_nrm = 866.0254;
  int _u_Nt = (_u_scale*200);
  /* [Ex, Ey, Ez, Hx, Hy, Hz, Ets] = fdtd(Lx, Ly, Lz, Nx, Ny, Nz, nrm, Nt); */
  double _u_eps0 = 8.8541878E-12;
  double _u_mu0 = (4E-7*SCILAB_PI);
  double _u_c0 = 299792458.0;
  double _u_Cx = (_u_Nx / _u_Lx);
  double _u_Cy = (_u_Ny / _u_Ly);
  double _u_Cz = (_u_Nz / _u_Lz);
  double _tmpxx0 = (_u_c0*_u_nrm);
  double _u_Dt = (1 / _tmpxx0);
  double _u_Ex[25][21][16];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,(_u_Ny+1),(_u_Nz+1),25,21,16,_u_Ex);
  double _u_Ey[26][20][16];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),_u_Ny,(_u_Nz+1),26,20,16,_u_Ey);
  double _u_Ez[26][21][15];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),(_u_Ny+1),_u_Nz,26,21,15,_u_Ez);
  double _u_Hx[26][20][15];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),_u_Ny,_u_Nz,26,20,15,_u_Hx);
  double _u_Hy[25][21][15];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,(_u_Ny+1),_u_Nz,25,21,15,_u_Hy);
  double _u_Hz[25][20][16];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,_u_Ny,(_u_Nz+1),25,20,16,_u_Hz);
  double _u_Ets[200][3];
  scilab_rt_zeros_i0i0_d2(_u_Nt,3,200,3,_u_Ets);
  _u_Ex[0][1][1] = 1;
  _u_Ey[1][0][1] = 2;
  _u_Ez[1][1][0] = 3;
  double _u_Hxt[26][20][15];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),_u_Ny,_u_Nz,26,20,15,_u_Hxt);
  double _u_Hyt[25][21][15];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,(_u_Ny+1),_u_Nz,25,21,15,_u_Hyt);
  double _u_Hzt[25][20][16];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,_u_Ny,(_u_Nz+1),25,20,16,_u_Hzt);
  double _u_Ext[25][21][16];
  scilab_rt_zeros_i0i0i0_d3(_u_Nx,(_u_Ny+1),(_u_Nz+1),25,21,16,_u_Ext);
  double _u_Eyt[26][20][16];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),_u_Ny,(_u_Nz+1),26,20,16,_u_Eyt);
  double _u_Ezt[26][21][15];
  scilab_rt_zeros_i0i0i0_d3((_u_Nx+1),(_u_Ny+1),_u_Nz,26,21,15,_u_Ezt);
  for (int _u_n=1; _u_n<=_u_Nt; _u_n++) {
    /* Hx = Hx + (Dt/mu0) * ( (Ey(:, :, 2:Nz+1) - Ey(:, :, 1:Nz)) * Cz - (Ez(:, 2:Ny+1, :) - Ez(:, 1:Ny, :)) * Cy );  */
    for (int _u_i=1; _u_i<=(_u_Nx+1); _u_i++) {
      for (int _u_j=1; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=1; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx1 = (((_u_Ey[_u_i-1][_u_j-1][(_u_k+1)-1]-_u_Ey[_u_i-1][_u_j-1][_u_k-1])*_u_Cz)-((_u_Ez[_u_i-1][(_u_j+1)-1][_u_k-1]-_u_Ez[_u_i-1][_u_j-1][_u_k-1])*_u_Cy));
          _u_Hxt[_u_i-1][_u_j-1][_u_k-1] = _tmpxx1;
        }
      }
    }
    double _tmpxx2 = (_u_Dt / _u_mu0);
    double _tmpxx3[26][20][15];
    scilab_rt_mul_d0d3_d3(_tmpxx2,26,20,15,_u_Hxt,26,20,15,_tmpxx3);
    double _tmpxx4[26][20][15];
    scilab_rt_add_d3d3_d3(26,20,15,_u_Hx,26,20,15,_tmpxx3,26,20,15,_tmpxx4);
    
    scilab_rt_assign_d3_d3(26,20,15,_tmpxx4,26,20,15,_u_Hx);
    /* Hy = Hy + (Dt/mu0) * ( (Ez(2:Nx+1, :, :) - Ez(1:Nx, :, :)) * Cx - (Ex(:, :, 2:Nz+1) - Ex(:, :, 1:Nz)) * Cz ); */
    for (int _u_i=1; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=1; _u_j<=(_u_Ny+1); _u_j++) {
        for (int _u_k=1; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx5 = (((_u_Ez[(_u_i+1)-1][_u_j-1][_u_k-1]-_u_Ez[_u_i-1][_u_j-1][_u_k-1])*_u_Cx)-((_u_Ex[_u_i-1][_u_j-1][(_u_k+1)-1]-_u_Ex[_u_i-1][_u_j-1][_u_k-1])*_u_Cz));
          _u_Hyt[_u_i-1][_u_j-1][_u_k-1] = _tmpxx5;
        }
      }
    }
    double _tmpxx6 = (_u_Dt / _u_mu0);
    double _tmpxx7[25][21][15];
    scilab_rt_mul_d0d3_d3(_tmpxx6,25,21,15,_u_Hyt,25,21,15,_tmpxx7);
    double _tmpxx8[25][21][15];
    scilab_rt_add_d3d3_d3(25,21,15,_u_Hy,25,21,15,_tmpxx7,25,21,15,_tmpxx8);
    
    scilab_rt_assign_d3_d3(25,21,15,_tmpxx8,25,21,15,_u_Hy);
    /* Hz = Hz + (Dt/mu0) * ( (Ex(:, 2:Ny+1, :) - Ex(:, 1:Ny, :)) * Cy - (Ey(2:Nx+1, :, :) - Ey(1:Nx, :, :)) * Cx ); */
    for (int _u_i=1; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=1; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=1; _u_k<=(_u_Nz+1); _u_k++) {
          double _tmpxx9 = (((_u_Ex[_u_i-1][(_u_j+1)-1][_u_k-1]-_u_Ex[_u_i-1][_u_j-1][_u_k-1])*_u_Cy)-((_u_Ey[(_u_i+1)-1][_u_j-1][_u_k-1]-_u_Ey[_u_i-1][_u_j-1][_u_k-1])*_u_Cx));
          _u_Hzt[_u_i-1][_u_j-1][_u_k-1] = _tmpxx9;
        }
      }
    }
    double _tmpxx10 = (_u_Dt / _u_mu0);
    double _tmpxx11[25][20][16];
    scilab_rt_mul_d0d3_d3(_tmpxx10,25,20,16,_u_Hzt,25,20,16,_tmpxx11);
    double _tmpxx12[25][20][16];
    scilab_rt_add_d3d3_d3(25,20,16,_u_Hz,25,20,16,_tmpxx11,25,20,16,_tmpxx12);
    
    scilab_rt_assign_d3_d3(25,20,16,_tmpxx12,25,20,16,_u_Hz);
    /*   Ex(:, 2:Ny, 2:Nz) = Ex(:, 2:Ny, 2:Nz) + (Dt/eps0) *  */
    /* ((Hz(:, 2:Ny, 2:Nz)-Hz(:, 1:Ny-1, 2:Nz)) *  */
    /* Cy - (Hy(:, 2:Ny, 2:Nz) - Hy(:, 2:Ny, 1:Nz-1)) * Cz); */
    for (int _u_i=1; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=2; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=2; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx13 = (((_u_Hz[_u_i-1][_u_j-1][_u_k-1]-_u_Hz[_u_i-1][(_u_j-1)-1][_u_k-1])*_u_Cy)-((_u_Hy[_u_i-1][_u_j-1][_u_k-1]-_u_Hy[_u_i-1][_u_j-1][(_u_k-1)-1])*_u_Cz));
          _u_Ext[_u_i-1][_u_j-1][_u_k-1] = _tmpxx13;
        }
      }
    }
    for (int _u_i=1; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=2; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=2; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx14 = (_u_Ex[_u_i-1][_u_j-1][_u_k-1]+((_u_Dt / _u_eps0)*_u_Ext[_u_i-1][_u_j-1][_u_k-1]));
          _u_Ex[_u_i-1][_u_j-1][_u_k-1] = _tmpxx14;
        }
      }
    }
    /*    Ey(2:Nx, :, 2:Nz) = Ey(2:Nx, :, 2:Nz)+(Dt/eps0) * */
    /* ((Hx(2:Nx, :, 2:Nz)-Hx(2:Nx, :, 1:Nz-1)) *  */
    /* Cz -(Hz(2:Nx, :, 2:Nz)-Hz(1:Nx-1, :, 2:Nz))*Cx); */
    for (int _u_i=2; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=1; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=2; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx15 = (((_u_Hx[_u_i-1][_u_j-1][_u_k-1]-_u_Hx[_u_i-1][_u_j-1][(_u_k-1)-1])*_u_Cz)-((_u_Hz[_u_i-1][_u_j-1][_u_k-1]-_u_Hz[(_u_i-1)-1][_u_j-1][_u_k-1])*_u_Cx));
          _u_Eyt[_u_i-1][_u_j-1][_u_k-1] = _tmpxx15;
        }
      }
    }
    for (int _u_i=2; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=1; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=2; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx16 = (_u_Ey[_u_i-1][_u_j-1][_u_k-1]+((_u_Dt / _u_eps0)*_u_Eyt[_u_i-1][_u_j-1][_u_k-1]));
          _u_Ey[_u_i-1][_u_j-1][_u_k-1] = _tmpxx16;
        }
      }
    }
    /*      Ez(2:Nx, 2:Ny, :) = Ez(2:Nx, 2:Ny, :)+(Dt/eps0) * */
    /*  ((Hy(2:Nx, 2:Ny, :)-Hy(1:Nx-1, 2:Ny, :)) *  */
    /* Cx -(Hx(2:Nx, 2:Ny, :)-Hx(2:Nx, 1:Ny-1, :))*Cy); */
    for (int _u_i=2; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=2; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=1; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx17 = (((_u_Hy[_u_i-1][_u_j-1][_u_k-1]-_u_Hy[(_u_i-1)-1][_u_j-1][_u_k-1])*_u_Cx)-((_u_Hx[_u_i-1][_u_j-1][_u_k-1]-_u_Hx[_u_i-1][(_u_j-1)-1][_u_k-1])*_u_Cy));
          _u_Ezt[_u_i-1][_u_j-1][_u_k-1] = _tmpxx17;
        }
      }
    }
    for (int _u_i=2; _u_i<=_u_Nx; _u_i++) {
      for (int _u_j=2; _u_j<=_u_Ny; _u_j++) {
        for (int _u_k=1; _u_k<=_u_Nz; _u_k++) {
          double _tmpxx18 = (_u_Ez[_u_i-1][_u_j-1][_u_k-1]+((_u_Dt / _u_eps0)*_u_Ezt[_u_i-1][_u_j-1][_u_k-1]));
          _u_Ez[_u_i-1][_u_j-1][_u_k-1] = _tmpxx18;
        }
      }
    }
    /* Ets(n, :) = [Ex(4, 4, 4) Ey(4, 4, 4) Ez(4, 4, 4)];   */
    _u_Ets[_u_n-1][0] = _u_Ex[3][3][3];
    _u_Ets[_u_n-1][1] = _u_Ey[3][3][3];
    _u_Ets[_u_n-1][2] = _u_Ez[3][3][3];
  }
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean of matrix Ets: ");
  double _tmpxx19;
  scilab_rt_mean_d2_d0(200,3,_u_Ets,&_tmpxx19);
  scilab_rt_disp_d0_(_tmpxx19);

  scilab_rt_terminate();
}

