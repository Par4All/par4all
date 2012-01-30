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

void palloc_i0d0d0i0i0i0_d0i0(int _u_tcnt, double _u_tmin, double _u_th, int _u_fcnt, int _u_fmin, int _u_fh, double* _u_tmax, int* _u_fmax)
{
  /*  compute tmax */
  double _tmpxx0 = (_u_tcnt*_u_th);
  *_u_tmax = (_u_tmin+_tmpxx0);
  /*  compute fmax */
  int _tmpxx1 = (_u_fcnt*_u_fh);
  *_u_fmax = (_u_fmin+_tmpxx1);
}


void setlhs_i0d0d0d2i0d0_d2(int _u_R, double _u_L, double _u_C, int _c_0_n0,int _c_0_n1,double _c_0[_c_0_n0][_c_0_n1], int _u_k, double _u_h, int _u_lhs_n0,int _u_lhs_n1,double _u_lhs[_u_lhs_n0][_u_lhs_n1])
{
  
  scilab_rt_assign_d2_d2(_u_lhs_n0,_u_lhs_n1,_c_0,_u_lhs_n0,_u_lhs_n1,_u_lhs);
  /*  set reference index */
  int _u_idx = (5*_u_k);
  /*  set resistor equation at k+1 */
  _u_lhs[(_u_idx+2)-1][(_u_idx+1)-1] = 1.0;
  double _tmpxx2 = (-1.0);
  _u_lhs[(_u_idx+2)-1][(_u_idx+5)-1] = _tmpxx2;
  int _tmpxx3 = (-_u_R);
  _u_lhs[(_u_idx+2)-1][(_u_idx+2)-1] = _tmpxx3;
  /*  set inductor equation at k+2 */
  double _tmpxx4 = (2.0*_u_L);
  double _u_rl = (_u_h / _tmpxx4);
  _u_lhs[(_u_idx+3)-1][(_u_idx+3)-1] = 1.0;
  double _tmpxx5 = (-_u_rl);
  _u_lhs[(_u_idx+3)-1][(_u_idx+5)-1] = _tmpxx5;
  _u_lhs[(_u_idx+3)-1][(_u_idx+6)-1] = _u_rl;
  /*  set capacitor equation at k+3 */
  double _tmpxx6 = (2.0*_u_C);
  double _u_rc = (_u_h / _tmpxx6);
  _u_lhs[(_u_idx+4)-1][(_u_idx+6)-1] = 1.0;
  double _tmpxx7 = (-_u_rc);
  _u_lhs[(_u_idx+4)-1][(_u_idx+4)-1] = _tmpxx7;
  /*  set input kcl */
  _u_lhs[(_u_idx+1)-1][(_u_idx+2)-1] = 1.0;
  /*  set kcl between R and L */
  double _tmpxx8 = (-1.0);
  _u_lhs[(_u_idx+5)-1][(_u_idx+2)-1] = _tmpxx8;
  _u_lhs[(_u_idx+5)-1][(_u_idx+3)-1] = 1.0;
  /*  set kcl between l and c */
  double _tmpxx9 = (-1.0);
  _u_lhs[(_u_idx+6)-1][(_u_idx+3)-1] = _tmpxx9;
  _u_lhs[(_u_idx+6)-1][(_u_idx+4)-1] = 1.0;
}


void newlhs_i0i0d0d0d0i0_d2(int _u_N, int _u_R, double _u_L, double _u_C, double _u_h, int _u_msiz, int _u_lhs_n0,int _u_lhs_n1,double _u_lhs[_u_lhs_n0][_u_lhs_n1])
{
  /*  get system size */
  /*   msiz = 5 * N + 2; */
  /*  set initial matrix to zero */
  
  scilab_rt_zeros_i0i0_d2(_u_msiz,_u_msiz,_u_lhs_n0,_u_lhs_n1,_u_lhs);
  /*  set the lhs */
  for (int _u_k=0; _u_k<=(_u_N-1); _u_k++) {
    double _u_lhs2[_u_msiz][_u_msiz];
    scilab_rt_assign_d2_d2(_u_lhs_n0,_u_lhs_n1,_u_lhs,_u_msiz,_u_msiz,_u_lhs2);
    double _tmpxx10[_u_msiz][_u_msiz];
    setlhs_i0d0d0d2i0d0_d2(_u_R,_u_L,_u_C,_u_lhs_n0,_u_lhs_n1,_u_lhs,_u_k,_u_h,_u_msiz,_u_msiz,_tmpxx10);
    
    scilab_rt_assign_d2_d2(_u_msiz,_u_msiz,_tmpxx10,_u_lhs_n0,_u_lhs_n1,_u_lhs);
  }
  /*  set kvl at source */
  _u_lhs[_u_msiz-1][0] = 1.0;
  /*  set kcl at source */
  _u_lhs[0][_u_msiz-1] = 1.0;
}


void newrhs_i0i0d0d0i0d0d0d2i0_d2(int _u_N, int _u_R, double _u_L, double _u_C, int _u_F, double _u_h, double _u_t, int _u_x_n0,int _u_x_n1,double _u_x[_u_x_n0][_u_x_n1], int _u_msiz, int _u_rhs_n0,int _u_rhs_n1,double _u_rhs[_u_rhs_n0][_u_rhs_n1])
{
  double _u_TPI = (2*3.1415926534);
  /*  get system size */
  /*   vsiz = 5 * N + 2; */
  /*  initialize vector */
  
  scilab_rt_zeros_i0i0_d2(_u_msiz,1,_u_rhs_n0,_u_rhs_n1,_u_rhs);
  /*  set the sinusoidal source at t */
  double _tmpxx11 = (5.0*scilab_rt_sin_d0_(((_u_TPI*_u_F)*_u_t)));
  _u_rhs[(_u_msiz-1)][0] = _tmpxx11;
  /*  set the kvl for all devices */
  for (int _u_k=0; _u_k<=(_u_N-1); _u_k++) {
    /*  set reference index */
    int _u_idx = (5*_u_k);
    /*  set inductor kvl */
    double _tmpxx12 = (2.0*_u_L);
    double _u_rl = (_u_h / _tmpxx12);
    double _tmpxx13 = (_u_x[((_u_idx+3)-1)][0]+(_u_rl*(_u_x[((_u_idx+5)-1)][0]-_u_x[((_u_idx+6)-1)][0])));
    _u_rhs[((_u_idx+3)-1)][0] = _tmpxx13;
    /*  set capacitor kvl */
    double _tmpxx14 = (2.0*_u_C);
    double _u_rc = (_u_h / _tmpxx14);
    double _tmpxx15 = (_u_x[((_u_idx+6)-1)][0]+(_u_rc*_u_x[((_u_idx+4)-1)][0]));
    _u_rhs[((_u_idx+4)-1)][0] = _tmpxx15;
  }
}


void rlcloop_i0i0d0d0i0i0d0d0_d2d2d2(int _u_N, int _u_R, double _u_L, double _u_C, int _u_F, int _u_tcnt, double _u_tmin, double _u_th, int _u_rt_n0,int _u_rt_n1,double _u_rt[_u_rt_n0][_u_rt_n1], int _u_rs_n0,int _u_rs_n1,double _u_rs[_u_rs_n0][_u_rs_n1], int _u_rv_n0,int _u_rv_n1,double _u_rv[_u_rv_n0][_u_rv_n1])
{
  /*  get system size */
  int _tmpxx16 = (5*_u_N);
  int _u_msiz = (_tmpxx16+2);
  /*  initial vector */
  double _u_x[_u_msiz][1];
  scilab_rt_zeros_i0i0_d2(_u_msiz,1,_u_msiz,1,_u_x);
  /*  get the lhs */
  double _u_lhs[7][7];
  newlhs_i0i0d0d0d0i0_d2(_u_N,_u_R,_u_L,_u_C,_u_th,_u_msiz,7,7,_u_lhs);
  /*  get the result matrix */
  
  scilab_rt_zeros_i0i0_d2(_u_tcnt,1,_u_rt_n0,_u_rt_n1,_u_rt);
  
  scilab_rt_zeros_i0i0_d2(_u_tcnt,1,_u_rs_n0,_u_rs_n1,_u_rs);
  
  scilab_rt_zeros_i0i0_d2(_u_tcnt,1,_u_rv_n0,_u_rv_n1,_u_rv);
  double _u_t = _u_th;
  /*  solveur loop */
  for (int _u_k=1; _u_k<=_u_tcnt; _u_k++) {
    /*  solve x vector */
    double _u_rhs[_u_msiz][1];
    newrhs_i0i0d0d0i0d0d0d2i0_d2(_u_N,_u_R,_u_L,_u_C,_u_F,_u_th,_u_t,_u_msiz,1,_u_x,_u_msiz,_u_msiz,1,_u_rhs);
    
    scilab_rt_leftdivide_d2d2_d2(7,7,_u_lhs,_u_msiz,1,_u_rhs,_u_msiz,1,_u_x);
    /*  get output voltage and source current */
    double _u_s = _u_x[(1-1)][0];
    double _u_v = _u_x[((_u_msiz-1)-1)][0];
    double _tmpxx17 = _u_x[(_u_msiz-1)][0];
    double _u_i = (-_tmpxx17);
    /*  set result matrix */
    _u_rt[(_u_k-1)][0] = _u_t;
    _u_rs[(_u_k-1)][0] = _u_i;
    _u_rv[(_u_k-1)][0] = _u_v;
    /* printf ("%f\t%f\t%f\t%f\n",t, s, v, i); */
    /*  next step */
    _u_t = (_u_t+_u_th);
  }
}


void rlcdemo_i0i0d0d0i0d0d0i0i0i0_(int _u_N, int _u_R, double _u_L, double _u_C, int _u_tcnt, double _u_tmin, double _u_th, int _u_fcnt, int _u_fmin, int _u_fh)
{
  /*  allocate operating vectors and  */
  double _u_tmax;
  int _u_fmax;
  palloc_i0d0d0i0i0i0_d0i0(_u_tcnt,_u_tmin,_u_th,_u_fcnt,_u_fmin,_u_fh,&_u_tmax,&_u_fmax);
  /*  set initial frequency */
  int _u_f = _u_fmin;
  /*  solve a transmission line in a frequency range */
  for (int _u_k=1; _u_k<=_u_fcnt; _u_k++) {
    /*  solve the system */
    double _u_rt[10000][1];
    double _u_rs[10000][1];
    double _u_rv[10000][1];
    rlcloop_i0i0d0d0i0i0d0d0_d2d2d2(_u_N,_u_R,_u_L,_u_C,_u_f,_u_tcnt,_u_tmin,_u_th,10000,1,_u_rt,10000,1,_u_rs,10000,1,_u_rv);
    /*  extract vector to plot */
    /*         clf (); */
    /*         plot2d (rt, rs); */
    /*  update frequency */
    _u_f = (_u_f+_u_fh);
    scilab_rt_disp_d2_(10000,1,_u_rs);
  }
}





/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t198.sce: rlc demo */
  /*  set the lhs kvl/kcl by index */
  /*  create a RLC lhs matrix by size */
  /*  create a RLC rhs matrix by size */
  /*  main loop solver execution */
  /*  compute tmax and frequency max */
  /*  run the main frequency loop */
  /*  set demo parameters */
  int _u_N = 1;
  /*  number of elements */
  int _u_R = 10;
  /*  element resistance */
  double _u_L = 1E-3;
  /*  element inductance */
  double _u_C = 1E-5;
  /*  element capacitance */
  int _u_TCNT = 10000;
  /*  time count */
  double _u_TMIN = 0.000;
  /*  min time */
  double _u_TH = 1E-8;
  /*  integration time step */
  int _u_FCNT = 3;
  /*  frequency count */
  int _u_FMIN = 1000;
  /*  frequency min */
  int _u_FH = 1000;
  /*  frequency step */
  /*  run the demo */
  rlcdemo_i0i0d0d0i0d0d0i0i0i0_(_u_N,_u_R,_u_L,_u_C,_u_TCNT,_u_TMIN,_u_TH,_u_FCNT,_u_FMIN,_u_FH);

  scilab_rt_terminate();
}

