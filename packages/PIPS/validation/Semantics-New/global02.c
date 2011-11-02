int __lv0 = 0;
int __lv1 = 1;
int __lv2 = 2;
int __lv3 = 3;


int main(int argc, char* argv[])
{
  //scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  int _u_N = 100;
  double _u_A[100][100];
  //scilab_rt_zeros_i0i0_d2(_u_N,_u_N,100,100,_u_A);
  for (int _u_ii=1; _u_ii<=_u_N; _u_ii++) {
    for (int _u_jj=1; _u_jj<=_u_N; _u_jj++) {
      _u_A[_u_ii-1][_u_jj-1] = 1.0 + (double) (__lv0+__lv1+__lv2+__lv3);
    }
  }
  //scilab_rt_disp_d2_(100,100,_u_A);

  //scilab_rt_terminate();
  return 0;
}
