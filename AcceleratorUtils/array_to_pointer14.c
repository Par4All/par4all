extern  int P4A_vp_0,P4A_vp_1;
 void p4a_kernel_0(int m, int n, int i, int j, double
 a[n][m])
 {
    // Loop nest P4A end
    if (i<=-1+n&&j<=-1+m)
       a[i][j] = i+100+j;
 }

void p4a_kernel_wrapper_0(int m, int n, int i,
 int j, double a[n][m])
 {
    // Index has been replaced by P4A_vp_0:
    i = P4A_vp_0;
    // Index has been replaced by P4A_vp_1:
    j = P4A_vp_1;
    // Loop nest P4A end
    p4a_kernel_0(m, n, i, j, a);
 }
void P4A_call_accel_kernel_2d(void (*w)(int m, int n, int i, int j, double a[n][m]),
        int n, int m, int m0, int n0, int i, int j, double a[n][m]) {
}
 void p4a_kernel_launcher_0(int m, int n, double a[n][m])
 {
    //PIPS generated variable
    int i, j;
    P4A_call_accel_kernel_2d(p4a_kernel_wrapper_0, n,m, m, n, i, j, a);
 }

