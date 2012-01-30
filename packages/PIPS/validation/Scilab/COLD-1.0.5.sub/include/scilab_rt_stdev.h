extern double scilab_rt_stdev_i0_(int in0);
extern double scilab_rt_stdev_d0_(double in0);
extern double scilab_rt_stdev_i0i0_(int in0, int in1);
extern double scilab_rt_stdev_d0i0_(double in0, int in1);
extern double scilab_rt_stdev_i0s0_(int in0, char *in1);
extern double scilab_rt_stdev_d0s0_(double in0, char *in1);
extern void scilab_rt_stdev_i2_d0(int sin00, int sin01, int in0[sin00][sin01], double *out0);
extern void scilab_rt_stdev_d2_d0(int sin00, int sin01, double in0[sin00][sin01], double *out0);
extern void scilab_rt_stdev_i3_d0(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02], double *out0);
extern void scilab_rt_stdev_d3_d0(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02], double *out0);
extern void scilab_rt_stdev_i2s0_d2(int sin00, int sin01, int in0[sin00][sin01], char* in1, int sout00, int sout01, double out0[sout00][sout01]);
extern void scilab_rt_stdev_d2s0_d2(int sin00, int sin01, double in0[sin00][sin01], char* in1, int sout00, int sout01, double out0[sout00][sout01]);
extern void scilab_rt_stdev_i2i0_d2(int sin00, int sin01, int in0[sin00][sin01], int in1, int sout00, int sout01, double out0[sout00][sout01]);
extern void scilab_rt_stdev_d2i0_d2(int sin00, int sin01, double in0[sin00][sin01], int in1, int sout00, int sout01, double out0[sout00][sout01]);
extern void scilab_rt_stdev_i2s0_d0(int sin00, int sin01, int in0[sin00][sin01], char* in1, double *out0);
extern void scilab_rt_stdev_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01], char* in1, double *out0);
extern void scilab_rt_stdev_i2i0_d0(int sin00, int sin01, int in0[sin00][sin01], int in1, double *out0);
extern void scilab_rt_stdev_d2i0_d0(int sin00, int sin01, double in0[sin00][sin01], int in1, double *out0);

