/* (c) HPC Project 2010 */

extern void scilab_rt_mean_i2_d0(int sin00, int sin01, int in0[sin00][sin01],
    double *out0);

extern double scilab_rt_mean_i2_(int sin00, int sin01, int in0[sin00][sin01]);

extern void scilab_rt_mean_i3_d0(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02],
    double *out0);

extern double scilab_rt_mean_i3_(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02]);

extern void scilab_rt_mean_d2_d0(int sin00, int sin01, double in0[sin00][sin01],
    double *out0);

extern double scilab_rt_mean_d2_(int sin00, int sin01, double in0[sin00][sin01]);

extern void scilab_rt_mean_d3_d0(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02],
    double *out0);

extern double scilab_rt_mean_d3_(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02]);



extern void scilab_rt_mean_i2i0_d2(int sin00, int sin01, int in0[sin00][sin01],
    double in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_mean_i2s0_d2(int sin00, int sin01, int in0[sin00][sin01],
    char * in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_mean_d2i0_d2(int sin00, int sin01, double in0[sin00][sin01],
    double in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_mean_d2s0_d2(int sin00, int sin01, double in0[sin00][sin01],
    char * in1,
    int sout00, int sout01, double out0[sout00][sout01]);



extern void scilab_rt_mean_i2i0_d0(int sin00, int sin01, int in0[sin00][sin01],
    double in1,
    double* out0);

extern void scilab_rt_mean_i2s0_d0(int sin00, int sin01, int in0[sin00][sin01],
    char * in1,
    double* out0);

extern void scilab_rt_mean_d2i0_d0(int sin00, int sin01, double in0[sin00][sin01],
    double in1,
    double* out0);

extern void scilab_rt_mean_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01],
    char* in1,
    double* out0);



extern void scilab_rt_mean_i3i0_d3(int sin00,int sin01, int sin02, double in0[sin00][sin01][sin02],
    double in1,
    int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02]);

extern void scilab_rt_mean_i3s0_d3(int sin00,int sin01, int sin02, double in0[sin00][sin01][sin02],
    char* in1,
    int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02]);

extern void scilab_rt_mean_d3i0_d3(int sin00,int sin01, int sin02, double in0[sin00][sin01][sin02],
    double in1,
    int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02]);

extern void scilab_rt_mean_d3s0_d3(int sin00,int sin01, int sin02, double in0[sin00][sin01][sin02],
    char* in1,
    int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02]);

