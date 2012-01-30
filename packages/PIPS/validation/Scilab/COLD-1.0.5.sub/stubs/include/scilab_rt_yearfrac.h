/*---------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */



extern double scilab_rt_yearfrac_d0d0_(double in0, double in1);

extern void scilab_rt_yearfrac_d0d0_d0(double in0, double in1, double* out0);

extern void scilab_rt_yearfrac_i2i2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_yearfrac_i2d2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_yearfrac_d2i2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_yearfrac_d2d2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

