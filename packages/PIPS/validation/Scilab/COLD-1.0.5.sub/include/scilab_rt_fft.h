/*---------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */

void scilab_rt_fft_i2_z2(int sin00, int sin01, int in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_fft_d2_z2(int sin00, int sin01, double in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_fft_z2_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_fft_i2d0_z2(int sin00, int sin01, int in0[sin00][sin01], double direction,
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_fft_d2d0_z2(int sin00, int sin01, double in0[sin00][sin01], double direction,
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_fft_z2d0_z2(int sin00, int sin01, double complex in0[sin00][sin01], double direction,
    int sout00, int sout01, double complex out0[sout00][sout01]);

