/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010-2011
*
*/

void scilab_rt_lsq_i2i2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_lsq_i2d2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_lsq_i2z2_z2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double complex in1[sin10][sin11],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_lsq_d2i2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_lsq_d2d2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_lsq_d2z2_z2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double complex in1[sin10][sin11],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_lsq_z2i2_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_lsq_z2d2_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_lsq_z2z2_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sin10, int sin11, double complex in1[sin10][sin11],
    int sout00, int sout01, double complex out0[sout00][sout01]);

