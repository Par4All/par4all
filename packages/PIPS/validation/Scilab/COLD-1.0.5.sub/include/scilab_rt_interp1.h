
#include <stdio.h>
#include <stdlib.h>

#include "scilab_rt_interp_intern.h"

void scilab_rt_interp1_i2i2i2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_d2i2i2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_i2d2i2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_d2d2i2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sin20, int sin21, int in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_i2i2d2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, double in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_d2i2d2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    int sin20, int sin21, double in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_i2d2d2_d2(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sin20, int sin21, double in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_d2d2d2_d2(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    int sin20, int sin21, double in2[sin20][sin21],
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_interp1_i2i2d0_d0(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    double in2,
    double* out0);
void scilab_rt_interp1_d2i2d0_d0(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    double in2,
    double* out0);
void scilab_rt_interp1_i2d2d0_d0(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    double in2,
    double* out0);
void scilab_rt_interp1_d2d2d0_d0(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    double in2,
    double* out0);

