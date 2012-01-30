
void scilab_rt_inttrap_i2_d0(int sin00, int sin01, int in0[sin00][sin01],
    double *out0);

void scilab_rt_inttrap_d2_d0(int sin00, int sin01, double in0[sin00][sin01],
    double *out0);

void scilab_rt_inttrap_i2i2_d0(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    double* out0);

void scilab_rt_inttrap_i2d2_d0(int sin00, int sin01, int in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    double* out0);

void scilab_rt_inttrap_d2i2_d0(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, int in1[sin10][sin11],
    double* out0);

void scilab_rt_inttrap_d2d2_d0(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11],
    double* out0);

