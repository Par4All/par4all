/* (c) HPC Project 2010 */

#ifdef OPENMP

extern void scilab_rt_max_i2_i0(int sin00, int sin01, int in0[sin00][sin01],
    int *out0);

extern int scilab_rt_max_i2_(int sin00, int sin01, int in0[sin00][sin01]);

extern void scilab_rt_max_i3_i0(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02],
    int *out0);

extern int scilab_rt_max_i3_(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02]);

extern void scilab_rt_max_d2_d0(int sin00, int sin01, double in0[sin00][sin01],
    double *out0);

extern double scilab_rt_max_d2_(int sin00, int sin01, double in0[sin00][sin01]);

extern void scilab_rt_max_d3_d0(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02],
    double *out0);

extern double scilab_rt_max_d3_(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02]);


extern void scilab_rt_max_i2d0_d2(int sin00, int sin01, int in0[sin00][sin01],
    double in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_max_d2d0_d2(int sin00, int sin01, double in0[sin00][sin01],
    double in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_max_d0i2_d2(double in0,
    int sin10, int sin11, int in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_max_d0d2_d2(double in0,
    int sin10, int sin11, double in1[sin10][sin11],
    int sout00, int sout01, double out0[sout00][sout01]);


extern void scilab_rt_max_i2s0_i2(int sin00, int sin01, int in0[sin00][sin01],
    char * in1,
    int sout00, int sout01, int out0[sout00][sout01]);

extern void scilab_rt_max_d2s0_d2(int sin00, int sin01, double in0[sin00][sin01],
    char * in1,
    int sout00, int sout01, double out0[sout00][sout01]);

extern void scilab_rt_max_i2s0_i0(int sin00, int sin01, int in0[sin00][sin01],
    char * in1,
    int* out0);

extern void scilab_rt_max_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01],
    char* in1,
    double* out0);

#else
#define scilab_rt_max_i2_i0(si00, si01, in0, out0) \
{ \
  int lv0, lv1; \
  *out0 = INT_MIN; \
  for (lv0=0; lv0<si00; lv0++) { \
    for (lv1=0; lv1<si01; lv1++) { \
      *out0 = MAX(*out0, in0[lv0][lv1]); \
    } \
  } \
}

extern int scilab_rt_max_i2_(int si00, int si01, int in0[si00][si01]);

#define scilab_rt_max_i3_i0(si00, si01, si02, in0, out0) \
{ \
  int lv0, lv1, lv2; \
  *out0 = INT_MIN; \
  for (lv0=0; lv0<si00; lv0++) { \
    for (lv1=0; lv1<si01; lv1++) { \
      for (lv2=0; lv2<si02; lv2++) { \
        *out0 = MAX(*out0, in0[lv0][lv1][lv2]); \
      } \
    } \
  } \
}

extern int scilab_rt_max_i3_(int si00, int si01, int si02, int in0[si00][si01][si02]);

#define scilab_rt_max_d2_d0(si00, si01, in0, out0) \
{ \
  int lv0, lv1; \
  *out0 = -HUGE_VAL; \
  for (lv0=0; lv0<si00; lv0++) { \
    for (lv1=0; lv1<si01; lv1++) { \
      *out0 = MAX(*out0, in0[lv0][lv1]); \
    } \
  } \
}

extern double scilab_rt_max_d2_(int si00, int si01, double in0[si00][si01]);

#define scilab_rt_max_d3_d0(si00, si01, si02, in0, out0) \
{ \
  int lv0, lv1, lv2; \
  *out0 = -HUGE_VAL; \
  for (lv0=0; lv0<si00; lv0++) { \
    for (lv1=0; lv1<si01; lv1++) { \
      for (lv2=0; lv2<si02; lv2++) { \
        *out0 = MAX(*out0, in0[lv0][lv1][lv2]); \
      } \
    } \
  } \
}

extern double scilab_rt_max_d3_(int si00, int si01, int si02, double in0[si00][si01][si02]);

#define scilab_rt_max_i2s0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
  int lv1, lv2; \
  if (*in1 == 'r') { \
    assert (si01 == so01); \
    for (lv2=0; lv2<so01; lv2++) { \
      out0[0][lv2] = INT_MIN; \
      for (lv1=0; lv1<si00; lv1++) { \
        out0[0][lv2] = MAX(out0[0][lv2], in0[lv1][lv2]); \
      } \
    } \
    \
  } else if (*in1 == 'c') { \
    assert (si00 == so00); \
    for (lv1=0; lv1<so00; lv1++) { \
      out0[lv1][0] = INT_MIN; \
      for (lv2=0; lv2<si01; lv2++) { \
        out0[lv1][0] = MAX(out0[lv1][0], in0[lv1][lv2]); \
      } \
    } \
    \
  } else exit(1); \
}

#define scilab_rt_max_d2s0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
  int lv1, lv2; \
  if (*in1 == 'r') { \
    assert (si01 == so01); \
    for (lv2=0; lv2<so01; lv2++) { \
      out0[0][lv2] = -HUGE_VAL; \
      for (lv1=0; lv1<si00; lv1++) { \
        out0[0][lv2] = MAX(out0[0][lv2], in0[lv1][lv2]); \
      } \
    } \
    \
  } else if (*in1 == 'c') { \
    assert (si00 == so00); \
    for (lv1=0; lv1<so00; lv1++) { \
      out0[lv1][0] = -HUGE_VAL; \
      for (lv2=0; lv2<si01; lv2++) { \
        out0[lv1][0] = MAX(out0[lv1][0], in0[lv1][lv2]); \
      } \
    } \
    \
  } else exit(1); \
}

extern void scilab_rt_max_i2d0_d2(int si00, int si01, int in0[si00][si01],
    double in1,
    int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_max_d0i2_d2(double in0,
    int si10, int si11, int in1[si10][si11],
    int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_max_d2d0_d2(int si00, int si01, double in0[si00][si01],
    double in1,
    int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_max_d0d2_d2(double in0,
    int si10, int si11, double in1[si10][si11],
    int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_max_i2s0_i0(int sin00, int sin01, int in0[sin00][sin01],
    char * in1,
    int* out0);

extern void scilab_rt_max_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01],
    char* in1,
    double* out0);

#endif
