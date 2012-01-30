/* (c) HPC Project 2010 */

#define scilab_rt_size_i0_i0i0(in0, out0, out1) \
{ \
	*out0 = (int) 1.0; \
	*out1 = (int) 1.0; \
}

#define scilab_rt_size_d0_i0i0(in0, out0, out1) \
{ \
	*out0 = (int) 1.0; \
	*out1 = (int) 1.0; \
}

#define scilab_rt_size_z0_i0i0(in0, out0, out1) \
{ \
	*out0 = (int) 1.0; \
	*out1 = (int) 1.0; \
}

#define scilab_rt_size_i2_i0i0(si00, si01, in0, out0, out1) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
}

#define scilab_rt_size_d2_i0i0(si00, si01, in0, out0, out1) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
}

#define scilab_rt_size_z2_i0i0(si00, si01, in0, out0, out1) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
}

#define scilab_rt_size_i3_i0i0i0(si00, si01, si02, in0, out0, out1, out2) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
	*out2 = (int)si02; \
}

#define scilab_rt_size_d3_i0i0i0(si00, si01, si02, in0, out0, out1, out2) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
	*out2 = (int)si02; \
}

#define scilab_rt_size_z3_i0i0i0(si00, si01, si02, in0, out0, out1, out2) \
{ \
	*out0 = (int)si00; \
	*out1 = (int)si01; \
	*out2 = (int)si02; \
}

#define scilab_rt_size_i0_i2(in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int) 1.0; \
}

#define scilab_rt_size_d0_i2(in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int) 1.0; \
}

#define scilab_rt_size_z0_i2(in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int) 1.0; \
}

#define scilab_rt_size_i1_i2(si00, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int)si00; \
}

#define scilab_rt_size_d1_i2(si00, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int)si00; \
}

#define scilab_rt_size_c1_i2(si00, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int) 1.0; \
	out0[0][1] = (int)si00; \
}

#define scilab_rt_size_i2_i2(si00, si01, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
}

#define scilab_rt_size_d2_i2(si00, si01, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
}

#define scilab_rt_size_z2_i2(si00, si01, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
}

#define scilab_rt_size_i3_i2(si00, si01, si02, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
}

#define scilab_rt_size_d3_i2(si00, si01, si02, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
}

#define scilab_rt_size_z3_i2(si00, si01, si02, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
}

#define scilab_rt_size_i4_i2(si00, si01, si02, si03, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
	out0[0][3] = (int)si03; \
}

#define scilab_rt_size_d4_i2(si00, si01, si02, si03, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
	out0[0][3] = (int)si03; \
}

#define scilab_rt_size_z4_i2(si00, si01, si02, si03, in0, so00, so01, out0) \
{ \
	out0[0][0] = (int)si00; \
	out0[0][1] = (int)si01; \
	out0[0][2] = (int)si02; \
	out0[0][3] = (int)si03; \
}

#define scilab_rt_size_i2i0_i0(si00, si01, in0, in1, out0) \
{ \
	if (in1 == 1) { \
	*out0 = (int)si00; \
	} else if (in1 == 2) { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_i2i0_(int si00, int si01, int in0[si00][si01],
	int in1);

#define scilab_rt_size_d2i0_i0(si00, si01, in0, in1, out0) \
{ \
	if (in1 == 1) { \
	*out0 = (int)si00; \
	} else if (in1 == 2) { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_d2i0_(int si00, int si01, double in0[si00][si01],
	int in1);

#define scilab_rt_size_z2i0_i0(si00, si01, in0, in1, out0) \
{ \
	if (in1 == 1) { \
	*out0 = (int)si00; \
	} else if (in1 == 2) { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_z2i0_(int si00, int si01, double complex in0[si00][si01],
	int in1);

#define scilab_rt_size_i2s0_i0(si00, si01, in0, in1, out0) \
{ \
	if (*in1 == 'r') { \
	*out0 = (int)si00; \
	} else if (*in1 == 'c') { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_i2s0_(int si00, int si01, int in0[si00][si01],
	char * in1);

#define scilab_rt_size_d2s0_i0(si00, si01, in0, in1, out0) \
{ \
	if (*in1 == 'r') { \
	*out0 = (int)si00; \
	} else if (*in1 == 'c') { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_d2s0_(int si00, int si01, double in0[si00][si01],
	char * in1);

#define scilab_rt_size_z2s0_i0(si00, si01, in0, in1, out0) \
{ \
	if (*in1 == 'r') { \
	*out0 = (int)si00; \
	} else if (*in1 == 'c') { \
	*out0 = (int)si01; \
	} else exit(1); \
}

extern int scilab_rt_size_z2s0_(int si00, int si01, double complex in0[si00][si01],
	char * in1);

