/* (c) HPC Project 2010 */

#define scilab_rt_mul_i0i0_i0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_mul_i0i0_(in0, in1) (in0 * in1)

#define scilab_rt_mul_i0d0_d0(in0, in1, out0) (*out0 = (double)in0 * in1)

#define scilab_rt_mul_i0d0_(in0, in1) ((double)in0 * in1)

#define scilab_rt_mul_i0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 * in1)

#define scilab_rt_mul_i0z0_(in0, in1) ((double complex)in0 * in1)

#define scilab_rt_mul_d0i0_d0(in0, in1, out0) (*out0 = in0 * (double)in1)

#define scilab_rt_mul_d0i0_(in0, in1) (in0 * (double)in1)

#define scilab_rt_mul_d0d0_d0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_mul_d0d0_(in0, in1) (in0 * in1)

#define scilab_rt_mul_d0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 * in1)

#define scilab_rt_mul_d0z0_(in0, in1) ((double complex)in0 * in1)

#define scilab_rt_mul_z0i0_z0(in0, in1, out0) (*out0 = in0 * (double complex)in1)

#define scilab_rt_mul_z0i0_(in0, in1) (in0 * (double complex)in1)

#define scilab_rt_mul_z0d0_z0(in0, in1, out0) (*out0 = in0 * (double complex)in1)

#define scilab_rt_mul_z0d0_(in0, in1) (in0 * (double complex)in1)

#define scilab_rt_mul_z0z0_z0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_mul_z0z0_(in0, in1) (in0 * in1)

#define scilab_rt_mul_i2i2_i0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (int)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern int scilab_rt_mul_i2i2_(int si00, int si01, int in0[si00][si01],
	int si10, int si11, int in1[si10][si11]);

#define scilab_rt_mul_i2d2_d0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += (double)in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double scilab_rt_mul_i2d2_(int si00, int si01, int in0[si00][si01],
	int si10, int si11, double in1[si10][si11]);

#define scilab_rt_mul_i2z2_z0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double complex)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += (double complex)in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double complex scilab_rt_mul_i2z2_(int si00, int si01, int in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11]);

#define scilab_rt_mul_d2i2_d0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * (double)in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double scilab_rt_mul_d2i2_(int si00, int si01, double in0[si00][si01],
	int si10, int si11, int in1[si10][si11]);

#define scilab_rt_mul_d2d2_d0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double scilab_rt_mul_d2d2_(int si00, int si01, double in0[si00][si01],
	int si10, int si11, double in1[si10][si11]);

#define scilab_rt_mul_d2z2_z0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double complex)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += (double complex)in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double complex scilab_rt_mul_d2z2_(int si00, int si01, double in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11]);

#define scilab_rt_mul_z2i2_z0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double complex)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * (double complex)in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double complex scilab_rt_mul_z2i2_(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, int in1[si10][si11]);

#define scilab_rt_mul_z2d2_z0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double complex)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * (double complex)in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double complex scilab_rt_mul_z2d2_(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, double in1[si10][si11]);

#define scilab_rt_mul_z2z2_z0(si00, si01, in0, si10, si11, in1, out0) \
{ \
	assert (si01 == si10); \
	*out0 = (double complex)0.0; \
	for (__lv1=0; __lv1<si00; __lv1++) { \
		for (__lv2=0; __lv2<si10; __lv2++) { \
			for (__lv3=0; __lv3<si11; __lv3++) { \
				*out0 += in0[__lv1][__lv2] * in1[__lv2][__lv3]; \
			} \
		} \
	} \
}

extern double complex scilab_rt_mul_z2z2_(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11]);

extern void scilab_rt_mul_i2i2_i2(int si00, int si01, int in0[si00][si01],
	int si10, int si11, int in1[si10][si11],
	int so00, int so01, int out0[so00][so01]);

extern void scilab_rt_mul_i2d2_d2(int si00, int si01, int in0[si00][si01],
	int si10, int si11, double in1[si10][si11],
	int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_mul_i2z2_z2(int si00, int si01, int in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11],
	int so00, int so01, double complex out0[so00][so01]);

extern void scilab_rt_mul_d2i2_d2(int si00, int si01, double in0[si00][si01],
	int si10, int si11, int in1[si10][si11],
	int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_mul_d2d2_d2(int si00, int si01, double in0[si00][si01],
	int si10, int si11, double in1[si10][si11],
	int so00, int so01, double out0[so00][so01]);

extern void scilab_rt_mul_d2z2_z2(int si00, int si01, double in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11],
	int so00, int so01, double complex out0[so00][so01]);

extern void scilab_rt_mul_z2i2_z2(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, int in1[si10][si11],
	int so00, int so01, double complex out0[so00][so01]);

extern void scilab_rt_mul_z2d2_z2(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, double in1[si10][si11],
	int so00, int so01, double complex out0[so00][so01]);

extern void scilab_rt_mul_z2z2_z2(int si00, int si01, double complex in0[si00][si01],
	int si10, int si11, double complex in1[si10][si11],
	int so00, int so01, double complex out0[so00][so01]);

#define scilab_rt_mul_i2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_i2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_i2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_d2i0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double)in1; \
		} \
	} \
}

#define scilab_rt_mul_d2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_d2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_z2i0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1; \
		} \
	} \
}

#define scilab_rt_mul_z2d0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1; \
		} \
	} \
}

#define scilab_rt_mul_z2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_mul_i0i2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_d0i2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_z0i2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_i0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_d0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_z0d2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_i0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_d0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_z0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_mul_i0i3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d0i3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * (double)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z0i3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * (double complex)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_i0d3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double)in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d0d3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z0d3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * (double complex)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_i0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_mul_i3i0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_i3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double)in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_i3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d3i0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double)in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_d3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z3i0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double complex)in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z3d0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double complex)in1; \
			} \
		} \
	} \
}

#define scilab_rt_mul_z3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1; \
			} \
		} \
	} \
}

