/* (c) HPC Project 2010 */

#define scilab_rt_eltmul_i0i0_i0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_eltmul_i0i0_(in0, in1) (in0 * in1)

#define scilab_rt_eltmul_i0d0_d0(in0, in1, out0) (*out0 = (double)in0 * in1)

#define scilab_rt_eltmul_i0d0_(in0, in1) ((double)in0 * in1)

#define scilab_rt_eltmul_i0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 * in1)

#define scilab_rt_eltmul_i0z0_(in0, in1) ((double complex)in0 * in1)

#define scilab_rt_eltmul_d0i0_d0(in0, in1, out0) (*out0 = in0 * (double)in1)

#define scilab_rt_eltmul_d0i0_(in0, in1) (in0 * (double)in1)

#define scilab_rt_eltmul_d0d0_d0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_eltmul_d0d0_(in0, in1) (in0 * in1)

#define scilab_rt_eltmul_d0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 * in1)

#define scilab_rt_eltmul_d0z0_(in0, in1) ((double complex)in0 * in1)

#define scilab_rt_eltmul_z0i0_z0(in0, in1, out0) (*out0 = in0 * (double complex)in1)

#define scilab_rt_eltmul_z0i0_(in0, in1) (in0 * (double complex)in1)

#define scilab_rt_eltmul_z0d0_z0(in0, in1, out0) (*out0 = in0 * (double complex)in1)

#define scilab_rt_eltmul_z0d0_(in0, in1) (in0 * (double complex)in1)

#define scilab_rt_eltmul_z0z0_z0(in0, in1, out0) (*out0 = in0 * in1)

#define scilab_rt_eltmul_z0z0_(in0, in1) (in0 * in1)

#define scilab_rt_eltmul_i2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_i2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_i2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_d2i0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double)in1; \
		} \
	} \
}

#define scilab_rt_eltmul_d2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_d2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_z2i0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1; \
		} \
	} \
}

#define scilab_rt_eltmul_z2d0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1; \
		} \
	} \
}

#define scilab_rt_eltmul_z2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1; \
		} \
	} \
}

#define scilab_rt_eltmul_i0i2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d0i2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z0i2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z0d2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i2i2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i2d2_d2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i2z2_z2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d2i2_d2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d2d2_d2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_d2z2_z2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z2i2_z2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z2d2_z2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * (double complex)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_z2z2_z2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] * in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_eltmul_i3i3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_i3d3_d3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double)in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_i3z3_z3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_d3i3_d3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_d3d3_d3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_d3z3_z3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_z3i3_z3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double complex)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_z3d3_z3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * (double complex)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_z3z3_z3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] * in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_eltmul_i3i0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_i3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_i3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d3i0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z3i0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z3d0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_i0i3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d0i3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z0i3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_i0d3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d0d3_d3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z0d3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_i0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_d0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

#define scilab_rt_eltmul_z0z3_z3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
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

