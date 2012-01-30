/* (c) HPC Project 2010 */

#define scilab_rt_div_i0d0_d0(in0, in1, out0) (*out0 = (double)in0 / in1)

#define scilab_rt_div_i0d0_(in0, in1) ((double)in0 / in1)

#define scilab_rt_div_d0i0_d0(in0, in1, out0) (*out0 = in0 / (double)in1)

#define scilab_rt_div_d0i0_(in0, in1) (in0 / (double)in1)

#define scilab_rt_div_d0d0_d0(in0, in1, out0) (*out0 = in0 / in1)

#define scilab_rt_div_d0d0_(in0, in1) (in0 / in1)

#define scilab_rt_div_i0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 / in1)

#define scilab_rt_div_i0z0_(in0, in1) ((double complex)in0 / in1)

#define scilab_rt_div_z0i0_z0(in0, in1, out0) (*out0 = in0 / (double complex)in1)

#define scilab_rt_div_z0i0_(in0, in1) (in0 / (double complex)in1)

#define scilab_rt_div_d0z0_z0(in0, in1, out0) (*out0 = (double complex)in0 / in1)

#define scilab_rt_div_d0z0_(in0, in1) ((double complex)in0 / in1)

#define scilab_rt_div_z0d0_z0(in0, in1, out0) (*out0 = in0 / (double complex)in1)

#define scilab_rt_div_z0d0_(in0, in1) (in0 / (double complex)in1)

#define scilab_rt_div_z0z0_z0(in0, in1, out0) (*out0 = in0 / in1)

#define scilab_rt_div_z0z0_(in0, in1) (in0 / in1)

#define scilab_rt_div_i2i0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0[__lv1][__lv2] / (double)in1; \
		} \
	} \
}

#define scilab_rt_div_i2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double)in0[__lv1][__lv2] / in1; \
		} \
	} \
}

#define scilab_rt_div_i2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] / in1; \
		} \
	} \
}

#define scilab_rt_div_d2i0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] / (double)in1; \
		} \
	} \
}

#define scilab_rt_div_d2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] / in1; \
		} \
	} \
}

#define scilab_rt_div_d2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (double complex)in0[__lv1][__lv2] / in1; \
		} \
	} \
}

#define scilab_rt_div_z2i0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] / (double complex)in1; \
		} \
	} \
}

#define scilab_rt_div_z2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] / in1; \
		} \
	} \
}

#define scilab_rt_div_i3i0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double)in0[__lv1][__lv2][__lv3] / (double)in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_i3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double)in0[__lv1][__lv2][__lv3] / in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_i3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] / in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_d3i0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] / (double)in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_d3d0_d3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] / in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_d3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (double complex)in0[__lv1][__lv2][__lv3] / in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_z3i0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] / (double complex)in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_z3d0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] / (double complex)in1; \
			} \
		} \
	} \
}

#define scilab_rt_div_z3z0_z3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] / in1; \
			} \
		} \
	} \
}

