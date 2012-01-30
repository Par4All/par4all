/* (c) HPC Project 2010 */

#define scilab_rt_pow_i0i0_i0(in0, in1, out0) (*out0 = pow(in0, in1))

#define scilab_rt_pow_i0i0_(in0, in1) (pow(in0, in1))

#define scilab_rt_pow_i0d0_d0(in0, in1, out0) (*out0 = pow(in0, in1))

#define scilab_rt_pow_i0d0_(in0, in1) (pow(in0, in1))

#define scilab_rt_pow_i0z0_d0(in0, in1, out0) (*out0 = cpow(in0, in1))

#define scilab_rt_pow_i0z0_(in0, in1) (cpow(in0, in1))

#define scilab_rt_pow_d0i0_d0(in0, in1, out0) (*out0 = pow(in0, in1))

#define scilab_rt_pow_d0i0_(in0, in1) (pow(in0, in1))

#define scilab_rt_pow_d0d0_d0(in0, in1, out0) (*out0 = pow(in0, in1))

#define scilab_rt_pow_d0d0_(in0, in1) (pow(in0, in1))

#define scilab_rt_pow_d0z0_d0(in0, in1, out0) (*out0 = cpow(in0, in1))

#define scilab_rt_pow_d0z0_(in0, in1) (cpow(in0, in1))

#define scilab_rt_pow_z0i0_z0(in0, in1, out0) (*out0 = cpow(in0, in1))

#define scilab_rt_pow_z0i0_(in0, in1) (cpow(in0, in1))

#define scilab_rt_pow_z0d0_z0(in0, in1, out0) (*out0 = cpow(in0, in1))

#define scilab_rt_pow_z0d0_(in0, in1) (cpow(in0, in1))

#define scilab_rt_pow_z0z0_z0(in0, in1, out0) (*out0 = cpow(in0, in1))

#define scilab_rt_pow_z0z0_(in0, in1) (cpow(in0, in1))

#define scilab_rt_pow_i0i2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_d0i2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_z0i2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_i0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_d0d2_d2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_z0d2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_i0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_d0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_z0z2_z2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0, in1[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_pow_i2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_i2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_i2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_d2i0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_d2d0_d2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = pow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_d2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_z2i0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_z2d0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

#define scilab_rt_pow_z2z0_z2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = cpow(in0[__lv1][__lv2], in1); \
		} \
	} \
}

