/* (c) HPC Project 2010 */

#define scilab_rt_sinh_i0_d0(in0, out0) (*out0 = sinh(in0))

#define scilab_rt_sinh_i0_(in0) (sinh(in0))

#define scilab_rt_sinh_d0_d0(in0, out0) (*out0 = sinh(in0))

#define scilab_rt_sinh_d0_(in0) (sinh(in0))

#define scilab_rt_sinh_z0_z0(in0, out0) (*out0 = csinh(in0))

#define scilab_rt_sinh_z0_(in0) (csinh(in0))

#define scilab_rt_sinh_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = sinh(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_sinh_z2_z2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = csinh(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_sinh_i2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = sinh(in0[__lv1][__lv2]); \
		} \
	} \
}

