/* (c) HPC Project 2010 */

#define scilab_rt_tan_i0_d0(in0, out0) (*out0 = tan(in0))

#define scilab_rt_tan_i0_(in0) (tan(in0))

#define scilab_rt_tan_d0_d0(in0, out0) (*out0 = tan(in0))

#define scilab_rt_tan_d0_(in0) (tan(in0))

#define scilab_rt_tan_z0_z0(in0, out0) (*out0 = ctan(in0))

#define scilab_rt_tan_z0_(in0) (ctan(in0))

#define scilab_rt_tan_i2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = tan(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_tan_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = tan(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_tan_z2_z2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = ctan(in0[__lv1][__lv2]); \
		} \
	} \
}

