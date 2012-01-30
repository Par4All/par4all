/* (c) HPC Project 2010 */

#define scilab_rt_transpose_i2_i2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so01 && si01 == so00); \
	for (__lv2=0; __lv2<so00; __lv2++) { \
		for (__lv1=0; __lv1<so01; __lv1++) { \
			out0[__lv2][__lv1] = (in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_transpose_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so01 && si01 == so00); \
	for (__lv2=0; __lv2<so00; __lv2++) { \
		for (__lv1=0; __lv1<so01; __lv1++) { \
			out0[__lv2][__lv1] = (in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_transpose_z2_z2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so01 && si01 == so00); \
	for (__lv2=0; __lv2<so00; __lv2++) { \
		for (__lv1=0; __lv1<so01; __lv1++) { \
			out0[__lv2][__lv1] = (in0[__lv1][__lv2]); \
		} \
	} \
}

