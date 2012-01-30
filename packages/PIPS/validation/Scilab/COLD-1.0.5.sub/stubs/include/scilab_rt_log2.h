/* (c) HPC Project 2010 */

#define scilab_rt_log2_i0_d0(in0, out0) (*out0 = log2(in0))

#define scilab_rt_log2_i0_(in0) (log2(in0))

#define scilab_rt_log2_d0_d0(in0, out0) (*out0 = log2(in0))

#define scilab_rt_log2_d0_(in0) (log2(in0))

#define scilab_rt_log2_i2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = log2(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_log2_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = log2(in0[__lv1][__lv2]); \
		} \
	} \
}

#define scilab_rt_log2_i3_d3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = log2(in0[__lv1][__lv2][__lv3]); \
			} \
		} \
	} \
}

#define scilab_rt_log2_d3_d3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = log2(in0[__lv1][__lv2][__lv3]); \
			} \
		} \
	} \
}

