/* (c) HPC Project 2010 */

#define scilab_rt_fix_d0_i0(in0, out0) (*out0 = FIX(in0))

#define scilab_rt_fix_d0_(in0) (FIX(in0))

#define scilab_rt_fix_d2_i2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	 \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = FIX(in0[__lv1][__lv2]); \
		} \
	} \
}

