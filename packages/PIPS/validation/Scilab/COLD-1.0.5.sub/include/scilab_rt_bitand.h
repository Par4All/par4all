/* (c) HPC Project 2010 */

#define scilab_rt_bitand_i0i0_i0(in0, in1, out0) (*out0 = BITAND(in0, in1))

#define scilab_rt_bitand_i0i0_(in0, in1) (BITAND(in0, in1))

#define scilab_rt_bitand_i2i2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = BITAND(in0[__lv1][__lv2], in1[__lv1][__lv2]); \
		} \
	} \
}

