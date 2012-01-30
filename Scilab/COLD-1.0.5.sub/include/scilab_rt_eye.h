/* (c) HPC Project 2010 */

#define scilab_rt_eye_i0i0_d2(in0, in1, so00, so01, out0) \
{ \
	assert ((int)in0 == so00 && (int)in1 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			if (__lv1 == __lv2) \
				out0[__lv1][__lv2] = (double)1.0; \
			else \
				out0[__lv1][__lv2] = (double)0.0; \
		} \
	} \
}

#define scilab_rt_eye_i0i0i0_d3(in0, in1, in2, so00, so01, so02, out0) \
{ \
	assert ((int)in0 == so00 && (int)in1 == so01 && (int)in2 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				if (__lv1 == __lv2 && __lv2 == _lv3) \
				out0[__lv1][__lv2][__lv3] = (double)1.0; \
			else \
				out0[__lv1][__lv2][__lv3] = (double)0.0; \
			} \
		} \
	} \
}

#define scilab_rt_eye_i2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			if (__lv1 == __lv2) \
				out0[__lv1][__lv2] = (double)1.0; \
			else \
				out0[__lv1][__lv2] = (double)0.0; \
		} \
	} \
}

#define scilab_rt_eye_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			if (__lv1 == __lv2) \
				out0[__lv1][__lv2] = (double)1.0; \
			else \
				out0[__lv1][__lv2] = (double)0.0; \
		} \
	} \
}

#define scilab_rt_eye_z2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			if (__lv1 == __lv2) \
				out0[__lv1][__lv2] = (double)1.0; \
			else \
				out0[__lv1][__lv2] = (double)0.0; \
		} \
	} \
}

#define scilab_rt_eye_s2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			if (__lv1 == __lv2) \
				out0[__lv1][__lv2] = (double)1.0; \
			else \
				out0[__lv1][__lv2] = (double)0.0; \
		} \
	} \
}

