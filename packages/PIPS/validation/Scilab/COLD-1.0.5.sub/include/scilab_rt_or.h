/* (c) HPC Project 2010 */

#define scilab_rt_or_i0i0_i0(in0, in1, out0) (*out0 = in0 || in1)

#define scilab_rt_or_i0i0_(in0, in1) (in0 || in1)

#define scilab_rt_or_i1i1_i1(si00, in0, si10, in1, so00, out0) \
{ \
	assert (si00 == si10 && si10 == so00); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		out0[__lv1] = in0[__lv1] || in1[__lv1]; \
	} \
}

#define scilab_rt_or_d1d1_i1(si00, in0, si10, in1, so00, out0) \
{ \
	assert (si00 == si10 && si10 == so00); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		out0[__lv1] = (int)in0[__lv1] || (int)in1[__lv1]; \
	} \
}

#define scilab_rt_or_c1c1_i1(si00, in0, si10, in1, so00, out0) \
{ \
	assert (si00 == si10 && si10 == so00); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		out0[__lv1] = (int)in0[__lv1] || (int)in1[__lv1]; \
	} \
}

#define scilab_rt_or_i2i2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] || in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_or_d2d2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0[__lv1][__lv2] || (int)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_or_z2z2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0[__lv1][__lv2] || (int)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_or_i3i3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] || in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_or_i3d3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] || (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_or_d3i3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] || in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_or_d3d3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] || (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_or_i2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 |= in0[__lv0][__lv1] != (int) 0; \
		} \
	} \
}

extern int scilab_rt_or_i2_(int si00, int si01, int in0[si00][si01]);

#define scilab_rt_or_d2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 |= in0[__lv0][__lv1] != (double) 0; \
		} \
	} \
}

extern int scilab_rt_or_d2_(int si00, int si01, double in0[si00][si01]);

#define scilab_rt_or_z2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 |= in0[__lv0][__lv1] != (double complex) 0; \
		} \
	} \
}

extern int scilab_rt_or_z2_(int si00, int si01, double complex in0[si00][si01]);

#define scilab_rt_or_i3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 |= in0[__lv0][__lv1][__lv2] != (int) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_or_i3_(int si00, int si01, int si02, int in0[si00][si01][si02]);

#define scilab_rt_or_d3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 |= in0[__lv0][__lv1][__lv2] != (double) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_or_d3_(int si00, int si01, int si02, double in0[si00][si01][si02]);

#define scilab_rt_or_z3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 0; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 |= in0[__lv0][__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_or_z3_(int si00, int si01, int si02, double complex in0[si00][si01][si02]);

#define scilab_rt_or_i2s0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (*in1 == 'r') { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (int) 0; \
			} \
		} \
	 \
	} else if (*in1 == 'c') { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (int) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

#define scilab_rt_or_d2s0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (*in1 == 'r') { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (double) 0; \
			} \
		} \
	 \
	} else if (*in1 == 'c') { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (double) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

#define scilab_rt_or_z2s0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (*in1 == 'r') { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	 \
	} else if (*in1 == 'c') { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

#define scilab_rt_or_i2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (in1 == 1) { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (int) 0; \
			} \
		} \
	 \
	} else if (in1 == 2) { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (int) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

#define scilab_rt_or_d2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (in1 == 1) { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (double) 0; \
			} \
		} \
	 \
	} else if (in1 == 2) { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (double) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

#define scilab_rt_or_z2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	if (in1 == 1) { \
		assert (si01 == so01); \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[0][__lv2] = (int) 0; \
			for (__lv1=0; __lv1<si00; __lv1++) { \
				out0[0][__lv2] |= in0[__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	 \
	} else if (in1 == 2) { \
		assert (si00 == so00); \
		for (__lv1=0; __lv1<so00; __lv1++) { \
			out0[__lv1][0] = (int) 0; \
			for (__lv2=0; __lv2<si01; __lv2++) { \
				out0[__lv1][0] |= in0[__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	 \
	} else exit(1); \
}

