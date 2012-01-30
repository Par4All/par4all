/* (c) HPC Project 2010 */

#define scilab_rt_ge_i0i0_i0(in0, in1, out0) (*out0 = in0 >= in1)

#define scilab_rt_ge_i0i0_(in0, in1) ( in0 >= in1)

#define scilab_rt_ge_i0d0_i0(in0, in1, out0) (*out0 = in0 >= (int)in1)

#define scilab_rt_ge_i0d0_(in0, in1) ( in0 >= (int)in1)

#define scilab_rt_ge_d0i0_i0(in0, in1, out0) (*out0 = (int)in0 >= in1)

#define scilab_rt_ge_d0i0_(in0, in1) ( (int)in0 >= in1)

#define scilab_rt_ge_d0d0_i0(in0, in1, out0) (*out0 = (int)in0 >= (int)in1)

#define scilab_rt_ge_d0d0_(in0, in1) ( (int)in0 >= (int)in1)

#define scilab_rt_ge_i1i1_i1(si00, in0, si10, in1, so00, out0) \
{ \
	assert (si00 == si10 && si10 == so00); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		out0[__lv1] = in0[__lv1] >= in1[__lv1]; \
	} \
}

#define scilab_rt_ge_d1d1_i1(si00, in0, si10, in1, so00, out0) \
{ \
	assert (si00 == si10 && si10 == so00); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		out0[__lv1] = (int)in0[__lv1] >= (int)in1[__lv1]; \
	} \
}

#define scilab_rt_ge_i2i2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] >= in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_d2d2_i2(si00, si01, in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0[__lv1][__lv2] >= (int)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_i2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] >= in1; \
		} \
	} \
}

#define scilab_rt_ge_d2i0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0[__lv1][__lv2] >= in1; \
		} \
	} \
}

#define scilab_rt_ge_i2d0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0[__lv1][__lv2] >= (int)in1; \
		} \
	} \
}

#define scilab_rt_ge_d2d0_i2(si00, si01, in0, in1, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0[__lv1][__lv2] >= (int)in1; \
		} \
	} \
}

#define scilab_rt_ge_i0i2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 >= in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_i0d2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = in0 >= (int)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_d0i2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0 >= in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_d0d2_i2(in0, si10, si11, in1, so00, so01, out0) \
{ \
	assert (si10 == so00 && si11 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = (int)in0 >= (int)in1[__lv1][__lv2]; \
		} \
	} \
}

#define scilab_rt_ge_i0i3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 >= in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_i0d3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0 >= (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d0i3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0 >= in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d0d3_i3(in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si10 == so00 && si11 == so01 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0 >= (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_i3i0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] >= in1; \
			} \
		} \
	} \
}

#define scilab_rt_ge_i3d0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] >= (int)in1; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d3i0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] >= in1; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d3d0_i3(si00, si01, si02, in0, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] >= (int)in1; \
			} \
		} \
	} \
}

#define scilab_rt_ge_i3i3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] >= in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_i3d3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = in0[__lv1][__lv2][__lv3] >= (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d3i3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] >= in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

#define scilab_rt_ge_d3d3_i3(si00, si01, si02, in0, si10, si11, si12, in1, so00, so01, so02, out0) \
{ \
	assert (si00 == si10 && si10 == so00 && si01 == si11 && si11 == so01 && si02 == si12 && si12 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = (int)in0[__lv1][__lv2][__lv3] >= (int)in1[__lv1][__lv2][__lv3]; \
			} \
		} \
	} \
}

