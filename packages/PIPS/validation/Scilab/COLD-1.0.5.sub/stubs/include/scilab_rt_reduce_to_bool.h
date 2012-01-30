/* (c) HPC Project 2010 */

#define scilab_rt_reduce_to_bool_i2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 &= in0[__lv0][__lv1] != (int) 0; \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_i2_(int si00, int si01, int in0[si00][si01]);

#define scilab_rt_reduce_to_bool_d2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 &= in0[__lv0][__lv1] != (double) 0; \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_d2_(int si00, int si01, double in0[si00][si01]);

#define scilab_rt_reduce_to_bool_z2_i0(si00, si01, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			*out0 &= in0[__lv0][__lv1] != (double complex) 0; \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_z2_(int si00, int si01, double complex in0[si00][si01]);

#define scilab_rt_reduce_to_bool_i3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 &= in0[__lv0][__lv1][__lv2] != (int) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_i3_(int si00, int si01, int si02, int in0[si00][si01][si02]);

#define scilab_rt_reduce_to_bool_d3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 &= in0[__lv0][__lv1][__lv2] != (double) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_d3_(int si00, int si01, int si02, double in0[si00][si01][si02]);

#define scilab_rt_reduce_to_bool_z3_i0(si00, si01, si02, in0, out0) \
{ \
	*out0 = (int) 1; \
	for (__lv0=0; __lv0<si00; __lv0++) { \
		for (__lv1=0; __lv1<si01; __lv1++) { \
			for (__lv2=0; __lv2<si02; __lv2++) { \
				*out0 &= in0[__lv0][__lv1][__lv2] != (double complex) 0; \
			} \
		} \
	} \
}

extern int scilab_rt_reduce_to_bool_z3_(int si00, int si01, int si02, double complex in0[si00][si01][si02]);

