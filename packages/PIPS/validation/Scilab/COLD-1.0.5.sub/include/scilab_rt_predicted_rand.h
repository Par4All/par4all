/* (c) HPC Project 2010 */

#define scilab_rt_predicted_rand_i0_d0(in0, out0) (*out0 = PREDICTEDRAND1())

#define scilab_rt_predicted_rand_i0_(in0) (PREDICTEDRAND1())

#define scilab_rt_predicted_rand_d0_d0(in0, out0) (*out0 = PREDICTEDRAND1())

#define scilab_rt_predicted_rand_d0_(in0) (PREDICTEDRAND1())

#define scilab_rt_predicted_rand_z0_z0(in0, out0) (*out0 = PREDICTEDZRAND1())

#define scilab_rt_predicted_rand_z0_(in0) (PREDICTEDZRAND1())

#define scilab_rt_predicted_rand_i2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = PREDICTEDRAND1(); \
		} \
	} \
}

#define scilab_rt_predicted_rand_d2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = PREDICTEDRAND1(); \
		} \
	} \
}

#define scilab_rt_predicted_rand_z2_z2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = PREDICTEDZRAND1(); \
		} \
	} \
}

#define scilab_rt_predicted_rand_s2_d2(si00, si01, in0, so00, so01, out0) \
{ \
	assert (si00 == so00 && si01 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = PREDICTEDRAND1(); \
		} \
	} \
}

#define scilab_rt_predicted_rand_i3_d3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = PREDICTEDRAND1(); \
			} \
		} \
	} \
}

#define scilab_rt_predicted_rand_d3_d3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = PREDICTEDRAND1(); \
			} \
		} \
	} \
}

#define scilab_rt_predicted_rand_z3_z3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = PREDICTEDZRAND1(); \
			} \
		} \
	} \
}

#define scilab_rt_predicted_rand_s3_d3(si00, si01, si02, in0, so00, so01, so02, out0) \
{ \
	assert (si00 == so00 && si01 == so01 && si02 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = PREDICTEDRAND1(); \
			} \
		} \
	} \
}

#define scilab_rt_predicted_rand_i0i0_d2(in0, in1, so00, so01, out0) \
{ \
	assert ((int)in0 == so00 && (int)in1 == so01); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			out0[__lv1][__lv2] = PREDICTEDRAND1(); \
		} \
	} \
}

#define scilab_rt_predicted_rand_i0i0i0_d3(in0, in1, in2, so00, so01, so02, out0) \
{ \
	assert ((int)in0 == so00 && (int)in1 == so01 && (int)in2 == so02); \
	for (__lv1=0; __lv1<so00; __lv1++) { \
		for (__lv2=0; __lv2<so01; __lv2++) { \
			for (__lv3=0; __lv3<so02; __lv3++) { \
				out0[__lv1][__lv2][__lv3] = PREDICTEDRAND1(); \
			} \
		} \
	} \
}

