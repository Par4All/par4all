/* (c) HPC Project 2010 */

#define scilab_rt_pmodulo_i0i0_i0(in0, in1, out0) (*out0 = ((in0 >= 0 ) ? (in0 % in1) : (in1 + in0 % in1)))

#define scilab_rt_pmodulo_i0i0_(in0, in1) ( ((in0 >= 0 ) ? (in0 % in1) : (in1 + in0 % in1)))

#define scilab_rt_pmodulo_i0d0_i0(in0, in1, out0) (*out0 = ((in0 >= 0 ) ? (in0 % (int)in1) : ((int)in1 + in0 % (int)in1)))

#define scilab_rt_pmodulo_i0d0_(in0, in1) ( ((in0 >= 0 ) ? (in0 % (int)in1) : ((int)in1 + in0 % (int)in1)))

#define scilab_rt_pmodulo_d0i0_i0(in0, in1, out0) (*out0 = (((int)in0 >= 0 ) ? ((int)in0 % in1) : (in1 + (int)in0 % in1)))

#define scilab_rt_pmodulo_d0i0_(in0, in1) ( (((int)in0 >= 0 ) ? ((int)in0 % in1) : (in1 + (int)in0 % in1)))

#define scilab_rt_pmodulo_d0d0_i0(in0, in1, out0) (*out0 = (((int)in0 >= 0 ) ? ((int)in0 % (int)in1) : ((int)in1 + (int)in0 % (int)in1)))

#define scilab_rt_pmodulo_d0d0_(in0, in1) ( (((int)in0 >= 0 ) ? ((int)in0 % (int)in1) : ((int)in1 + (int)in0 % (int)in1)))

