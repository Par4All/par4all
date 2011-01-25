/** @addtogroup P4AUpscaling P4A version

    @{
*/

/** @defgroup ChrUpscaling Chrominance upscaling.

    @{
    Kernel to upscale the chrominance of a frame.
*/

#include "p4a_accel_wrapper.h"

//typedef unsigned char uchar;

#ifdef P4A_ACCEL_OPENMP
typedef unsigned char uchar;
#endif

#ifdef P4A_ACCEL_CUDA
typedef unsigned char uchar;
#endif

#define WIDTH 400
#define HEIGHT 226
#define OFFSET 3

#define SIZE WIDTH*HEIGHT

#define W_Y_IN (WIDTH+OFFSET*2)
#define H_Y_IN (HEIGHT+OFFSET*2)
#define W_UV_IN (WIDTH/2)
#define H_UV_IN (HEIGHT/2)
#define SIZE_UV_IN ((W_UV_IN)*(H_UV_IN))
#define SIZE_Y_IN ((W_Y_IN)*(H_Y_IN))

#define W_Y_OUT (WIDTH*2)
#define H_Y_OUT (HEIGHT*2)
#define W_UV_OUT ((W_Y_OUT)/2)
#define H_UV_OUT ((H_Y_OUT)/2)
#define SIZE_UV_OUT ((W_UV_OUT)*(H_UV_OUT))
#define SIZE_Y_OUT ((W_Y_OUT)*(H_Y_OUT))

typedef struct type_yuv_frame_in type_yuv_frame_in;
typedef struct type_yuv_frame_out type_yuv_frame_out;

struct type_yuv_frame_in {
	uchar y[SIZE_Y_IN];
	uchar u[SIZE_UV_IN];
	uchar v[SIZE_UV_IN];
};

struct type_yuv_frame_out {
	uchar y[SIZE_Y_OUT];
	uchar u[SIZE_UV_OUT];
	uchar v[SIZE_UV_OUT];
};

P4A_accel_kernel void upscale_chrominance(P4A_accel_global_address uchar u_fin[SIZE_UV_IN],P4A_accel_global_address uchar u_fout[SIZE_UV_OUT], P4A_accel_global_address uchar v_fin[SIZE_UV_IN],P4A_accel_global_address uchar v_fout[SIZE_UV_OUT],int i,int j)
{
  int jj = W_UV_OUT*j*2;
  int ll = W_UV_IN*j;
  
  // First line of frame_out
  int indice  = ll+i;
  u_fout[jj+2*i] = u_fout[jj+2*i+1] = u_fin[indice];
  v_fout[jj+2*i] = v_fout[jj+2*i+1] = v_fin[indice];
 
  // Second line of frame_out
  jj += W_UV_OUT;
  u_fout[jj+2*i] = u_fout[jj+2*i+1] = u_fin[indice];
  v_fout[jj+2*i] = v_fout[jj+2*i+1] = v_fin[indice];
}

P4A_accel_kernel_wrapper chrominance_wrapper(P4A_accel_global_address type_yuv_frame_in *frame_in,P4A_accel_global_address type_yuv_frame_out *frame_out)
{
  int j = P4A_vp_0;
  int i = P4A_vp_1;
  
  if (j < H_UV_IN && i < W_UV_IN)
    upscale_chrominance(frame_in->u,frame_out->u,frame_in->v,frame_out->v,i,j);
}


/** @} */
/** @} */
