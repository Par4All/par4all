#include "p4a_accel_wrapper.h"

#define WIDTH 400
#define HEIGHT 226
#define OFFSET 2

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



typedef unsigned char uint8;
typedef struct type_yuv_frame_in type_yuv_frame_in;
typedef struct type_yuv_frame_out type_yuv_frame_out;


struct type_yuv_frame_in {
	uint8 y[SIZE_Y_IN];
	uint8 u[SIZE_UV_IN];
	uint8 v[SIZE_UV_IN];
};

struct type_yuv_frame_out {
	uint8 y[SIZE_Y_OUT];
	uint8 u[SIZE_UV_OUT];
	uint8 v[SIZE_UV_OUT];
};

P4A_accel_kernel_wrapper upscale_chrominance(P4A_accel_global_address type_yuv_frame_in *frame_in,P4A_accel_global_address type_yuv_frame_out *frame_out)
{
  uint8 *u_fin = frame_in->u;
  uint8 *v_fin = frame_in->v;
  uint8 *u_fout = frame_out->u;
  uint8 *v_fout = frame_out->v;
  
  
  // Respecte mieux la localité des données
  for (int j = 0;j < H_UV_OUT;j+=2) {
    int l = j >> 1;
    int jj = W_UV_OUT*j;
    int ll = W_UV_IN*l;
    // première ligne de frame_out
    for (int i = 0; i < W_UV_OUT;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_fout[jj+i] = u_fout[jj+i+1] = u_fin[indice];
      v_fout[jj+i] = v_fout[jj+i+1] = v_fin[indice];
    }
    jj += W_UV_OUT;
    // deuxième ligne de frame_out
    for (int i = 0; i < W_UV_OUT;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_fout[jj+i] = u_fout[jj+i+1] = u_fin[indice];
      v_fout[jj+i] = v_fout[jj+i+1] = v_fin[indice];
    }
  }
}

