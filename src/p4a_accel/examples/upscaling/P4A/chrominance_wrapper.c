#include "p4a_accel_wrapper.h"

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

typedef unsigned char uint8;

P4A_accel_kernel upscale_chrominance(P4A_accel_global_address uint8 *u_fin,P4A_accel_global_address uint8 *u_fout, P4A_accel_global_address uint8 *v_fin,P4A_accel_global_address uint8 *v_fout,int i,int j)
{
  int jj = W_UV_OUT*j*2;
  int ll = W_UV_IN*j;
  
  // première ligne de frame_out
  int indice  = ll+i;
  u_fout[jj+2*i] = u_fout[jj+2*i+1] = u_fin[indice];
  v_fout[jj+2*i] = v_fout[jj+2*i+1] = v_fin[indice];
 
  // deuxième ligne de frame_out
  jj += W_UV_OUT;
  u_fout[jj+2*i] = u_fout[jj+2*i+1] = u_fin[indice];
  v_fout[jj+2*i] = v_fout[jj+2*i+1] = v_fin[indice];
}

P4A_accel_kernel_wrapper chrominance_wrapper(P4A_accel_global_address uint8 *u_fin,P4A_accel_global_address uint8 *u_fout, P4A_accel_global_address uint8 *v_fin,P4A_accel_global_address uint8 *v_fout)
{
  int j = P4A_vp_0;
  int i = P4A_vp_1;

  if (j < H_UV_IN && i < W_UV_IN)
    upscale_chrominance(u_fin,u_fout,v_fin,v_fout,i,j);
}

