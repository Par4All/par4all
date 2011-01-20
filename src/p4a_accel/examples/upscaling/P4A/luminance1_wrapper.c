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


#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))

uint8 interpolation_luminance1(uint8 *y,int im2,int im1,int i,int ip1,int ip2,int ip3)
{
  const short normalisation=5;
  const short bias=16;

  short f = (short)y[im2] + (short)y[ip3] -5*((short)y[im1]+(short)y[ip2] - (((short)y[i]+(short)y[ip1])<<2));
  f = (f+bias)>>normalisation;
  f=clip((f));
  return (uint8)f;
}

P4A_accel_kernel_wrapper upscale_luminance_centre(P4A_accel_global_address uint8 *y_fin ,P4A_accel_global_address uint8 *y_fout)
{
  for (int y=0; y < H_Y_OUT; y +=2) {
    int ydiv2=(y>>1)+OFFSET;
    int wydiv2=W_Y_IN*ydiv2;
    int Wy = y*W_Y_OUT;
    for(int x=0;x< W_Y_OUT;x += 2) {
      int xdiv2=(x>>1)+OFFSET;
      int indice = xdiv2+wydiv2;
      // (x,y)
      y_fout[Wy+x]=y_fin[xdiv2+wydiv2];
      //(x+1,y)
      y_fout[Wy+x+1]=interpolation_luminance1(y_fin,indice-2,indice-1,indice,indice+1,indice+2,indice+3);
      //(x,y+1)
      y_fout[x+Wy+W_Y_OUT]=interpolation_luminance1(y_fin,indice-W_Y_IN*2,indice-W_Y_IN,indice,indice+W_Y_IN,indice+2*W_Y_IN,indice+3*W_Y_IN);
    }
  }
}

