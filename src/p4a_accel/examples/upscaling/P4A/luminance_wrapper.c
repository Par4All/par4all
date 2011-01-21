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

#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))

P4A_accel_kernel uchar interpolation_luminance(uchar a,uchar b,uchar c,uchar d,uchar e,uchar g)
{
  const short normalisation=5;
  const short bias=16;

  short f = (short)a+(short)g-5*((short)b+(short)e-(((short)c+(short)d)<<2));
  f = (f+bias)>>normalisation;
  f=clip((f));
  return (uchar)f;
}

P4A_accel_kernel void upscale_luminance_centre(P4A_accel_global_address uchar y_fin[SIZE_Y_IN],P4A_accel_global_address uchar y_fout[SIZE_Y_OUT],int x,int y)
{ 
  int ydiv2=y+OFFSET;
  int wydiv2=W_Y_IN*ydiv2;
  int Wy = (2*y)*W_Y_OUT;
  
  int xdiv2=x+OFFSET;
  int indice = xdiv2+wydiv2;

  // (x,y)
  y_fout[Wy+2*x]=y_fin[xdiv2+wydiv2];
  //(x+1,y)
  y_fout[Wy+2*x+1]=interpolation_luminance(y_fin[indice-2],y_fin[indice-1],y_fin[indice],y_fin[indice+1],y_fin[indice+2],y_fin[indice+3]);
  //(x,y+1)
  y_fout[2*x+Wy+W_Y_OUT]=interpolation_luminance(y_fin[indice-W_Y_IN*2],y_fin[indice-W_Y_IN],y_fin[indice],y_fin[indice+W_Y_IN],y_fin[indice+2*W_Y_IN],y_fin[indice+3*W_Y_IN]);
  // To compute (x+1,y+1) from (x-4:x-2:x:x+2:x+4:x+6,y+1)
  //(x-4,y+1)
  uchar a =  interpolation_luminance(y_fin[indice-2-W_Y_IN*2],y_fin[indice-2-W_Y_IN],y_fin[indice-2],y_fin[indice-2+W_Y_IN],y_fin[indice-2+2*W_Y_IN],y_fin[indice-2+3*W_Y_IN]);
    //(x-2,y+1)
  uchar b =  interpolation_luminance(y_fin[indice-1-W_Y_IN*2],y_fin[indice-1-W_Y_IN],y_fin[indice-1],y_fin[indice-1+W_Y_IN],y_fin[indice-1+2*W_Y_IN],y_fin[indice-1+3*W_Y_IN]);
  uchar c = y_fout[2*x+Wy+W_Y_OUT];
  //(x+2,y+1)
  uchar d =  interpolation_luminance(y_fin[indice+1-W_Y_IN*2],y_fin[indice+1-W_Y_IN],y_fin[indice+1],y_fin[indice+1+W_Y_IN],y_fin[indice+1+2*W_Y_IN],y_fin[indice+1+3*W_Y_IN]);
  //(x+4,y+1)
  uchar e =  interpolation_luminance(y_fin[indice+2-W_Y_IN*2],y_fin[indice+2-W_Y_IN],y_fin[indice+2],y_fin[indice+2+W_Y_IN],y_fin[indice+2+2*W_Y_IN],y_fin[indice+2+3*W_Y_IN]);
  //(x+6,y+1)
  uchar f =  interpolation_luminance(y_fin[indice+3-W_Y_IN*2],y_fin[indice+3-W_Y_IN],y_fin[indice+3],y_fin[indice+3+W_Y_IN],y_fin[indice+3+2*W_Y_IN],y_fin[indice+3+3*W_Y_IN]);

  //Finaly (x+1,y+1)
  y_fout[Wy+W_Y_OUT+2*x+1]=interpolation_luminance(a,b,c,d,e,f);
}

P4A_accel_kernel_wrapper luminance_wrapper(P4A_accel_global_address type_yuv_frame_in *frame_in,P4A_accel_global_address type_yuv_frame_out *frame_out)
{
  int y = P4A_vp_0;
  int x = P4A_vp_1;
  int k;

  if (y < HEIGHT && x < WIDTH)
    upscale_luminance_centre(frame_in->y,frame_out->y,x,y);
}

