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

uint8 interpolation_luminance2(uint8 *y,int im2,int im1,int i,int ip1,int ip2,int ip3)
{  
  const short normalisation=5;
  const short bias=16;

  short f = (short)y[im2] + (short)y[ip3] -5*((short)y[im1]+(short)y[ip2] - (((short)y[i]+(short)y[ip1])<<2));
  f = (f+bias)>>normalisation;
  f=clip((f));
  return (uint8)f;
}

P4A_accel_kernel_wrapper upscale_luminance_xplus1yplus1(P4A_accel_global_address uint8 *y_fout)
{
  const int offset_max = WIDTH-OFFSET-1;
  
  for(int y=0;y<HEIGHT;y++){
    
    int Wy = W_Y_OUT*(2*y+1);
    
    //for(int x=0;x < 2*OFFSET; x += 2) {
    for(int x=0;x < OFFSET; x++) {
      int indice = 2*x+Wy;
      y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,Wy,Wy,indice,indice+2,indice+4,indice+6);
    }
    
    //for(int x=2*OFFSET;x <= offset_max-2;x=x+2) {
    for(int x=OFFSET;x < offset_max;x++) {
      int indice = 2*x+Wy;
      y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,indice+6);
    }

  /*
  //const int offset_max = W_Y_OUT-2*OFFSET-2;
  const int offset_max = WIDTH-OFFSET-1;

  for(int y=0;y<HEIGHT;y++){

    int Wy = W_Y_OUT*(2*y+1);

    //for(int x=0;x < 2*OFFSET; x += 2) {
    for(int x=0;x < OFFSET; x++) {
      int indice = 2*x+Wy;
      y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,Wy,Wy,indice,indice+2,indice+4,indice+6);
    }

    //for(int x=2*OFFSET;x <= offset_max-2;x=x+2) {
    for(int x=OFFSET;x < offset_max;x++) {
      int indice = 2*x+Wy;
      y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,indice+6);
    }
  */

    int lim = W_Y_OUT-2+W_Y_OUT*(2*y+1);
    int x = offset_max;
    int indice = 2*x+Wy;
    y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,lim);

    x = offset_max+1;
    indice = 2*x+Wy;
    y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,lim,lim);

    x = offset_max+2;
    indice = 2*x+Wy;
    y_fout[Wy+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,lim,lim,lim);
  }

  /*
  const int offset_max = W_Y_OUT-2*OFFSET-2;

  for(int y=0;y<H_Y_OUT;y=y+2){
    int Wy = W_Y_OUT*(y+1);

    for(int x=0;x < 2*OFFSET; x += 2) {
      int indice = x+Wy;
      y_fout[Wy+x+1]=interpolation_luminance2(y_fout,Wy,Wy,indice,indice+2,indice+4,indice+6);
    }

    for(int x=2*OFFSET;x <= offset_max-2;x=x+2) {
      int indice = x+Wy;
      y_fout[Wy+x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,indice+6);
    }


    int lim = W_Y_OUT-2+W_Y_OUT*(y+1);
    int x = offset_max;
    int indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,lim);

    x = offset_max+2;
    indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,lim,lim);

    x = offset_max+4;
    indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,lim,lim,lim);
  }
  */
}



