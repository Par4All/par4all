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
#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))

//uint8 interpolation_luminance1(uint8 *y,int im2,int im1,int i,int ip1,int ip2,int ip3)
uint8 interpolation_luminance1(uint8 a,uint8 b,uint8 c,uint8 d,uint8 e,uint8 g)
{
  const short normalisation=5;
  const short bias=16;

  short f = (short)a+(short)g-5*((short)b+(short)e-(((short)c+(short)d)<<2));
  f = (f+bias)>>normalisation;
  f=clip((f));
  return (uint8)f;
}

P4A_accel_kernel upscale_luminance_centre(P4A_accel_global_address uint8 *y_fin ,P4A_accel_global_address uint8 *y_fout,int x,int y)
{ 
  int ydiv2=y+OFFSET;
  int wydiv2=W_Y_IN*ydiv2;
  int Wy = (2*y)*W_Y_OUT;
  
  int xdiv2=x+OFFSET;
  int indice = xdiv2+wydiv2;
  // (x,y)
  y_fout[Wy+2*x]=y_fin[xdiv2+wydiv2];
  //(x+1,y)
  y_fout[Wy+2*x+1]=interpolation_luminance1(y_fin[indice-2],y_fin[indice-1],y_fin[indice],y_fin[indice+1],y_fin[indice+2],y_fin[indice+3]);
  //(x,y+1)
  y_fout[2*x+Wy+W_Y_OUT]=interpolation_luminance1(y_fin[indice-W_Y_IN*2],y_fin[indice-W_Y_IN],y_fin[indice],y_fin[indice+W_Y_IN],y_fin[indice+2*W_Y_IN],y_fin[indice+3*W_Y_IN]);

  // To compute (x+1,y+1) from (x-4:x-2:x:x+2:x+4:x+6,y+1)
  //(x-4,y+1)
  double a =  interpolation_luminance1(y_fin[indice-2-W_Y_IN*2],y_fin[indice-2-W_Y_IN],y_fin[indice-2],y_fin[indice-2+W_Y_IN],y_fin[indice-2+2*W_Y_IN],y_fin[indice-2+3*W_Y_IN]);
  //(x-2,y+1)
  double b =  interpolation_luminance1(y_fin[indice-1-W_Y_IN*2],y_fin[indice-1-W_Y_IN],y_fin[indice-1],y_fin[indice-1+W_Y_IN],y_fin[indice-1+2*W_Y_IN],y_fin[indice-1+3*W_Y_IN]);
  double c = y_fout[2*x+Wy+W_Y_OUT];
  //(x+2,y+1)
  double d =  interpolation_luminance1(y_fin[indice+1-W_Y_IN*2],y_fin[indice+1-W_Y_IN],y_fin[indice+1],y_fin[indice+1+W_Y_IN],y_fin[indice+1+2*W_Y_IN],y_fin[indice+1+3*W_Y_IN]);
  //(x+4,y+1)
  double e =  interpolation_luminance1(y_fin[indice+2-W_Y_IN*2],y_fin[indice+2-W_Y_IN],y_fin[indice+2],y_fin[indice+2+W_Y_IN],y_fin[indice+2+2*W_Y_IN],y_fin[indice+2+3*W_Y_IN]);
  //(x+6,y+1)
  double f =  interpolation_luminance1(y_fin[indice+3-W_Y_IN*2],y_fin[indice+3-W_Y_IN],y_fin[indice+3],y_fin[indice+3+W_Y_IN],y_fin[indice+3+2*W_Y_IN],y_fin[indice+3+3*W_Y_IN]);

  //Finaly (x+1,y+1)
  y_fout[Wy+W_Y_OUT+2*x+1]=interpolation_luminance1(a,b,c,d,e,f);
  /*
  //(x+1,y+1)
  indice = 2*x+Wy+W_Y_OUT;
  y_fout[Wy+W_Y_OUT+2*x+1]=interpolation_luminance2(y_fout,indice-4,indice-2,indice,indice+2,indice+4,indice+6);
  */
}

P4A_accel_kernel_wrapper luminance1_wrapper(P4A_accel_global_address uint8 *y_fin ,P4A_accel_global_address uint8 *y_fout)
{
  int y = P4A_vp_0;
  int x = P4A_vp_1;

  if (y < HEIGHT && x < WIDTH)
    upscale_luminance_centre(y_fin,y_fout,x,y);
}

