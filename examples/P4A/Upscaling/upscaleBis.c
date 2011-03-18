/** @addtogroup CUpscaling Classic C version

    @{
*/

/** @defgroup FUpscaling Upscaling functions.

    @{
    Functions to upscale the video.
    Chrominances are just copied.
    The Luminance is interpolated with an interpolation function at 6 points.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "yuv.h"
#include "upscale.h"

/* Upscaling at the half-pixel */
/* frame_in: input frame  */
/* frame_out: output frame */
const short normalisation=5;
const short bias=16;


void upscale_chrominance(type_yuv_frame_in frame_in, uint8 u_out[SIZE_UV_OUT], uint8 v_out[SIZE_UV_OUT])
{
  int sizeChrW = W_UV_IN;
  int sizeChrWout = W_UV_OUT;
  int sizeChrHout = H_UV_OUT;
  int sizeChrH = H_UV_IN;

   // Respecte mieux la localité des données
  for (int j = 0;j < sizeChrHout;j+=2) {
    int l = j >> 1;
    int jj = sizeChrWout*j;
    int ll = sizeChrW*l;
    // première ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_out[jj+i] = frame_in.u[indice];
      u_out[jj+i+1] = frame_in.u[indice];

      v_out[jj+i] = frame_in.v[indice];
      v_out[jj+i+1] = frame_in.v[indice];
    }
    jj += sizeChrWout;
    // deuxième ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_out[jj+i] = frame_in.u[indice];
      u_out[jj+i+1] = frame_in.u[indice];

      v_out[jj+i] = frame_in.v[indice];
      v_out[jj+i+1] = frame_in.v[indice];
    }
  }
}

uint8 interpolation_luminance(uint8 a, uint8 b,uint8 c,uint8 d,uint8 e,uint8 f)
{
  short res = (short)a + (short)f -5*((short)b+(short)e - (((short)c+(short)d)<<2));
  res = (res+bias)>>normalisation;
  res=clip((res));
  return (uint8)res;
}

void upscale_luminance(type_yuv_frame_in frame_in,uint8 y_out[SIZE_Y_OUT])
{
  int h = H_Y_IN;
  int w = W_Y_IN;

  int H = H_Y_OUT;
  int W = W_Y_OUT;
  
  int offset_y = OFFSET;
  int offset_x = OFFSET;

  for (int y=0; y < H; y +=2) {
    int ydiv2=(y>>1)+offset_y;
    int wydiv2=w*ydiv2;
    int Wy = y*W;
    for(int x=0;x< W;x += 2) {
      int xdiv2=(x>>1)+offset_x;
      int indice = xdiv2+wydiv2;
      // (x,y)
      y_out[Wy+x]=frame_in.y[xdiv2+wydiv2];
      //(x+1,y)
      y_out[Wy+x+1]=interpolation_luminance(frame_in.y[indice-2],frame_in.y[indice-1],frame_in.y[indice],frame_in.y[indice+1],frame_in.y[indice+2],frame_in.y[indice+3]);
      //(x,y+1)
      y_out[x+Wy+W]=interpolation_luminance(frame_in.y[indice-w*2],frame_in.y[indice-w],frame_in.y[indice],frame_in.y[indice+w],frame_in.y[indice+2*w],frame_in.y[indice+3*w]);
      // To compute (x+1,y+1) from (x-4:x-2:x:x+2:x+4:x+6,y+1)
      int j = indice-2;
      uint8 a = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      j = indice-1;
      uint8 b = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      j = indice;
      uint8 c = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      j = indice+1;
      uint8 d = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      j = indice+2;
      uint8 e = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      j = indice+3;
      uint8 f = interpolation_luminance(frame_in.y[j-w*2],frame_in.y[j-w],frame_in.y[j],frame_in.y[j+w],frame_in.y[j+2*w],frame_in.y[j+3*w]);
      
      y_out[x+1+Wy+W]=interpolation_luminance(a,b,c,d,e,f);


    }
  }
}

void upscale(type_yuv_frame_in frame_in,uint8 y[SIZE_Y_OUT], uint8 u[SIZE_UV_OUT], uint8 v[SIZE_UV_OUT])
{
  upscale_luminance(frame_in,y);
  upscale_chrominance(frame_in,u,v);
}


/** @} */
/** @} */
