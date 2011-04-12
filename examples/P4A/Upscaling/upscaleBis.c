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


void upscale_chrominance(uint8 u_in[SIZE_UV_IN], uint8 v_in[SIZE_UV_IN], uint8 u_out[SIZE_UV_OUT], uint8 v_out[SIZE_UV_OUT])
{
  int sizeChrW = W_UV_IN;
  int sizeChrWout = W_UV_OUT;
  int sizeChrHout = H_UV_OUT;

   // Respecte mieux la localité des données
  for (int j = 0;j < sizeChrHout;j+=2) {
    int l = j >> 1;
    int jj = sizeChrWout*j;
    int ll = sizeChrW*l;
    // première ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_out[jj+i] = u_in[indice];
      u_out[jj+i+1] = u_in[indice];

      v_out[jj+i] = v_in[indice];
      v_out[jj+i+1] = v_in[indice];
    }
    jj += sizeChrWout;
    // deuxième ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_out[jj+i] = u_in[indice];
      u_out[jj+i+1] = u_in[indice];

      v_out[jj+i] = v_in[indice];
      v_out[jj+i+1] = v_in[indice];
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

void upscale_luminance(uint8 y_in[SIZE_Y_IN],uint8 y_out[SIZE_Y_OUT])
{
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
      y_out[Wy+x]=y_in[xdiv2+wydiv2];
      //(x+1,y)
      y_out[Wy+x+1]=interpolation_luminance(y_in[indice-2],y_in[indice-1],y_in[indice],y_in[indice+1],y_in[indice+2],y_in[indice+3]);
      //(x,y+1)
      y_out[x+Wy+W]=interpolation_luminance(y_in[indice-w*2],y_in[indice-w],y_in[indice],y_in[indice+w],y_in[indice+2*w],y_in[indice+3*w]);
      // To compute (x+1,y+1) from (x-4:x-2:x:x+2:x+4:x+6,y+1)
      int j = indice-2;
      uint8 a = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      j = indice-1;
      uint8 b = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      j = indice;
      uint8 c = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      j = indice+1;
      uint8 d = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      j = indice+2;
      uint8 e = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      j = indice+3;
      uint8 f = interpolation_luminance(y_in[j-w*2],y_in[j-w],y_in[j],y_in[j+w],y_in[j+2*w],y_in[j+3*w]);
      
      y_out[x+1+Wy+W]=interpolation_luminance(a,b,c,d,e,f);


    }
  }
}

void upscale(uint8 y_in[SIZE_Y_IN], uint8 u_in[SIZE_UV_IN], uint8 v_in[SIZE_UV_IN],uint8 y_out[SIZE_Y_OUT], uint8 u_out[SIZE_UV_OUT], uint8 v_out[SIZE_UV_OUT])
{
  upscale_luminance(y_in,y_out);
  upscale_chrominance(u_in,v_in,u_out,v_out);

  
  for (int i = 0;i < SIZE_UV_OUT;i++)
    printf("UV %d : %d %d\n",i,u_out[i],v_out[i]);

  exit(0);

}


/** @} */
/** @} */
