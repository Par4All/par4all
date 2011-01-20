#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "yuv.h"
#include "upscale.h"

/* fonction d'upcaling au demi pixel */
/* frame_in: frame initiale */
/* frame_out: frame upscalée */
const short normalisation=5;
const short bias=16;


void upscale_chrominance(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  int sizeChrW = W_UV_IN;
  int sizeChrWout = W_UV_OUT;
  int sizeChrHout = H_UV_OUT;
  int sizeChrH = H_UV_IN;

  uint8 *u_fin = frame_in->u;
  uint8 *v_fin = frame_in->v;
  uint8 *u_fout = frame_out->u;
  uint8 *v_fout = frame_out->v;
  
  
  // Respecte mieux la localité des données
  for (int j = 0;j < sizeChrHout;j+=2) {
    int l = j >> 1;
    int jj = sizeChrWout*j;
    int ll = sizeChrW*l;
    // première ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_fout[jj+i] = u_fout[jj+i+1] = u_fin[indice];

      v_fout[jj+i] = v_fout[jj+i+1] = v_fin[indice];
    }
    jj += sizeChrWout;
    // deuxième ligne de frame_out
    for (int i = 0; i < sizeChrWout;i+=2) {
      int k = i>>1;
      int indice  = ll+k;
      u_fout[jj+i] = u_fout[jj+i+1] = u_fin[indice];

      v_fout[jj+i] = v_fout[jj+i+1] = v_fin[indice];
    }
  }
}

uint8 interpolation_luminance(uint8 *y,int im2,int im1,int i,int ip1,int ip2,int ip3)
{
  short f = (short)y[im2] + (short)y[ip3] -5*((short)y[im1]+(short)y[ip2] - (((short)y[i]+(short)y[ip1])<<2));
  f = (f+bias)>>normalisation;
  f=clip((f));
  return (uint8)f;
}

void upscale_luminance_centre(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  uint8 *y_fin = frame_in->y;
  uint8 *y_fout = frame_out->y;

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
      y_fout[Wy+x]=y_fin[xdiv2+wydiv2];
      //(x+1,y)
      y_fout[Wy+x+1]=interpolation_luminance(y_fin,indice-2,indice-1,indice,indice+1,indice+2,indice+3);
      //(x,y+1)
      y_fout[x+Wy+W]=interpolation_luminance(y_fin,indice-w*2,indice-w,indice,indice+w,indice+2*w,indice+3*w);
    }
  }
}

void  upscale_luminance_xplus1yplus1(uint8 *y_fout,int W,int H)
{
  const int offset_max = W-OFFSET-2;

  for(int y=0;y<H;y=y+2){
    int Wy = W*(y+1);

    for(int x=0;x < OFFSET; x += 2) {
      int indice = x+Wy;
      y_fout[Wy+x+1]=interpolation_luminance(y_fout,Wy,Wy,indice,indice+2,indice+4,indice+6);
    }

    for(int x=OFFSET;x <= offset_max-2;x=x+2) {
      int indice = x+Wy;
      y_fout[Wy+x+1]=interpolation_luminance(y_fout,indice-4,indice-2,indice,indice+2,indice+4,indice+6);
    }


    int lim = W-2+W*(y+1);
    int x = offset_max;
    int indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance(y_fout,indice-4,indice-2,indice,indice+2,indice+4,lim);

    x = offset_max+2;
    indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance(y_fout,indice-4,indice-2,indice,indice+2,lim,lim);

    x = offset_max+4;
    indice = x+Wy;
    y_fout[Wy+x+1]=interpolation_luminance(y_fout,indice-4,indice-2,indice,lim,lim,lim);
  }
}

void upscale_luminance(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  
  int W=W_Y_OUT;
  int H=H_Y_OUT;

  // Tout à la fin (x+1,y+1)
  upscale_luminance_centre(frame_in,frame_out);
  upscale_luminance_xplus1yplus1(frame_out->y,W,H);  
}

void upscale(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  upscale_luminance(frame_in,frame_out);
  upscale_chrominance(frame_in,frame_out);
}


