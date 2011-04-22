#include <stdio.h>
#include <stdlib.h>
#include "yuv.h"

#define _clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))

upsc_uint8 _interpolation_luminance(upsc_uint8 a, upsc_uint8 b,upsc_uint8 c,upsc_uint8 d,upsc_uint8 e,upsc_uint8 f)
{
  short _normalisation=5;
  short _bias=16;
  short res = (short)a + (short)f -5*((short)b+(short)e - (((short)c+(short)d)<<2));
  res = (res+_bias)>>_normalisation;
  res=_clip((res));
  return (upsc_uint8)res;
}

void video_processing(FILE* fpin,FILE* fpout)
{
  short _normalisation=5;
  short _bias=16;
  type_yuv_frame_in frame_in[NBFRAMES];
  type_yuv_frame_out frame_out[NBFRAMES];

  printf("Begin reading input video\n");
  // Reading ... data dependence
  for(int i = 0; i < NBFRAMES; i++) {
    printf("Reading image %d\n",i);
    if (read_yuv_frame(fpin,frame_in[i].y,frame_in[i].u,frame_in[i].v)) {
      fprintf(stderr,"erreur read_yuv_frame No frame=%d\n",i);
      exit(0);
    }
  }
  printf("End of reading\n");  

   printf("Begin computation\n");
   
  // Computation ... no dependence
   for(int n=0;n<NBFRAMES;n++) {
     for (int indice = 0;indice < SIZE_UV_IN;indice++) {
       int i = indice/W_UV_IN;
       int j = indice - i*W_UV_IN;
       // Column
       int jj = j*2;
       // Line
       int ii = i*W_UV_OUT*2;
       // First line
       frame_out[n].u[jj+ii] = frame_in[n].u[indice];
       frame_out[n].u[jj+ii+1] = frame_in[n].u[indice];
       //Second line
       frame_out[n].u[jj+W_UV_OUT+ii] = frame_in[n].u[indice];
       frame_out[n].u[jj+W_UV_OUT+ii+1] = frame_in[n].u[indice];

       // First line
       frame_out[n].v[jj+ii] = frame_in[n].v[indice];
       frame_out[n].v[jj+ii+1] = frame_in[n].v[indice];
       //Second line
       frame_out[n].v[jj+W_UV_OUT+ii] = frame_in[n].v[indice];
       frame_out[n].v[jj+W_UV_OUT+ii+1] = frame_in[n].v[indice];
     }
   }
   
  
for(int n=0;n<NBFRAMES;n++) {
     for (int index = 0;index < SIZE;index++) {
       //Line in
       int i = index/WIDTH;
       // Column in
       int j = index - i*WIDTH;
       // Column out
       int jj = j*2;
       // Line out
       int ii = i*W_Y_OUT*2;
       
       j += OFFSET;
       i += OFFSET;
       int indice = i*W_Y_IN+j;
       
       // (x,y)
       frame_out[n].y[jj+ii]=frame_in[n].y[indice];
       //(x+1,y)
       frame_out[n].y[jj+ii+1]=_interpolation_luminance(frame_in[n].y[indice-2],frame_in[n].y[indice-1],frame_in[n].y[indice],frame_in[n].y[indice+1],frame_in[n].y[indice+2],frame_in[n].y[indice+3]);
       //(x,y+1)
       frame_out[n].y[jj+W_Y_OUT+ii]=_interpolation_luminance(frame_in[n].y[indice-W_Y_IN*2],frame_in[n].y[indice-W_Y_IN],frame_in[n].y[indice],frame_in[n].y[indice+W_Y_IN],frame_in[n].y[indice+2*W_Y_IN],frame_in[n].y[indice+3*W_Y_IN]);
       
       
       frame_out[n].y[jj+ii+1+W_Y_OUT] = _interpolation_luminance(frame_in[n].y[(indice-2)-W_Y_IN*2],frame_in[n].y[(indice-2)-W_Y_IN],frame_in[n].y[indice-2],frame_in[n].y[(indice-2)+W_Y_IN],frame_in[n].y[(indice-2)+2*W_Y_IN],frame_in[n].y[(indice-2)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] -= 5*_interpolation_luminance(frame_in[n].y[(indice-1)-W_Y_IN*2],frame_in[n].y[(indice-1)-W_Y_IN],frame_in[n].y[indice-1],frame_in[n].y[(indice-1)+W_Y_IN],frame_in[n].y[(indice-1)+2*W_Y_IN],frame_in[n].y[(indice-1)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] += 20*_interpolation_luminance(frame_in[n].y[(indice)-W_Y_IN*2],frame_in[n].y[(indice)-W_Y_IN],frame_in[n].y[indice],frame_in[n].y[(indice)+W_Y_IN],frame_in[n].y[(indice)+2*W_Y_IN],frame_in[n].y[(indice)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] += 20*_interpolation_luminance(frame_in[n].y[(indice+1)-W_Y_IN*2],frame_in[n].y[(indice+1)-W_Y_IN],frame_in[n].y[indice+1],frame_in[n].y[(indice+1)+W_Y_IN],frame_in[n].y[(indice+1)+2*W_Y_IN],frame_in[n].y[(indice+1)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] -= 5*_interpolation_luminance(frame_in[n].y[(indice+2)-W_Y_IN*2],frame_in[n].y[(indice+2)-W_Y_IN],frame_in[n].y[indice+2],frame_in[n].y[(indice+2)+W_Y_IN],frame_in[n].y[(indice+2)+2*W_Y_IN],frame_in[n].y[(indice+2)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] += _interpolation_luminance(frame_in[n].y[(indice+3)-W_Y_IN*2],frame_in[n].y[(indice+3)-W_Y_IN],frame_in[n].y[indice+3],frame_in[n].y[(indice+3)+W_Y_IN],frame_in[n].y[(indice+3)+2*W_Y_IN],frame_in[n].y[(indice+3)+3*W_Y_IN]);
       frame_out[n].y[jj+ii+1+W_Y_OUT] = (upsc_uint8)_clip((((short)frame_out[n].y[jj+ii+1+W_Y_OUT])+_bias)>>_normalisation);
       
     }
   }
  printf("End of computation\n");

  printf("Begin writing output video\n");
  // Writing ... data dependence
  for(int i = 0;i < NBFRAMES;i++) {
    if (write_yuv_frame(fpout,frame_out[i].y,frame_out[i].u,frame_out[i].v)) {
      fprintf(stderr,"erreur write_yuv_frame No frame=%d\n",i);
      exit(0);
    } 
  }
}

int main (int argc, char *argv[])
{
   FILE *fpin, *fpout;

   if (argc!=3) {
      fprintf(stderr, "Usage: %s infile outfile\n", argv[0]);
      return 1;
   }
   if ((fpin = fopen(argv[1], "rb"))==NULL) {
      fprintf(stderr, "Wrong input file name or path\n");
      return 1;
   }
   if ((fpout = fopen(argv[2], "wb"))==NULL) {
      fprintf(stderr, "Wrong output file name or path\n");
      return 1;
   }
   video_processing(fpin, fpout);

   fclose(fpin);
   fclose(fpout);

   return 0;
}

