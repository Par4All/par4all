#include <stdlib.h>
#include <stdio.h>
#include "yuv.h"

void buffer_copy_s(uint8 * buffer,uint8 *y)
{
  for (int j = 0;j < HEIGHT;j++) {
    for (int i = 0;i < WIDTH;i++) {
      y[(j+OFFSET)*W_Y_IN+i+OFFSET]=buffer[j*WIDTH+i];
    }
  }
  
  for (int j = OFFSET;j < HEIGHT+OFFSET;j++) {
    // Bordure gauche
    for (int i = 0;i < OFFSET;i++) {
      y[j*W_Y_IN+i]=y[j*W_Y_IN+OFFSET];
    }
    // Bordure droite
    for (int i = WIDTH+OFFSET;i < WIDTH+OFFSET*2;i++) {
      y[j*W_Y_IN+i]=y[j*W_Y_IN+(WIDTH+OFFSET)-1];
    }
  }
  
  // Bordure haute
  for (int j = 0;j < OFFSET;j++) {
    for (int i = 0;i < WIDTH+(OFFSET*2);i++) {
      y[j*W_Y_IN+i]=y[OFFSET*W_Y_IN+i];
    }
  }
  // Bordure basse
  for (int j = HEIGHT+OFFSET;j < HEIGHT+(OFFSET*2);j++) {
    for (int i = 0;i < WIDTH+(OFFSET*2);i++) {
      y[j*W_Y_IN+i]=y[(HEIGHT+OFFSET-1)*W_Y_IN+i];
    }
  }
}

int read_yuv_frame(FILE* fp,type_yuv_frame_in *frame)
{  
  uint8 * buffer=(uint8*)malloc(SIZE*sizeof(uint8));
  
  // lecture luminance
  int rd=fread(buffer,1,SIZE,fp);
  if(rd != SIZE) {
    fprintf(stderr,"erreur lecture y fichier rd=%d rd attendu=%d \n",rd,SIZE);
    return -1;
  }
  // Copy of the buffer in Y with offset
  buffer_copy_s(buffer,frame->y);

  // lecture u
  rd = fread(frame->u,1,SIZE_UV_IN,fp);
  if(rd != SIZE_UV_IN) {
    fprintf(stderr,"erreur lecture u fichier rd=%d rd attendu=%d \n",rd,SIZE_UV_IN);
    return -1;
  }

  // lecture v
  rd = fread(frame->v,1,SIZE_UV_IN,fp);
  if(rd != SIZE_UV_IN) {
    fprintf(stderr,"erreur lecture v fichier rd=%d rd attendu=%d \n",rd,SIZE_UV_IN);
    return -1;
  }

  /*
  for (int k = 0;k < HEIGHT;k++) {
    for (int j = 0;j < WIDTH;j++)
      printf("pixel Y %d = %d %d, U = %d et V = %d\n",k*WIDTH+j,buffer[k*WIDTH+j],frame->y[k*WIDTH+j],frame->u[k/2*WIDTH/2+j/2],frame->v[k/2*WIDTH/2+j/2]);
  }
  exit(0);
  */
  free(buffer);
  return 0;
}

/* écriture d'une trame dans un fichier*/

int write_yuv_frame(FILE* fp,type_yuv_frame_out *frame)
{
  int wr = fwrite(frame->y,1,SIZE_Y_OUT,fp);
  if(wr != SIZE_Y_OUT) {
    fprintf(stderr,"erreur ecriture y fichier wr=%d wr attendu=%d\n",wr,SIZE_Y_OUT);
    return -1; 
  } 

  wr = fwrite(frame->u,1,SIZE_UV_OUT,fp);
  if(wr != SIZE_UV_OUT){ 
    fprintf(stderr,"erreur ecriture u fichier wr=%d wr attendu=%d\n",wr,SIZE_UV_OUT);
    return -1;
  }

  //v
  wr=fwrite(frame->v,1,SIZE_UV_OUT,fp);
  if(wr != SIZE_UV_OUT) {
    fprintf(stderr,"erreur ecriture v fichier wr=%d wr attendu=%d\n",wr,SIZE_UV_OUT);
    return -1;
  }
  return 0;
}
