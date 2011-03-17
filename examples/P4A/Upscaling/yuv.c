/** @addtogroup CUpscaling Classic C version

    @{
*/

/** @defgroup yuvUpscaling Read and Write.

    @{
    Functions to manipulate the video (read an write).
*/
#include <stdlib.h>
#include <stdio.h>
#include "yuv.h"

/** The luminance is read in a buffer.
    The luminance data as input is increased by an OFFSET for a better
    parallelization of the interpolation function.
    This function copies and initializes the input
    luminance data.
 */

void buffer_copy_s(uint8 * buffer,uint8 *y)
{
  for (int j = 0;j < HEIGHT;j++) {
    for (int i = 0;i < WIDTH;i++) {
      y[(j+OFFSET)*W_Y_IN+i+OFFSET]=buffer[j*WIDTH+i];
    }
  }
  
  for (int j = OFFSET;j < HEIGHT+OFFSET;j++) {
    // Left border
    for (int i = 0;i < OFFSET;i++) {
      y[j*W_Y_IN+i]=y[j*W_Y_IN+OFFSET];
    }
    // Rigth border
    for (int i = WIDTH+OFFSET;i < WIDTH+OFFSET*2;i++) {
      y[j*W_Y_IN+i]=y[j*W_Y_IN+(WIDTH+OFFSET)-1];
    }
  }
  
  // Top border
  for (int j = 0;j < OFFSET;j++) {
    for (int i = 0;i < WIDTH+(OFFSET*2);i++) {
      y[j*W_Y_IN+i]=y[OFFSET*W_Y_IN+i];
    }
  }
  // Bottom border
  for (int j = HEIGHT+OFFSET;j < HEIGHT+(OFFSET*2);j++) {
    for (int i = 0;i < WIDTH+(OFFSET*2);i++) {
      y[j*W_Y_IN+i]=y[(HEIGHT+OFFSET-1)*W_Y_IN+i];
    }
  }
}

int read_yuv_frame(FILE* fp,type_yuv_frame_in *frame)
{  
  uint8 * buffer=(uint8*)malloc(SIZE*sizeof(uint8));
  
  // Read the luminance
  int rd=fread(buffer,1,SIZE,fp);
  if(rd != SIZE) {
    fprintf(stderr,"erreur lecture y fichier rd=%d rd attendu=%d \n",rd,SIZE);
    return -1;
  }
  // Copy of the buffer in Y with offset
  buffer_copy_s(buffer,frame->y);

  // Read the chrominance U
  rd = fread(frame->u,1,SIZE_UV_IN,fp);
  if(rd != SIZE_UV_IN) {
    fprintf(stderr,"erreur lecture u fichier rd=%d rd attendu=%d \n",rd,SIZE_UV_IN);
    return -1;
  }

  // Read the chrominance V
  rd = fread(frame->v,1,SIZE_UV_IN,fp);
  if(rd != SIZE_UV_IN) {
    fprintf(stderr,"erreur lecture v fichier rd=%d rd attendu=%d \n",rd,SIZE_UV_IN);
    return -1;
  }
  free(buffer);
  return 0;
}

/* Write the output video to a file */

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

/** @} */
/** @} */
