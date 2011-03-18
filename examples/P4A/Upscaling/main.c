/** @defgroup Examples

    @{

*/

/** @defgroup Upscaling

    @{
    A two folds upscaling function. The video format is YUV.

    This was developped and tested in the framework of the transmedi@ project.

    "mailto:Stephanie.Even@enstb.org"
*/

/** @defgroup CUpscaling Classic C version

    @{
    An original C version of the upscaling.
*/

/** @defgroup mainUpscaling The main.

    @{
    Call to the main and video processing functions.
*/

#include <stdio.h>
#include <stdlib.h>
#include "yuv.h"
#include "upscale.h"

/* NBFRAMES: number of frames in the video  */
#define NBFRAMES 3525

/* Realize the processing of the video */
/* fpin: input file */
/* fpout: output file */

void video_processing(FILE* fpin,FILE* fpout)
{
  type_yuv_frame_in frame_in[NBFRAMES];
  type_yuv_frame_out frame_out[NBFRAMES];

  printf("Begin reading input video\n");
  // Reading ... data dependence
  for(int i = 0; i < NBFRAMES; i++) {
    //printf("Reading image %d\n",i);
    if (read_yuv_frame(fpin,&frame_in[i])) {
      fprintf(stderr,"erreur read_yuv_frame No frame=%d\n",i);
      exit(0);
    }
  }
  printf("End of reading\n");

   printf("Begin computation\n");
  // Computation ... no dependence
  for(int i=0;i<NBFRAMES;i++) { 
    upscale(&frame_in[i],&frame_out[i]);
  }
  printf("End of computation\n");

  printf("Begin writing output video\n");
  // Writing ... data dependence
  for(int i = 0;i < NBFRAMES;i++) {    
    if (write_yuv_frame(fpout,&frame_out[i])) {
      fprintf(stderr,"erreur write_yuv_frame No frame=%d\n",i);
      exit(0);
    }  
  }
}

int main ( int argc, char *argv[] )
{
  FILE *fpin,*fpout;

  if (argc != 3) {
    fprintf(stderr,"Usage: %s infile outfile\n",argv[0]);
    return EXIT_FAILURE;
  }
  
  if ((fpin = fopen(argv[1],"rb")) == NULL) {
    fprintf(stderr,"Wrong input file name or path\n");
    return EXIT_FAILURE;
  }
  if ((fpout = fopen(argv[2],"wb")) == NULL) {
    fprintf(stderr,"Wrong output file name or path\n");
    return EXIT_FAILURE;
  }
  video_processing(fpin,fpout);
 
  fclose(fpin);
  fclose(fpout);

  return EXIT_SUCCESS;
}

/** @} */
/** @} */
/** @} */
/** @} */
