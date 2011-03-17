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

/* Realize the processing of the video */
/* fpin: input file */
/* fpout: output file */
/* nbframes: number of frames in the video  */

void video_processing(FILE* fpin,FILE* fpout,int nbframes)
{
  type_yuv_frame_in frame_in[nbframes];
  type_yuv_frame_out frame_out[nbframes];

  printf("Begin reading input video\n");
  // Reading ... data dependance
  for(int i = 0; i < nbframes; i++) {
    if (read_yuv_frame(fpin,&frame_in[i])) {
      fprintf(stderr,"erreur read_yuv_frame No frame=%d\n",i);
      break;
    }
  }
  printf("End of reading\n");
  
  printf("Begin computation\n");
  // Computation ... no dependance
  for(int i=0;i<nbframes;i++) { 
    upscale(&frame_in[i],&frame_out[i]);
  }
  printf("End of computation\n");

  printf("Begin writing output video\n");
  // Writing ... data dependance
  for(int i = 0;i < nbframes;i++) {    
    if (write_yuv_frame(fpout,&frame_out[i])) {
      fprintf(stderr,"erreur write_yuv_frame No frame=%d\n",i);
      break;
    }  
  }
}

int main ( int argc, char *argv[] )
{
  FILE *fpin,*fpout;
  int nbframes;

  if (argc != 4 ) {
    fprintf(stderr,"Usage: upscale infile outfile sizew sizeh nbframes\n" );
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
  nbframes=atoi(argv[3]);
  
  video_processing(fpin,fpout,nbframes);
 
  fclose(fpin);
  fclose(fpout);

  return EXIT_SUCCESS;
}

/** @} */
/** @} */
/** @} */
/** @} */
