/** @addtogroup Upscaling

    @{
*/

/** @defgroup P4AUpscaling P4A version

    @{
    The P4A version for upscaling a video.
*/

/** @defgroup mainUpscaling The main.

    @{
    Call to the main and video processing functions.
*/
#include <p4a_accel.h>

#include <stdio.h>
#include <stdlib.h>
#include "yuv.h"

// Prototypes for the two kernels 

P4A_wrapper_proto(luminance_wrapper,P4A_accel_global_address type_yuv_frame_in *frame_in,P4A_accel_global_address type_yuv_frame_out *frame_out);

P4A_wrapper_proto(chrominance_wrapper,P4A_accel_global_address type_yuv_frame_in *frame_in, P4A_accel_global_address type_yuv_frame_out *frame_out);


void upscale(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  P4A_call_accel_kernel_2d(luminance_wrapper,HEIGHT,WIDTH,frame_in,frame_out);
  P4A_call_accel_kernel_2d(chrominance_wrapper,H_UV_IN,W_UV_IN,frame_in,frame_out);
}


/* réalise le processing de la video */
/* fpin: fichier d'entrée */
/* fpout: fichier de sortie */
/* nbframes: nombre de frames dans la video  */
void video_processing(FILE* fpin,FILE* fpout,int nbframes)
{
  type_yuv_frame_in frame_in[nbframes];
  type_yuv_frame_out frame_out[nbframes];

  P4A_init_accel;

  printf("Begin reading input video\n");
  // Reading ... data dependance
  for(int i = 0; i < nbframes; i++) {
    if (read_yuv_frame(fpin,&frame_in[i])) {
      fprintf(stderr,"erreur read_yuv_frame No frame=%d\n",i);
      break;
    }
  }
  printf("End of reading\n");
  
  type_yuv_frame_in *p4a_in;
  type_yuv_frame_out *p4a_out;
  P4A_accel_malloc((void **) &p4a_in, sizeof(type_yuv_frame_in));
  P4A_accel_malloc((void **) &p4a_out, sizeof(type_yuv_frame_out));

  double copy_time = 0.;
  double execution_time = 0.;
  printf("Begin computation\n");
  // Computation ... no dependance
  for(int i=0;i<nbframes;i++) { 
    P4A_accel_timer_start;
    P4A_copy_to_accel(sizeof(type_yuv_frame_in),&frame_in[i],p4a_in);
    copy_time += P4A_accel_timer_stop_and_float_measure();
    
    P4A_accel_timer_start;
    upscale(p4a_in,p4a_out);
    execution_time += P4A_accel_timer_stop_and_float_measure();
    
    P4A_accel_timer_start;
    P4A_copy_from_accel(sizeof(type_yuv_frame_out),&frame_out[i],p4a_out);
    copy_time += P4A_accel_timer_stop_and_float_measure();
  }
  printf("End of computation\n");
  fprintf(stderr, "Time of execution : %f s\n", execution_time);
  fprintf(stderr, "Time for copy : %f s\n", copy_time);

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
