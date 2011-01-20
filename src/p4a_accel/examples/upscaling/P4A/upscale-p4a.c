#include <p4a_accel.h>

#include <stdio.h>
#include <stdlib.h>
#include "yuv.h"
//#include "upscale.h"

// Prototypes for the two kernels 
P4A_wrapper_proto(upscale_luminance_centre,P4A_accel_global_address uint8 *y_fin,P4A_accel_global_address uint8 *y_fout);
P4A_wrapper_proto(upscale_luminance_xplus1yplus1,P4A_accel_global_address uint8 *y_fout);
P4A_wrapper_proto(upscale_chrominance,P4A_accel_global_address type_yuv_frame_in *frame_in,P4A_accel_global_address type_yuv_frame_out *frame_out);


void upscale(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out)
{
  upscale_luminance_centre(frame_in->y,frame_out->y);
  upscale_luminance_xplus1yplus1(frame_out->y);
  upscale_chrominance(frame_in,frame_out);
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

  printf("Begin computation\n");
  // Computation ... no dependance
  for(int i=0;i<nbframes;i++) { 
    P4A_accel_timer_start;
    P4A_copy_to_accel(sizeof(type_yuv_frame_in),&frame_in[i],p4a_in);
    //upscale(&frame_in[i],&frame_out[i]);
    upscale(p4a_in,p4a_out);
    P4A_copy_from_accel(sizeof(type_yuv_frame_out),&frame_out[i],p4a_out);
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
