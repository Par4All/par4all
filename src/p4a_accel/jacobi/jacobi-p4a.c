/* Adapted from Terapix PIPS output */

#include <p4a_accel.h>

#include <stdio.h>
#include <stdlib.h>
typedef float float_t;
#define SIZE 501
#define T 400

float_t space[SIZE][SIZE];
// For the dataparallel semantics:
float_t save[SIZE][SIZE];

// Signatures des fonctions exportées dans les kernels 
// et appelées depuis le launch_kernel
P4A_wrapper_proto(kernel1_wrapper,P4A_accel_global_address float_t space[SIZE][SIZE],P4A_accel_global_address float_t save[SIZE][SIZE]);
P4A_wrapper_proto(kernel2_wrapper,P4A_accel_global_address float_t space[SIZE][SIZE],P4A_accel_global_address float_t save[SIZE][SIZE]);


void get_data(char filename[]) {
  int i, j, nx, ny;
  unsigned char c;
  FILE *fp;

  if ((fp = fopen(filename, "r")) == NULL) {
    perror("Error loading file");
    exit(0);
  }

  /* Get *.pgm file type */
  c = fgetc(fp);
  c = fgetc(fp);

  /* Skip comment lines */
  do {
    while((c = fgetc(fp)) != '\n');
  } while((c = fgetc(fp)) == '#');

  /* Put back good char */
  ungetc(c,fp);

  /* Get image dimensions */
  fscanf(fp, "%d %d\n", &nx, &ny);
  /* Get grey levels */
  fscanf(fp,"%d",&i);
  /* Get ONE carriage return */
  fgetc(fp);
  printf("Input image  : x=%d y=%d grey=%d\n", nx, ny, i);

  /* Erase the memory, in case the image is not big enough: */
  for(i = 0; i < SIZE; i++)
    for(j = 0; j < SIZE; j++) {
      space[i][j] = 0;
      save[i][j] = 0;
    }

  /* Read the pixel grey value: */
  for(j = 0; j < ny; j++)
    for(i = 0; i < nx; i++) {
      c = fgetc(fp);
      /* Truncate the image if too big: */
      if (i < SIZE && j < SIZE)
	space[i][j] = c;
    }

  fclose(fp);
}


void write_data(char filename[]) {
  int i,j;
  unsigned char c;
  FILE *fp;

  if ((fp = fopen(filename, "w")) == NULL) {
    perror("Error opening file");
    exit(0);
  }

  /* Write the PGM header: */
  fprintf(fp,"P5\n%d %d\n255\n", SIZE, SIZE);

  for(j = 0; j < SIZE; j++)
    for(i = 0; i < SIZE; i++) {
      c = space[i][j];
      fputc(c, fp);
    }
  fclose(fp);
}


#define MIN(a,b) (a < b ? a : b )

/*
 * file for launch_kernel1.c
 */
void launch_kernel1(float_t space[SIZE][SIZE], float_t save[SIZE][SIZE]) {
  kernel1:
  P4A_call_accel_kernel_2d(kernel1_wrapper, SIZE, SIZE, space, save);
}

/*
 * file for launch_kernel2.c
 */
void launch_kernel2(float_t space[SIZE][SIZE], float_t save[SIZE][SIZE])
{
  
kernel2:
  P4A_call_accel_kernel_2d(kernel2_wrapper, SIZE, SIZE, space, save);
}

void compute(float_t space[SIZE][SIZE], float_t save[SIZE][SIZE]) {
  int i, j;

  /* Use 2 array in flip-flop to have dataparallel forall semantics. I
     could use also a flip-flop dimension instead... */
 kernel1:   launch_kernel1(space, save);
 kernel2:   launch_kernel2(space, save);
}


int main(int argc, char *argv[]) {
  int t, i;

  P4A_init_accel;

  if (argc != 2) {
    fprintf(stderr,
	    "%s needs only one argument that is the PGM image input file\n",
	    argv[0]);
    exit(0);
  }
  get_data(argv[1]);


  // Initialize the border of the destination image, since it is used but
  // never written to: 
  for(i = 0; i < SIZE; i++)
    save[i][0] = save[0][i] = save[i][SIZE - 1] = save[SIZE - 1][i] = 0;

  // Useless to transfer and allocate data everytime... So this should be
  //  put at the highest level. It needs the interprocedural PIPS region
  //  analysis... :-) 
  P4A_accel_timer_start;
  float_t (*p4a_var_space)[SIZE][SIZE];
  P4A_accel_malloc((void **) &p4a_var_space, sizeof(space));
  P4A_copy_to_accel(sizeof(space), space, p4a_var_space);

  float_t (*p4a_var_save)[SIZE][SIZE];
  P4A_accel_malloc((void **) &p4a_var_save, sizeof(save));
  P4A_copy_to_accel(sizeof(space), save, p4a_var_save);
  double copy_time = P4A_accel_timer_stop_and_float_measure();

  P4A_accel_timer_start;

  
  for(t = 0; t < T; t++)
    compute(*p4a_var_space, *p4a_var_save);
  

  double execution_time = P4A_accel_timer_stop_and_float_measure();
  fprintf(stderr, "Temps d'exécution : %f s\n", execution_time);
  fprintf(stderr, "GFLOPS : %f\n",
	  4e-9/execution_time*T*(SIZE - 1)*(SIZE - 1));
  
  P4A_accel_timer_start;
  P4A_copy_from_accel((size_t)sizeof(space), (void *)space, (void *)p4a_var_space);
  copy_time += P4A_accel_timer_stop_and_float_measure();
  fprintf(stderr, "Temps de copie : %f s\n", copy_time);

  P4A_accel_free(p4a_var_space);
  P4A_accel_free(p4a_var_save);

  write_data("output.pgm");

  P4A_release_accel;
  return 0;
}
