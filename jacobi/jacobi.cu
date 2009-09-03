/* Adapted fro Terapix PIPS output */


#define P4A_DEBUG
#define P4A_ACCEL_CUDA
#include <p4a_accel.h>

#include <stdio.h>
#include <stdlib.h>
typedef float float_t;
#define SIZE 64
#define T 64

float_t space[SIZE][SIZE];
// For the dataparallel semantics:
float_t save[SIZE][SIZE];

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


  for(i = 0;i < SIZE; i++)
    for(j = 0;j < SIZE; j++) {
      c = fgetc(fp);
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

  for(i = 0;i < SIZE; i++)
    for(j = 0;j < SIZE; j++) {
      c = save[i][j];
      fputc(c, fp);
    }
  fclose(fp);
}


#define MIN(a,b) (a < b ? a : b )
/*
 * file for kernel1.c
 */

P4A_ACCEL_KERNEL void kernel1(float_t save[64][64], float_t space[64][64], int i, int j)
{
   //int j;
   {
      //int i_1;
      // No need of strip mining in CUDA
      // for(i_1 = i; i_1 <= MIN(i+9, 62); i_1 += 1)
      // Already 2D
      //   for(j = 1; j <= 62; j += 1)
     save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
   }
}


/*
 * file for launch_kernel1.c
 */
P4A_ACCEL_KERNEL_WRAPPER void kernel1_wrapper(float_t save[64][64], float_t space[64][64])
{
  int j;
  int i;
   /* Use 2 array in flip-flop to have dataparallel forall semantics. I
           could use also a flip-flop dimension instead... */
kernel1:
   //for(i = 1; i <= 62; i += 10)
  /* We need this wrapper to get the virtual processor coordinates

     The Cuda compiler inline P4A_ACCEL_KERNEL functions by default, so
     there is no overhead */
  i = P4A_VP_X;
  j = P4A_VP_Y;
  // Oops. I forgotten a loop normalize since the GPU iterate in [0..SIZE-1]...
  /* We need a phase to generate this clamping too: */
  if (i >= 1 && i <= 62 && j >= 1 && j <= 62)
    kernel1(save, space, i, j);
}

/*
 * file for launch_kernel1.c
 */
void launch_kernel1(float_t save[64][64], float_t space[64][64])
{
  // int j;
  // int i;
   /* Use 2 array in flip-flop to have dataparallel forall semantics. I
           could use also a flip-flop dimension instead... */
kernel1:
   //for(i = 1; i <= 62; i += 10)
  // Oops. I forgotten a loop normalize since the GPU iterate in [0..SIZE-1]...
  P4A_CALL_ACCEL_KERNEL_2D(kernel1_wrapper, SIZE, SIZE, save, space);
}

void compute(float_t save[64][64], float_t space[64][64]) {
  int i, j;

  /* Use 2 array in flip-flop to have dataparallel forall semantics. I
     could use also a flip-flop dimension instead... */
kernel1:   launch_kernel1(save, space);

  // The same should be done on this kernel2...
#pragma omp parallel for private(j)
   for(i = 1; i <= 62; i += 1)
#pragma omp parallel for 
      for(j = 1; j <= 62; j += 1)

         space[i][j] = 0.25*(save[i-1][j]+save[i+1][j]+save[i][j-1]+save[i][j+1]);
}


int main(int argc, char *argv[]) {
  int t;

  P4A_INIT_ACCEL;

  if (argc != 2) {
    fprintf(stderr,
	    "%s needs only one argument that is the PGM image input file\n",
	    argv[0]);
    exit(0);
  }
  get_data(argv[1]);

  /* Useless to transfer and allocate data everytime... So this should be
     put at the highest level. It needs the interprocedural PIPS region
     analysis... :-) */
  float_t (*p4a_var_space)[SIZE][SIZE];
  P4A_ACCEL_MALLOC(&p4a_var_space, sizeof(space));
  P4A_COPY_TO_ACCEL(space, p4a_var_space, sizeof(space));

  float_t (*p4a_var_save)[SIZE][SIZE];
  P4A_ACCEL_MALLOC(&p4a_var_save, sizeof(save));

  P4A_ACCEL_TIMER_START;

  for(t = 0; t < T; t++)
    compute(*p4a_var_space, *p4a_var_save);

  double execution_time = P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE();
  fprintf(stderr, "Temps d'exécution : %f ms\n", execution_time);

  P4A_COPY_FROM_ACCEL(save, p4a_var_save, sizeof(save));

  P4A_ACCEL_FREE(p4a_var_space);
  P4A_ACCEL_FREE(p4a_var_save);

  write_data("output.pgm");

  P4A_RELEASE_ACCEL;
  return 0;
}
