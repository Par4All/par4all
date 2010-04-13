#include <stdio.h>
#include <stdlib.h>
typedef float float_t;
#define SIZE 501
#define T 400


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

  /* Erase the memory, in case the image is not big enough: */
  for(i = 0; i < SIZE; i++)
    for(j = 0; j < SIZE; j++)
      space[i][j] = 0;

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


void iteration(float_t space[SIZE][SIZE], float_t save[SIZE][SIZE]) {
  int i, j;

  /* Use 2 arrays in flip-flop to have dataparallel forall semantics. I
     could use also a flip-flop dimension instead... */
  for(i = 1;i < SIZE - 1; i++)
    for(j = 1;j < SIZE - 1; j++) {
      save[i][j] = 0.25*(space[i - 1][j] + space[i + 1][j]
			  + space[i][j - 1] + space[i][j + 1]);
    }

  for(i = 1;i < SIZE - 1; i++)
    for(j = 1;j < SIZE - 1; j++) {
      space[i][j] = 0.25*(save[i - 1][j] + save[i + 1][j]
			  + save[i][j - 1] + save[i][j + 1]);
    }
}


void compute(float_t space[SIZE][SIZE], float_t save[SIZE][SIZE]) {
  int t;

  /* Apply the relaxation T times: */
  for(t = 0; t < T; t++)
    iteration(space, save);
}



int main(int argc, char *argv[]) {
  int i;

  if (argc != 2) {
    fprintf(stderr,
	    "%s needs only one argument that is the PGM image input file\n",
	    argv[0]);
    exit(0);
  }
  get_data(argv[1]);

  /* Initialize the border of the destination image, since it is used but
     never written to: */
  for(i = 0; i < SIZE; i++)
    save[i][0] = save[0][i] = save[i][SIZE - 1] = save[SIZE - 1][i] = 0;

  compute(space, save);

  write_data("output.pgm");

  return 0;
}
