#include <stdio.h>
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
      c = space[i][j];
      fputc(c, fp);
    }
  fclose(fp);
}


void compute() {
  int i, j;

  /* Use 2 array in flip-flop to have dataparallel forall semantics. I
     could use also a flip-flop dimension instead... */
 kernel1:
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


int main(int argc, char *argv[]) {
  int t;

  if (argc != 2) {
    fprintf(stderr,
	    "%s needs only one argument that is the PGM image input file\n",
	    argv[0]);
    exit(0);
  }
  get_data(argv[1]);

  for(t = 0; t < T; t++)
    compute();

  write_data("output.pgm");

  return 0;
}
