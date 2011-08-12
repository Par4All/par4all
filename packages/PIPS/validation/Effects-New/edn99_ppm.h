#include <stdlib.h>
#include <stdio.h>

typedef float FLOAT;
typedef FLOAT RGB[3];

typedef unsigned char IMAGE_RGB[3];


typedef struct {
  int width;
  int height;
} ppm_dim;

ppm_dim get_ppm_dim(FILE *f);
int read_ppm(FILE *f, int *max_color_value, int height, int width, IMAGE_RGB image[height][width]);
int write_ppm(FILE *f, char *comment, int max_color_value, int height, int width, IMAGE_RGB image[height][width]);
void split_filename(char *fn, char *head, char *tail);
