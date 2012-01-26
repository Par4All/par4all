#include <string.h>
#include <ctype.h>
#include <time.h>

#include "edn99_ppm.h"

/* PPM graphic file format I/O routines */

/**
 * Scan the next meaningful character from a PPM header.  This might
 * mean skipping comments.
 */
static int get_ppm_char(FILE *f) {
  int ch = getc(f);
  while(ch == '#') {
    do {
      ch = getc(f);
    } while(ch != EOF && ch != '\n');
  }
  return ch;
}

/**
 * Scan the next unsigned integer from a PPM header.  This might mean
 * skipping leading whitespace. In turn, this might mean skipping
 * comments. Stops when scanning another digit would cause overflow.
 */
static int get_ppm_unsigned(FILE *f) {
  int ch;

  // Find the first non-whitespace.
  do {
    ch = get_ppm_char(f);
  } while(isspace(ch));

  // If it's not a digit, we have a problem.
  if(!isdigit(ch)) {
    ungetc(ch, f);
    return -1;
  }
  // Scan all digits, converting to an integer.
  int i = (ch - '0');
  for (;;) {
    ch = get_ppm_char(f);
    if(!isdigit(ch))
      break;
    i = 10 * i + (ch - '0');
    // Crufty overflow handling.
    if(i < 0) {
      i = -1;
      break;
    }
  }
  ungetc(ch, f);
  return i;
}

/*
 * From the PPM format spec:

 * A PPM file consists of a sequence of one or more PPM images. There
 * are no data, delimiters, or padding before, after, or between
 * images.
 *
 * Each PPM image consists of the following: 
 *
 * 1. A "magic number" for identifying the file type. A ppm image's
 *    magic number is the two characters "P6".
 * 2. Whitespace (blanks, TABs, CRs, LFs). 
 * 3. A width, formatted as ASCII characters in decimal. 
 * 4. Whitespace. 
 * 5. A height, again in ASCII decimal. 
 * 6. Whitespace.
 * 7. The maximum color value (Maxval), again in ASCII decimal. Must
 *    be less than 65536 and more than zero.  8.  A single whitespace
 *    character (usually a newline).
 * 9. A raster of Height rows, in order from top to bottom. Each row
 *    consists of Width pixels, in order from left to right. Each pixel
 *    is a triplet of red, green, and blue samples, in that order. Each
 *    sample is represented in pure binary by either 1 or 2 bytes. If the
 *    Maxval is less than 256, it is 1 byte. Otherwise, it is 2
 *    bytes. The most significant byte is first.
 * A. (Taken from the PBM spec, which the PPM references.) Before the
 *    whitespace character that delimits the raster, any characters from
 *    a "#" through the next carriage return or newline character, is a
 *    comment and is ignored. Note that this is rather unconventional,
 *    because a comment can actually be in the middle of what you might
 *    consider a token. Note also that this means if you have a comment
 *    right before the raster, the newline at the end of the comment is
 *    not sufficient to delimit the raster.
 */

/**
 * Read a PPM file into an initialized image.  Return 0 on
 * success. Otherwise return a code indicating the error.  Old image
 * data are always cleared regardless of success or failure.
 */
int read_ppm(FILE *f,
             int *max_color_value,
             int height,
             int width,
             IMAGE_RGB image[height][width]) {
  ppm_dim dim = get_ppm_dim(f);

  // Check magic number.
  if(dim.height == -1 && dim.width == -1)
    return 1;

  // Get three numeric header values.
  if(dim.height == -2 && dim.width == -2 || dim.height != height)
    return 2;

  if(dim.height == -3 && dim.width == -3 || dim.width != width)
    return 3;

  if(dim.height < 0 || dim.width < 0)
    return 4;

  *max_color_value = get_ppm_unsigned(f);
  if(*max_color_value != 255)
    return 5;

  // Get the whitespace delimiter before the data.
  int delimiter = get_ppm_char(f);
  if(!isspace(delimiter))
    return 6;

  // Read the data.
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      if(fread(&image[i][j], sizeof image[i][j], 1, f) != 1)
        return 100 + i;

  // Success!
  return 0;
}

/**
 * Get the dimensions for a ppm file
 */
ppm_dim get_ppm_dim(FILE *f) {
  rewind(f);
  ppm_dim dim;
  // Check magic number.
  if(get_ppm_char(f) != 'P' || get_ppm_char(f) != '6'
      ||
      !isspace(get_ppm_char(f))) {
    dim.height = -1;
    dim.width = -1;
    return dim;
  }
  // Get three numeric header values.
  int _width = get_ppm_unsigned(f);
  if(_width == -1) {
    dim.height = -2;
    dim.width = -2;
    return dim;
  }

  int _height = get_ppm_unsigned(f);
  if(_height == -1) {
    dim.height = -3;
    dim.width = -3;
    return dim;
  }

  dim.height = _height;
  dim.width = _width;
  return dim;
}

/**
 * Write an image to a PPM file, which is assumed open for binary
 * write.  Return 0 on success, otherwise an error code.  Image
 * is unchanged.
 */
int write_ppm(FILE *f,
              char *comment,
              int max_color_value,
              int height,
              int width,
              IMAGE_RGB image[height][width]) {
  if(fprintf(f, "P6\n"
    "# %s\n"
    "%d %d\n"
    "%d\n", comment, width, height, max_color_value) < 0)
    return 1;

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      if(fwrite(image[i][j], sizeof image[i][j], 1, f) != 1)
        return 100 + i;

  return 0;
}

/**
 * Find the portion of a filename path up to the dot and the portion after the dot.
 */
void split_filename(char *fn, char *head, char *tail) {
  char *dot = strrchr(fn, '.');
  if(dot == NULL) {
    strcpy(head, fn);
    strcpy(tail, "");
  } else {
    strncpy(head, fn, dot - fn);
    strcpy(tail, dot);
  }
}

