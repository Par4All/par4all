#include <stdio.h>
#include <sys/time.h>


void copy (int w, int h, unsigned char bytes[h*w], unsigned char dest[h*w])
{
  int i,j;

  for (i = 0; i < w; i++)  
    for (j = 0; j < h; j++)
      {
	dest[j*w + i] = bytes[j*w + i];
      }
}

void light (int w, int h, unsigned char bytes[h*w], unsigned char val)
{
  int i,j;

  for (i = 0; i < w; i++)  
    for (j = 0; j < h; j++)
      {
	unsigned char current = bytes[j*w + i];
	bytes[j*w + i] = (((int)current + val)>255)?255:current+val;
      }
}

void curve (int w, int h, unsigned char bytes[h*w], unsigned char lut[h*w])
{
  int i,j;

  for (i = 0; i < w; i++)  
    for (j = 0; j < h; j++)
      {
	unsigned char current = bytes[j*w + i];
	bytes[j*w + i] = lut[current];
      }
}

void transfo (int w, int h, unsigned char bytes[h*w], unsigned char dest[h*w], unsigned char lut[h*w], unsigned char val)
{
  copy (w, h, bytes, dest);
  curve (w, h, dest, lut);
  light (w, h, dest, val);
}

/* No support yet for assembly code */
int main (int ac, char *av[])
{
  FILE *in;
  FILE *map;
  FILE *out;
  int height;
  int width;
  int maxval;
  char c1;
  char c2;
  double t;
  long i, size;

  if (ac != 4)
    {
      printf ("Usage: light infile.pgm mapfile.amp outfile.pgm\n");
      exit (1);
    }
  
  in = fopen (av[1], "r");
  if (in == NULL)
    {
      perror ("fopen");
      exit (1);
    }

  map = fopen (av[2], "r");
  if (map == NULL)
    {
      perror ("fopen");
      exit (1);
    }
  
  fscanf (in, "%c", &c1);
  fscanf (in, "%c", &c2);
  if (c1 != 'P' || c2 != '5')
    {
      fprintf (stderr, "Error, input file is not PGM\n");
      exit (1);
    }
  
  fscanf (in, "%d %d", &height, &width);
  fscanf (in, "%d", &maxval);

  printf ("w=%d, h=%d, max=%d\n", width, height, maxval);

  {
  unsigned char (*bytes)[height*width];
  unsigned char (*dest)[height*width];
  unsigned char (*lut)[height*width];
  size = width * height;

  bytes = (unsigned char (*) [height*width]) malloc (sizeof (unsigned char) * size);
  if (bytes == NULL)
    {
      perror ("malloc");
      exit (1);
    }

  dest = (unsigned char (*)[height*width]) malloc (sizeof (unsigned char) * size);
  if (dest == NULL)
    {
      perror ("malloc");
      exit (1);
    }

  lut = (unsigned char (*)[height*width]) malloc (sizeof (unsigned char) * 256);
  if (lut == NULL)
    {
      perror ("malloc");
      exit (1);
    }

  {
    int n = 0;
    unsigned char val;
    while (fread (&val, 1, 1, map) != 0) {
      (*lut)[n] = val;
      n++;
    }
  }

  fseek (in, 1, SEEK_CUR);
  for (i = 0; i < size; i ++)
    {
      if (fread (*bytes + i, 1, 1, in) == 0)
	{
	  perror ("fread");
	  exit (1);
	}
    }
  fclose (in);

  {
      struct timeval start,stop;
      gettimeofday(&start,0);
      transfo (width, height, *bytes, *dest, *lut, 5);
      gettimeofday(&stop,0);
      fprintf(stderr,"%ld us\n",((stop.tv_sec-start.tv_sec) * 1000000) + stop.tv_usec-start.tv_usec);
  }


  out = fopen (av[3], "w");
  if (out == NULL)
    {
      perror ("fopen");
      exit (1);
    }
  fprintf (out, "P5\n");
  fprintf (out, "%d %d\n", height, width);
  fprintf (out, "%d\n", maxval);
  for (i = 0; i < size; i ++)
    {
      if (fwrite (*dest + i, 1, 1, out) == EOF)
	{
	  perror ("fwrite");
	  exit (1);
	}
    }
  fclose (out);
  }
}
