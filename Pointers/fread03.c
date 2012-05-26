/* For IEF - Ter@ops */

#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>
typedef unsigned char byte;

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

byte** bmatrix(int nrl, int nrh, int ncl, int nch)
/* ------------------------------------------------ */
/* allocate an uchar matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  int i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  byte **m;

  /* allocate pointers to rows */
  m = (byte **) malloc((size_t)((nrow+ncol)*sizeof(byte*)));
  for(i=nrl+1;i<=nrh;i++)
    m[i] = m[i-1]+ncol;
  /* return pointer to array of pointers to rows */
  return m;
 
}


void ReadPGMrow(FILE  *file, int width, byte  *line)
{
     fread(&(line[0]), sizeof(byte), width, file);
}

byte ** LoadPGM_bmatrix(char *filename, int *nrl, int *nrh, int *ncl, int *nch)
{
  int height = 1, width = 2;
  byte **m;
  FILE *file;
  char *buffer;
  int i;
  
  buffer = (char*) calloc(80, sizeof(char));
  file = fopen(filename,"rb");
  if (file==NULL)
    nrerror("ouverture du fichier impossible\n");

  // The code synthesis fails because of type_supporting_entities() which explodes the stack
  //readitem(file, buffer);
  if(strcmp(buffer, "P5") != 0)
    nrerror("entete du fichier %s invalide\n");

  //width  = atoi(readitem(file, buffer));
  //height = atoi(readitem(file, buffer));
  //gris   = atoi(readitem(file, buffer));

  *nrl = 0;
  *nrh = height - 1;
  *ncl = 0;
  *nch = width - 1;
  m = bmatrix(*nrl, *nrh, *ncl, *nch);
  
  for(i=0; i<height; i++) {
    ReadPGMrow(file, width, m[i]);
  }

  fclose(file);
  free(buffer);

  return m;
}
