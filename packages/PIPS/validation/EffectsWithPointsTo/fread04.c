/* For IEF - Ter@ops */

#include <stdio.h>

typedef unsigned char byte;

void ReadPGMrow(FILE  *file, int width, byte  *line)
{
     fread(&(line[0]), sizeof(byte), width, file);
}


byte ** LoadPGM_bmatrix(char *filename, int *nrl, int *nrh, int *ncl, int *nch)
{
  int height, width, gris;
  byte **m;
  FILE *file;
  int i;

  file = fopen(filename,"rb");
  if (file==NULL)
    nrerror("ouverture du fichier impossible\n");
  
  m = bmatrix(*nrl, *nrh, *ncl, *nch);
  
  for(i=0; i<height; i++) {
    ReadPGMrow(file, width, m[i]);
  }
  fclose(file);
 return m;
}
