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
