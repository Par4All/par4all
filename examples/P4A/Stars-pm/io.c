#include <stdio.h>
#include <stdlib.h>
#include "io.h"

// FIXME .cpp ?


void readDataFile(unsigned int n,
                  float mp[NCELL][NCELL][NCELL],
                  coord pos[NCELL][NCELL][NCELL],
                  coord vel[NCELL][NCELL][NCELL],
                  char *filename) {
  unsigned int np = n;

  // Initialisation des particules depuis un fichier
#ifdef VERBOSE
  printf("Reading %d particules in %s\n",np, filename);
#endif

  FILE *ic;
  ic = fopen(filename, "r");

  // Test d'accès au fichier
  if(ic == NULL) {
    perror("Impossible d'ouvrir le fichier.");
    exit(-1);
  }

  // Nombre de particules dans le fichier
  if(1 != fread(&n, sizeof(int), 1, ic)) {
    perror("Erreur de lecture du nombre de particules dans le fichier.");
    fclose(ic);
    exit(-1);
  }
  // Initialise les tableaux de données

  if(n != np) {
    fprintf(stderr,
            "Input file contains %d particle while we expected %d, abort\n",
            n,
            np);
    fclose(ic);
    exit(-1);
  }


  // Masses
  if(np != fread(mp, sizeof(float), np, ic)) {
    perror("Erreur de lecture des masses dans le fichier.");
    fclose(ic);
    exit(-1);
  }

  np = 3 * np;
  // Positions
  if(np != fread(*pos, sizeof(float), np, ic)) {
    perror("Erreur de lecture des positions dans le fichier.");
    fclose(ic);
    exit(-1);
  }

  // Vélocité
  if(np != fread(*vel, sizeof(float), np, ic)) {
    perror("Erreur de lecture des vélocités dans le fichier.");
    fclose(ic);
    exit(-1);
  }

  // Fin de lecture du fichier
  fclose(ic);

#ifdef VERBOSE
  fprintf(stderr,"Lecture ok");
#endif

}

int writepart(char fname[], float field[]) {
  FILE *fp;
  int np = NPART;

  printf("dump part in %s\n", fname);

  fp = fopen(fname, "w");
  fwrite(&np, sizeof(int), 1, fp);
  fwrite(field, sizeof(float), NPART, fp);
  fclose(fp);

  return 0;

}

int writeparti(char fname[], int field[]) {
  FILE *fp;
  int np = NPART;

  printf("dump part in %s\n", fname);

  fp = fopen(fname, "w");
  fwrite(&np, sizeof(int), 1, fp);
  fwrite(field, sizeof(int), NPART, fp);
  fclose(fp);

  return 0;

}

void dumpfield(char fname[], float *field) {
  FILE *fp;
  int ncell = NCELL;
  float lbox = LBOX;

  printf("dumping field in %s\n", fname);

  fp = fopen(fname, "w");
  fwrite(&ncell, sizeof(int), 1, fp);
  fwrite(&lbox, sizeof(float), 1, fp);
  fwrite(field, sizeof(float), NCELL * NCELL * NCELL, fp);
  fclose(fp);
}

int writepos(int np, char fname[], float *pos, float *vel, float time) {
  FILE *fp;

  printf("dump part in %s\n", fname);

  fp = fopen(fname, "w");
  fwrite(&np, sizeof(int), 1, fp);
  fwrite(pos, sizeof(float), 3 * NPART, fp);
  fwrite(vel, sizeof(float), 3 * NPART, fp);
  fwrite(&time, sizeof(float), 1, fp);
  fclose(fp);

  return 0;

}

int dump_pos(coord pos[NCELL][NCELL][NCELL], int npdt) {
  FILE *fp;
  char fname[255];
  snprintf(fname,255,"out%d",npdt);
  printf("dump part in %s\n", fname);

  fp = fopen(fname, "w");
  fwrite(pos, sizeof(coord), NCELL*NCELL*NCELL, fp);
  fclose(fp);

  return 0;

}
int read_pos(coord pos[NCELL][NCELL][NCELL], int npdt, char *fname_) {
  FILE *fp;
  char static_fname[255];
  char *fname = static_fname;

  if(fname_) {
    fname = fname_;
  } else {
    snprintf(fname,255,"out%d",npdt);
  }

  fp = fopen(fname, "r");
  if(fp==NULL) {
    printf("No file named %s\n", fname);
    return 0;
  }

  printf("Reading %s\n", fname);
  fread(pos, sizeof(coord), NCELL*NCELL*NCELL, fp);
  fclose(fp);

  return 1;

}
