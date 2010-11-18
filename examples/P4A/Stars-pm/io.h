#ifndef IO_H_
#define IO_H_
#include "varglob.h"

#ifdef __cplusplus
 extern "C" {
#endif


int dump_pos(coord pos[NCELL][NCELL][NCELL], int npdt);
int read_pos(coord pos[NCELL][NCELL][NCELL], int npdt, char *fname_);

void readDataFile(unsigned int n,
                  float mp[NCELL][NCELL][NCELL],
                  coord pos[NCELL][NCELL][NCELL],
                  coord vel[NCELL][NCELL][NCELL],
                  char *filename);

int writepart(char fname[], float field[]);

int writeparti(char fname[], int field[]);

void dumpfield(char fname[], float *field);

int writepos(int np, char fname[], float *pos, float *vel, float time);

#ifdef __cplusplus
 }
#endif

#endif /*IO_H_*/
