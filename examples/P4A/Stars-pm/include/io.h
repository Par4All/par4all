#ifndef IO_H_
#define IO_H_
#include "varglob.h"

#ifdef __cplusplus
 extern "C" {
#endif


int dump_pos(coord pos[NP][NP][NP], int npdt);
int read_pos(coord pos[NP][NP][NP], int npdt, char *fname_);

void readDataFile(float mp[NP][NP][NP],
                  coord pos[NP][NP][NP],
                  coord vel[NP][NP][NP],
                  char *filename);

int writepart(char fname[], float field[]);

int writeparti(char fname[], int field[]);

void dumpfield(char fname[], float *field);

int writepos(char fname[], float *pos, float *vel, float time);

#ifdef __cplusplus
 }
#endif

#endif /*IO_H_*/
