#include "varglob.h"
#ifdef __cplusplus
extern "C" {
#endif
/* common/io.c */
void readDataFile(float mp[32][32][32], coord pos[32][32][32], coord vel[32][32][32], char *filename);
int writepart(char fname[], float field[]);
int writeparti(char fname[], int field[]);
void dumpfield(char fname[], float *field);
int writepos(char fname[], float *pos, float *vel, float time);
int dump_pos(coord pos[32][32][32], int npdt);
int read_pos(coord pos[32][32][32], int npdt, char *fname_);
/* sequential/pm.c */
void printUsage(int argc, char **argv);
int main(int argc, char **argv);
/* sequential/1-discretization.c */
void discretization(coord pos[32][32][32], int data[32][32][32]);
/* sequential/2-histogramme.c */
void histogram(int data[32][32][32], int histo[32][32][32]);
/* sequential/3-potential.c */
void potential_init_plan(float cdens[32][32][32][2]);
void potential_free_plan(void);
void potential(int histo[32][32][32], float dens[32][32][32], float cdens[32][32][32][2], float mp[32][32][32]);
/* sequential/4-updateforce.c */
void forcex(float pot[32][32][32], float fx[32][32][32]);
void forcey(float pot[32][32][32], float fx[32][32][32]);
void forcez(float pot[32][32][32], float fx[32][32][32]);
/* sequential/4-updatevel.c */
void updatevel(coord vel[32][32][32], float force[32][32][32], int data[32][32][32], int c, float dt);
/* sequential/6-updatepos.c */
void updatepos(coord pos[32][32][32], coord vel[32][32][32]);
#ifdef __cplusplus
}
#endif
