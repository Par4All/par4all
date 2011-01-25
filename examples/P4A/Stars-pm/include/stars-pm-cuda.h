#include "varglob.h"
#include "kernel_tools.h"
extern "C" {
/* common/io.c */
double get_time(void);
void readDataFile(float mp[NP][NP][NP], coord pos[NP][NP][NP], coord vel[NP][NP][NP], char *filename);
int writepart(char fname[], float field[]);
int writeparti(char fname[], int field[]);
void dumpfield(char fname[], float *field);
int writepos(char fname[], float *pos, float *vel, float time);
int dump_pos(coord pos[NP][NP][NP], int npdt);
int read_pos(coord pos[NP][NP][NP], int npdt, char *fname_);
/* common/glgraphics.c */
void *mainloop(void *unused);
void graphic_gldraw(int argc_, char **argv_, coord pos_[NP][NP][NP]);
void graphic_gldraw_histo(int argc_, char **argv_, int histo_[NP][NP][NP]);
void graphic_gldestroy(void);
void graphic_glupdate(coord pos_[NP][NP][NP]);
/* common/graphics.c */
void graphic_destroy(void);
void graphic_draw(int argc, char **argv, int histo[NP][NP][NP]);
}


/* sequential/pm.c */
void printUsage(int argc, char **argv);
int main(int argc, char **argv);
/* sequential/1-discretization.c */
void discretization(coord pos[NP][NP][NP], int data[NP][NP][NP]);
/* sequential/2-histogramme.c */
void histogram(int data[NP][NP][NP], int histo[NP][NP][NP]);
/* sequential/3-potential.c */
void potential_init_plan(float cdens[NP][NP][NP][2]);
void potential_free_plan(void);
void potential(int histo[NP][NP][NP], float dens[NP][NP][NP], float cdens[NP][NP][NP][2], float mp[NP][NP][NP]);
/* sequential/4-updateforce.c */
void forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]);
void forcey(float pot[NP][NP][NP], float fx[NP][NP][NP]);
void forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]);
/* sequential/4-updatevel.c */
void updatevel(coord vel[NP][NP][NP], float force[NP][NP][NP], int data[NP][NP][NP], int c, float dt);
/* sequential/6-updatepos.c */
void updatepos(coord pos[NP][NP][NP], coord vel[NP][NP][NP]);
