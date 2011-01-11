#include "varglob.h"
#include "kernel_tools.h"
extern "C" {
/* common/io.c */
void readDataFile(float mp[128][128][128], coord pos[128][128][128], coord vel[128][128][128], char *filename);
int writepart(char fname[], float field[]);
int writeparti(char fname[], int field[]);
void dumpfield(char fname[], float *field);
int writepos(char fname[], float *pos, float *vel, float time);
int dump_pos(coord pos[128][128][128], int npdt);
int read_pos(coord pos[128][128][128], int npdt, char *fname_);
/* common/glgraphics.c */
void *mainloop(void *unused);
void graphic_gldraw(int argc_, char **argv_, coord pos_[128][128][128]);
void graphic_gldraw_histo(int argc_, char **argv_, int histo_[128][128][128]);
void graphic_gldestroy(void);
void graphic_glupdate(coord pos_[128][128][128]);
}
/* sequential/pm.c */
void printUsage(int argc, char **argv);
int main(int argc, char **argv);
/* sequential/1-discretization.c */
void discretization(coord pos[128][128][128], int data[128][128][128]);
/* sequential/2-histogramme.c */
void histogram(int data[128][128][128], int histo[128][128][128]);
/* sequential/3-potential.c */
void potential_init_plan(float cdens[128][128][128][2]);
void potential_free_plan(void);
void potential(int histo[128][128][128], float dens[128][128][128], float cdens[128][128][128][2], float mp[128][128][128]);
/* sequential/4-updateforce.c */
void forcex(float pot[128][128][128], float fx[128][128][128]);
void forcey(float pot[128][128][128], float fx[128][128][128]);
void forcez(float pot[128][128][128], float fx[128][128][128]);
/* sequential/4-updatevel.c */
void updatevel(coord vel[128][128][128], float force[128][128][128], int data[128][128][128], int c, float dt);
/* sequential/6-updatepos.c */
void updatepos(coord pos[128][128][128], coord vel[128][128][128]);
