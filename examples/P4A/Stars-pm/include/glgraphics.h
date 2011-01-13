#include "varglob.h"


#ifdef __cplusplus
 extern "C" {
#endif


void graphic_gldestroy();
void graphic_gldraw(int argc, char **argv, coord pos_[NP][NP][NP]);
void graphic_gldraw_histo(int argc_, char **argv_, int histo_[NP][NP][NP]);
void graphic_glupdate(coord pos_[NP][NP][NP]);

#ifdef __cplusplus
 }
#endif

