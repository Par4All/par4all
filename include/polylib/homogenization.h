/** homogenization.h -- Bavo Nootaert **/
#ifndef HOMOGENIZATION_H
#define HOMOGENIZATTON_H

#include <polylib/polylib.h>

void dehomogenize_evalue(evalue *ep,  int nb_param);
void dehomogenize_enode(enode *p, int nb_param);
void dehomogenize_enumeration(Enumeration *en, int nb_param, int maxRays);
Polyhedron *dehomogenize_polyhedron(Polyhedron *p, int maxRays);

#endif
