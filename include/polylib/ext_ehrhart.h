#ifndef _EXT_EHRHART_H_
#define _EXT_EHRHART_H_

extern Enumeration *Domain_Enumerate(Polyhedron *D, Polyhedron *C, unsigned MAXRAYS, char **pn);

extern 
Enumeration *Domain_Image_Enumerate(Polyhedron *D,  Polyhedron *C,Matrix *T, unsigned MAXRAYS, char **par_name) ;

extern
void new_eadd(evalue *e1,evalue *res);

#endif
