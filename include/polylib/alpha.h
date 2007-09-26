#ifndef _alpha_h_
#define _alpha_h_

#if defined(__cplusplus)
extern "C" {
#endif

extern int GaussSimplify ( Matrix *M, Matrix *M2 );
extern int PolyhedronLTQ ( Polyhedron *P1, Polyhedron *P2, int INDEX, int
                           PDIM, int MAXRAYS );
extern int PolyhedronTSort ( Polyhedron ** L, unsigned int n, unsigned
                             int index, unsigned int pdim, int * time,
                             int * pvect, unsigned int MAXRAYS );
extern int Polyhedron_Not_Empty ( Polyhedron *P, Polyhedron *C, int
                                  MAXRAYS );
#if defined(__cplusplus)
}
#endif

#endif /* _alpha_h_ */
