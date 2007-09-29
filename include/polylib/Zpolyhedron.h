#ifndef _Zpolyhedron_h_
#define _Zpolyhedron_h_

#if defined(__cplusplus)
extern "C" {
#endif

extern void CanonicalForm(ZPolyhedron *Zpol, ZPolyhedron **Result,
			  Matrix **Basis);
extern ZPolyhedron *EmptyZPolyhedron (int dimension);
extern ZPolyhedron *IntegraliseLattice (ZPolyhedron *A);
extern Bool isEmptyZPolyhedron (ZPolyhedron *Zpol);
extern ZPolyhedron *ZDomainDifference (ZPolyhedron *A, ZPolyhedron *B);
extern ZPolyhedron *ZDomainImage ( ZPolyhedron *A, Matrix *Func );
extern Bool ZDomainIncludes ( ZPolyhedron *A, ZPolyhedron *B );
extern ZPolyhedron *ZDomainIntersection ( ZPolyhedron *A, ZPolyhedron *B );
extern ZPolyhedron *ZDomainPreimage ( ZPolyhedron *A, Matrix *Func );
extern void ZDomainPrint(FILE *fp, const char *format, ZPolyhedron *A);
extern ZPolyhedron *ZDomainSimplify ( ZPolyhedron *ZDom );
extern ZPolyhedron *ZDomainUnion ( ZPolyhedron *A, ZPolyhedron *B );
extern ZPolyhedron *ZDomain_Copy ( ZPolyhedron *Head );
extern void ZDomain_Free ( ZPolyhedron *Head );
extern Bool ZPolyhedronIncludes ( ZPolyhedron *A, ZPolyhedron *B );
extern ZPolyhedron *ZPolyhedron_Alloc ( Lattice *Lat, Polyhedron *Poly );
extern ZPolyhedron *SplitZpolyhedron(ZPolyhedron *ZPol, Lattice *B);

#if defined(__cplusplus)
}
#endif

#endif /* _Zpolyhedron_h_ */
