
/*
 union_convex computes the convex hull of the list of domains given as
 input :
		(integer) # of polyhedra
		list of polyhedra in the usual matrix (constraints) format
*/

#include <stdio.h>
#include <stdlib.h>

#include <polylib/polylib.h>

#define WS 200

int main() {
	
  int np, i;
  
  Matrix *a;
  Polyhedron *P, *P1;
  Polyhedron *C, *D;
  
  scanf("%d", &np);
  
  P = NULL;
  
  for(i=0;i<np;++i) {
		
    a = Matrix_Read();
    P1 = Constraints2Polyhedron(a,WS);
    if(!P)
      P = Empty_Polyhedron(P1->Dimension);
    P = DomainUnion(P,P1,WS);
    Polyhedron_Free(P1);
    Matrix_Free(a);
  }
  
  C = DomainConvex(P,WS);
  
  D = DomainDifference(C,P,WS);
  if(D)
    if(!emptyQ(D)) {
      printf("WARNING: ca marche pas!!!!\n");
      printf("Voici le sous domaine de l'union non couvert:\n");
      Polyhedron_Print(stdout,P_VALUE_FMT,D);
    }
  
  printf("---------------------------------------------\n");
  Polyhedron_Print(stdout,P_VALUE_FMT,C);
  
  /* free.... :-) */
  
  return 0;
}






