/*       Polyhedron disjoint intersections
 */

/*
 union_disjointe computes the disjoint union of the given list of domains.
 input :
		(integer) # of polyhedra
		list of polyhedra in the usual matrix (constraints) format

 This version keeps the intersections in the result
*/

#include <stdio.h>
#include <stdlib.h>

#include <polylib/polylib.h>

#define WS 200

typedef struct LP_ { struct LP_ *next; Polyhedron *P; } LP;


/* Procedure to print constraints of a polyhedron */
void AffContraintes(Polyhedron *p) {
  
  int i,j;
  for( ;p;p=p->next){
    
    printf( "%d %d\n", p->NbConstraints, p->Dimension+2 );
    for(i=0;i<p->NbConstraints;i++) {
      for(j=0;j<p->Dimension+2;j++ )
	printf(P_VALUE_FMT,p->Constraint[i][j]);
      printf("\n");
    }
    printf("\n");
  }
}


int main() {
  
	int np, i;

	Matrix *a;
	LP *P, *lP;
	LP *Result, *lR, *tmp;
	Polyhedron *reste;
	Polyhedron *d1,*d2,*dx;

	scanf( "%d", &np );

	P = Result = NULL;
	for(i=0;i<np;++i) {
	 a = Matrix_Read();
	 lP = (LP *) malloc(sizeof(LP));
	 lP->next = P;
	 lP->P = Constraints2Polyhedron(a,WS);
	 Matrix_Free(a);
	 P = lP;
	}


	for(lP=P;lP;lP=lP->next)
	{
    reste = lP->P;
    
    /* Intersection avec chacun des domaines deja trouves (dans Result) */
    for(lR=Result;lR && reste;lR=lR->next)
	 {
      dx = PDomainIntersection(reste,lR->P,WS);
      if(!dx) continue;
      if (emptyQ(dx)) {	
			Domain_Free(dx);
			continue;
      }
      
      d1 = PDomainDifference(reste,lR->P,WS);	/* dans reste */
      d2 = PDomainDifference(lR->P,reste,WS);	/* dans lR->P */
      
      if(!d1 || emptyQ(d1)) {
			if(d1)
			  Domain_Free(d1);
	
			if(!d2 || emptyQ(d2)) {
	  
			  /* d2 = d1 = vide. dx = reste */
			  /* dx est le courant. */
			  /* on ne fait rien. */
			}
			else {
	  
			  /* ajoute l'intersection en tete : */
			  tmp = (LP *)malloc(sizeof(LP));
			  tmp->next = Result;
			  tmp->P = dx;
			  Result = tmp;
	  
			  /* remplace le courant par d2 */
			  lR->P = d2;
			}
			reste = NULL;
      }
      else {
			if(!d2 || emptyQ(d2)) {
			  if(d2)
	   		 Domain_Free(d2);

			  /* remplace le courant par dx. */
			  lR->P = dx;
			  reste = d1;
			}
			else {

			  /* ajoute d2 en tete */
			  tmp = (LP *)malloc( sizeof(LP) );
			  tmp->next = Result;
			  tmp->P = d2;
			  Result = tmp;

			  /* remplace le courant par dx. */
			  lR->P = dx;	  
			  reste = d1;
			}
      }
    }  /* fin intersection result précédent */
    
    if(reste)
      if(!emptyQ(reste))
		{
			Polyhedron *r, *rn;
			lR = (LP *)malloc(sizeof(LP));
			lR->next = Result;
			lR->P = reste;
			r = lR->P->next;
			lR->P->next = NULL;
			Result = lR;
			/* ajoute la fin du reste dans lp->next */
			while( r )
			{
				lR = (LP *)malloc(sizeof(LP));
				lR->next = lP->next;
				lP->next = lR;

				rn = r->next;
				r->next = NULL;
				lR->P = r;
				r = rn;
			}
      }
	}


	printf( "######################################################\n" );
	for(lR = Result;lR;lR=lR->next) {
	 AffContraintes(lR->P);

	 /* Polyhedron_Print(stdout,P_VALUE_FMT,lR->P); */
	}
	printf( "######################################################\n" );

	/* free's.... :-) */

	return 0;
}



