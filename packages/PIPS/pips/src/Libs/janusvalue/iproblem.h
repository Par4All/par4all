/* ========================================================================= */
/*                       SIMPLEXE for integer variables                      */
/*                            ALL-INTEGER METHODS                            */
/*                             Jean Claude SOGNO                             */
/*                     Projet CHLOE -- INRIA ROCQUENCOURT                    */
/*                                Juin 1994                                  */
/* ========================================================================= */

/* ========================================================================= */
/*                             Duong NGUYEN QUE                              */
/*                 Adaption to abstract computation: janusvalue              */
/*                               CRI-ENSMP                                   */
/* ========================================================================= */

typedef struct problem
{ int result;
  int jinfini; /* column in case of infinite vale of function */
  int nvar;    /* nombre de variables originales */
  int mcontr;  /* nombre de fonctions et contraintes originales */
  int nvpos;   /* nombre de variables originales contraintes (rangees en
		  tete). Usuellement =0 en calcul de dependances */
  FILE *ftrace;  /* numero "file system" pour "trace" */
  /**** options solveur * entre parentheses: choix "recommande".
              Parfois plusieurs choix plus ou moins equivalents */
  int dyn;
  int dyit;
  int ntrace;   /* niveau "trace" (execution accompagnee de commentaires)
		      0: aucune impression
		   >= 1: volume d'information croit avec le niveau
		      4: comportement probleme decrit en "latex" */
  int ntrac2;
  int ntrac3;
  int fourier;  /* (3) elimination FOURIER-MOTZKIN
		      0: elimination non appliquee
		      1: cas triviaux (
		   >= 2: lorsque le nombre d'inequations decroit
		   >= 3: lorsque le nombre d'inequations n'augmente pas */
  int varc ;    /* (0-1) introduction variables contraintes
		      0 "strategie simplexe"
		      1 pivot unitaire sinon "strategie simplexe"
		      2 pivot unitaire sinon variable unique ajoutee
		   >= 3 mise au point */
  int forcer;  /* in any case non negative variables */
  int remove;  /* redundant inequalities are removed */
  int meth;     /* (0-1) methode finale de resolution */
  int met2,met3,met4,met5,met6,met7,met8;
  int turbo,critermax;
  int choixpiv; /* (0-1) choix du pivot (simplexe dual)
		   0 plus petit pivot
		   1 plus grand pivot */
  int choixprim; /* (-) choix du pivot (simplexe primal)*/

#define MAXCOLONNES 200 /*108   variables 10-10 */
#define MAXLIGNES 500 /*250    contraintes 10-10 */
#define MAXNX MAXCOLONNES+9
#define MAXMX MAXLIGNES+9
#define AKCOLONNES MAXNX+1
#define AKLIGNES MAXMX+1
  /***************** tableaux de donnees du probleme ******************/
        /* nature contraintes
		      0 egalite
		      1 inequation <=
		     -1 inequation >=
		      2 fonction a minimiser
		     -2 fonction a maximiser
		      3 fonction d'accompagnement */
  /*int e[MAXLIGNES+1] ;*/
  /*int d[MAXLIGNES+1] ;*/              /* rhs */
  /*int a[MAXLIGNES+1][MAXCOLONNES+1] ;*/ /* matrix coefficients */
/* ........... local data .............................................*/
  int negal;
  int icout;
  int minimum;
  int mx,nx,ic1;
  int tmax;
  int niter;
  int itdirect,nredun;
  int numero,numax;//Value?DN
  int lastfree;
  int nub;
  int ntp;
  int vdum;
  int nturb;
  Value ak[AKLIGNES][AKCOLONNES] ;
  Value dk[AKLIGNES] ;
  int b[AKCOLONNES+1];
  int g[AKLIGNES];/* note g[0] inutilise */ 
  int vredun[AKLIGNES];   
  Value ibound[AKCOLONNES+1+AKLIGNES];
  Value ilbound[AKCOLONNES+1+AKLIGNES];
  int utilb[AKCOLONNES+1+AKLIGNES];
  float rbound[AKCOLONNES+1+AKLIGNES];
  float rlbound[AKCOLONNES+1+AKLIGNES];
  int frequency[AKCOLONNES+1+AKLIGNES];
  Value tturb[30][AKCOLONNES] ;//DN
  Value dturb[30];//DN
  int jturb[30];
  int state,tilt,marque,repere,lu,cu,vbest,bloquer;
  Value decrement,bestlow,bestup;
  float rrbest;
/* ........... visualization local data ....................................*/
  int nstep,majstep;
/********** temporary data *************************************/
  int itemp; float ftemp;
/********** pour statistiques *************************************/
      /*int croix;*/
  int mcdeb, nxdeb, mcfin, nxfin;
  int niveau;
} *Pproblem,problem;
#define MTDAI 0	/* methode DUAL ALL INTEGERS */
#define MTDI2 1 /* methode surrogate uniforme dual all integers */
/* ..................... resultat du solveur ....................... */
#define VRFIN 0	/* solution finie */
#define VRVID 1	/* polyedre vide */
#define VRINF 2	/* solution infinie */
#define VRINS 3	/* nombre de pivotages insuffisant, bouclage possible */
#define VRDEB 4	/* debordement de tableaux */
#define VRCAL 5	/* appel errone */
#define VRBUG 6	/* bug */
#define VROVF 7	/* overflow */
typedef struct initpb
{ 
  Value a[MAXLIGNES+1][MAXCOLONNES+1] ; /* matrix coefficients */
  Value d[MAXLIGNES+1] ;              /* rhs */
  int e[MAXLIGNES+1] ;
} *Pinitpb,initpb;
