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

/* ............. tailles maxima du probleme simplexe .............. */
#define RMAXCOLONNES 48		/* variables */
#define RMAXLIGNES 90 /* contraintes */
/*#define RMAXCOLONNES 800 */		/* variables */
/*#define RMAXLIGNES 2000 *//* contraintes */
typedef struct rproblem
{ int nvar;    /* nombre de variables originales */
  int mcontr;  /* nombre de fonctions et contraintes originales */
  int nvpos;   /* nombre de variables originales contraintes (rangees en
		  tete). Usuellement =0 en calcul de dependances */
  FILE *ftrace;  /* numero "file system" pour "trace" */
  int ntrace;   /* niveau "debug" (execution accompagnee de commentaires)
		      0: aucune impression
		   >= 1: volume d'information croit avec le niveau
		      4: comportement probleme decrit en "latex" */
  int meth;     /* 0 primal simplex - 1 dual simplex */
  int base;
  int copie;  /* variables libres doublees */
  int testp;  /* pour evaluation perte precision */
  int tfrac ; /* utilisation marginale pour mise au point */
  int state,vc,vl,namev;
  /***************** tableaux de donnees du probleme ******************/
  int e[RMAXLIGNES+2];       /* nature contraintes
		      0 egalite
		      1 inequation <=
		     -1 inequation >=
		      1 ou 2 fonction a minimiser (ligne 0)
		     -1 ou -2 fonction a maximiser (ligne 0)
		      3 fonction d'accompagnement (non operationnel) */
  float d[RMAXLIGNES+2] ;      /* seconds membres */
  float a[RMAXLIGNES+2][RMAXCOLONNES+1] ;
/* ............... variables ou constantes de depart .............. */
  float eps ;
  int iname[RMAXLIGNES+RMAXCOLONNES+2];/*information application particuliere*/
      /************************* resultats ******************************/
  int vresult,iter ;
  float x[RMAXLIGNES+RMAXCOLONNES+2] ;
  float pi[RMAXLIGNES+RMAXCOLONNES+2] ;
  int rfrac ; /* information marginale pour mise au point */
      /************************* local data *****************************/
  int b[RMAXCOLONNES+1],g[RMAXLIGNES+2] ;	/* note g[0] inutilise */
  float inf[RMAXLIGNES+2] ;
  float tinfcolonn[RMAXCOLONNES+2] ; /* accede par macro */
  float tconor[RMAXLIGNES+RMAXCOLONNES+2];/*par macro,permet simuler conorm[-1]*/
  float *pconor,*pinfcolonn ; /* pointeurs pour tableaux par macros */
  int m,n,np ;
  int m2 /*,i0*/,j0;
  int nb,mb,tmax ;
  int sommaire ;	/* utilise comme booleen si tests sommaires */
  int cost,rhs ; /* pour reutilisation de tableau simplexe */
  float eps1,eps2,epss;/*eps1:precision,prec.col *eps2:pivotage*epss frac piv*/
} *Prproblem,rproblem;
/* ..................... resultat du solveur ....................... */
#define VRFIN 0	/* solution finie */
#define VRVID 1	/* polyedre vide */
#define VRINF 2	/* solution infinie */
#define VRINS 3	/* nombre de pivotages insuffisant, bouclage possible */
#define VRDEB 4	/* debordement de tableaux */
#define VRCAL 5	/* appel errone */
#define VRBUG 6	/* bug */
#define VROVF 7	/* overflow */
#define VREPS 8	/* pivot anormalement petit */
/*................................................................ */
