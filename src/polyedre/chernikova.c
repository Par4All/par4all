
#include <malloc.h> 
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"

/* IRISA  data structures.
 */
#include "polylib/polylib.h"

/* maximum number of rays allowed in chernikova... (was 20000)
 * it does not look a good idea to move the limit up, as it
 * makes both time and memory consumption to grow a lot.
 */
#define MAX_NB_RAYS (20000)

/* Irisa is based on int. We would like to change this to 
 * some other type, say "long long" if desired, as VALUE may
 * also be changed. It is currently an int. Let us assume
 * that the future type will be be called "IRINT" (Irisa Int)
 */
/*
#define VALUE_TO_IRINT(val) VALUE_TO_INT(val)
#define IRINT_TO_VALUE(i) ((Value)i)
*/

#define VALUE_TO_IRINT(val) (val)
#define IRINT_TO_VALUE(i) (i)

/* should be ANSI C headers...
 */
/*
extern Matrix * Matrix_Alloc();
extern void Matrix_Print();
extern void Matrix_Free();
extern void Polyhedron_Print();
extern Polyhedron * Constraints2Polyhedron();
extern Polyhedron * Rays2Polyhedron();
extern Matrix * Polyhedron2Constraints();
extern void Polyhedron_Free();
*/

/*  Fonctions de conversion traduisant une ligne de la structure 
 * Matrix de l'IRISA en un Pvecteur
 */

static Pvecteur matrix_ligne_to_vecteur(mat,i,base,dim)
Matrix *mat;
int i;
Pbase base;
int dim;
{
    int j;
    Pvecteur pv,pvnew = NULL;
    boolean NEWPV = TRUE;

    for (j=1,pv=base ;j<dim;j++,pv=pv->succ) {
	if (mat->p[i][j]) 
	    if (NEWPV) { pvnew= vect_new(vecteur_var(pv),
					 IRINT_TO_VALUE(mat->p[i][j]));
			 NEWPV =FALSE;  }		
	    else vect_add_elem(&pvnew,vecteur_var(pv),
			       IRINT_TO_VALUE(mat->p[i][j]));
    }
    return(pvnew);
}



/*  Fonctions de conversion traduisant une ligne de la structure 
 * Matrix de l'IRISA en une Pcontrainte
*/


static Pcontrainte matrix_ligne_to_contrainte(mat,i,base)
Matrix * mat;
int i;
Pbase base;
{
    Pcontrainte pc=NULL;
    int dim = vect_size(base) +1;

    Pvecteur pvnew = matrix_ligne_to_vecteur(mat,i,base,dim);
    vect_add_elem(&pvnew,TCST,IRINT_TO_VALUE(mat->p[i][dim]));
    vect_chg_sgn(pvnew);
    pc = contrainte_make(pvnew);
    return (pc);
}


/*  Fonctions de conversion traduisant une ligne de la structure 
 *  Polyhedron de l'IRISA en un Pvecteur
*/


static Pvecteur polyhedron_ligne_to_vecteur(pol,i,base,dim)
Polyhedron *pol;
int i;
Pbase base;
int dim;
{
    int j;
    Pvecteur pv,pvnew = NULL;
    boolean NEWPV = TRUE;

    for (j=1,pv=base ;j<dim;j++,pv=pv->succ) {
	if (pol->Ray[i][j]) 
	    if (NEWPV) { pvnew= vect_new(vecteur_var(pv),
					 IRINT_TO_VALUE(pol->Ray[i][j]));
			 NEWPV =FALSE;  }		
	    else vect_add_elem(&pvnew,vecteur_var(pv),
			       IRINT_TO_VALUE(pol->Ray[i][j]));
    }
    return(pvnew);
}


/*  Fonctions de conversion traduisant une Pray_dte en une 
 * ligne de la structure matrix de l'IRISA 
 */ 
static void ray_to_matrix_ligne(pr,mat,i,base)
Pray_dte pr;
Matrix *mat;
int i;
Pbase base;
{
    Pvecteur pb;
    int j;
    Value v;

    for (pb = base, j=1; 
	 !VECTEUR_NUL_P(pb) && j< mat->NbColumns-1; 
	 pb = pb->succ,j++)
    {
	v = vect_coeff(vecteur_var(pb),pr->vecteur);
	mat->p[i][j] = VALUE_TO_IRINT(v);
    }
}


/*  Fonctions de conversion traduisant une Pcontrainte en une 
 * ligne de la structure matrix de l'IRISA 
 */
static void contrainte_to_matrix_ligne(pc,mat,i,base)
Pcontrainte pc;
Matrix *mat;
int i;
Pbase base;
{
    Pvecteur pv;
    int j;
    Value v;

    for (pv=base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
    {
	v = value_uminus(vect_coeff(vecteur_var(pv),pc->vecteur));
	mat->p[i][j] = VALUE_TO_IRINT(v);
    }

    v = value_uminus(vect_coeff(TCST,pc->vecteur));
    mat->p[i][j]= VALUE_TO_IRINT(v);
}



/* Passage du systeme lineaire sc a une matrice matrix (structure Irisa)
 * Cette fonction de conversion est utilisee par la fonction 
 * sc_to_sg_chernikova
 */ 
static void sc_to_matrix(sc,mat)
Psysteme sc;
Matrix *mat;
{

    int nbrows, nbcolumns, i, j;
    Pcontrainte peq;
    Pvecteur pv;

    nbrows = mat->NbRows;
    nbcolumns = mat->NbColumns;

    /* Differentiation equations and inequations */
    for (i=0; i<=sc->nb_eq-1;i++)
	mat->p[i][0] =0;
    for (; i<=nbrows-1;i++)
	mat->p[i][0] =1;

    /* Matrix initialisation */
    for (peq = sc->egalites,i=0;
	 !CONTRAINTE_UNDEFINED_P(peq);
	 contrainte_to_matrix_ligne(peq,mat,i,sc->base),
	 peq=peq->succ,i++);
    for (peq =sc->inegalites;
	 !CONTRAINTE_UNDEFINED_P(peq);
	 contrainte_to_matrix_ligne(peq,mat,i,sc->base),
	 peq=peq->succ,i++);

    for (pv=sc->base,j=1;!VECTEUR_NUL_P(pv);
	 mat->p[i][j] = 0,pv=pv->succ,j++);
    mat->p[i][j]=1;
}

/* Fonction de conversion traduisant un systeme generateur sg 
 * en une matrice de droites, rayons et sommets utilise'e par la 
 * structure  de Polyhedron de l'Irisa.
 * Cette fonction de conversion est utilisee par la fonction 
 * sg_to_sc_chernikova
*/ 


static void sg_to_polyhedron(sg,mat)
Ptsg sg;
Matrix * mat;
{

    Pray_dte pr;
    Psommet ps;
    int  nbC=0;
    int nbcolumns = mat->NbColumns;

    /* Traitement des droites */
    if (sg_nbre_droites(sg)) {
	pr = sg->dtes_sg.vsg;
	for (pr = sg->dtes_sg.vsg; pr!= NULL; 
	     mat->p[nbC][0] = 0, 
	     mat->p[nbC][nbcolumns-1] =0,
	     ray_to_matrix_ligne(pr,mat,nbC,sg->base),
	     nbC++,
	     pr = pr->succ); 
    } 
    /* Traitement des rayons */
    if (sg_nbre_rayons(sg)) {
	pr =sg->rays_sg.vsg;
	for (pr = sg->rays_sg.vsg; pr!= NULL; 
	     mat->p[nbC][0] = 1, 
	     mat->p[nbC][nbcolumns-1] =0,
	     ray_to_matrix_ligne(pr,mat,nbC,sg->base),
	     nbC++,
	     pr = pr->succ);
    } 
    /* Traitement des sommets */
    if (sg_nbre_sommets(sg)) {
	for (ps = sg->soms_sg.ssg; ps!= NULL; 
	     mat->p[nbC][0] = 1, 
	     mat->p[nbC][nbcolumns-1] = VALUE_TO_IRINT(ps->denominateur),
	     ray_to_matrix_ligne(ps,mat,nbC,sg->base),
	     nbC++,
	     ps = ps->succ);
    }
}

/* Fonction de conversion traduisant une matrix structure Irisa 
 * sous forme d'un syste`me lineaire sc. Cette fonction est 
 * utilisee paar la fonction sg_to_sc_chernikova 
 */ 
static void matrix_to_sc(mat,sc)
Matrix *mat;
Psysteme sc;
{
    Pcontrainte pce=NULL;
    Pcontrainte pci=NULL;
    Pcontrainte pc_tmp=NULL;
    boolean neweq = TRUE;
    boolean newineq = TRUE;
    int i,nbrows;
    int nbeq=0;
    int nbineq=0;

    /* Premiere droite */
    if ((nbrows= mat->NbRows)) {
	for (i=0; i<nbrows; i++) {
	    switch (mat->p[i][0]) {
	    case 0:
		nbeq ++;
		if (neweq) { 
		    pce= pc_tmp  = 
			matrix_ligne_to_contrainte(mat, i, sc->base);
		    neweq = FALSE;} 
		else {
		    pc_tmp->succ = 
			matrix_ligne_to_contrainte(mat, i, sc->base);
		    pc_tmp = pc_tmp->succ;	}
		break;
	    case 1:
		nbineq++;
		if (newineq) {pci = pc_tmp =
				  matrix_ligne_to_contrainte(mat,
							     i,sc->base); 
				  newineq = FALSE; } 
		else {
		    pc_tmp->succ = matrix_ligne_to_contrainte(mat,
							      i,sc->base);
		    pc_tmp = pc_tmp->succ;}
		break;
	    default:
		printf("error in matrix interpretation in Matrix_to_sc\n");
		break;
	    }
	}
	sc->nb_eq = nbeq;
	sc->egalites = pce;
	sc->nb_ineq = nbineq;
	sc->inegalites = pci;
    }
}



/* Fonction de conversion traduisant un polyhedron structure Irisa 
 * sous forme d'un syste`me ge'ne'rateur. Cette fonction est 
 * utilisee paar la fonction sc_to_sg_chernikova 
 */ 


static void polyhedron_to_sg(pol,sg)
Polyhedron  *pol;
Ptsg sg;
{

    Pray_dte ldtes_tmp=NULL,ldtes = NULL;
    Pray_dte lray_tmp=NULL,lray = NULL;
    Psommet lsommet_tmp=NULL,lsommet=NULL;
    Stsg_vects dtes,rays;
    Stsg_soms sommets;
    Pvecteur pvnew;
    int i;
    int nbsommets =0;
    int  nbrays=0;
    int dim = vect_size(sg->base) +1;
    boolean newsommet = TRUE;
    boolean newray = TRUE;
    boolean newdte = TRUE;

    for (i=0; i< pol->NbRays; i++) {
	switch (pol->Ray[i][0]) {
	case 0:
	    /* Premiere droite */
	    pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
	    if (newdte) {
		ldtes_tmp= ldtes = ray_dte_make(pvnew);
		newdte = FALSE; 
	    } else {
		/* Pour chaque droite suivante */
		ldtes_tmp->succ = ray_dte_make(pvnew);
		ldtes_tmp =ldtes_tmp->succ;
	    }
	    break;
	case 1:
	    switch (pol->Ray[i][dim]) {
	    case 0:
		nbrays++;
		/* premier rayon */
		pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
		if (newray) {
		    lray_tmp = lray = ray_dte_make(pvnew);
		    newray = FALSE;
		} else {
		    lray_tmp->succ= ray_dte_make(pvnew);
		    lray_tmp =lray_tmp->succ;    }
		break;
	    default:
		nbsommets ++;
		pvnew = polyhedron_ligne_to_vecteur(pol,i,sg->base,dim);
		if (newsommet) {
		    lsommet_tmp=lsommet=
			sommet_make(IRINT_TO_VALUE(pol->Ray[i][dim]),
				    pvnew);
		    newsommet = FALSE;
		} else {
		    lsommet_tmp->succ=
			sommet_make(IRINT_TO_VALUE(pol->Ray[i][dim]),
				    pvnew);
		    lsommet_tmp = lsommet_tmp->succ;
		}
		break;
	    }
	    break;

	default: printf("error in rays elements \n");
	    break;
	}
    }
    if (nbsommets) {
	sommets.nb_s = nbsommets;
	sommets.ssg = lsommet;
	sg->soms_sg = sommets;  }
    if (nbrays) {
	rays.nb_v = nbrays;
	rays.vsg=lray;
	sg->rays_sg = rays;    }
    if (pol->NbBid) {
	dtes.vsg=ldtes;
	dtes.nb_v=pol->NbBid;
	sg->dtes_sg = dtes;
    }
}

/* Fonction de conversion d'un systeme lineaire a un systeme generateur
 * par chenikova
*/

Ptsg  sc_to_sg_chernikova(sc)
Psysteme sc;
{
    Matrix *a;
    int nbrows = 0;
    int nbcolumns = 0;
    Ptsg sg = sg_new();
    Polyhedron *A;

    /* mem_spy_begin(); */

    assert(!SC_UNDEFINED_P(sc) && (sc_dimension(sc) != 0));

    nbrows = sc->nb_eq + sc->nb_ineq + 1;
    nbcolumns = sc->dimension +2;
    sg->base = base_dup(sc->base);
    a = Matrix_Alloc(nbrows, nbcolumns);
    sc_to_matrix(sc,a);
    
    /*  printf("\na =");
      Matrix_Print(stderr, "%4d", a); */
      
    A = Constraints2Polyhedron(a, MAX_NB_RAYS);
    Matrix_Free(a);

/*    Polyhedron_Print(stderr, "%4d",A);
*/
    polyhedron_to_sg(A,sg);
    Polyhedron_Free(A);
    /*   printf(" systeme generateur\n");
	 sg_fprint(stdout,sg,variable_dump_name);
	 */

    /* mem_spy_end("sc_to_sg_chernikova"); */

    return(sg);
}

/* Fonction de conversion d'un systeme generateur a un systeme lineaire.
 * par chernikova
*/

Psysteme sg_to_sc_chernikova(sg)
Ptsg sg;
{


    int nbrows = sg_nbre_droites(sg)+ sg_nbre_rayons(sg)+sg_nbre_sommets(sg);
    int nbcolumns = base_dimension(sg->base)+2;
    Matrix *a;
    Psysteme sc= sc_new();
    Polyhedron *A;

    sc->base = base_dup(sg->base);
    sc->dimension = vect_size(sc->base);

    if (sg_nbre_droites(sg)+sg_nbre_rayons(sg)+sg_nbre_sommets(sg)) {

	a = Matrix_Alloc(nbrows, nbcolumns);
	sg_to_polyhedron(sg,a);	 
   
	/* printf("\na =");
	Matrix_Print(stderr, "%4d,", a); */

	A = Rays2Polyhedron(a, MAX_NB_RAYS);
	Matrix_Free(a);

/*	Polyhedron_Print(stderr, "%4d",A);
*/
	a= Polyhedron2Constraints(A);
	Polyhedron_Free(A);
	/* printf("\na =");
		Matrix_Print(stderr, "%4d", a); 	*/

	matrix_to_sc(a,sc);
	Matrix_Free(a);
	sc=sc_normalize(sc);
	if (sc == NULL) {
	    Pcontrainte pc = contrainte_make(vect_new(TCST, VALUE_ONE));
	    sc = sc_make(pc, CONTRAINTE_UNDEFINED);
	    sc->base = base_dup(sg->base);
	    sc->dimension = vect_size(sc->base);	}

    }
    else {
	sc->egalites = contrainte_make(vect_new(TCST,VALUE_ONE));
	sc->nb_eq ++;
    }	


    /*  printf(" impression du systeme \n"); 
	sc_dump(sc); */

    return(sc);
}


Psysteme sc_convex_hull(sc1,sc2)
Psysteme sc1,sc2;
{

    Matrix *a1,*a2;
    int nbrows1 = 0;
    int nbcolumns1 = 0;
    int nbrows2 = 0;
    int nbcolumns2 = 0;
    Polyhedron *A1,*A2;
    Matrix *a;
    Psysteme sc= sc_new();
    Polyhedron *A;
    int i1,i2,j;
    int Dimension,cp;

    /* mem_spy_begin(); */

    assert(!SC_UNDEFINED_P(sc1) && (sc_dimension(sc1) != 0));
    assert(!SC_UNDEFINED_P(sc2) && (sc_dimension(sc2) != 0));
    
    ifscdebug(7) {
	fprintf(stderr, "[sc_convex_hull] considering:\n");
	sc_default_dump(sc1);
	sc_default_dump(sc2);
    }

    /* comme on il ne faut pas que les structures irisa 
       apparaissent dans le fichier polyedre.h, une sous-routine 
       renvoyant un polyhedron n'est pas envisageable.
       Le code est duplique*/

    nbrows1 = sc1->nb_eq + sc1->nb_ineq + 1;
    nbcolumns1 = sc1->dimension +2;
    a1 = Matrix_Alloc(nbrows1, nbcolumns1);
    sc_to_matrix(sc1,a1);

    nbrows2 = sc2->nb_eq + sc2->nb_ineq + 1;
    nbcolumns2 = sc2->dimension +2;
    a2 = Matrix_Alloc(nbrows2, nbcolumns2);
    sc_to_matrix(sc2,a2);

    ifscdebug(8) {
	fprintf(stderr, "[sc_convex_hull]\na1 =");
	Matrix_Print(stderr, "%4d",a1);
	fprintf(stderr, "\na2 =");
	Matrix_Print(stderr, "%4d",a2);
    }
    
    A1 = Constraints2Polyhedron(a1, MAX_NB_RAYS);
    Matrix_Free(a1); 
    A2 = Constraints2Polyhedron(a2, MAX_NB_RAYS);
    Matrix_Free(a2);

    ifscdebug(8) {
	fprintf(stderr, "[sc_convex_hull]\nA1 =");
	Polyhedron_Print(stderr, "%4d",A1);
	fprintf(stderr, "\nA2 =");
	Polyhedron_Print(stderr, "%4d",A2);
    }
    
    sc->base = base_dup(sc1->base);
    sc->dimension = vect_size(sc->base);

    if (A1->NbRays == 0) {
	a= Polyhedron2Constraints(A2); 
    } else  if (A2->NbRays == 0) {
	a= Polyhedron2Constraints(A1); 
    } else {
	int i1p;
	int cpp;

	Dimension = A1->Dimension+2;
	a = Matrix_Alloc(A1->NbRays + A2->NbRays,Dimension);

	/* Tri des contraintes de A1->Ray et A2->Ray, pour former 
	   l'union de ces contraintes dans un meme format 
	   Line , Ray , vertex */
	cp = 0;
	i1 = 0;
	i2 = 0;
	while (i1 < A1->NbRays && A1->Ray[i1][0] ==0) {
	    for (j=0; j < Dimension ; j++)
		a->p[cp][j] = A1->Ray[i1][j]; 
	    cp++; i1++; 
	}
	/*
	while (i2 < A2->NbRays && A2->Ray[i2][0] ==0) {
	    boolean equal_rays = FALSE;
	    for(i1p = 0, cpp=0; i1p < A1->NbRays && A1->Ray[i1p][0] ==0
		&& !equal_rays; i1p++, cpp++) {
		equal_rays = TRUE;
		for (j=0 ; j < Dimension && equal_rays ; j++) {
		    equal_rays = (a->p[cpp][j] == A2->Ray[i2][j]);
		}
	    }
	    if(!equal_rays) {
		for (j=0 ; j < Dimension ; j++) 
		    a->p[cp][j] = A2->Ray[i2][j];
		cp++; i2++; }
	}
	*/
	while (i2 < A2->NbRays && A2->Ray[i2][0] ==0) {
	    for (j=0 ; j < Dimension ; j++) 
		a->p[cp][j] = A2->Ray[i2][j];
	    cp++; i2++;
	}

	i1p = i1;
	cpp = cp;
	while (i1 < A1->NbRays && A1->Ray[i1][0] ==1 
	       && A1->Ray[i1][Dimension-1]==0) {
	    for (j=0; j < Dimension ; j++) 
		a->p[cp][j] = A1->Ray[i1][j]; 
	    cp++; i1++; 
	}

	/*
	while (i2 < A2->NbRays && A2->Ray[i2][0] == 1 
	       && A2->Ray[i2][Dimension-1]==0) {
	    boolean equal_rays = FALSE;
	    for(; i1p < A1->NbRays && A1->Ray[i1p][0] == 1 
		&& A1->Ray[i1p][Dimension-1]==0
		&& !equal_rays; i1p++, cpp++) {
		equal_rays = TRUE;
		for (j=0 ; j < Dimension && equal_rays ; j++) {
		    equal_rays = (a->p[cpp][j] == A2->Ray[i2][j]);
		}
	    }

	    if(!equal_rays) {
		for (j=0; j < Dimension ; j++) 
		    a->p[cp][j] = A2->Ray[i2][j]; 
		cp++; i2++;
	    }
	}
	*/
	while (i2 < A2->NbRays && A2->Ray[i2][0] == 1 
	       && A2->Ray[i2][Dimension-1]==0) {
	    for (j=0; j < Dimension ; j++) 
		a->p[cp][j] = A2->Ray[i2][j]; 
	    cp++; i2++;
	}

	i1p = i1;
	cpp = cp;
	while (i1 < A1->NbRays && A1->Ray[i1][0] == 1 
	       && A1->Ray[i1][Dimension-1]!= 0) {
	    for (j=0; j < Dimension ; j++)  
		a->p[cp][j] = A1->Ray[i1][j]; 
	    cp++; i1++; 
	}
	/*
	while (i2 < A2->NbRays && A2->Ray[i2][0] == 1 
	       && A2->Ray[i2][Dimension-1]!=0) {

	    boolean equal_rays = FALSE;
	    for(; i1p < A1->NbRays && A1->Ray[i1p][0] == 1 
		&& A1->Ray[i1p][Dimension-1]!=0
		&& !equal_rays; i1p++, cpp++) {
		equal_rays = TRUE;
		for (j=0 ; j < Dimension && equal_rays ; j++) {
		    equal_rays = (a->p[cpp][j] == A2->Ray[i2][j]);
		}
	    }

	    if(!equal_rays) {
		for (j=0; j < Dimension ; j++)  
		    a->p[cp][j] = A2->Ray[i2][j]; 
		cp++;  i2++;
	    }
	}
	*/
	while (i2 < A2->NbRays && A2->Ray[i2][0] == 1 
	       && A2->Ray[i2][Dimension-1]!=0) {
	    for (j=0; j < Dimension ; j++)  
		a->p[cp][j] = A2->Ray[i2][j]; 
	    cp++;  i2++;
	}
  
	Polyhedron_Free(A1);
	Polyhedron_Free(A2);
	
	/*    printf("\na =");
		Matrix_Print(stderr, "%4d,",a); */

	A = Rays2Polyhedron(a, MAX_NB_RAYS);
/*	Polyhedron_Print(stderr, "%4d",A);
*/

	Matrix_Free(a);
	a = Polyhedron2Constraints(A);    
	Polyhedron_Free(A);
    }

   /*    printf("\na =");
	 Matrix_Print(stderr, "%4d", a); 	*/
    matrix_to_sc(a,sc);
    Matrix_Free(a);
    sc=sc_normalize(sc);


    if (sc == NULL) {
	Pcontrainte pc = contrainte_make(vect_new(TCST, VALUE_ONE));
	sc = sc_make(pc, CONTRAINTE_UNDEFINED);
	sc->base = base_dup(sc1->base);
	sc->dimension = vect_size(sc->base);
	}

    /*    printf(" impression du systeme \n"); 
	  sc_dump(sc); */

    /* mem_spy_end("sc_convex_hull"); */

    return(sc);
}
