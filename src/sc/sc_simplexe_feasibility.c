/* test du simplex : ce test s'appelle par :
 *  programme fichier1.data fichier2.data ... fichiern.data
 * ou bien : programme<fichier.data
 * Si on compile grace a` "make sim" dans le directory
 *  /home/users/pips/C3/Linear/Development/polyedre.dir/test.dir
 * alors on peut tester l'execution dans le meme directory
 * en faisant : tests|more
 */

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <limits.h>
#include <setjmp.h>

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

extern jmp_buf overflow_error;

/* To replace #define NB_EQ and #define NB_INEQ - BC, 2/4/96 - 
 * NB_EQ and NB_INEQ are initialized at the beginning of the main subroutine.
 * they represent the number of non-NULL constraints in sc. This is useful
 * to allocate the minimum amount of memory necessary.  */
static int NB_EQ = 0;
static int NB_INEQ = 0;

#define DEBUG 0
#define DEBUG1 0
#define DEBUG2 0
#define PTR_NIL -32001
#define INFINI 32700
/* #define NB_INEQ sc->nb_ineq */
/* #define NB_EQ sc->nb_eq */
#define DIMENSION sc->dimension
#define SIMPL(A,B) {if(A!=1 && B!=1){Value I1,J1,K;I1=A,J1=B;while((K=I1%J1)!=0)I1=J1,J1=K;A=A/J1;B=B/J1;if(B<0)A=-A,B=-B;}}
#define G(J1,A,B) {Value I1,K;if(B>1){I1=A,J1=B;while((K=I1%J1)!=0)I1=J1,J1=K;if(J1<0)J1=-J1;}else J1=B;}
#define SIMPLIFIE(FRAC) SIMPL(FRAC.num,FRAC.den)
#define NUMERO hashtable[h].numero
/*#define MAX_VAR 197 nombre max de variables */
#define MAX_VAR 1971 /* nombre max de variables */
#define MAXVAL 24  /* seuil au dela duquel on se mefie d'un overflow */
#define SOLUBLE(N) soluble=N;goto FINSIMPLEX ;
#define EGAL1(x) (x.num==x.den)
#define EGAL0(x) (x.num==0)
#define EGAL(x,y) ((x.num==0 && y.num==0) || (x.den!=0 && y.den!=0 && x.num*y.den==x.den*y.num))
#define NEGATIF(x) ((x.num<0&&x.den>0)||(x.num>0&&x.den<0))
#define POSITIF(x) ((x.num>0&&x.den>0)||(x.num<0&&x.den<0))
#define SUP1(x) ((x.num>0) && (x.den>0) && (x.num>x.den)||(x.num<0) && (x.den<0) && (x.den>x.num))
#define INF(x,y) (x.num*y.den<x.den*y.num)
#define NUL(x) (x.num==0)
#define AFF(x,y) {x.num=y.num;x.den=y.den;}
#define METINFINI(x) {x.num=INFINI;x.den=1;}
#define DIV(x,y,z) {if(y.num==0)x.num=0,x.den=1;else{x.num=y.num*z.den;x.den=y.den*z.num;SIMPLIFIE(x);}}
#define MUL(x,y,z) {if(y.num==0||z.num==0)x.num=0,x.den=1;else{x.num=y.num*z.num;x.den=y.den*z.den;SIMPLIFIE(x);}}
   /* Pivot :  x = a - b c / d    */
#define PIVOT(X,A,B,C,D) {if(A.num==0){if(B.num==0||C.num==0||D.den==0)X.num=0, X.den=1;else if(B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL){X.num=-B.num*C.num*D.den;X.den=B.den*C.den*D.num;SIMPLIFIE(X);}else{frac uu;if(DEBUG2)printf("++ "),printfrac(A),printfrac(B),printfrac(C),printfrac(D),printf(" \n");MUL(uu,B,C);DIV(X,uu,D);X.num=-X.num;if(DEBUG2)printfrac(X);}} \
else if(B.num==0||C.num==0||D.den==0)X.num=A.num,X.den=A.den; \
else if(D.num==1&&A.den==1&&B.den==1&&C.den==1)X.den=1,X.num=A.num-B.num*C.num*D.den; \
else if(A.den<MAXVAL && B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL){X.num=A.num*B.den*C.den*D.num-A.den*B.num*C.num*D.den;X.den=A.den*B.den*C.den*D.num;SIMPLIFIE(X);} \
else{frac uu,vv,ww;if(DEBUG2)printfrac(A),printfrac(B),printfrac(C),printfrac(D),printf(" \n"); \
uu.num=B.num;vv.num=C.num;ww.num=D.den;uu.den=B.den;vv.den=C.den;ww.den=D.num; \
SIMPL(uu.num,vv.den);SIMPL(uu.num,ww.den);SIMPL(vv.num,uu.den);SIMPL(vv.num,ww.den);SIMPL(ww.num,uu.den);SIMPL(ww.num,vv.den); \
vv.num*=uu.num*ww.num;vv.den*=uu.den*ww.den; \
SUB(X,A,vv);if(DEBUG2)printfrac(X);}\
}
#define SUB(X,A,B) { \
if(A.num==0)X.num=-B.num,X.den=B.den; \
else if(B.num==0)X.num=A.num,X.den=A.den; \
else if(A.den==1&&B.den==1)X.num=A.num-B.num,X.den=1; \
else{Value GDEN,AD,BD;AD=A.den,BD=B.den; \
  if(A.den>B.den)G(GDEN,AD,BD) \
  else G(GDEN,BD,AD); \
  if(GDEN!=1)AD=AD/GDEN,BD=BD/GDEN; \
  X.num=A.num*BD-B.num*AD;X.den=AD*BD; \
  if(GDEN!=1){SIMPLIFIE(X);SIMPL(X.num,GDEN);X.den=X.den*GDEN;} \
}}
#define CREVARVISIBLE variables[compteur-3]=compteur-2;
#define CREVARCACHEE { variablescachees[nbvariables]=nbvariables + MAX_VAR ; \
			 if (nbvariables ++ >= MAX_VAR) abort(); }


#define MULTOFL(RES,A,B) {if ((B==0) || (ABS(A)<VALUE_MAX/ABS(B))) RES=A*B; else longjmp(overflow_error3, 5);}
#define SIMPLOFL(A,B) {if(A!=1 && B!=1){Value I1,J1,K;I1=A,J1=B;if(!J1)longjmp(overflow_error3,6);while((K=I1%J1)!=0) I1=J1,J1=K;if(!J1)longjmp(overflow_error3,6); A=A/J1; B=B/J1;if(B<0)A=-A,B=-B;}}
#define GOFL(J1,A,B) {Value I1,K;if(B>1){I1=A,J1=B;if(!J1)longjmp(overflow_error3,6);while((K=I1%J1)!=0)I1=J1,J1=K;if(J1<0)J1=-J1;}else J1=B;}
#define SIMPLIFIEOFL(FRAC) SIMPLOFL(FRAC.num,FRAC.den)
#define DIVOFL(x,y,z) {if(y.num==0)x.num=0,x.den=1;else{MULTOFL(x.num,y.num,z.den);MULTOFL(x.den,y.den,z.num);SIMPLIFIEOFL(x);}}
#define MULOFL(x,y,z) {if(y.num==0||z.num==0)x.num=0,x.den=1;else{MULTOFL(x.num,y.num,z.num);MULTOFL(x.den,y.den,z.den);SIMPLIFIEOFL(x);}}
   /* Pivot :  x = a - b c / d    */
#define PIVOTOFL(X,A,B,C,D) {if(A.num==0){if(B.num==0||C.num==0||D.den==0)X.num=0, X.den=1;else if(B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL) {MULTOFL(tp,C.num,D.den);MULTOFL(X.num,-B.num,tp);MULTOFL(tp,C.den,D.num);MULTOFL( X.den,B.den,tp);SIMPLIFIEOFL(X);}else{frac uu;if(DEBUG2)printf("++ "),printfrac(A),printfrac(B),printfrac(C),printfrac(D),printf(" \n");MULOFL(uu,B,C);DIVOFL(X,uu,D);X.num=-X.num;if(DEBUG2)printfrac(X);}} \
else if(B.num==0||C.num==0||D.den==0)X.num=A.num,X.den=A.den; \
else if(D.num==1&&A.den==1&&B.den==1&&C.den==1) {X.den=1;MULTOFL(tp,C.num,D.den);MULTOFL(tp,B.num,tp);X.num=A.num-tp;} \
else if(A.den<MAXVAL && B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL) {MULTOFL(tp,C.den,D.num); MULTOFL(tp,B.den,tp);MULTOFL(tp,A.num,tp);MULTOFL(tp1,C.num,D.den);MULTOFL(tp1,B.num,tp1);MULTOFL(tp1,A.den,tp1);X.num=tp-tp1;MULTOFL(tp,C.den,D.num);MULTOFL(tp,B.den,tp);MULTOFL(X.den,A.den,tp);SIMPLIFIEOFL(X);} \
else{frac uu,vv,ww;if(DEBUG2)printfrac(A),printfrac(B),printfrac(C),printfrac(D),printf(" \n"); \
uu.num=B.num;vv.num=C.num;ww.num=D.den;uu.den=B.den;vv.den=C.den;ww.den=D.num; \
SIMPLOFL(uu.num,vv.den);SIMPLOFL(uu.num,ww.den);SIMPLOFL(vv.num,uu.den);SIMPLOFL(vv.num,ww.den);SIMPLOFL(ww.num,uu.den);SIMPLOFL(ww.num,vv.den); \
MULTOFL(tp,uu.num,ww.num);MULTOFL(vv.num,vv.num,tp);MULTOFL(tp,uu.den,ww.den);MULTOFL(vv.den,vv.den,tp); \
SUBOFL(X,A,vv);if(DEBUG2)printfrac(X);} \
}
#define SUBOFL(X,A,B) { \
if(A.num==0)X.num=-B.num,X.den=B.den; \
else if(B.num==0)X.num=A.num,X.den=A.den; \
else if(A.den==1&&B.den==1)X.num=A.num-B.num,X.den=1; \
else{Value GDEN,AD,BD;AD=A.den,BD=B.den; \
  if(A.den>B.den)GOFL(GDEN,AD,BD) \
  else GOFL(GDEN,BD,AD); \
  if(!GDEN)longjmp(overflow_error3,8);/* just before the fault */\
  if(GDEN!=1)AD=AD/GDEN,BD=BD/GDEN; \
  MULTOFL(tp1,A.num,BD); MULTOFL(tp,B.num,AD);X.num = tp1-tp;MULTOFL(X.den,AD,BD); \
  if(GDEN!=1){SIMPLIFIEOFL(X);SIMPLOFL(X.num,GDEN);MULTOFL(X.den,X.den,GDEN);} \
}}



/* Le nombre de variables visibles est : compteur-2
 * La i-eme variable visible a le numero : variables[i+1]=i
 *   (0 <= i < compteur-2)
 * Le nombre de variables cachees est : nbvarables
 * La i-eme variable cachee a le numero : variablescachees[i+1]=MAX_VAR+i-1
 *   (0 <= i < nbvariables)
 */
/* utilise'es par dump_tableau ; a rendre local */
static int nbvariables, variablescachees[MAX_VAR], variables[MAX_VAR] ; 
static frac frac0={0,0,0} ;

static void printfrac(frac x) {
    printf(" ");
    print_Value(x.num);
    printf("/");
    print_Value(x.den);
    /* printf(" %3.1ld/%-3.1ld",x.num,x.den) ; */
}

static void dump_tableau(tableau *t,int colonnes) {
    int i,j, k, w;
    int max=0;
    for(i=0;i<colonnes;i++) 
      if(t[i].colonne[t[i].taille-1].numero>max)max=t[i].colonne[t[i].taille-1].numero ; 
    printf("Dump du tableau ------ %d colonnes  %d lignes\n",colonnes,max) ;
    printf("%d Variables  visibles :\n",colonnes-2) ;
    for(i=0;i<colonnes-2;i++) printf("%7d",variables[i]) ;
    printf("\n") ;
    printf("%d Variables cachees :\n",nbvariables);
    for(i=0;i<nbvariables;i++) printf("%7d",variablescachees[i]) ;
    printf("\n") ;
if(DEBUG){
    for(i=0;i<colonnes;i++) {
      if(t[i].existe != 0) {
        printf("Colonne %d Existe=%d Taille=%d\n",i,t[i].existe,t[i].taille) ;
        for(j=0 ; j<t[i].taille ; j++)
	    printf("ligne %d valeur", t[i].colonne[j].numero),
	    printfrac(t[i].colonne[j]),
	    printf("\n");
      }
    }
} /* DEBUG */
  
    printf("Nb lignes: %d\n", max);
    for(j=0;j<=max;j++) { printf("\nLigne %d ",j) ;
        for(i=0;i<colonnes;i++) {
            w=1 ;
            for(k=0;k<t[i].taille;k++)
                if(t[i].colonne[k].numero==j)
		    printfrac(t[i].colonne[k]) , w=0 ;
            if(w!=0)printfrac(frac0) ;
        }
    }
    printf("\n");
} /* dump_tableau */


/* calcule le hashcode d'un pointeur
   sous forme d'un nombre compris entre 0 et  MAX_VAR */
static int hash(Variable s) 
{ int i ;
  i=(long)s % MAX_VAR ;
  return (i) ;
}

                 
/* fonction de calcul de la faisabilite' d'un systeme
 * d'equations et d'inequations
 * Auteur : Robert Mahl, Date : janvier 1994
 */
/* Retourne : 1 si le systeme est soluble (faisable)
 *  en rationnels,
 * 0 s'il n'y a pas de solution.
 */
/* overflow control :
 *  ofl_ctrl == NO_OFL_CTRL  => no overflow control
 *  ofl_ctrl == FWD_OFL_CTRL  
 *           => overflow control is made (longjmp(overflow_error,5))
 * BC, 13/12/94
 */
int sc_simplexe_feasibility_ofl_ctrl(Psysteme sc, int ofl_ctrl) {
    Pcontrainte pc, pc_tmp ;
    Pvecteur pv ;
    int premier_hash = PTR_NIL ; /* tete de liste des noms de variables */
    /* Necessaire de declarer "hashtable" static 
     *  pour initialiser tout automatiquement a` 0.
     * Necessaire de chainer les enregistrements
     *  pour reinitialiser a 0
     *  en sortie de la procedure.
     */
    static struct
    {
	Variable nom;
	int numero; int hash ;
	int val ;
	int succ ;
    } hashtable[MAX_VAR] ;
    jmp_buf overflow_error3;
    tableau *eg ; /* tableau des egalite's  */
    tableau *t ; /* tableau des inegalite's  */
    /* les colonnes 0 et 1 sont reservees au terme const: */
    int compteur = 2 ;
    long i, j, k, h, trouve, valeur, hh=0, ligne, poidsM, i0, i1, jj, ii ;
    long w ;
    int soluble = 1 ; /* valeur retournee par feasible */
    frac *nlle_colonne , *colo ;
    frac objectif[2] ; /* objectif de max pour simplex : 
			  somme des (b2,c2) termes constants "inferieurs" */
    frac rapport1, rapport2, min1, min2, pivot, cc ;
    
    /* Allocation a priori du tableau des egalites.
     * "eg" : tableau a "nb_eq" lignes et "dimension"+2 colonnes.
     */
    
    long tp,tp1;

    /* the input Psysteme must be consistent; this is not the best way to
     * do this; array bound checks should be added instead in proper places;
     * no time to do it properly for the moment. BC.
     */
    assert(sc_weak_consistent_p(sc));

    /* Do not allocate place for NULL constraints */
    NB_EQ = 0;
    NB_INEQ = 0;
    for(pc_tmp = sc->egalites; pc_tmp!= NULL; pc_tmp=pc_tmp->succ) 
    {
	if (pc_tmp->vecteur != NULL)
	    NB_EQ++;
    }
    for(pc_tmp = sc->inegalites; pc_tmp!= NULL; pc_tmp=pc_tmp->succ) 
    {
	if (pc_tmp->vecteur != NULL)
	    NB_INEQ++;
    }
    
    if (setjmp(overflow_error3))
    {
	for(i=premier_hash ; i!=PTR_NIL; i=hashtable[i].succ)
	    hashtable[i].nom = 0 ;
	if(NB_EQ > 0) {
	    for(i=0 ; i<(3+DIMENSION) ; i++)
		free(eg[i].colonne);
	    free(eg);
	}

	/* I have noticed that when pips core dumps here, it is because
	 * a setjmp(overflow_error) has been forgotten. bc.
	 */
	for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++)  
	    free(t[i].colonne); 
	free(t); 
	free(nlle_colonne); 
	if (ofl_ctrl == FWD_OFL_CTRL) 
	    longjmp(overflow_error,5);
    }
    else
    {
	if(NB_EQ != 0)
	{
	    eg=(tableau*)malloc((3+DIMENSION)*sizeof(tableau)) ;
	    for(i=0 ; i<(3+DIMENSION) ; i++)
	    {
		eg[i].colonne=(frac*)malloc(NB_EQ*sizeof(frac)) ;
		eg[i].existe = 0 ;
		eg[i].taille = 0 ;
	    }
	}
    /* Determination d'un numero pour chaque variable */
    
    for(pc=sc->egalites, ligne=1 ; pc!=0; pc=pc->succ, ligne++)
    {
	pv=pc->vecteur;
	if (pv!=NULL) /* skip if empty */
	{
	    j=0 ; /* compteur du nb de variables de l'equation */
	    valeur=0 ; /* le terme cst vaut 0 par defaut */
	    for(; pv !=0 ; pv=pv->succ) {
		if(vect_coeff(pv->var,sc_base(sc))) {
		    j++ ;
		    h = hash((Variable) pv->var) ; trouve=0 ;
		    while (hashtable[h].nom != 0)  {
			if (hashtable[h].nom==pv->var) {
			    trouve=1 ;
			    break ;
			}
			else { h = (h+1) % MAX_VAR ; }
		    }
		    if(!trouve) {
			hashtable[h].succ=premier_hash ;
			premier_hash = h ;
			hashtable[h].val = PTR_NIL ;
			hashtable[h].numero=compteur++ ;
			CREVARVISIBLE;
			hashtable[h].nom=pv->var ;
		    }
		    hh = h ;
		    assert((NUMERO) < (3+DIMENSION));
		    eg[NUMERO].existe = 1 ;
		    eg[NUMERO].colonne[eg[NUMERO].taille].numero=ligne ;
		    eg[NUMERO].colonne[eg[NUMERO].taille].num = pv->val ;
		    eg[NUMERO].colonne[eg[NUMERO].taille].den = 1 ;
		    eg[NUMERO].taille++ ;
		} 
		else { valeur= - pv->val; 
		       eg[0].existe = 1 ;
		       eg[0].colonne[eg[0].taille].numero=ligne ;
		       eg[0].colonne[eg[0].taille].num = valeur ;
		       eg[0].colonne[eg[0].taille].den = 1 ;
		       eg[0].taille++ ;
		   }
	    }
	    /* Cas ou` valeur de variable est connue : */
	    if(j==1) hashtable[hh].val = valeur ;
	    if(DEBUG1&&sc->egalites!=0)dump_tableau(eg,compteur) ;
	}
	else
	    ligne--;
    }
    
    /* Allocation a priori du tableau du simplex "t" par
     * colonnes. Soit
     * "dimension" le nombre de variables, la taille maximum
     * du tableau est de (1 + nb_ineq) lignes
     * et de (2 + dimension + nb_ineq + nb_eq) colonnes
     * On y ajoute en fait le double du nombre d'egalite's.
     * Ce tableau sera rempli de la facon suivante :
     * - ligne 0 : critere d'optimisation
     * - lignes 1 a nb_ineq : les inequations
     * - colonne 0 : le terme constant (composante de poids 1)
     * - colonne 1 : le terme constant (composante de poids M)
     * - colonnes 2 et suivantes : les elements initiaux
     *   et les termes d'ecart
     * Le tableau a une derniere colonne temporaire pour 
     *  pivoter un vecteur unitaire.
     */
    
    t = (tableau*)malloc((3 + NB_INEQ + NB_EQ + DIMENSION)*sizeof(tableau));
    for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++) {
        t[i].colonne=(frac *) malloc((4 + 2*NB_EQ + NB_INEQ)*sizeof(frac)) ;
        t[i].existe = 0 ;
        t[i].taille = 1 ;
        t[i].colonne[0].numero = 0 ;
        t[i].colonne[0].num = 0 ;
    }
    nbvariables= 0 ;
    /* Initialisation de l'objectif */

    for(i=0;i<=1;i++) 
	objectif[i].num=0, 
	objectif[i].den=1;

    for(i=0;i<MAX_VAR;i++) if(hashtable[i].nom != 0)
	if(DEBUG) printf("%s %d %d\n",
			 hashtable[i].nom,
			 hashtable[i].numero,
			 hashtable[i].val) ;

    /* Entree des inegalites dans la table */
    
    for(pc=sc->inegalites, ligne=1; pc!=0; pc=pc->succ, ligne++) 
    {
	pv=pc->vecteur;
	if (pv!=NULL) /* skip if empty */
	{
	    valeur = 0 ;
	    poidsM=0 ;
	    for(; pv !=0 ; pv=pv->succ) 
		if(vect_coeff(pv->var,sc_base(sc)))
		    poidsM += pv->val ;
		else valeur = - pv->val ; /* val terme const */

	    for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
		if(vect_coeff(pv->var,sc_base(sc))) {
		    h = hash((Variable)  pv->var) ; trouve=0 ;
		    while (hashtable[h].nom != 0)  {
			if (hashtable[h].nom==pv->var) {
			    trouve=1 ;
			    break ;
			}
			else { h = (h+1) % MAX_VAR ; }
		    }
		    if(!trouve) {
			hashtable[h].succ=premier_hash ;
			premier_hash = h ;
			hashtable[h].val = PTR_NIL ;
			hashtable[h].numero=compteur++ ;
			hashtable[h].nom=pv->var ;
			CREVARVISIBLE ;
		    }
		    assert((NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
		    if(poidsM < 0 || (poidsM==0 && valeur<0))
			t[NUMERO].colonne[0].num += pv->val,
			/*
			   if(DEBUG)printf("pv->val = %ld, t[NUMERO].colonne[0].num = %ld\n",pv->val,t[NUMERO].colonne[0].num),
			   */
			t[NUMERO].colonne[0].den = 1 ;
		    t[NUMERO].existe = 1 ;
		    t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
		    if(poidsM < 0 || (poidsM==0 && valeur<0))
			i = -pv->val ; else i = pv->val ;
		    t[NUMERO].colonne[t[NUMERO].taille].num=i ;
		    t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
		    t[NUMERO].taille++ ;
		}
	    }
	    /* Creation de variable d'ecart ? */
	    if(poidsM < 0 || (poidsM==0 && valeur<0)) {
		if(DEBUG)dump_tableau(t, compteur) ;
		i=compteur++ ;
		CREVARVISIBLE ;
		t[i].existe = 1 ; t[i].taille = 2 ;
		t[i].colonne[0].num = 1 ;
		t[i].colonne[0].den = 1 ;
		if(DEBUG)printf("ligne ecart = %ld, colonne %ld\n",ligne,i) ;
		t[i].colonne[1].numero = ligne ;
		t[i].colonne[1].num = -1 ;
		t[i].colonne[1].den = 1 ;
		poidsM = - poidsM, valeur= - valeur ;
		objectif[0].num+=valeur ; 
		objectif[1].num+= poidsM ;
	    }
	    /* Mise a jour des colonnes 0 et 1 */
	    t[0].colonne[t[0].taille].numero = ligne ;
	    t[0].colonne[t[0].taille].den = 1 ;
	    t[0].colonne[t[0].taille].num = valeur ;
	    t[0].existe = 1 ;
	    t[0].taille++ ;
	    /* Element de poids M en 1ere colonne */
	    t[1].colonne[t[1].taille].numero = ligne ;
	    t[1].colonne[t[1].taille].num = poidsM ;
	    t[1].colonne[t[1].taille].den = 1 ;
	    t[1].existe = 1 ;
	    t[1].taille++ ;
	    /* Creation d'une colonne cachee */
	    CREVARCACHEE ;
	    if(DEBUG) dump_tableau(t, compteur) ;
	}
	else
	    ligne--;
    }

    if (DEBUG)
    {
	for(i=0;i<MAX_VAR;i++) 
	    if(hashtable[i].nom != 0)
		printf("%s %d %d\n",
		       hashtable[i].nom,
		       hashtable[i].numero,
		       hashtable[i].val) ;
    }
	
    if(DEBUG1)dump_tableau(t, compteur) ;


    
    /* NON IMPLEMENTE' */
    
    /* Elimination de Gauss-Jordan dans le tableau "eg"
     *  Chaque variable a` eliminer est marquee
     *  eg[ ].existe = 2
     *  Si le processus d'elimination ne revele pas
     *  d'impossibilite', il est suivi du processus
     *  d'elimination dans les inegalites.
     */
    /* FIN DE NON IMPLEMENTE' */
    
    /* SOLUTION PROVISOIRE
     *  Pour chaque egalite on introduit
     *  2 inequations complementaires
     */
    
    for(pc=sc->egalites ; pc!=0; pc=pc->succ, ligne++)
    {
	/* Added by bc: do nothing for nul equalities */
	if (pc->vecteur != NULL)
	{
        valeur = 0 ;
        poidsM=0 ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                poidsM += pv->val ;
            else valeur = - pv->val ; /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) {
                h = hash((Variable) pv->var) ; trouve=0 ;
                while (hashtable[h].nom != 0)  {
                    if (hashtable[h].nom==pv->var) {
                        trouve=1 ;
                        break ;
                    }
                    else { h = (h+1) % MAX_VAR ; }
                }
                if(!trouve) {
                    hashtable[h].succ=premier_hash ;
                    premier_hash = h ;
                    hashtable[h].val = PTR_NIL ;
                    hashtable[h].numero=compteur++ ;
                    CREVARVISIBLE ;
                    hashtable[h].nom=pv->var ;
                }
		assert((NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
                if(poidsM < 0 || (poidsM==0 && valeur<0))
                    t[NUMERO].colonne[0].num += pv->val,
                    t[NUMERO].colonne[0].den = 1 ;
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(poidsM < 0 || (poidsM==0 && valeur<0))
                    i = -pv->val ; else i = pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num=i ;
                t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
                t[NUMERO].taille++ ;
            }
        }
	/* Creation de variable d'ecart ? */
        if(poidsM < 0 || (poidsM==0 && valeur<0)) {
            i=compteur++ ;
            CREVARVISIBLE ;
            t[i].existe = 1 ; t[i].taille = 2 ;
            t[i].colonne[0].num = 1 ;
            t[i].colonne[0].den = 1 ;
            t[i].colonne[1].numero = ligne ;
            t[i].colonne[1].num = -1 ;
            t[i].colonne[1].den = 1 ;
            poidsM = - poidsM, valeur= - valeur ;
            objectif[0].num+=valeur ;
            objectif[1].num+= poidsM ;
        }
	/* Mise a jour des colonnes 0 et 1 */
        t[0].colonne[t[0].taille].numero = ligne ;
        t[0].colonne[t[0].taille].num = valeur ;
        t[0].colonne[t[0].taille].den = 1 ;
        t[0].existe = 1 ;
        t[0].taille++ ;
	/* Element de poids M en 1ere colonne */
        t[1].colonne[t[1].taille].numero = ligne ;
        t[1].colonne[t[1].taille].num = poidsM ;
        t[1].colonne[t[1].taille].den = 1 ;
        t[1].existe = 1 ;
        t[1].taille++ ;
	/* Creation d'une colonne cachee */
        CREVARCACHEE ;
	if(DEBUG)dump_tableau(t, compteur) ;
        }
    }
    
    for(pc=sc->egalites ; pc!=0; pc=pc->succ, ligne++)
    {
	/* Added by bc: do nothing for nul equalities */
	if (pc->vecteur != NULL)
	{
        valeur = 0 ;
        poidsM=0 ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                poidsM -= pv->val ;
            else valeur = pv->val ; /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if (vect_coeff(pv->var,sc_base(sc))) {
                h = hash((Variable) pv->var) ; trouve=0 ;
                while (hashtable[h].nom != 0)  {
                    if (hashtable[h].nom==pv->var) {
                        trouve=1 ;
                        break ;
                    }
                    else { h = (h+1) % MAX_VAR ; }
                }
                if(!trouve) {
                    hashtable[h].succ=premier_hash ;
                    premier_hash = h ;
                    hashtable[h].val = PTR_NIL ;
                    hashtable[h].numero=compteur++ ;
                    hashtable[h].nom=pv->var ;
                    CREVARVISIBLE ;
                }
		assert((NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
                if(poidsM < 0 || (poidsM==0 && valeur<0))
                    t[NUMERO].colonne[0].num -= pv->val,
                    t[NUMERO].colonne[0].den = 1 ;
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(poidsM < 0 || (poidsM==0 && valeur<0))
                    i = pv->val ; else i = - pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num=i ;
                t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
                t[NUMERO].taille++ ;
            }
        }
	/* Creation de variable d'ecart ? */
        if(poidsM < 0 || (poidsM==0 && valeur<0)) {
            i=compteur++ ;
            CREVARVISIBLE ;
            t[i].existe = 1 ; t[i].taille = 2 ;
            t[i].colonne[0].num = 1 ;
            t[i].colonne[0].den = 1 ;
            t[i].colonne[1].numero = ligne ;
            t[i].colonne[1].num = -1 ;
            t[i].colonne[1].den = 1 ;
            poidsM = - poidsM, valeur= - valeur ;
            objectif[0].num+=valeur ;
            objectif[1].num+= poidsM ;
        }
	/* Mise a jour des colonnes 0 et 1 */
        t[0].colonne[t[0].taille].numero = ligne ;
        t[0].colonne[t[0].taille].num = valeur ;
        t[0].colonne[t[0].taille].den = 1 ;
        t[0].existe = 1 ;
        t[0].taille++ ;
	/* Element de poids M en 1ere colonne */
        t[1].colonne[t[1].taille].numero = ligne ;
        t[1].colonne[t[1].taille].num = poidsM ;
        t[1].colonne[t[1].taille].den = 1 ;
        t[1].existe = 1 ;
        t[1].taille++ ;
	/* Creation d'une colonne cachee */
        CREVARCACHEE ;
	if(DEBUG)dump_tableau(t, compteur) ;
        }
    }
    
    /* FIN DE SOLUTION PROVISOIRE */
    
    /* Algorithme du simplexe - methode primale simple.
     * L'objectif est d'etudier la faisabilite' d'un systeme
     * de contraintes sans trouver l'optimum.
     *   Les contraintes ont la forme : a x <= b
     *      et  d x = e
     * La methode de resolution procede comme suit :
     *
     *  1) Creer un tableau
     *       a  b
     *       d  e
     *     Eliminer autant de variables que posible par
     *    Gauss-Jordan
     *
     *  2) Travailler sur les inegalites seulement.
     *      Poser  x = x' - M 1
     *    ou` 1 est le vecteur de chiffres 1.
     *     Les inequations prennent alors la forme :
     *      a1 x <= b1 + M c1
     *      a2 x >= b2 + M c2
     *    avec c1 et c2 positifs
     *     On introduit les variables d'ecart y (autant que 
     *    d'inequations du 2eme type) et on cherche
     *      max{1(a2 x - y) | x,y >= 0 ; a1 x <= b1 + M c1 ;
     *                                a2 x - y <= b2 + M c2}
     *     On cree donc le tableau :
     *        0  0  1 a2     1  0  0
     *        b1 c1  a1      0  I  0
     *        b2 c2  a2     -I  0  I
     *
     *     On applique ensuite l'algorithme du simplex en
     *    se souvenant que c1 et c2 sont a multiplier par un
     *    infiniment grand.
     *     Si l'optimum est egal a (1 b2 , 1 c2), il y a une
     *    solution.
     *
     * Structures de donnees : on travaille sur des tableaux
     * de fractions rationnelles.
     */
    nlle_colonne=(frac *) malloc((4 + 2*NB_EQ + NB_INEQ)*sizeof(frac)) ;
    while(1) {

        /*  Recherche d'un nombre negatif 1ere ligne  */
        for(j=2, jj= -1 ;j<compteur;j++)
            if(t[j].existe && NEGATIF(t[j].colonne[0]))
            {  jj=j ; break ;
	   }
        /*  Terminaison  */
        if(jj == -1) { 
            if(DEBUG1)printf ("solution :\n") ;
            if(DEBUG1)dump_tableau(t, compteur) ;
            if(DEBUG1){ printf("objectif : "); printfrac(objectif[0]) ; 
			printfrac(objectif[1]) ; printf("\n") ;}
	    if ( (ofl_ctrl == FWD_OFL_CTRL) && 
		((t[0].colonne[0].den && 
		 (ABS(objectif[0].num) > ((VALUE_MAX)/ABS(t[0].colonne[0].den)))) 
		|| (t[0].colonne[0].num && 
		 (ABS(objectif[0].den) > VALUE_MAX/ABS(t[0].colonne[0].num)))
		|| (t[1].colonne[0].den &&
		    (ABS(objectif[1].num) > VALUE_MAX/ABS(t[1].colonne[0].den))) ||
		(t[0].colonne[0].num && 
		 (ABS(objectif[0].den) > VALUE_MAX/ABS(t[0].colonne[0].num))))) 
		longjmp(overflow_error3,5);
	    
	    if(EGAL(objectif[0],t[0].colonne[0]) &&
               EGAL(objectif[1],t[1].colonne[0]))
             {
                if(DEBUG1)printf("Systeme soluble (faisable) en rationnels\n") ;
                SOLUBLE(1)
            } else { if(DEBUG1)printf("Systeme insoluble (infaisable)\n") ;
		     SOLUBLE(0)
		 }
	    if(DEBUG1)printf("fin\n");
        }
	if(DEBUG)printf("1 : jj= %ld\n",jj) ;
	if(DEBUG2)dump_tableau(t, compteur) ;
        /*  Recherche de la ligne de pivot  */
        METINFINI(min1) ; METINFINI(min2) ;

        for(i=1, i0=1, i1=1, ii=-1 ; i<t[jj].taille ; )
        {
            if(((i0<t[0].taille && t[jj].colonne[i].numero<= t[0].colonne[i0].numero)  || i0>=t[0].taille)
	       && ((i1<t[1].taille && t[jj].colonne[i].numero<= t[1].colonne[i1].numero) || i1>=t[1].taille)) {
		if( POSITIF(t[jj].colonne[i])) {
		    if (ofl_ctrl == FWD_OFL_CTRL) {
DIVOFL(rapport1,((i0<t[0].taille&&t[jj].colonne[i].numero==t[0].colonne[i0].numero)?t[0].colonne[i0]:frac0),t[jj].colonne[i])
DIVOFL(rapport2,((i1<t[1].taille&&t[jj].colonne[i].numero==t[1].colonne[i1].numero)?t[1].colonne[i1]:frac0),t[jj].colonne[i])
}
		    else {
DIV(rapport1,((i0<t[0].taille&&t[jj].colonne[i].numero==t[0].colonne[i0].numero)?t[0].colonne[i0]:frac0),t[jj].colonne[i])
DIV(rapport2,((i1<t[1].taille&&t[jj].colonne[i].numero==t[1].colonne[i1].numero)?t[1].colonne[i1]:frac0),t[jj].colonne[i])	
}
    

  if ( (ofl_ctrl == FWD_OFL_CTRL) && 
      ((min2.den && (ABS(rapport2.num) > VALUE_MAX/ABS(min2.den))) || 
	(min1.den && (ABS(rapport1.num) > VALUE_MAX/ABS(min1.den))) ||
	(min2.num && (ABS(rapport2.den) > VALUE_MAX/ABS(min2.num))) ||
	(min1.num && (ABS(rapport1.den) > VALUE_MAX/ABS(min1.num)))))
      longjmp(overflow_error3,5);

      if(INF(rapport2,min2)||(EGAL(rapport2,min2)&&INF(rapport1,min1))){
	  AFF(min1,rapport1) ;
	  AFF(min2,rapport2) ;
	  AFF(pivot,t[jj].colonne[i]) ;
	  ii=t[jj].colonne[i].numero ;
      }
  } /* POSITIF(t[jj].colonne[i])) */
i++ ;
}
else {
    if(i0<t[0].taille && t[jj].colonne[i].numero> t[0].colonne[i0].numero) i0++ ;
if(i1<t[1].taille && t[jj].colonne[i].numero > t[1].colonne[i1].numero) i1++ ;
}
	    if(DEBUG)printf("i=%ld i0=%ld i1=%ld   %d %d %d\n",i,i0,i1,t[jj].colonne[i].numero,t[0].colonne[i0].numero,t[1].colonne[i1].numero) ;
        }
        /* Cas d'impossibilite'  */
if(ii==-1) {
	    if(DEBUG1)dump_tableau(t, compteur) ;
	    if(DEBUG1)printf("Solution infinie\n");
	    SOLUBLE(1)
	}
	/* Modification des numeros des variables */
        j=variables[jj-2];k=variablescachees[ii-1];
        variables[jj-2]=k;variablescachees[ii-1]=j;
        if(DEBUG2){printf("Visibles :");for(j=0;j<compteur-2;j++) printf(" %d",variables[j]);printf("\nCachees :"); for(j=0;j<nbvariables;j++) printf(" %d",variablescachees[j]);printf("\n");}
        /*  Pivot autour de la ligne ii / colonne jj
         * Dans ce qui suit, j = colonne courante,
         *  k = numero element dans la nouvelle colonne
         *     qui remplacera la colonne j,
         *  cc = element (colonne j, ligne ii)
         */
	if(DEBUG2)printf("Pivoter %ld %ld\n",ii,jj) ;
        
	/* Remplir la derniere colonne temporaire de t
	 *   qui contient un 1 en position ligne ii
	 */
        t[compteur].taille = 2 ;
        t[compteur].colonne[0].num = 0 ;
        t[compteur].colonne[0].numero = 0 ;
        t[compteur].colonne[1].numero = ii;
        t[compteur].colonne[1].num = 1 ;
        t[compteur].colonne[1].den = 1 ;
        t[compteur].existe = 1 ;

        for(j=0 ; j<=compteur ; j=(j==(jj-1))?(j+2):(j+1)) {
	    if(t[j].existe)
	    {
		k=0 ;
		cc.num=0 ; cc.den=1 ;
		for(i=1;i<t[j].taille;i++)
		    if(t[j].colonne[i].numero==ii)
                    { AFF(cc,t[j].colonne[i]); break ; }
		    else if(t[j].colonne[i].numero>ii)
                    {cc.num=0 ; cc.den=1 ; break ; }
		for(i=0,i1=0;i<t[j].taille || i1<t[jj].taille ;) {
		    if(i<t[j].taille &&  i1<t[jj].taille && t[j].colonne[i].numero == t[jj].colonne[i1].numero) 
		    {   if(DEBUG)printf("k=%ld, j=%ld, i=%ld i1=%ld\n",k,j,i,i1);
			if(DEBUG)printfrac(t[j].colonne[i]) ;
			if(DEBUG)printfrac(t[jj].colonne[i1]) ;
			if(DEBUG)printfrac(cc);
			if(DEBUG)printfrac(pivot) ;
			if(t[j].colonne[i].numero == ii) {
			    AFF(nlle_colonne[k],t[j].colonne[i])
			} else {
			    if (ofl_ctrl == FWD_OFL_CTRL) {
				PIVOTOFL(nlle_colonne[k],t[j].colonne[i],t[jj].colonne[i1],cc,pivot) ;}
			    else {
				PIVOT(nlle_colonne[k],t[j].colonne[i],t[jj].colonne[i1],cc,pivot) ;}
			}
			if(i==0||nlle_colonne[k].num!=0) {
			    nlle_colonne[k].numero = t[j].colonne[i].numero ;
			    if(DEBUG)printfrac(nlle_colonne[k]) ;
			    if(DEBUG)printf(" ligne numero %d ", nlle_colonne[k].numero) ;
			    if(DEBUG)printf("\n") ;
			    k++ ;
			}
			i++ ; i1++ ;
		    }
		    else if(i>=t[j].taille || (i1<t[jj].taille && t[j].colonne[i].numero > t[jj].colonne[i1].numero))
		    {  
 if(DEBUG)printf("t[j].colonne[i].numero > t[jj].colonne[i1].numero , k=%ld, j=%ld, i=%ld i1=%ld\n",k,j,i,i1);
			if(DEBUG){ printf("j = %ld  t[j].taille=%d , t[jj].taille=%d\n",j,t[j].taille,t[jj].taille);
				   printf("t[j].colonne[i].numero=%d , t[jj].colonne[i1].numero=%d\n",
					  t[j].colonne[i].numero,t[jj].colonne[i1].numero);}
                        /* 0 en colonne j  ligne t[jj].colonne[i1].numero */
			if(t[jj].colonne[i1].numero == ii) {
			    AFF(nlle_colonne[k],frac0)
			} else {
			    if(ofl_ctrl == FWD_OFL_CTRL) {
				PIVOTOFL(nlle_colonne[k],frac0,t[jj].colonne[i1],cc,pivot) ;}
			    else {
				PIVOT(nlle_colonne[k],frac0,t[jj].colonne[i1],cc,pivot) ;}
			}
			if(i==0||nlle_colonne[k].num!=0) {
			    nlle_colonne[k].numero = t[jj].colonne[i1].numero ;
			    if(DEBUG){printfrac(nlle_colonne[k]); printf(" ligne numero %d ", nlle_colonne[k].numero);printf("\n");}
			    k++ ;
			}
			if(i1<t[jj].taille) i1++ ; else i++ ;
}
else if(i1>=t[jj].taille || t[j].colonne[i].numero < t[jj].colonne[i1].numero)
		    {       /* 0 en colonne jj  ligne t[j].colonne[i].numero */
			if(DEBUG){ printf("t[j].colonne[i].numero < t[jj].colonne[i1].numero , k=%ld, j=%ld, i=%ld i1=%ld\n",k,j,i,i1);
				   printf("j = %ld  t[j].taille=%d , t[jj].taille=%d\n",j,t[j].taille,t[jj].taille);}
			AFF(nlle_colonne[k],t[j].colonne[i]) ;
			if(DEBUG)printfrac(nlle_colonne[k]) ;
			if(i==0||nlle_colonne[k].num!=0) {
			    nlle_colonne[k].numero = t[j].colonne[i].numero ;
			    if(DEBUG)printf(" ligne numero %d \n", nlle_colonne[k].numero) ;
			    k++ ;
			}
			if(i<t[j].taille) i++ ; else i1++ ;
	    }

		}
		if(j==compteur) w = jj ; else w = j ;
		colo = t[w].colonne ;
		t[w].colonne=nlle_colonne ;
		nlle_colonne = colo ;
		t[w].taille=k ;
		if(DEBUG){ printf("w = %ld  t[w].taille=%d \n",w,t[w].taille); dump_tableau(t, compteur);}
}
        }
	
    }

    /* Restauration des entrees vides de la table hashee  */
    FINSIMPLEX :
	for(i=premier_hash ; i!=PTR_NIL; i=hashtable[i].succ)
	    hashtable[i].nom = 0 ;
    if(DEBUG2)dump_tableau(t, compteur) ;

    if(NB_EQ > 0) {
      for(i=0 ; i<(3+DIMENSION) ; i++)
	  free(eg[i].colonne);
      free(eg);
  }

    for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++) 
        free(t[i].colonne);
    free(t);
    free(nlle_colonne);
    return (soluble) ;
}
}     /* main */
