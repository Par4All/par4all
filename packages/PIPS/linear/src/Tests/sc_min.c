/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* test sc_min     : ce test s'appelle par :
 *  programme fichier1.data fichier2.data ... fichiern.data
 * ou bien : programme<fichier.data
 *  Se compile grace a` "make min" dans le directory
 *  /home/users/pips/C3/Linear/Development/polyedre.dir/test.dir
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
extern int fprintf();
extern int printf();
extern char * strdup();

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sc.h"
#include "sg.h"
#include "types.h"
#include "polyedre.h"

#define DEBUG 0
#define DEBUG1 0
#define DEBUG2 0
/* Hmmm. To be compatible with some weird old 16-bit constants... RK */
#define PTR_NIL (INTPTR_MIN+767)
#define INFINI (INTPTR_MAX-767)
#define NB_INEQ sc->nb_ineq
#define NONTCST (vect_coeff(pv->var,sc_base(sc)))
#define SIMPL(A,B) {if(A!=1 && B!=1){long I1,J1,K;I1=A,J1=B;while((K=I1%J1)!=0)I1=J1,J1=K;A=A/J1;B=B/J1;if(B<0)A=-A,B=-B;}}
#define G(J1,A,B) {long I1,K;if(B>1){I1=A,J1=B;while((K=I1%J1)!=0)I1=J1,J1=K;if(J1<0)J1=-J1;}else J1=B;}
#define SIMPLIFIE(FRAC) SIMPL(FRAC.num,FRAC.den)
#define NB_EQ sc->nb_eq
#define DIMENSION sc->dimension
#define NUMERO hashtable[h].numero
#define MAX_VAR 197 /* nombre max de variables */
#define MAXVAL 24  /* seuil au dela duquel on se mefie d'un overflow */
#define SOLUBLE(N) soluble=N;goto FINSIMPLEX ;
#define EGAL1(x) (x.num==x.den)
#define EGAL0(x) (x.num==0)
#define EGAL(x,y) (x.num==0 && y.num==0 ||x.den!=0 && y.den!=0 && x.num*y.den==x.den*y.num)
#define NEGATIF(x) (x.num<0&&x.den>0||x.num>0&&x.den<0)
#define POSITIF(x) (x.num>0&&x.den>0||x.num<0&&x.den<0)
#define SUP1(x) ((x.num>0) && (x.den>0) && (x.num>x.den)||(x.num<0) && (x.den<0) && (x.den>x.num))
#define INF(x,y) (x.num*y.den<x.den*y.num)
#define NUL(x) (x.num==0)
#define AFF(x,y) {x.num=y.num;x.den=y.den;}
#define METINFINI(x) {x.num=INFINI;x.den=1;}
#define DIV(x,y,z) {if(y.num==0)x.num=0,x.den=1;else{x.num=y.num*z.den;x.den=y.den*z.num;SIMPLIFIE(x);}}
#define MUL(x,y,z) {if(y.num==0||z.num==0)x.num=0,x.den=1;else{x.num=y.num*z.num;x.den=y.den*z.den;SIMPLIFIE(x);}}
   /* Pivot :  x = a - b c / d    */
#define PIVOT(X,A,B,C,D) {if(A.num==0){if(B.num==0||C.num==0||D.den==0)X.num=0, X.den=1;else if(B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL){X.num=-B.num*C.num*D.den;X.den=B.den*C.den*D.num;SIMPLIFIE(X);}else{frac uu;if(DEBUG2)printf("++ %d/%d %d/%d %d/%d %d/%d \n",A.num,A.den,B.num,B.den,C.num,C.den,D.num,D.den);MUL(uu,B,C);DIV(X,uu,D);X.num=-X.num;if(DEBUG2)printf("%d/%d\n",X.num,X.den);}} \
else if(B.num==0||C.num==0||D.den==0)X.num=A.num,X.den=A.den; \
else if(D.num==1&&A.den==1&&B.den==1&&C.den==1)X.den=1,X.num=A.num-B.num*C.num*D.den; \
else if(A.den<MAXVAL && B.den<MAXVAL && C.den<MAXVAL && D.num<MAXVAL){X.num=A.num*B.den*C.den*D.num-A.den*B.num*C.num*D.den;X.den=A.den*B.den*C.den*D.num;SIMPLIFIE(X);} \
else{frac uu,vv,ww;if(DEBUG2)printf("%d/%d %d/%d %d/%d %d/%d \n",A.num,A.den,B.num,B.den,C.num,C.den,D.num,D.den); \
uu.num=B.num;vv.num=C.num;ww.num=D.den;uu.den=B.den;vv.den=C.den;ww.den=D.num; \
SIMPL(uu.num,vv.den);SIMPL(uu.num,ww.den);SIMPL(vv.num,uu.den);SIMPL(vv.num,ww.den);SIMPL(ww.num,uu.den);SIMPL(ww.num,vv.den); \
vv.num*=uu.num*ww.num;vv.den*=uu.den*ww.den; \
SUB(X,A,vv);if(DEBUG2)printf("%d/%d\n",X.num,X.den);}\
}
#define SUB(X,A,B) { \
if(A.num==0)X.num=-B.num,X.den=B.den; \
else if(B.num==0)X.num=A.num,X.den=A.den; \
else if(A.den==1&&B.den==1)X.num=A.num-B.num,X.den=1; \
else{long GDEN,AD,BD;AD=A.den,BD=B.den; \
  if(A.den>B.den)G(GDEN,AD,BD) \
  else G(GDEN,BD,AD); \
  if(GDEN!=1)AD=AD/GDEN,BD=BD/GDEN; \
  X.num=A.num*BD-B.num*AD;X.den=AD*BD; \
  if(GDEN!=1){SIMPLIFIE(X);SIMPL(X.num,GDEN);X.den=X.den*GDEN;} \
}}
#define VIDE 3
#define EQUATION 1
#define INEQUATION 2

Psysteme sc_min(Psysteme sc)
{
  /* nature vide=3 , equation=1 , inequation=2 */
    typedef struct {int numero ; int nature ; } rangee ;
    typedef struct {long num ; long den ; } f ;
 
    rangee ligne[MAX_VAR] ; int taille ;
    rangee colonne[MAX_VAR] ;
    f t[MAX_VAR][MAX_VAR] ;
    f frac0={0,1}, frac1={1,1} ;
    
    Pcontrainte pc ;
    Pvecteur pv ;
    int premier_hash = PTR_NIL ; /* tete de liste des noms de variables */
    static struct { Variable nom; int numero; int hash ; int val ; int succ ; } hashtable[MAX_VAR] ;
      /* Necessaire de declarer "hashtable" static 
       *  pour initialiser tout automatiquement a` 0.
       * Necessaire de chainer les enregistrements
       *  pour reinitialiser a 0
       *  en sortie de la procedure.
       */
    frac rapport1, rapport2, min1, min2, pivot, quot1, quot2, cc ;
    long compteur ; /* compte les variables */
    long valeur,i,j,k,h, numeroligne ;

    void printfrac(f x) {
        printf(" %3.1d/%-3.1d",x.num,x.den) ;
    }
    
    void dump_table(int taille,rangee ligne[MAX_VAR],rangee colonne[MAX_VAR],f t[MAX_VAR][MAX_VAR]) {
      int i,j ;

      printf("Taille=%2.1d\n",taille) ;
      if(taille>0) {
        for(i=0;i<taille;i++) { printf( " %4.1d(%1.1d)",colonne[i].numero,colonne[i].nature) ; }
        printf("\n\n");
        for(i=0;i<taille;i++) { printf( " %4.1d(%1.1d)",ligne[i].numero,ligne[i].nature) ; 
          for(j=0;j<taille;j++) printfrac(t[i][j]) ;
          printf("\n") ;
        }
      }
    }

    int hash(Variable s) /* calcule le hashcode d'une chaine
     sous forme d'un nombre compris entre 0 et  MAX_VAR */
    { int i ;
      i=((long)s % MAX_VAR);
      return (i) ;
    }

    int entree() {
      /* retourne le nume'ro de l'entree de la Variable pv->var du
       * Psysteme sc dans la table hashcodee, apres l'avoir creee si
       * elle n'existait pas
       */
        int h, trouve ;
        h = hash(pv->var) ; trouve=0 ;
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
        }
        return (h) ;
    }

    void pivoter(int taille, rangee ligne[], rangee colonne[], f *t[],
                 int i, int j)
    {  /* Pivote le tableau t ayant les caracteristiques
        * taille ligne[] et colonne[] autour du pivot de
        * coordonnees i et j  */
        
        int i1,j1 ;
        f x ;

        for(i1=0; i1<taille;i1++)
            if((i1 != i) && (j1 != j)) { /* Cas general */
                PIVOT(x,t[i1][j1],t[i1][j],t[i][j1],t[i][j])
                AFF(t[i1][j1],x)
            } else if((i1 != i) && (j1 == j)) {
                PIVOT(x,frac0,t[i1][j],frac1,t[i][j])
                AFF(t[i1][j1],x)
            } else if(i1==i) AFF(t[i][j],frac1) ;
      /* intervertir les variables de la ligne et de la colonne pivot */
        i1=colonne[j].numero ;
        colonne[j].numero=ligne[i].numero ;
        ligne[i].numero=i1 ;
        i1=colonne[j].nature ;
        colonne[j].nature=ligne[i].nature ;
        ligne[i].nature=i1 ;
    }

    if(sc_empty_p(sc)) return sc_empty(sc->base) ;
    taille=NB_EQ+NB_INEQ ;

/* 1) Recherche des variables a` valeur totalement determinee */

    for(pc=sc->egalites, i=0 ; pc!=0 ; pc=pc->succ,i++ ) {
        valeur = 0 ;   /* le terme cst vaut 0 par defaut */
        for(pv=pc->vecteur, j=0 ; pv!=0 ; pv=pv->succ) {
            if(NONTCST) { h=entree() ; j++ ; }
            else valeur = - pv->val ;
        }
        if(j==1) { ligne[i].nature = VIDE ;
            if(hashtable[h].val == PTR_NIL)
                hashtable[h].val = valeur ;
            else if(hashtable[h].val != valeur)
                return(sc_empty(sc->base)) ;
        }
    }

/* 2) Enregistrement des egalites */

    for(pc=sc->egalites, numeroligne=1 ; pc!=0 ; pc=pc->succ, numeroligne++ ) {
        if( ligne[numeroligne].nature==VIDE ) continue ;
        valeur = 0 ;   /* le terme cst vaut 0 par defaut */
        for(pv=pc->vecteur, j=0 ; pv!=0 ; pv=pv->succ) {
            if(NONTCST) { h=entree() ; t[numeroligne][NUMERO].num=pv->val ;
                t[numeroligne][NUMERO].den=1 ;
            }
            else t[numeroligne][0].num= - pv->val,t[numeroligne][NUMERO].den=1;
        }
    }
    dump_table(compteur, ligne, colonne, t) ;

/* 3) Enregistrement des inegalites */

    for(pc=sc->inegalites ; pc!=0 ; pc=pc->succ, numeroligne++ ) {
        valeur = 0 ;   /* le terme cst vaut 0 par defaut */
        for(pv=pc->vecteur, j=0 ; pv!=0 ; pv=pv->succ) {
            if(NONTCST) { h=entree() ; t[numeroligne][NUMERO].num=pv->val ;
                t[numeroligne][NUMERO].den=1 ;
            }
            else t[numeroligne][0].num= - pv->val,t[numeroligne][NUMERO].den=1;
        }
    }

}     /* FIN DE sc_min */

main(int argc, char *argv[])
{
/*  Programme de test de faisabilite'
 *  d'un ensemble d'equations et d'inequations.
 */
    FILE * f1;
    Psysteme sc=sc_new(),sc1; 
    int i; /* compte les systemes, chacun dans un fichier */

/* lecture et test de la faisabilite' de systemes sur fichiers */

    if(argc>=2) for(i=1;i<argc;i++){
        if((f1 = fopen(argv[i],"r")) == NULL) {
    	      fprintf(stdout,"Ouverture fichier %s impossible\n",
    		      argv[1]);
    	      exit(4);
        }
        printf("systeme initial \n");
        if(sc_fscan(f1,&sc)) {
            fprintf(stdout,"syntaxe correcte dans %s\n",argv[i]);
            sc_fprint(stdout, sc, *variable_default_name);
            printf("Nb_eq %d , Nb_ineq %d, dimension %d\n",
                NB_EQ, NB_INEQ, DIMENSION) ;
            if(sc_empty_p(sc1=sc_min(sc)))
                printf("Systeme infaisable (insoluble) en rationnels\n") ;
            else { printf("Systeme minimum :\n");
                sc_fprint(stdout,sc1,*variable_default_name) ;
            }
            fclose(f1) ;
        }
        else {
            fprintf(stderr,"erreur syntaxe dans %s\n",argv[1]);
            exit(1);
        }
    }
    else { f1=stdin ;

/* lecture et test de la faisabilite' du systeme sur stdin */

        printf("systeme initial \n");
        if(sc_fscan(f1,&sc)) {
	    fprintf(stdout,"syntaxe correcte dans %s\n",argv[1]);
	    sc_fprint(stdout, sc, *variable_default_name);
            printf("Nb_eq %d , Nb_ineq %d, dimension %d\n",
                NB_EQ, NB_INEQ, DIMENSION) ;
            if(sc_empty_p(sc1=sc_min(sc)))
                printf("Systeme infaisable (insoluble) en rationnels\n") ;
            else { printf("Systeme minimum :\n");
                sc_fprint(stdout,sc1,*variable_default_name) ;
            }

            exit(0) ;
        }
        else {
	    fprintf(stderr,"erreur syntaxe dans %s\n",argv[1]);
	    exit(1);
        }
    }
    exit(0) ;
} /*  FIN DE main */

