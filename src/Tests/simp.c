
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
#define PTR_NIL -32001
#define INFINI 32700
#define NB_INEQ sc->nb_ineq
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
#define CREVARVISIBLE variables[compteur-3]=compteur-2;
#define CREVARCACHEE { variablescachees[nbvariables]=nbvariables + MAX_VAR ; nbvariables ++ ; }

frac frac0={0,0,0} ;

/* Le nombre de variables visibles est : compteur-2
 * La i-eme variable visible a le numero : variables[i+1]=i
 *   (0 <= i < compteur-2)
 * Le nombre de variables cachees est : nbvariables
 * La i-eme variable cachee a le numero : variablescachees[i+1]=MAX_VAR+i-1
 *   (0 <= i < nbvariables)
 */
int nbvariables, variablescachees[MAX_VAR], variables[MAX_VAR] ; /* utilise'es par dump_tableau ; a rendre local */

printfrac(frac x) {
    printf(" %3.1d/%-3.1d",x.num,x.den) ;
}

void dump_tableau(tableau *t,int colonnes) {
    int i,j, k, w, max = 0 ;
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
          printf("ligne %d valeur %d/%d\n",
            t[i].colonne[j].numero,t[i].colonne[j].num,
            t[i].colonne[j].den) ;
      }
    }
} /* DEBUG */
  
    for(j=0;j<=max;j++) { printf("\nLigne %d ",j) ;
        for(i=0;i<colonnes;i++) {
            w=1 ;
            for(k=0;k<t[i].taille;k++)
                if(t[i].colonne[k].numero==j) printfrac(t[i].colonne[k]) , w=0 ;
            if(w!=0)printfrac(frac0) ;
        }
    }
    printf("\n");
} /* dump_tableau */


int hash(char* s) ;
                 
int feasible(Psysteme sc) {
/* fonction de calcul de la faisabilite' d'un systeme
 * d'equations et d'inequations
 * Auteur : Robert Mahl, Date : janvier 1994
 */
/* Retourne : 1 si le systeme est soluble (faisable)
 *  en rationnels,
 * 0 s'il n'y a pas de sulution.
 */

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
    tableau *eg ; /* tableau des egalite's  */
    tableau *t ; /* tableau des inegalite's  */
     /* les colonnes 0 et 1 sont reservees au terme const: */
    int compteur = 2 ;
    long i, j, k, p, h, trouve, valeur, hh, ligne, poidsM, i0, i1, jj, ii ;
    long w ;
    int soluble ; /* valeur retournee par feasible */
    frac *nlle_colonne , *colo ;
    frac objectif[2] ; /* objectif de max pour simplex : 
        somme des (b2,c2) termes constants "inferieurs" */
    frac rapport1, rapport2, min1, min2, pivot, quot1, quot2, cc ;

/* Allocation a priori du tableau des egalites.
 * "eg" : tableau a "nb_eq" lignes et "dimension"+2 colonnes.
 */

    if(NB_EQ != 0)
    {   eg=(tableau*)malloc((3+DIMENSION)*sizeof(tableau)) ;
        for(i=0 ; i<(3+DIMENSION) ; i++) {
            eg[i].colonne=(frac*)malloc(NB_EQ*sizeof(frac)) ;
            eg[i].existe = 0 ;
            eg[i].taille = 0 ;
        }
    }

/* Determination d'un numero pour chaque variable */

    for(pc=sc->egalites, ligne=1 ; pc!=0; pc=pc->succ, ligne++) {
        j=0 ; /* compteur du nb de variables de l'equation */
        valeur=0 ; /* le terme cst vaut 0 par defaut */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) { j++ ;
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
                    CREVARVISIBLE;
                    hashtable[h].nom=pv->var ;
                }
                hh = h ;
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
        t[i].colonne=(frac *) malloc((1 + 2*NB_EQ + NB_INEQ)*sizeof(frac)) ;
        t[i].existe = 0 ;
        t[i].taille = 1 ;
        t[i].colonne[0].numero = 0 ;
        t[i].colonne[0].num = 0 ;
    }
    nbvariables= 0 ;
/* Initialisation de l'objectif */
    for(i=0;i<=1;i++) objectif[i].num=0, objectif[i].den=1 ;

    for(i=0;i<MAX_VAR;i++) if(hashtable[i].nom != 0) if(DEBUG)printf("%s %d %d\n",hashtable[i].nom,hashtable[i].numero,hashtable[i].val) ;
/* Entree des inegalites dans la table */

    for(pc=sc->inegalites, ligne=1; pc!=0; pc=pc->succ, ligne++) {
        valeur = 0 ;
        poidsM=0 ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) 
            if(vect_coeff(pv->var,sc_base(sc))) 
                poidsM += pv->val ;
            else valeur = - pv->val ; /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) {
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
                    CREVARVISIBLE ;
                }
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    t[NUMERO].colonne[0].num += pv->val,
/*
if(DEBUG)printf("pv->val = %d, t[NUMERO].colonne[0].num = %d\n",pv->val,t[NUMERO].colonne[0].num),
*/
                    t[NUMERO].colonne[0].den = 1 ;
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    i = -pv->val ; else i = pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num=i ;
                t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
                t[NUMERO].taille++ ;
            }
        }
              /* Creation de variable d'ecart ? */
        if(poidsM < 0 || poidsM==0 && valeur<0) {
    if(DEBUG)dump_tableau(t, compteur) ;
            i=compteur++ ;
            CREVARVISIBLE ;
            t[i].existe = 1 ; t[i].taille = 2 ;
            t[i].colonne[0].num = 1 ;
            t[i].colonne[0].den = 1 ;
if(DEBUG)printf("ligne ecart = %d, colonne %d\n",ligne,i) ;
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

    for(i=0;i<MAX_VAR;i++) if(hashtable[i].nom != 0) if(DEBUG)printf("%s %d %d\n",hashtable[i].nom,hashtable[i].numero,hashtable[i].val) ;
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
        valeur = 0 ;
        poidsM=0 ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                poidsM += pv->val ;
            else valeur = - pv->val ; /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) {
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
                    CREVARVISIBLE ;
                    hashtable[h].nom=pv->var ;
                }
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    t[NUMERO].colonne[0].num += pv->val,
                    t[NUMERO].colonne[0].den = 1 ;
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    i = -pv->val ; else i = pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num=i ;
                t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
                t[NUMERO].taille++ ;
            }
        }
              /* Creation de variable d'ecart ? */
        if(poidsM < 0 || poidsM==0 && valeur<0) {
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

    for(pc=sc->egalites ; pc!=0; pc=pc->succ, ligne++)
    {
        valeur = 0 ;
        poidsM=0 ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                poidsM -= pv->val ;
            else valeur = pv->val ; /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) {
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
                    CREVARVISIBLE ;
                }
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    t[NUMERO].colonne[0].num -= pv->val,
                    t[NUMERO].colonne[0].den = 1 ;
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(poidsM < 0 || poidsM==0 && valeur<0)
                    i = pv->val ; else i = - pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num=i ;
                t[NUMERO].colonne[t[NUMERO].taille].den = 1 ;
                t[NUMERO].taille++ ;
            }
        }
              /* Creation de variable d'ecart ? */
        if(poidsM < 0 || poidsM==0 && valeur<0) {
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

    nlle_colonne=(frac *) malloc((1 + 2*NB_EQ + NB_INEQ)*sizeof(frac)) ;
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
            if(DEBUG1){ printf("objectif : "); printfrac(objectif[0]) ; printfrac(objectif[1]) ; printf("\n") ;}
            if(EGAL(objectif[0],t[0].colonne[0]) &&
               EGAL(objectif[1],t[1].colonne[0])) {
                if(DEBUG1)printf("Systeme soluble (faisable) en rationnels\n") ;
                SOLUBLE(1)
            } else { if(DEBUG1)printf("Systeme insoluble (infaisable)\n") ;
                SOLUBLE(0)
            }
if(DEBUG1)printf("fin\n");
        }
if(DEBUG)printf("1 : jj= %d\n",jj) ;
    if(DEBUG2)dump_tableau(t, compteur) ; 


        /*  Recherche de la ligne de pivot  */
        METINFINI(min1) ; METINFINI(min2) ;
        for(i=1, i0=1, i1=1, ii=-1 ; i<t[jj].taille ; )
        {
            if((i0<t[0].taille && t[jj].colonne[i].numero<= t[0].colonne[i0].numero  || i0>=t[0].taille)
             && (i1<t[1].taille && t[jj].colonne[i].numero<= t[1].colonne[i1].numero || i1>=t[1].taille)) {
              if( POSITIF(t[jj].colonne[i])) {
                DIV(rapport1,((i0<t[0].taille&&t[jj].colonne[i].numero==t[0].colonne[i0].numero)?t[0].colonne[i0]:frac0),t[jj].colonne[i])
                DIV(rapport2,((i1<t[1].taille&&t[jj].colonne[i].numero==t[1].colonne[i1].numero)?t[1].colonne[i1]:frac0),t[jj].colonne[i])
                if(INF(rapport2,min2)||EGAL(rapport2,min2)&&INF(rapport1,min1)){
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
if(DEBUG)printf("i=%d i0=%d i1=%d   %d %d %d\n",i,i0,i1,t[jj].colonne[i].numero,t[0].colonne[i0].numero,t[1].colonne[i1].numero) ;
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
if(DEBUG2)printf("Pivoter %d %d\n",ii,jj) ;
        
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
                {   if(DEBUG)printf("k=%d, j=%d, i=%d i1=%d\n",k,j,i,i1);
                    if(DEBUG)printfrac(t[j].colonne[i]) ;
                    if(DEBUG)printfrac(t[jj].colonne[i1]) ;
                    if(DEBUG)printfrac(cc);
                    if(DEBUG)printfrac(pivot) ;
                    if(t[j].colonne[i].numero == ii) {
                        AFF(nlle_colonne[k],t[j].colonne[i])
                    } else {
                        PIVOT(nlle_colonne[k],t[j].colonne[i],t[jj].colonne[i1],cc,pivot) ;
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
                else if(i>=t[j].taille || i1<t[jj].taille && t[j].colonne[i].numero > t[jj].colonne[i1].numero)
                {   if(DEBUG)printf("t[j].colonne[i].numero > t[jj].colonne[i1].numero , k=%d, j=%d, i=%d i1=%d\n",k,j,i,i1);
            if(DEBUG)printf("j = %d  t[j].taille=%d , t[jj].taille=%d\n",j,t[j].taille,t[jj].taille);
            if(DEBUG)printf("t[j].colonne[i].numero=%d , t[jj].colonne[i1].numero=%d\n",t[j].colonne[i].numero,t[jj].colonne[i1].numero);
                        /* 0 en colonne j  ligne t[jj].colonne[i1].numero */
                    if(t[jj].colonne[i1].numero == ii) {
                        AFF(nlle_colonne[k],frac0)
                    } else {
                        PIVOT(nlle_colonne[k],frac0,t[jj].colonne[i1],cc,pivot) ;
                    }
                    if(i==0||nlle_colonne[k].num!=0) {
                        nlle_colonne[k].numero = t[jj].colonne[i1].numero ;
                        if(DEBUG)printfrac(nlle_colonne[k]) ;
                        if(DEBUG)printf(" ligne numero %d ", nlle_colonne[k].numero) ;
                        if(DEBUG)printf("\n") ;
                        k++ ;
                    }
                    if(i1<t[jj].taille) i1++ ; else i++ ;
                }
                else if(i1>=t[jj].taille || t[j].colonne[i].numero < t[jj].colonne[i1].numero)
                {       /* 0 en colonne jj  ligne t[j].colonne[i].numero */
                    if(DEBUG)printf("t[j].colonne[i].numero < t[jj].colonne[i1].numero , k=%d, j=%d, i=%d i1=%d\n",k,j,i,i1);
            if(DEBUG)printf("j = %d  t[j].taille=%d , t[jj].taille=%d\n",j,t[j].taille,t[jj].taille);
                    AFF(nlle_colonne[k],t[j].colonne[i]) ;
                    if(DEBUG)printfrac(nlle_colonne[k]) ;
                    if(i==0||nlle_colonne[k].num!=0) {
                        nlle_colonne[k].numero = t[j].colonne[i].numero ;
                        if(DEBUG)printf(" ligne numero %d ", nlle_colonne[k].numero) ;
                        if(DEBUG)printf("\n") ;
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
            if(DEBUG)printf("w = %d  t[w].taille=%d \n",w,t[w].taille);
                if(DEBUG)dump_tableau(t, compteur) ;
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
}     /* main */

int hash(Variable s) /* calcule le hashcode d'une chaine
 sous forme d'un nombre compris entre 0 et  MAX_VAR */
{ int i ;
  i=((long)s % MAX_VAR);
  return (i) ;
}

main(int argc, char *argv[])
{
/*  Programme de test de faisabilite'
 *  d'un ensemble d'equations et d'inequations.
 */
    FILE * f1;
    Psysteme sc=sc_new(); 
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
            if(feasible(sc)) printf("Systeme faisable (soluble) en rationnels\n") ;
            else printf("Systeme insoluble\n");
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
            if(feasible(sc)) printf("Systeme faisable (soluble) en rationnels\n") ;
            else printf("Systeme insoluble\n");
            exit(0) ;
        }
        else {
	    fprintf(stderr,"erreur syntaxe dans %s\n",argv[1]);
	    exit(1);
        }
    }
    exit(0) ;
}

