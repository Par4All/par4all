 /* Traitement des affectations
  *
  * Malik Imadache
  *
  * Modifie par Francois Irigoin, le 30 mars 1989:
  *  - reprise des includes
  *  - compatibilisation des appels a la bibliotheque "vecteur" (liblin);
  *    modification de "affect_eq"
  *  - reecriture de affect_eq (buggee); remplacement par affect_eg pour
  *    les equations et affect_ineg pour les inegalites;
  *  - modification de affect_sc_av
  *  - a moyen terme, il faudrait remplacer une succession d'affectations
  *    contenus dans un BAS (block of assignments) par une unique affectation
  *    multiple
  */
 
 /* Probleme mathematique: effet des transformations lineaires inversibles
  * sur les predicats de type egalite.
  *
  * Avec comme cas particulier l'affectation lineaire inversible
  *
  * Soit u un vecteur contenant les variables entieres et representant
  * l'etat de la memoire. Soit t la forme lineaire definissant la
  * precondition:
  *    t . u + tk = 0
  * Soit u' l'etat de la memoire apres l'affectation et t' la post-condition:
  *    t' . u' + tk' = 0
  * Soit A la matrice et ak le vecteur de termes constants donnant la
  * transformation lineaire appliquee a l'etat u:
  *    u' = A u + ak
  * On a donc:
  *    u = inv(A) (u'-ak)
  * et:
  *    t . inv(A) u' - t . inv(A) . ak + tk = 0
  * D'ou:
  *    t' = t . inv(A)
  *    tk' = tk - t . inv(A) . ak
  * Et reciproquement:
  *    t' . (A u + ak) + tk' = 0
  *    t' . A . u + t' . ak + tk' = 0
  * D'ou:
  *    t = t' . A
  *    tk = t' . ak + tk'
  *
  * Dans le cas d'une affectation lineaire inversible Fortran, la
  * matrice A et le vecteur ak ont une forme speciale:
  *
  *     ( a1 a2 a3 ... )       ( ak1 )
  * A = (  0  1  0 ... )  ak = (  0  )
  *     (  0  0  1 ... )       (  :  )
  *
  * On a alors:
  *
  *     -1   ( 1 -a2 -a3 ... )
  * a1 A   = ( 0  a1  0  ... )
  *          ( 0  0   a1 ... )
  *
  * (on a suppose que la variable transformee correspondait a la premiere
  * coordonnee de u)
  *
  * En multipliant l'invariant par a1 pour rester en nombres entiers,
  * on en deduit:
  *   t' = a1 t - t1 a + t1 e1
  * ou e1 est le premier vecteur de base et a = (a1 a2 a3 ...) la premiere
  * ligne de A, i.e. la partie lineaire de l'expression affectee. Pour tk':
  *   tk' = a1 tk - t1 ak1
  * Les termes constant peuvent donc etre traites comme le reste.
  * Il ne reste qu'a normaliser t' et tk' par le PGCD de t'.
  *
  * En analyse en avant, il faut donc effectuer une combinaison lineaire
  * et une addition, suivies d'une normalisation par le PGCD.
  * 
  * En analyse en arriere:
  *   t = t' + t'1 . a - t'1 . e1
  *   tk = tk' + t'1 ak1
  * C'est aussi unifiable en une combinaison lineaire, suivi d'une sous-traction
  * et d'une normalisation.
  *
  * Pour les predicats de type inegalite, et en analyse en avant, la 
  * multiplication par a1 n'est valable que si a1 est positif. Si
  * a1 est negatif l'inegalite change de signe. Il faut donc multiplier
  * le predicat par la valeur absolue de a1:
  *   t . u + tk <= 0
  *   t . |a1| u + |a1| tk <= 0
  * Si a1 est positif, le resultat precedente reste bon:
  *    t' = t . inv(A)
  *    tk' = tk - t . inv(A) . ak
  *    t' = a1 t - t1 a + t1 e1
  *    tk' = a1 tk - t1 ak1
  * Si a1 est negatif, on doit utiliser:
  *
  *       -1   ( -1   a2  a3 ... )
  * |a1| A   = (  0  -a1   0 ... )
  *            (  0    0 -a1 ... )
  *
  *    t' = t1 a - a1 t - t1 e1
  *    tk' = t1 ak1 - a1 tk
  * Il faut donc multiplier le resultat obtenu avec l'algorithme pour
  * a1 positif par -1.
  */

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "polyedre.h"

Value vect_apply_exp(vec,exp)
Pvecteur vec,exp;
{
    Value vs = vect_prod_scal(vec,exp),
          vc = vect_coeff(TCST,exp);
    return value_plus(vs,vc);
}

 /* expr_inverse transforme une expression en son inverse
  * (ie change tous les coefficients de signe) et
  * donne un coefficient de 1 a la variable parametre;
  *
  * retourne le coefficient de cette variable
  * avant transformation
  */
 /*
  * int expr_inverse(exp,v)
  * Pvecteur exp;
  * Variable v;
  * {
  *    Pvecteur var_val;
  *    int coeff=0;
  *    for (var_val=exp;var_val!=NULL;var_val=var_val->succ) {
  *	if (var_val->var==v) {
  *	    coeff = var_val->val;
  *	    var_val->val = 1;
  *	}
  *	else var_val->val = -var_val->val;
  *    }
  *    return(coeff);
  * }
  */

 /* expr_remake est exactement l'inverse de la precedente,
  * ie elle change le signe de tous les coefficients et rend
  * son coefficient initial a la variable parametre
  */
 /*
  * int expr_remake(exp,v,coeff)
  * Pvecteur exp;
  * Variable v;
  * int coeff;
  * {
  *     Pvecteur var_val;
  *     for (var_val=exp;var_val!=NULL;var_val=var_val->succ) {
  * 	if (var_val->var==v) var_val->val = coeff;
  * 	else var_val->val = -var_val->val;
  *     }
  * }
  */

/* affect_eg: modification de eg pour representer le nouveau predicat apres
 * l'affectation de l'expression lineaire exp a la variable v
 *
 * En entree de la routine, eg represente une pre-condition de l'affectation
 * v = exp; en sortie, eg represente la post-condition correspondante.
 *
 * Lire les explications donnees en debut de fichier
 */
void affect_eg(eg,v,exp)
Pcontrainte eg;
Pvecteur exp;
Variable v;
{
    Pvecteur new_pred = (Pvecteur) NULL;
    Pvecteur old_pred = eg->vecteur;
    /* coefficient associe a v dans exp */
    Value av = vect_coeff(v,exp);
    /* coefficient associe a v dans eg->vecteur */
    Value tv = vect_coeff(v,old_pred);
    /* terme constant du predicat t, remis en partie gauche */
    Value tk = value_uminus(vect_coeff(TCST,old_pred));
    /* terme constant du nouveau predicat t' */
    Value tkp ;

    /* comme le terme constant est implicitement a droite, il faut en
       changer le signe */
    vect_chg_coeff(&old_pred,TCST,tk);

    new_pred = vect_cl2(av,old_pred,value_uminus(tv),exp);
    /* ajout du vecteur de base tv ev */
    vect_chg_coeff(&new_pred,v,tv);

    /* passage du terme constant a droite */
    tkp = vect_coeff(TCST,new_pred);
    vect_chg_coeff(&new_pred,TCST,value_uminus(tkp));

    vect_rm(old_pred);
    eg->vecteur = new_pred;
    /* normalisation de l'equation eg; dans ce cas, la normalisation doit
       toujours pouvoir etre realisee car l'equation doit toujours etre
       faisable */
    if(egalite_normalize(eg)==FALSE) {
	(void) fprintf(stderr,"affect_eg: algorithmic error\n");
	abort();
    }
}

/* affect_ineg: modification de ineg pour representer le nouveau predicat apres
 * l'affectation de l'expression lineaire exp a la variable v
 *
 * En entree de la routine, ineg represente une pre-condition de l'affectation
 * v = exp; en sortie, ineg represente la post-condition correspondante.
 */
void affect_ineg(ineg,v,exp)
Pcontrainte ineg;
Pvecteur exp;
Variable v;
{
    Pvecteur new_pred = (Pvecteur) NULL;
    Pvecteur old_pred = ineg->vecteur;
    /* coefficient associe a v dans exp */
    Value av = vect_coeff(v,exp);
    /* coefficient associe a v dans eg->vecteur */
    Value tv = vect_coeff(v,old_pred);
    /* terme constant du predicat t, remis en partie gauche */
    Value tk = value_uminus(vect_coeff(TCST,old_pred));
    /* terme constant du nouveau predicat t' */
    Value tkp ;

    /* comme le terme constant est implicitement a droite, il faut en
       changer le signe dans le predicat; par contre, le signe du terme
       constant est correct dans l'expression */
    vect_chg_coeff(&old_pred,TCST,tk);

    new_pred = vect_cl2(av,old_pred,value_uminus(tv),exp);
    /* ajout du vecteur de base tv ev */
    vect_chg_coeff(&new_pred,v,tv);

    /* passage du terme constant a droite */
    tkp = vect_coeff(TCST,new_pred);
    vect_chg_coeff(&new_pred,TCST,value_uminus(tkp));

    /* correction de signe si le predicat a ete multiplie par un
       nombre negatif */
    if(value_neg_p(av)) 
	vect_chg_sgn(new_pred);

    vect_rm(old_pred);
    ineg->vecteur = new_pred;
    /* une inegalite est toujours faisable: inegalite_normalize ne peut pas rater */
    (void) inegalite_normalize(ineg);
}

/*
 effet d'une affectation inversible sur une contrainte pour la resolution
	en arriere :
	c'est la fonction de substitution inverse a la precedente;
	pour cela nous modifierons l'expression d'affectation afin 
	de nous ramener a une substitution.
*/

void back_affect_eq(eg,v,exp)

Pcontrainte eg;
Pvecteur exp;
Variable v;

{
    Value cv;
    Value c;
    cv = expr_inverse(exp,v);
    if ((c=vect_coeff(v,eg->vecteur))) {
	affect_eg(eg,v,exp);
	vect_chg_coeff(&(eg->vecteur),v,value_mult(cv,c));
    }
    expr_remake(exp,v,cv);
}


/* 
 effet d'une affectation inversible sur un sommet :
  - en avant
	il suffit de changer le coefficient de la variable cible par
	la valeur de l'expression d'affectation au point defini par
	les coordonnees du sommet.
  - en arriere
	la manipulation est la meme a ceci pret qu'il faut former
	l'expression inverse; ceci entraine une manipulation du
	denominateur afin de respecter les regles de calcul en
	nombres entiers.
*/

void back_affect_som(s,v,exp)
Psommet s;
Variable v;
Pvecteur exp;
{
    Value value,cv,den,tc,new_tc;
    cv = expr_inverse(exp,v);

    tc = vect_coeff(TCST,exp);
    den = sommet_denominateur(s);
    new_tc = value_mult(den,tc);
    /* change the value in expr */
    vect_chg_coeff(&exp,TCST,new_tc);

    value = vect_apply_exp(s->vecteur,exp);

    /* restore the initial value! */
    vect_chg_coeff(&exp,TCST,tc);
    (void) vect_multiply(s->vecteur,cv);
    vect_chg_coeff(&(s->vecteur),v,value);
    value_product(sommet_denominateur(s),value_abs(cv));
    expr_remake(exp,v,cv);
}

void affect_som(s,v,exp)
Psommet s;
Variable v;
Pvecteur exp;
{
    Value term_const = vect_coeff(TCST,exp), ntc,ntv;
    ntc = value_mult(sommet_denominateur(s),term_const);
    vect_chg_coeff(&exp,TCST,ntc);
    ntv = vect_apply_exp(s->vecteur,exp);
    vect_chg_coeff(&(s->vecteur),v,ntv);
    vect_chg_coeff(&exp,TCST,term_const);
}

/* affect_lvect: effet d'une affectation inversible sur un rayon
 * ou une droite
 *
 * aussi appele pour les affectations non-inversibles (Francois Irigoin)
 *
 * Attention! ca cree des maillons pointant vers des vecteurs nuls
 * qu'il faut eliminer ensuite
 *
 * comme pour les sommets sauf les problemes de denominateur
 * qui sont elimines.
 */
void affect_lvect(rd,v,exp)
Pray_dte rd;
Variable v;
Pvecteur exp;
{
    vect_chg_coeff(&(rd->vecteur),v,
				 vect_prod_scal(rd->vecteur,exp));
}


void back_affect_lvect(rd,v,exp)
Pray_dte rd;
Variable v;
Pvecteur exp;
{
    Value value,cv;
    cv = expr_inverse(exp,v);
    value = vect_prod_scal(rd->vecteur,exp);
    (void) vect_multiply(rd->vecteur,cv);
    vect_chg_coeff(&(rd->vecteur),v,value);
    expr_remake(exp,v,cv);
}

/* affectation en avant (sg puis sc, et enfin poly) */

/* affect_sg_av: effet d'une affectation inversible lors de l'analyse en avant 
 * sur un systeme generateur
 */
void affect_sg_av(sg,v,exp)
Ptsg sg;
Variable v;
Pvecteur exp;
{
    Psommet s;
    Pray_dte rd;
    int nv;

    (void) fprintf(stderr,"affect_sg_av: begin\n");

    /* traitement des sommets: peut-etre faudrait-il traiter la
       redondance qui peut s'en suivre? */
    for (s=sg_sommets(sg);s!=NULL;s=s->succ) 
	affect_som(s,v,exp);

    for (rd=sg_rayons(sg);rd!=NULL;rd=rd->succ) 
	affect_lvect(rd,v,exp);
    /* suppression des vecteurs nuls qui peuvent avoir apparus */
    sg_rayons(sg) = elim_null_vect(sg_rayons(sg),&nv);
    sg->rays_sg.nb_v = nv;

    for (rd=sg_droites(sg);rd!=NULL;rd=rd->succ) 
	affect_lvect(rd,v,exp);
    /* suppression des vecteurs nuls qui peuvent avoir apparus */
    sg_droites(sg) = elim_null_vect(sg_droites(sg),&nv);
    sg->dtes_sg.nb_v = nv;

    (void) fprintf(stderr,"affect_sg_av: end\n");
}

/* 
  effet d'une affectation inversible lors de l'analyse en avant 
	sur un systeme de contraintes
*/
void affect_sc_av(sc,v,exp)
Psysteme sc;
Variable v;
Pvecteur exp;
{
    Pcontrainte eg;
    for(eg=sc->egalites;eg!=NULL;eg=eg->succ) {
	if (vect_coeff(v,eg->vecteur)) affect_eg(eg,v,exp);
    }
    for(eg=sc->inegalites;eg!=NULL;eg=eg->succ) {
	if (vect_coeff(v,eg->vecteur)) affect_ineg(eg,v,exp);
    }
}

/* 
  effet d'une affectation inversible lors de l'analyse en avant
	sur tout un polyedre

 ATTENTION on agit physiquement sur le polyedre donne

*/

Ppoly affect_av(p,v,exp)
Ppoly p;
Variable v;
Pvecteur exp;
{
    if (p==NULL) return((Ppoly)NULL);
    else {
	if (poly_nbre_sommets(p)==0) return(p); /* polyedre vide */
	affect_sc_av(p->sc,v,exp);
	affect_sg_av(p->sg,v,exp);
	return(p);
    }
}

/**************************************************************************/
/*  ET MAINTENANT EN ARRIERE                                              */
/**************************************************************************/


/* 
  effet d'une affectation inversible lors de l'analyse en avant 
	sur un systeme de contraintes
*/
void affect_sg_ar(sg,v,exp)
Ptsg sg;
Variable v;
Pvecteur exp;
{
    Psommet s;
    Pray_dte rd;

    for (s=sg_sommets(sg);s!=NULL;s=s->succ) 
	back_affect_som(s,v,exp);
    for (rd=sg_rayons(sg);rd!=NULL;rd=rd->succ) 
	back_affect_lvect(rd,v,exp);
    for (rd=sg_droites(sg);rd!=NULL;rd=rd->succ) 
	back_affect_lvect(rd,v,exp);
}


/* 
  effet d'une affectation inversible lors de l'analyse en avant 
	sur un systeme de contraintes
*/
void affect_sc_ar(sc,v,exp)
Psysteme sc;
Variable v;
Pvecteur exp;
{
    Pcontrainte eg;
    for(eg=sc->egalites;eg!=NULL;eg=eg->succ) {
	if (vect_coeff(v,eg->vecteur)) back_affect_eq(eg,v,exp);
    }
    for(eg=sc->inegalites;eg!=NULL;eg=eg->succ) {
	if (vect_coeff(v,eg->vecteur)) back_affect_eq(eg,v,exp);
    }
}

Ppoly affect_ar(p,v,exp)
Ppoly p;
Variable v;
Pvecteur exp;
{
    if (p==NULL) return((Ppoly)NULL);
    else {
	if (poly_nbre_sommets(p)==0) return(p);
	affect_sc_ar(p->sc,v,exp);
	affect_sg_ar(p->sg,v,exp);
	return(p);
    }
}
