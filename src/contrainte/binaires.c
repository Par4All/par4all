 /* package contrainte - operations binaires */

#include <stdio.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

int contrainte_subst_ofl(v,def,c,eq_p)
Variable v;
Pcontrainte def,c;
boolean eq_p;
{
    return( contrainte_subst_ofl_ctrl(v,def,c,eq_p, FWD_OFL_CTRL)); 
}

int contrainte_subst(v,def,c,eq_p)
Variable v;
Pcontrainte def,c;
boolean eq_p;
{
    return( contrainte_subst_ofl_ctrl(v,def,c,eq_p, NO_OFL_CTRL)); 

}



Pcontrainte inegalite_comb(posit,negat,v)	
Pcontrainte posit, negat;
Variable v;
{
    return( inegalite_comb_ofl_ctrl(posit,negat,v, NO_OFL_CTRL));
}

Pcontrainte inegalite_comb_ofl(posit,negat,v)	
Pcontrainte posit, negat;
Variable v;
{
    return( inegalite_comb_ofl_ctrl(posit,negat,v, FWD_OFL_CTRL));

}


/* int contrainte_subst_ofl_ctrl(Variable v, Pcontrainte def, Pcontrainte c  
 * Boolean eq_p, int ofl_ctrl):
 * elimination d'une variable v entre une equation def et une contrainte,
 * egalite ou inegalite, c. La contrainte c est modifiee en substituant
 * v par sa valeur impliquee par def.
 *
 * La contrainte c est interpretee comme une inegalite et la valeur retournee
 * vaut:
 *  -1 si la contrainte c est trivialement verifiee et peut etre eliminee
 *   0 si la contrainte c est trivialement impossible
 *   1 sinon (tout va bien)
 * Si la contrainte c passee en argument etait trivialement impossible
 * ou trivialement verifiee, aucune subsitution n'a lieu et la valeur 1
 * est retournee.
 *
 * La substitution d'une variable dans une inegalite peut aussi introduire
 * une non-faisabilite testable par calcul de PGCD, mais cela n'est pas
 * fait puisqu'on ne sait pas si c est une egalite ou une inegalite. Le
 * traitement du terme constant n'est pas decidable.
 *
 * Note: il faudrait separer le probleme de la combinaison lineaire
 * a coefficients positifs de celui de la faisabilite et de la trivialite
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier setjmp(overflow_error) effectue.
 */
int contrainte_subst_ofl_ctrl(v,def,c,eq_p, ofl_ctrl)
Variable v;
Pcontrainte def,c;
boolean eq_p;
int ofl_ctrl;
{
    Pvecteur save_c;

    /* cv_def = coeff de v dans def */
    int cv_def = vect_coeff(v,def->vecteur);
    /* cv_c = coeff de v dans c */
    int cv_c = vect_coeff(v,c->vecteur);

    /* il faut que cv_def soit non nul pour que la variable v puisse etre
       eliminee */
    assert(cv_def!=0);

    /* il n'y a rien a faire si la variable v n'apparait pas dans la
       contrainte c */
    /* substitution inutile: variable v absente */
    if (cv_c==0) return (1);

    /* on garde trace de la valeur de c avant substitution pour pouvoir
       la desallouer apres le calcul de la nouvelle */
    save_c = c->vecteur;
    /* on ne fait pas de distinction entre egalites et inegalites, mais
       on prend soin de toujours multiplier la contrainte, inegalite
       potentielle, par un coefficient positif */
    if (cv_def<0) {
	c->vecteur = vect_cl2_ofl_ctrl(-cv_def,c->vecteur,cv_c,
				       def->vecteur,ofl_ctrl);
    }
    else {
	c->vecteur = vect_cl2_ofl_ctrl(cv_def,c->vecteur,-cv_c,
				  def->vecteur, ofl_ctrl);
    }
    vect_rm(save_c);

    /* reste malikien: cette partie ne peut pas etre faite sans savoir
       si on est en train de traiter une egalite ou une inegalite */
    if(contrainte_constante_p(c)) {
	if(contrainte_verifiee(c,eq_p)) 
	    /* => eliminer cette c inutile */
	    return(-1);	
	else 
	    /* => systeme non faisable      */
	    return(FALSE);
    }
    return(TRUE);
}




/* Pcontrainte inegalite_comb_ofl_ctrl(Pcontrainte posit, Pcontrainte negat, 
 *                            Variable v, int ofl_ctrl):
 * combinaison lineaire positive des deux inegalites posit et negat
 * eliminant la variable v.
 * 
 * Une nouvelle contrainte est allouee et renvoyee.
 *
 * Si le coefficient de v dans negat egale -1 ou si le coefficient de
 * v dans posit egale 1, la nouvelle contrainte est equivalente en
 * nombres entiers avec posit et negat.
 *
 * Modifications:
 *  - use gcd to reduce the combination coefficients in hope to reduce
 *    integer overflow risk (Francois Irigoin, 17 December 1991)
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier setjmp(overflow_error) effectue.
 */
Pcontrainte inegalite_comb_ofl_ctrl(posit,negat,v, ofl_ctrl)	
Pcontrainte posit, negat;
Variable v;
int ofl_ctrl;
{
    int cv_p, cv_n;
    int d;
    Pcontrainte ineg;

    cv_p = vect_coeff(v,posit->vecteur); 
    cv_n = vect_coeff(v,negat->vecteur);

    assert(cv_p>0 && cv_n<0);

    if((d = pgcd(cv_p, -cv_n))!=1) {
	cv_p = cv_p/d;
	cv_n = cv_n/d;
    }

    ineg = contrainte_new();

    ineg->vecteur = vect_cl2_ofl_ctrl(cv_p,negat->vecteur,-cv_n,
				      posit->vecteur, ofl_ctrl);
    return(ineg);
}



/* Value eq_diff_const(Pcontrainte c1, Pcontrainte c2):
 * calcul de la difference des deux termes constants des deux 
 * equations c1 et c2
 *
 * Notes:
 *  - cette routine fait l'hypothese que CONTRAINTE_UNDEFINED=>CONTRAINTE_NULLE
 *
 * Modifications:
 *  - renvoie d'une valeur non nulle meme si une des contraintes est nulles
 */
Value eq_diff_const(c1,c2)
Pcontrainte c1,c2;
{
    Value b;

    if(c1!=NULL) 
	if(c2!=NULL) {
	    int b1 = vect_coeff(TCST,c1->vecteur);
	    int b2 = vect_coeff(TCST,c2->vecteur);
	    b = b1 - b2;
	}
	else
	    b = vect_coeff(TCST,c1->vecteur);
    else
	if(c2!=NULL)
	    b = - vect_coeff(TCST,c2->vecteur);
	else
	    b = 0;

    return (b);
}

/* Pcontrainte contrainte_append(c1, c2)
 * Pcontrainte c1, c2;
 *
 * append directly c2 to c1. Both c1 and c2 are not relevant 
 * when the result is returned.
 */
Pcontrainte contrainte_append(c1, c2)
Pcontrainte c1, c2;
{
    Pcontrainte c;
    
    if (c1==NULL) return(c2);
    if (c2==NULL) return(c1);
    for (c=c1; c->succ!=NULL; c=c->succ);
    c->succ=c2;
    return(c1);
}
