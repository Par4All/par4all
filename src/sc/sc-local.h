 /* package sc sur les Systemes de Contraintes lineaires. Une contrainte
  * peut etre une egalite lineaire ou une inegalite lineaire
  *
  * Malik Imadache, Corinne Ancourt, Neil Butler, Francois Irigoin,
  * Remi Triolet
  *
  * Autres packages necessaires:
  *  - types.h
  *  - boolean.h
  *  - vecteur.h
  *  - contrainte.h
  *
  * Modifications:
  *  - redefinition de la structure "Ssysteme"; le champ "nbvars" est renomme
  *    "dimension"; il reste de type "int"; le champ "num_var" est remplace
  *    par un champ "base" de type "Pbase"; le champ "base" ne contient pas
  *    le terme constant; FI, 13/12/89;
  */

#ifndef SYSTEME
/* constante definissant le type Systeme */
#define SYSTEME 1001

#include "arithmetique.h"

/* Le systeme vide "sc_empty" est represente par l'egalite "0==-1".
 * le systeme representant l'espace Rn "sc_rn" correspond au  systeme 
 * ne contenant aucune contrainte. 
 * Avant ces deux systemes etaient representes par Le pointeur (Psysteme) NULL.
 * Progressivement, les (Psysteme) NULL sont replaces par des appels aux
 *  fonctions sc_empty et sc_rn. 
 *
 * Le champ dimension donne le nombre de variables utilisees dans les egalites
 * et les inegalites, ou si l'on prefere, la dimension de l'espace dans
 * lequel est defini le polyedre correspondant. Le terme constant ne
 * fait pas partie de l'espace.
 *
 * La champ base contient tous les vecteurs de base, i.e. toutes les
 * variables apparaissant dans les egalites et les inegalites., y compris
 * le terme constant. La dimension de la base est donc superieure de 1 a la
 * dimension du systeme. Ce champ est utilise par des algorithmes comme
 * celui du test de faisabilite mais il n'est pas entretenu automatiquement.
 * Il faut penser a le regenerer avant de l'utiliser.
 */
typedef struct Ssysteme { 
	Pcontrainte egalites;
	Pcontrainte inegalites;
	int nb_eq;
	int nb_ineq;
	int dimension;
	Pbase base;
	} *Psysteme,Ssysteme;

/* MACROS */

/* #define sc_print(sc,f) sc_fprint(stdout,sc,f) */

#define sc_nbre_egalites(psc) ((psc)->nb_eq)
#define sc_nbre_inegalites(psc) ((psc)->nb_ineq)
#define sc_egalites(psc) ((psc)->egalites)
#define sc_inegalites(psc) ((psc)->inegalites)
#define sc_base(psc) ((psc)->base)
#define sc_dimension(psc) ((psc)->dimension)

/* void sc_add_egalite(Psysteme p, Pcontrainte e): macro ajoutant une
 * egalite e a un systeme p; la base n'est pas mise a jour; il faut faire
 * ensuite un appel a sc_creer_base(); il vaut mieux utiliser sc_make()
 *
 * sc_add_eg est (a peu pres) equivalent a sc_add_egalite, mais le
 * parametre e n'est utilise qu'une fois ce qui permet d'eviter
 * des surprises en cas de e++ et autres effects de bords a chaque
 * evaluation de e; sc_add_egalite est donc plus sur que sc_add_eg
 */
#define sc_add_egalite(p,e) { Pcontrainte e_new= (e); \
                              e_new->succ=(p)->egalites; \
                              (p)->egalites=e_new; (p)->nb_eq++; }
#define sc_add_eg(p,e) { (e)->succ=(p)->egalites; (p)->egalites=(e); (p)->nb_eq += 1; }

/* void sc_add_inegalite(Psysteme p, Pcontrainte i): macro ajoutant une
 * inegalite i a un systeme p; la base n'est pas mise a jour; il faut
 * ensuite faire un appel a sc_creer_base(); il vaut mieux utiliser
 * sc_make();
 *
 * sc_add_ineg est (a peu pres) equivalent a sc_add_inegalite; cf supra
 * pour l'explication des differences
 */
#define sc_add_inegalite(p,i) { Pcontrainte i_new= (i); \
                              i_new->succ=(p)->inegalites; \
                              (p)->inegalites=i_new; (p)->nb_ineq++; }
#define sc_add_ineg(p,i) { (i)->succ=(p)->inegalites; (p)->inegalites=(i); (p)->nb_ineq += 1; }

/* ex-definition d'un systeme de contraintes infaisable, representant un
 * polyedre vide.
 *
 * Utiliser sc_empty() et sc_empty_p() plutot que ces macros obsoletes.
 */
#define SC_EMPTY ((Psysteme) NULL)
#define SC_EMPTY_P(sc) ((sc)==SC_EMPTY)

/* ex-definition d'un systeme de contraintes vide, representant tout l'espace,
 * dont la base se trouve eventuellement dans "base" (quand ce champ est 
 * alloue); quand la base et la dimension ne sont pas definies, cela
 * represente un espace de dimension quelconque.
 *
 * Utiliser sc_rn() et sc_rn_p() plutot que ces macros obsoletes.
 */
#define SC_RN ((Psysteme) NULL)
#define SC_RN_P(sc) ((sc)==(Psysteme) NULL)

/* definition du systeme de contraintes non initialise
 */
#define SC_UNDEFINED ((Psysteme) NULL)
#define SC_UNDEFINED_P(sc) ((sc)==(Psysteme) NULL)

/* nombre maximum d'inequations que doit comporter un systeme lineaire
pour que l'elimination des redondances en nombres REELS s'effectue en un
temps raisonnable */
#define NB_INEQ_MAX1 100

/* nombre maximum d'inequations que doit comporter un systeme lineaire
pour que l'elimination des redondances en nombres ENTIERS s'effectue en
un temps raisonnable */
#define NB_INEQ_MAX2  50

/*  Nombre de contraintes au dela duquel l'algorithme du simplexe
    est preferable a l'algorithme de Fourier-Motzkin: 20
    (However, the average optimal is lower, or another decision
    procedure should be investigated)
*/
#define NB_CONSTRAINTS_MAX_FOR_FM 20

/* ensemble de macros permettant de compiler les programmes utilisant
les anciens noms des fonctions */

#define sc_faisabilite(sc) sc_rational_feasibility_ofl_ctrl((sc), NO_OFL_CTRL,TRUE) 
#define sc_faisabilite_ofl(sc) \
 sc_rational_feasibility_ofl_ctrl((sc), FWD_OFL_CTRL, TRUE)
#define sc_feasible_ofl(sc, b) sc_rational_feasibility_ofl_ctrl((sc), OFL_CTRL, (b))
#define sc_elim_redond(ps) sc_elim_redund((ps))
#define sc_rm_empty_constraints( ps,b) sc_elim_empty_constraints((ps),(b))
#define sc_kill_db_eg( ps) sc_elim_db_constraints((ps))
#define non_redundent_subsystem( s1,  s2) extract_nredund_subsystem((s1), (s2))
#define sc_nredund_ofl( psc) build_sc_nredund_2pass_ofl_ctrl((psc),FWD_OFL_CTRL)
#define sc_nredund_optim( psc) build_sc_nredund_2pass((psc))
#define sc_nredund( psc) build_sc_nredund_2pass((psc))
#define sc_projection_on_list_of_variables(sc,ib,pv) \
 sc_projection_on_variables((sc),(ib),(pv))
#define combiner(sc, v) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),FALSE,FALSE,NO_OFL_CTRL)
#define combiner_ofl(sc, v) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),FALSE,FALSE,FWD_OFL_CTRL)
#define exact_combiner_ofl(sc, v, b) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),TRUE, (b), FWD_OFL_CTRL)
#define eq_v_min_coeff(c, v, cf) contrainte_var_min_coeff((c), (v), (cf), FALSE)
#define sc_projection_ofl_with_eq(sc, eq, v) \
 sc_variable_substitution_with_eq_ofl_ctrl((sc), (eq), (v), FWD_OFL_CTRL)
#define cond_suff_comb_integer(sc,pos,neg, v) \
 cond_suff_comb_integer_ofl_ctrl((sc),(pos),(neg), (v), NO_OFL_CTRL)
#define cond_suff_comb_integer_ofl(sc,pos,neg, v) \
 cond_suff_comb_integer_ofl_ctrl((sc),(pos),(neg), (v), FWD_OFL_CTRL)
#define sc_projection_int_along_vecteur(fsc,sc,ib,pv,ti,dim,n) \
 sc_integer_projection_along_variables((fsc),(sc),(ib),(pv),(ti),(dim),(n))
#define integer_projection(sci,sc,v) \
 sc_integer_projection_along_variable((sci),(sc),(v))

/* structures D'Arnauld Leservot */
typedef struct Ssyslist	 {
	Psysteme		psys;
	struct Ssyslist		*succ;
	} *Psyslist,Ssyslist;
#define SL_UNDEFINED	NULL



#endif SYSTEME
