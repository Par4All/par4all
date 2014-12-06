/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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


/*
 * Le champ dimension donne le nombre de variables utilisees dans 
 * les egalites  et les inegalites, ou si l'on prefere, la dimension 
 * de l'espace dans lequel est defini le polyedre correspondant. 
 * Le terme constant ne fait pas partie de l'espace.
 *
 * La champ base contient tous les vecteurs de base, i.e. toutes les
 * variables apparaissant dans les egalites et les inegalites.
 * Le terme constant ne fait pas partie de la base.  La taille  de la 
 * base est donc egale a la  dimension du systeme. 
 * Si certaines fonctions ajoutent temporairement le terme constant a la
 * base, elles doivent l'oter apres traitement.
 * Le champ base est utilise par des algorithmes comme  celui du test 
 * de faisabilite mais il n'est pas entretenu automatiquement lors de 
 * l'ajout de nouvelles contraintes. Il faut penser aux mises a jour. 
 *
 */
typedef struct Ssysteme { 
	Pcontrainte egalites;
	Pcontrainte inegalites;
	int nb_eq;
	int nb_ineq;
	int dimension;
	Pbase base;
} *Psysteme,Ssysteme;

/* - Traitement des overflows :
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
 * Pour ne pas dupliquer trop de fonctions pour le traitement des
 * overflows, nous avons fait une seule fonction prenant en parametre ofl_ctrl.
 *
 * NO_OFL_CTRL : pas de traitement des overflows.
 * 
 * FWD_OFL_CTRL : les overflows doivent e^tre re'cupe're's par les
 * fonctions appelantes.
 * OFL_CTRL : le setjmp est fait dans la fonction appele'e, et donc il est
 * inutile de faire un setjmp dans la fonction appelante. Cette dernie`re
 * option n'est pas disponible dans toutes les fonctions, car pour cela,
 * il faut rajouter un parame`tre permettant de savoir quoi retourner en
 * cas d'overflow, et ce n'e'tait pas toujours possible. Voir les
 * commentaires au dessus des fonctions pour cela.
 *
 * Toutes les fonctions qui acceptent ces valeurs en (dernier) parame`tre
 * ont leur nom qui finit par _ofl_ctrl.
 *
 * Ceci concerne quasiment toutes les fonctions de projection, les
 * fonctions de test de faisabilite', la fonction du simplexe, les
 * fonctions d'e'limination de redondances, la fonction de valeur absolue
 * (abs_ofl_ctrl), les fonctions de combinaisons line'aires de vecteurs.
 * Ceci n'est pas force'ment exhaustif...
 *
 *
 * - SC_EMPTY, SC_RN, sc_empty, sc_rn
 *   ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~
  * Le systeme vide "sc_empty" est represente par l'egalite "0==-1".
 * le systeme representant l'espace Rn "sc_rn" est un  systeme
 * ne contenant aucune contrainte.
 * Avant ces deux systemes etaient representes par Le pointeur (Psysteme) NULL.
 * Progressivement, les (Psysteme) NULL sont replaces par des appels aux
 *  fonctions sc_empty et sc_rn.
 * SC_EMPTY et SC_RN representent des valeurs obsoletes qu'ils faudraient
 * remplacer par les sc_empty et sc_rn.
 *
 * - entier ou rationnel ?
 *   ~~~~~~~~~~~~~~~~~~~~~
 * Les tests de faisabilite' e'taient fait en entier. Les deux tests
 * (entier et rationnel) sont maintenant disponibles (voir
 * sc_feasibility.c, il y a des commentaires).  Le test appele' par
 * l'ancienne fonction sc_faisabilite est celui en rationnel.
 * 
 * Les fonctions d'elimination des contraintes redondantes utilisent 
 * toujours la meme technique : inversion de la contrainte et ajout de 
 * la constante 1, avant de tester la faisabilite de cette contrainte inversee
 *  dans le contexte choisi. Ce contexte est soit le systeme entier pour 
 * "sc_elim_redund" ou le  systeme non redondant prealablement construit 
 * pour les fonctions  "build_sc_nredund..." et "sc_triang_elim_redund". 
 * Le test de faisabilite qui est applique est en rationnel pour les 
 * fonctions "build_sc_nredund..." et en entiers pour les 
 * "sc_triang_elim_redund". L'utilisation du test rationel permet de conserver 
 * des contraintes qui reduisent l'ensemble convexe des points entiers du 
 * polyedre. Ces contraintes peuvent etre redondantes en entiers, mais utiles 
 * si l'on veut tester l'exactitude de la projection. Ce type de contraintes 
 * est  considere comme redondant par "sc_triang_elim_redund".  
 * "sc_triang_elim_redund" est principalement utilise
 * pour parcourir les nids de boucles. En cas de projection non exacte, 
 * l'une des bornes inf. peut alors etre superieure a une borne sup. 
 * L'utilisation d'un test rationel pour "sc_triang_elim_redund" peut conduire 
 * a conserver des contraintes redondantes.
 * 
 * Note: l'inversion entiere de contrainte peut conduire a une augmentation 
 * du polyedre correspondant au systeme traite. Le nombre de contraintes est
 * minimise en entier mais le polyedre rationnel correspondant peut augmenter.
 * Si une enveloppe convexe est calculee ulterieurement, le resultat peut donc
 * etre degrade par une elimination de redondance anterieure.
 *
 */

#define get_sc_debug_level() sc_debug_level
#define ifscdebug(l) if (get_sc_debug_level()>=l)

/* MACROS */

#define sc_nbre_egalites(psc) ((psc)->nb_eq)
#define sc_nbre_inegalites(psc) ((psc)->nb_ineq)
#define sc_egalites(psc) ((psc)->egalites)
#define sc_inegalites(psc) ((psc)->inegalites)
#define sc_base(psc) ((psc)->base)
#define sc_dimension(psc) ((psc)->dimension)

void sc_add_egalite(Psysteme, Pcontrainte);
void sc_add_inegalite(Psysteme, Pcontrainte);

/* For the parsers: */
extern void sc_init_lex(void);
extern int syst_parse ();
extern void syst_restart(FILE * input_file );


/* old compatible... */
#define sc_add_eg(p,c) sc_add_egalite(p, c)
#define sc_add_ineg(p,c) sc_add_inegalite(p, c)

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

/* ensemble de macros permettant de compiler les programmes utilisant
les anciens noms des fonctions */

#define sc_faisabilite(sc) sc_rational_feasibility_ofl_ctrl((sc), NO_OFL_CTRL,true) 
#define sc_faisabilite_ofl(sc) \
 sc_rational_feasibility_ofl_ctrl((sc), FWD_OFL_CTRL, true)
#define sc_feasible_ofl(sc, b) sc_rational_feasibility_ofl_ctrl((sc), OFL_CTRL, (b))
#define sc_elim_redond(ps) sc_elim_redund((ps))
#define sc_triang_elim_redond(x,y) sc_triang_elim_redund(x,y)
#define sc_rm_empty_constraints( ps,b) sc_elim_empty_constraints((ps),(b))
#define sc_kill_db_eg( ps) sc_elim_db_constraints((ps))
#define sc_safe_kill_db_eg( ps) sc_safe_elim_db_constraints((ps))
#define non_redundent_subsystem( s1,  s2) extract_nredund_subsystem((s1), (s2))
#define sc_nredund_ofl( psc) build_sc_nredund_2pass_ofl_ctrl((psc),FWD_OFL_CTRL)
#define sc_nredund_optim( psc) build_sc_nredund_2pass((psc))
#define sc_nredund( psc) build_sc_nredund_2pass((psc))
#define sc_projection_on_list_of_variables(sc,ib,pv) \
 sc_projection_on_variables((sc),(ib),(pv))
#define combiner(sc, v) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),false,false,NO_OFL_CTRL)
#define combiner_ofl(sc, v) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),false,false,FWD_OFL_CTRL)
#define exact_combiner_ofl(sc, v, b) \
 sc_fourier_motzkin_variable_elimination_ofl_ctrl((sc),(v),true, (b), FWD_OFL_CTRL)
#define eq_v_min_coeff(c, v, cf) contrainte_var_min_coeff((c), (v), (cf), false)
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

typedef int (*two_int_infop)[2];

typedef int (* constraint_cmp_func_t)
(const Pcontrainte *, const Pcontrainte *, void *);

#endif /* SYSTEME */
