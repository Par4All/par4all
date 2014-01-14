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

/* package sur les vecteurs creux et les bases
 *
 * Malik Imadache, Corinne Ancourt, Neil Butler, Francois Irigoin,
 * Remi Triolet
 *
 * Modifications:
 *  - les fonctions d'interface avec GenPgm dont les noms commencent par
 *    "_gC_" ont ete deplacees dans _gC_lib
 *  - passage a "char *" pour le type Variable au lieu de "int" (Linear-C3)
 *    et de "entity *" (Linear/C3 Library); le package n'est pas independant de la
 *    definition du type "Variable"; il faudrait ameliorer ca avec un
 *    package "Variable"
 *  - ajout des fonctions d'interface avec Newgen: (RT, 27/11/89)
 *  - ajout de la notion de base, comme cas particulier de vecteur
 *    (FI, 27/11/89) ou le champ "Value" n'a pas de signification
 *  - suppression de l'include de vecteur-basic-types.h; la fusion entre
 *    les versions C3 et Linear/C3 Library ne necessite plus cette distinction; il y a
 *    tellement peu de code a ecrire pour les variables et les valeurs
 *    qu'il est inutile d'avoir une directory differente pour lui
 *  - rapatriement de la definition du terme constant TCST et de la macro
 *    term_cst (du package contrainte) (PB, 06/06/90)
 *
 * - trop creux a mon avis. il faudrait une liste de petits tableaux ? FC.
 */

#ifndef NEWGEN
#define VECTEUR 1006	/* constante associee a un vecteur	*/
#endif

/* arithmetique is a requirement for vecteur, but I do not want
 * to inforce it in all pips files... thus here it is
 */
#include "arithmetique.h"

/* le type des variables (ou coordonnees) dans les vecteurs */
typedef void * Variable;
// The method type that return the name of a variable:
typedef char * (*get_variable_name_t)(Variable);

#define VARIABLE_UNDEFINED ((Variable) 0)
#define VARIABLE_UNDEFINED_P(v) ((v)==VARIABLE_UNDEFINED)
#define VARIABLE_DEFINED_P(v) ((v)!=VARIABLE_UNDEFINED)

/* le type des coefficients dans les vecteurs:
 * Value est defini dans le package arithmetique
 */

/* STRUCTURE D'UN VECTEUR
 *
 * Un vecteur est defini par une suite de couples Variable (i.e. element
 * de la base) et Valeur (valeur du coefficient correspondant). Les
 * coordonnees nulles ne sont pas representees et n'existe qu'implicitement
 * par rapport a une base (hypothetique) definie via la package "variable".
 *
 * En consequence, le vecteur nul est (malencontreusement) represente par
 * NULL. Cela gene toutes les procedures succeptibles de retourner une
 * valeur vecteur nul par effet de bord. Il faut alors passer en argument
 * un POINTEUR vers un Pvecteur. En general, nous avons prefere retourner
 * explicitement le vecteur calcule, a la maniere de ce qui est fait
 * dans string.h
 *
 * Il n'existe pas non plus de VECTEUR_UNDEFINED, puisque sa valeur devrait
 * logiquement etre NULL.
 */
typedef struct Svecteur {
    Variable var;
    Value val;
    struct Svecteur *succ;
} Svecteur, *Pvecteur;

/* STRUCTURE D'UNE BASE
 *
 * Une base est definie par son vecteur diagonal
 *
 * Les tests d'appartenance sont effectues par comparaison des pointeurs
 * et non par des strcmp.
 *
 * Rien ne contraint les coefficients a valoir 1 et le package plint
 * mais meme certains coefficients a 0, ce qui devrait revenir a faire
 * disparaitre la variable (i.e. la coordonnee) correspondante de la
 * base.
 */
typedef struct Svecteur Sbase, * Pbase;

/* DEFINITION DU VECTEUR NUL */
#define VECTEUR_NUL ((Pvecteur) 0)
#define VECTEUR_NUL_P(v) ((v)==VECTEUR_NUL)
#define VECTEUR_UNDEFINED ((Pvecteur) 0)
#define VECTEUR_UNDEFINED_P(v) ((v)==VECTEUR_UNDEFINED)

/* definition de la valeur de type PlinX==Pvecteur qui correspond a un
 * vecteur indefini parce que l'expression correspondante n'est pas
 * lineaire (Malik Imadache, Jean Goubault ?)
 */
#define PlinX Pvecteur
#define NONEXPLIN ((PlinX)-1)

/* MACROS SUR LES VECTEURS */
#define print_vect(s) vect_fprint(stdout,(s))
#define var_of(varval) ((varval)->var)
#define val_of(varval) ((varval)->val)
#define vecteur_var(v) ((v)->var)
#define vecteur_val(v) ((v)->val)
#define vecteur_succ(v) ((v)->succ)

/* VARIABLE REPRESENTANT LE TERME CONSTANT */
#define TCST ((Variable) 0)
#define term_cst(varval) ((varval)->var == TCST)

/* MACROS SUR LES BASES */
#define BASE_NULLE VECTEUR_NUL
#define BASE_NULLE_P(b) ((b)==VECTEUR_NUL)
#define BASE_UNDEFINED ((Pbase) 0)
#define BASE_UNDEFINED_P(b) ((b)==BASE_UNDEFINED)
#define base_dimension(b) vect_size((Pvecteur)(b))
#define base_add_dimension(b,v) vect_chg_coeff((Pvecteur *)(b),(v),VALUE_ONE)
#define base_rm(b) (vect_rm((Pvecteur)(b)), (b)=BASE_NULLE)

/* I do thing that overflows are managed in a very poor manner. FC.
 * It should be all or not, as provided by any os that would raise
 * integer overflows. Thus we should have thought of a sofware
 * mecanism compatible with such a hardware and os approach.
 * maybe by defining a mult_Value macro to check explicitely for
 * overflows if needed, and defined to a simple product if not.
 * functions would have an additional argument for returning a
 * conservative answer in case of overflow. Maybe some global
 * variable could count the number of overflow that occured so that
 * some caller could check whether sg got wrong and thus could
 * warn about the result and this fact.
 * then we would have either the library compiled for these soft checks
 * or for none, but without any difference or explicite requirements
 * from the user of these functions.
 *
 * instead of that, we have the two versions at the same time with explicite
 * control required from the user. I heard that for some functions
 * this is not used... thus allowing good performance (each time some
 * result is false someone tracks down the not checked function and
 * checks overflow explicitely, thus it is not a very good approach).
 * moreover the most costly functions (simplexe, chernikova) are also
 * those in which the exceptions occurs thus they are all checked.
 * the the impact on performances is definitely low.
 * as far as software engineering is concerned, the current solution
 * adds low level switch for calling different versions (controled or not)
 * of pieces of code... this will have to be removed if some good os
 * is to host this software...
 */

/* OVERFLOW CONTROL
 */
#if (defined(LINEAR_NO_OVERFLOW_CONTROL))
#define OFL_CTRL 0
#define FWD_OFL_CTRL 0
#define NO_OFL_CTRL 0
#else /* some OVERFLOW CONTROL is allowed */
#define OFL_CTRL 2     /* overflows are treated in the called procedure */
#define FWD_OFL_CTRL 1 /* overflows are treated by the calling procedure */
#define NO_OFL_CTRL 0  /* overflows are not trapped at all  (dangerous !) */
#endif /* LINEAR_NO_OVERFLOW_CONTROL */

/* internal hash table for variable sets. */
struct linear_hashtable_st;
typedef struct linear_hashtable_st * linear_hashtable_pt;

/* end of vecteur-local.h */
