/* package sur les vecteurs creux et les bases
 *
 * Malik Imadache, Corinne Ancourt, Neil Butler, Francois Irigoin,
 * Remi Triolet
 *
 * Modifications:
 *  - les fonctions d'interface avec GenPgm dont les noms commencent par
 *    "_gC_" ont ete deplacees dans _gC_lib
 *  - passage a "char *" pour le type Variable au lieu de "int" (Linear-C3) 
 *    et de "entity *" (PIPS); le package n'est pas independant de la
 *    definition du type "Variable"; il faudrait ameliorer ca avec un
 *    package "Variable"
 *  - ajout des fonctions d'interface avec Newgen: (RT, 27/11/89)
 *  - ajout de la notion de base, comme cas particulier de vecteur
 *    (FI, 27/11/89) ou le champ "Value" n'a pas de signification
 *  - suppression de l'include de vecteur-basic-types.h; la fusion entre
 *    les versions C3 et PIPS ne necessite plus cette distinction; il y a
 *    tellement peu de code a ecrire pour les variables et les valeurs
 *    qu'il est inutile d'avoir une directory differente pour lui
  *  - rapatriement de la definition du terme constant TCST et de la macro
 *    term_cst (du package contrainte) (PB, 06/06/90)
*/

#ifndef NEWGEN
#define VECTEUR 1006	/* constante associee a un vecteur	*/
#endif

#ifndef VECTEUR_INCLUDED
#define VECTEUR_INCLUDED

/* le type des variables (ou coordonnees) dans les vecteurs */
typedef char * Variable;

#define VARIABLE_UNDEFINED ((Variable) 0)
#define VARIABLE_UNDEFINED_P(v) ((v)==VARIABLE_UNDEFINED)
#define VARIABLE_DEFINED_P(v) ((v)!=VARIABLE_UNDEFINED)

/* le type des coefficients dans les vecteurs */
typedef int Value;

/* STRUCTURE D'UN VECTEUR */

/* Un vecteur est defini par une suite de couples Variable (i.e. element
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
} Svecteur,*Pvecteur;

/* STRUCTURE D'UNE BASE */

/* Une base est definie par son vecteur diagonal
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

/* VARIABLE REPRESENTANT LE TERME CONSTANT */
#define TCST ((Variable) 0)
#define term_cst(varval) ((varval)->var == TCST)

/* MACROS SUR LES BASES */
#define BASE_NULLE VECTEUR_NUL
#define BASE_NULLE_P(b) ((b)==VECTEUR_NUL)
#define BASE_UNDEFINED ((Pbase) 0)
#define BASE_UNDEFINED_P(b) ((b)==BASE_UNDEFINED)
#define base_dimension(b) vect_size((Pvecteur)(b))
#define base_add_dimension(b,v) vect_chg_coeff((Pvecteur *)(b),(v),1)
#define base_rm(b) vect_rm((Pvecteur)(b))

/* OVERFLOW CONTROL */
#define OFL_CTRL 2     /* overflows are treated in the called procedure */
#define FWD_OFL_CTRL 1 /* overflows are treated by the calling procedure */
#define NO_OFL_CTRL 0  /* overflows are not trapped at all  (dangerous !) */

#endif VECTEUR_INCLUDED
