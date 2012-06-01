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

 /* package contrainte - operations d'entree-sortie
  */


/* modifications :  
 *  - ajout du parame`tre a_la_fortran pour une impression compatible avec 
 *    Fortran (permet a` certains logiciels externes de re'cupe'rer les 
 *    syste`mes sous un format compatible au leur. BA, avril 1994.
 */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

/* void contrainte_fprint(FILE * fp, Pcontrainte c, bool is_inegalite,
 *                        char * (*variable_name)()):
 *                                                                          
 * imprime dans le fichier fp la contrainte c, de type egalite ou inegalite
 * suivant la valeur du booleen is_inegalite, en utilisant la fonction
 * variable_name pour trouver les noms des variables.
 *
 * Pour suivre les convention usuelles, le terme constant est imprime
 * comme membre droit.
 *
 * On considere que CONTRAINTE_UNDEFINED => CONTRAINTE_NULLE
 *                                                                          
 * Resultat:
 *   2 * I - J = 4 LF
 *   -I + 3 * J <= 5 LF
 *
 * Note: l'impression se termine par un LF
 *       aucun routine ne sait lire ce format
 *       ancien nom eq_print
 */

/* returns the constant */
static Value 
fprint_contrainte_vecteur(
  FILE * fp,
  Pvecteur v,
  char * (*variable_name)(Variable))
{
    short int debut = 1;
    Value constante = VALUE_ZERO;

    while (!VECTEUR_NUL_P(v)) {
	Variable var = var_of(v);
	Value coeff = val_of(v);
	if (var!=TCST) {
	    char signe;

	    if (value_notzero_p(coeff)) {

		if (value_pos_p(coeff))
		    signe = (debut) ? ' ' : '+';
		else {
		    signe = '-';
		    coeff = value_uminus(coeff);
		};
		debut = 0;
		(void) fprintf(fp, "%c", signe);
		if (value_notone_p(coeff))
		    (void) fprintf(fp, " "), fprint_Value(fp, coeff);
		(void) fprintf(fp, " %s ", variable_name(var));
	    }
	}
	else
	    /* on admet plusieurs occurences du terme constant!?! */
	    value_addto(constante, coeff);

	v = v->succ;
    }

    /* To handle cases where the constraint only has constant (this is a bug 
       somewhere, we must remove the constraint). If we print: 
       "<= constant ," then sc_fscan cannot read this output, so let's print:
       " constant <= constant ," which is readable by sc_fscan, and doesn't change
       the sc.
    */
    if (debut) {
      (void) fprintf(fp, " "), fprint_Value(fp, constante), fprintf(fp, " ");
    }

    return constante;
}

void contrainte_fprint(fp,c,is_inegalite,variable_name)
FILE *fp;
Pcontrainte c;
bool is_inegalite;
char * (*variable_name)(Variable);
{
    Pvecteur v;
    Value constante = VALUE_ZERO;

    if (!CONTRAINTE_UNDEFINED_P(c))
	v = contrainte_vecteur(c);
    else
	v = VECTEUR_NUL;

    assert(vect_check(v));

    constante = fprint_contrainte_vecteur(fp, v, variable_name);

    (void) fprintf(fp, " %s ", is_inegalite? "<=": "==");
    fprint_Value(fp, value_uminus(constante));
    fprintf(fp, " ,\n");
}

/* void egalite_fprint(FILE * fp, Pcontrainte eg, char * (*variable_name)()):
 * impression d'une egalite eg dans le fichier fp avec des noms de
 * variables donnes par variable_name; voir contrainte_fprint
 *
 * Ancien nom: eg_print(), print_eq()
 */
void egalite_fprint(FILE *fp, Pcontrainte eg, char * (*variable_name)(Variable))
{
    contrainte_fprint(fp,eg,false,variable_name);
}

/* void egalite_dump(Pcontrainte c): impression "physique" d'une egalite;
 * utilise en debugging
 */
void egalite_dump(Pcontrainte c) {
    egalite_fprint(stderr, c, variable_debug_name);
}

/* void inegalite_fprint(FILE * fp, Pcontrainte ineg,
 *                       char * (*variable_name)()):
 * impression d'une inegalite ineg dans le fichier fp avec des noms de
 * variables donnes par variable_name; voir contrainte_fprint
 *
 * Ancien nom: ineg_print(), print_ineq()
 */
void inegalite_fprint(fp,ineg,variable_name)
FILE *fp;
Pcontrainte ineg;
char * (*variable_name)(Variable);
{
    contrainte_fprint(fp,ineg,true,variable_name);
}

/* void inegalite_dump(Pcontrainte c): impression "physique" d'une inegalite;
 * utilise en debugging
 */
void inegalite_dump(Pcontrainte c) {
    inegalite_fprint(stderr, c, variable_debug_name);
}

/* void egalites_fprint(FILE * fp, Pcontrainte eg, char * (*variable_name)()):
 * impression d'une liste d'egalites eg dans le fichier fp avec des noms de
 * variables donnes par variable_name; voir contrainte_fprint
 *
 * Ancien nom: fprint_leq()
 */
void egalites_fprint(fp,eg,variable_name)
FILE *fp;
Pcontrainte eg;
char * (*variable_name)(Variable);
{
    for( ; eg != NULL; eg = eg->succ)
	contrainte_fprint(fp,eg,false,variable_name);
}

void egalites_dump(Pcontrainte eg)
{egalites_fprint(stderr, eg, variable_debug_name);}

/* void inegalites_fprint(FILE * fp, Pcontrainte ineg,
 *                        char * (*variable_name)()):
 * impression d'une liste d'inegalites ineg dans le fichier fp avec des noms de
 * variables donnes par variable_name; voir contrainte_fprint
 *
 * Ancien nom: fprint_lineq()
 */
void inegalites_fprint(fp,ineg,variable_name)
FILE *fp;
Pcontrainte ineg;
char * (*variable_name)(Variable);
{
    for( ; ineg != NULL; ineg = ineg->succ)
	contrainte_fprint(fp,ineg,true,variable_name);
}

void inegalites_dump(Pcontrainte eg)
{inegalites_fprint(stderr, eg, variable_debug_name);}

void
sprint_operator(char *s, bool is_inegalite, bool a_la_fortran)
{
    (void) sprintf(s, "%s",(is_inegalite? (a_la_fortran? ".LE.": "<="):
			    (a_la_fortran? ".EQ.": "==")));
}

static char * 
heuristique_1(s, v, is_inegalite, variable_name, a_la_fortran)
char * s;
Pvecteur v;
bool is_inegalite;
char * (*variable_name)(Variable);
bool a_la_fortran;
{
    short int debut = 1;
    Value constante = VALUE_ZERO;

    while (!VECTEUR_NUL_P(v)) {
	Variable var = var_of(v);
	Value coeff = val_of(v);
	if (var!=TCST) {
	    char signe;

	    if (value_notzero_p(coeff)) {
		if (value_pos_p(coeff))
		    signe = (debut) ? ' ' : '+';
		else {
		    signe = '-';
		    coeff = value_uminus(coeff);
		};
		debut = 0;
		(void) sprintf(s+strlen(s),"%c", signe);
		if (value_notone_p(coeff))
		    (void) sprint_Value(s+strlen(s), coeff);
		(void) sprintf(s+strlen(s), "%s", variable_name(var));
	    }
	}
	else
	    /* on admet plusieurs occurences du terme constant!?! */
	    value_addto(constante, coeff);

	v = v->succ;
    }

    (void) sprint_operator(s+strlen(s), is_inegalite, a_la_fortran);
    (void) sprint_Value(s+strlen(s), value_uminus(constante));

    return s;
}

static char * 
heuristique_3(s, v, is_inegalite, variable_name, a_la_fortran)
char * s;
Pvecteur v;
bool is_inegalite;
char * (*variable_name)(Variable);
bool a_la_fortran;
{
    Pvecteur coord;
    short int debut = true;
    int positive_terms = 0;
    int negative_terms = 0;
    Value const_coeff = 0;
    bool const_coeff_p = false;

    if(!is_inegalite) {
	for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	    if(vecteur_var(coord)!= TCST) {
		if(value_pos_p(vecteur_val(coord)))
		    positive_terms++;
		else
		    negative_terms++;
	    }
	}

	if(negative_terms > positive_terms) {
	    vect_chg_sgn(v);
	}
    }

    positive_terms = 0;
    negative_terms = 0;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Value coeff = vecteur_val(coord);
	Variable var = vecteur_var(coord);

	if (value_pos_p(coeff)) {
	    positive_terms++;
	    if (debut) {
		debut = false;
		if (value_one_p(coeff) && var!=TCST)
		    (void) sprintf(s+strlen(s),"%s", 
				   variable_name(vecteur_var(coord)));
		else if(!term_cst(coord) || is_inegalite){
		    (void) sprint_Value(s+strlen(s), coeff);
		    (void) sprintf(s+strlen(s),"%s", variable_name(var));
		}
		else {
		    debut = true;
		    positive_terms--;
		}
	    }
	    else
		if (value_one_p(coeff) && var!=TCST)
		    (void) sprintf(s+strlen(s),"+%s", variable_name(var));
		else if(!term_cst(coord) || is_inegalite) {
		    (void) sprintf(s+strlen(s),"+");
		    (void) sprint_Value(s+strlen(s), coeff);
		    (void) sprintf(s+strlen(s),"%s", variable_name(var));
		}
		else
		    positive_terms--;
	}
    }

    if(positive_terms == 0)
	(void) sprintf(s+strlen(s),"0");

    (void) sprint_operator(s+strlen(s), is_inegalite, a_la_fortran);

    debut = true;
    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Value coeff = vecteur_val(coord);
	Variable var = var_of(coord);

	if(term_cst(coord) && !is_inegalite) {
	    /* Save the constant term for future use */
	    const_coeff_p = true;
	    const_coeff = coeff;
	    /* And now, a lie... In fact, rhs_terms++ */
	    negative_terms++;
	}
	else if (value_neg_p(coeff)) {
	    negative_terms++;
	    if (debut == true) {
		debut = false;
		if (value_mone_p(coeff) && var!=TCST)
		    (void) sprintf(s+strlen(s),"%s", variable_name(var));
		else {
		    (void) sprint_Value(s+strlen(s), value_uminus(coeff));
		    (void) sprintf(s+strlen(s),"%s", variable_name(var));
		}
	    }
	    else
		if (value_mone_p(coeff) && var!=TCST)
		    (void) sprintf(s+strlen(s),"+%s", variable_name(var));
		else {
		    (void) sprintf(s+strlen(s),"+");
		    (void) sprint_Value(s+strlen(s), value_uminus(coeff));
		    (void) sprintf(s+strlen(s),"%s", variable_name(var));
		}
	}
    }

    if(negative_terms == 0)
	(void) sprintf(s+strlen(s),"0");
    else if(const_coeff_p) {
	assert(value_notzero_p(const_coeff));
	
	if(!debut && value_neg_p(const_coeff))
	    (void) sprintf(s+strlen(s), "+");

	sprint_Value(s+strlen(s), value_uminus(const_coeff));
    }

    return s;
}

/* char * contrainte_sprint(char * s, Pcontrainte c, bool is_inegalite,
 *                          char * (*variable_name)()):
 * Traduction d'une contrainte c en une chaine s de caracteres ASCII.
 * Les noms des variables sont recuperes via la fonction variable_name().
 * Egalites et inegalites sont traitees.
 *
 * La chaine s doit avoir ete allouee par le programme appelant, avec une
 * longueur suffisante. Pour etre propre, il faudrait aussi passer cette
 * longueur.
 *
 * Plusieurs heuristiques d'impression ont ete proposees:
 *  - h1: mettre le terme constant en partie droite (Francois Irigoin);
 *    inconvenient: -I <= -1 au lieu de I >= 1
 *  - h2: minimiser le nombre de signes "moins" en prenant la contrainte
 *    opposee si necessaire; le signe du terme constant n'est pas
 *    pris en compte (Francois Irigoin); inconvenient: I == J est
 *    imprime sous la forme I - J == 0
 *  - h3: mettre les termes du bon cote pour ne pas avoir de signe "moins"
 *    (Michel Lenci); inconvenient: I == -1 est imprime comme I+1 == 0
 *
 * Pour avoir de bons resultats, il doit sans doute falloir faire du cas
 * par cas, distinguer les egalites des inegalites et prendre en compte
 * le nombre de termes de la contrainte. A ameliorer experimentalement.
 *
 * Note: variable_name() should return an empty string for constant terms
 *
 * Modifications:
 *  - suppression de l'impression de la virgule de separation en fin
 *    de chaine (Francois Irigoin, 7 mai 1990)
 */
char * 
contrainte_sprint(s, c, is_inegalite, variable_name)
char * s;
Pcontrainte c;
bool is_inegalite;
char * (*variable_name)(Variable);
{
    s = contrainte_sprint_format(s, c, is_inegalite, variable_name, false);
    return s;
}

char * 
contrainte_sprint_format(
    char * s,
    Pcontrainte c,
    bool is_inegalite,
    char * (*variable_name)(Variable),
    bool a_la_fortran)
{
    Pvecteur v;
    int heuristique = 3;

    if (!CONTRAINTE_UNDEFINED_P(c))
	v = contrainte_vecteur(c);
    else
	v = VECTEUR_NUL;

    assert(vect_check(v));

    switch(heuristique) {
    case 1: s = heuristique_1(s, v, is_inegalite, variable_name, a_la_fortran);
	break;
    case 3: s = heuristique_3(s, v, is_inegalite, variable_name, a_la_fortran);
	break;
    default: contrainte_error("contrainte_sprint", "unknown heuristics\n");
    }

    return s;
}

/* void egalite_fprint(FILE * fp, Pcontrainte eg, char * (*variable_name)()):
 * impression d'une egalite eg dans la chaine s avec des noms de
 * variables donnes par variable_name; voir contrainte_sprint
 */
char * 
egalite_sprint(s, eg, variable_name)
char *s;
Pcontrainte eg;
char * (*variable_name)(Variable);
{
    return contrainte_sprint(s, eg, false, variable_name);
}

char * 
inegalite_sprint(s, ineg, variable_name)
char * s;
Pcontrainte ineg;
char * (*variable_name)(Variable);
{
    return contrainte_sprint(s, ineg, true, variable_name);
}

char * 
egalite_sprint_format(s, eg, variable_name, a_la_fortran)
char *s;
Pcontrainte eg;
char * (*variable_name)(Variable);
bool a_la_fortran;
{
    return contrainte_sprint_format
	(s, eg, false, variable_name, a_la_fortran);
}

char * 
inegalite_sprint_format(s, ineg, variable_name, a_la_fortran)
char * s;
Pcontrainte ineg;
char * (*variable_name)(Variable);
bool a_la_fortran;
{
    return contrainte_sprint_format
	(s, ineg, true, variable_name, a_la_fortran);
}
