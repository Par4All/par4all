 /* package contrainte - operations d'entree-sortie
  */


/* modifications :  
 *  - ajout du parame`tre a_la_fortran pour une impression compatible avec Fortran
 *    (permet a` certains logiciels externes de re'cupe'rer les syste`mes sous un
 *    format compatible au leur. BA, avril 1994.
 */

/*LINTLIBRARY*/

#include <stdio.h>
#include <assert.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"

/* void contrainte_fprint(FILE * fp, Pcontrainte c, boolean is_inegalite,
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
void contrainte_fprint(fp,c,is_inegalite,variable_name)
FILE *fp;
Pcontrainte c;
boolean is_inegalite;
char * (*variable_name)();
{
    Pvecteur v;
    short int debut = 1;
    long int constante = 0;

    if (!CONTRAINTE_UNDEFINED_P(c))
	v = contrainte_vecteur(c);
    else
	v = VECTEUR_NUL;

    assert(vect_check(v));

    while (!VECTEUR_NUL_P(v)) {
	if (v->var!=TCST) {
	    char signe;
	    long int coeff = v->val;

	    if (coeff != 0) {
		if (coeff > 0)
		    signe = (debut) ? ' ' : '+';
		else {
		    signe = '-';
		    coeff = -coeff;
		};
		debut = 0;
		if (coeff == 1)
		    (void) fprintf(fp,"%c %s ", signe, variable_name(v->var));
		else 
		    (void) fprintf(fp,"%c %ld %s ", signe, coeff,
			    variable_name(v->var));
	    }
	}
	else
	    /* on admet plusieurs occurences du terme constant!?! */
	    constante += v->val;

	v = v->succ;
    }
    if (is_inegalite)
	(void) fprintf (fp," <= %ld ,\n", -constante);
    else 
	(void) fprintf (fp," == %ld ,\n", -constante);
}

/* void egalite_fprint(FILE * fp, Pcontrainte eg, char * (*variable_name)()):
 * impression d'une egalite eg dans le fichier fp avec des noms de
 * variables donnes par variable_name; voir contrainte_fprint
 *
 * Ancien nom: eg_print(), print_eq()
 */
void egalite_fprint(fp,eg,variable_name)
FILE *fp;
Pcontrainte eg;
char * (*variable_name)();
{
    contrainte_fprint(fp,eg,FALSE,variable_name);
}

/* void egalite_dump(Pcontrainte c): impression "physique" d'une egalite;
 * utilise en debugging
 */
void egalite_dump(c)
Pcontrainte c;
{
    egalite_fprint(stderr, c, variable_dump_name);
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
char * (*variable_name)();
{
    contrainte_fprint(fp,ineg,TRUE,variable_name);
}

/* void inegalite_dump(Pcontrainte c): impression "physique" d'une inegalite;
 * utilise en debugging
 */
void inegalite_dump(c)
Pcontrainte c;
{
    inegalite_fprint(stderr, c, variable_dump_name);
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
char * (*variable_name)();
{
    for( ; eg != NULL; eg = eg->succ)
	contrainte_fprint(fp,eg,FALSE,variable_name);
}

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
char * (*variable_name)();
{
    for( ; ineg != NULL; ineg = ineg->succ)
	contrainte_fprint(fp,ineg,TRUE,variable_name);
}

static char * heuristique_1(s, v, is_inegalite, variable_name, a_la_fortran)
char * s;
Pvecteur v;
boolean is_inegalite;
char * (*variable_name)();
boolean a_la_fortran;
{
    short int debut = 1;
    long int constante = 0;

    while (!VECTEUR_NUL_P(v)) {
	if (v->var!=TCST) {
	    char signe;
	    long int coeff = v->val;

	    if (coeff != 0) {
		if (coeff > 0)
		    signe = (debut) ? ' ' : '+';
		else {
		    signe = '-';
		    coeff = -coeff;
		};
		debut = 0;
		if (coeff == 1)
		    (void) sprintf(s+strlen(s),"%c%s", 
				   signe, variable_name(v->var));
		else 
		    (void) sprintf(s+strlen(s),"%c%ld%s", signe, coeff,
			    variable_name(v->var));
	    }
	}
	else
	    /* on admet plusieurs occurences du terme constant!?! */
	    constante += v->val;

	v = v->succ;
    }
    if (is_inegalite)	
	switch (a_la_fortran){
	case FALSE :
	    (void) sprintf(s+strlen(s) ,"<=%ld", -constante);
	    break;
	case TRUE : 
	    (void) sprintf(s+strlen(s) ,".LE.%ld", -constante);
	    break;
	}
    else 
	switch (a_la_fortran){
	case FALSE :
	    (void) sprintf(s+strlen(s) ,"==%ld", -constante);
	    break;
	case TRUE : 
	    (void) sprintf(s+strlen(s) ,".EQ.%ld", -constante);
	    break;
	}

    return s;
}

static char * heuristique_3(s, v, is_inegalite, variable_name, a_la_fortran)
char * s;
Pvecteur v;
boolean is_inegalite;
char * (*variable_name)();
boolean a_la_fortran;
{
    Pvecteur coord;
    short int debut = TRUE;
    int positive_terms = 0;
    int negative_terms = 0;

    if(!is_inegalite) {
	for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	    if(vecteur_var(coord)!= TCST) {
		if(vecteur_val(coord) >0 )
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
	int coeff = vecteur_val(coord);

	if (coeff > 0) {
	    positive_terms++;
	    if (debut == TRUE) {
		debut = FALSE;
		if (coeff == 1 && vecteur_var(coord) != TCST)
		    (void) sprintf(s+strlen(s),"%s", 
				   variable_name(vecteur_var(coord)));
		else if(!term_cst(coord) || is_inegalite)
		    (void) sprintf(s+strlen(s),"%d%s", coeff,
			    variable_name(vecteur_var(coord)));
		else
		    positive_terms--;
	    }
	    else
		if (coeff == 1 && vecteur_var(coord) != TCST)
		    (void) sprintf(s+strlen(s),"+%s", 
				   variable_name(vecteur_var(coord)));
		else if(!term_cst(coord) || is_inegalite)
		    (void) sprintf(s+strlen(s),"+%d%s", coeff,
				   variable_name(vecteur_var(coord)));
		else
		    positive_terms--;
	}
    }

    if(positive_terms == 0)
	(void) sprintf(s+strlen(s),"0");

    if (is_inegalite)
	switch (a_la_fortran){
	case FALSE :
	    (void) sprintf(s+strlen(s) ,"<=");
	    break;
	case TRUE : 
	    (void) sprintf(s+strlen(s) ,".LE.");
	    break;
	}
    else 
	switch (a_la_fortran){
	case FALSE :
	    (void) sprintf(s+strlen(s) ,"==");
	    break;
	case TRUE : 
	    (void) sprintf(s+strlen(s) ,".EQ.");
	    break;
	}

    debut = TRUE;
    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	int coeff = vecteur_val(coord);

	if (coeff < 0) {
	    negative_terms++;
	    if (debut == TRUE) {
		debut = FALSE;
		if (-coeff == 1 && vecteur_var(coord) != TCST)
		    (void) sprintf(s+strlen(s),"%s", 
				   variable_name(vecteur_var(coord)));
		else 
		    (void) sprintf(s+strlen(s),"%d%s", -coeff,
			    variable_name(vecteur_var(coord)));
	    }
	    else
		if (-coeff == 1 && vecteur_var(coord) != TCST)
		    (void) sprintf(s+strlen(s),"+%s", 
				   variable_name(vecteur_var(coord)));
		else 
		    (void) sprintf(s+strlen(s),"+%d%s", -coeff,
			    variable_name(vecteur_var(coord)));
	}
	else if(term_cst(coord) && !is_inegalite) {
	    (void) sprintf(s+strlen(s),"%d", -coeff);
	    /* And now, a lie... In fact, rhs_terms++ */
	    negative_terms++;
	}
	else
	    ;
    }

    if(negative_terms == 0)
	(void) sprintf(s+strlen(s),"0");

    return s;
}

/* char * contrainte_sprint(char * s, Pcontrainte c, boolean is_inegalite,
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
char * contrainte_sprint(s, c, is_inegalite, variable_name)
char * s;
Pcontrainte c;
boolean is_inegalite;
char * (*variable_name)();
{
    s = contrainte_sprint_format(s, c, is_inegalite, variable_name, FALSE);
    return s;
}

char * contrainte_sprint_format(s, c, is_inegalite, variable_name, a_la_fortran)
char * s;
Pcontrainte c;
boolean is_inegalite;
char * (*variable_name)();
boolean a_la_fortran;
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
char * egalite_sprint(s, eg, variable_name)
char *s;
Pcontrainte eg;
char * (*variable_name)();
{
    return contrainte_sprint(s, eg, FALSE, variable_name);
}

char * inegalite_sprint(s, ineg, variable_name)
char * s;
Pcontrainte ineg;
char * (*variable_name)();
{
    return contrainte_sprint(s, ineg, TRUE, variable_name);
}

char * egalite_sprint_format(s, eg, variable_name, a_la_fortran)
char *s;
Pcontrainte eg;
char * (*variable_name)();
boolean a_la_fortran;
{
    return contrainte_sprint_format(s, eg, FALSE, variable_name, a_la_fortran);
}

char * inegalite_sprint_format(s, ineg, variable_name, a_la_fortran)
char * s;
Pcontrainte ineg;
char * (*variable_name)();
boolean a_la_fortran;
{
    return contrainte_sprint_format(s, ineg, TRUE, variable_name, a_la_fortran);
}
