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

/******************************************************************** pnome-io.c
 *
 * POLYNOMIAL INPUT/OUTPUT FUNCTIONS
 *
 */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <assert.h>
#include <ctype.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"


/* void float_to_frac(float x, char** ps)
 *  PRIVATE
 *  returns the simplest representation of floating-point number x
 *  tries to make an integer, then a small fraction, then a floating-point,
 *  then a floating-point with exponent.
 */
void float_to_frac(x, ps)
char **ps;
float x;
{
    int i;
    float fprecision = (float) intpower(10.0, -PNOME_FLOAT_N_DECIMALES);

    if (((x <= fprecision) && (x >= -fprecision)) || 
	((x>PNOME_FLOAT_TO_EXP_LEVEL) | (x<-PNOME_FLOAT_TO_EXP_LEVEL)))
         /* if too little or too big print it with an exponent */
	sprintf(*ps, "%.*E", PNOME_FLOAT_N_DECIMALES, x);
    else {
	/* default printing: as a float */
	sprintf(*ps, "%.*f", PNOME_FLOAT_N_DECIMALES, x);
	for (i=1; i<PNOME_FLOAT_TO_FRAC_LEVEL; i++)
	    if ((((x*i) - ((int) (x*i+0.5))) < (fprecision)) &&  
		(((x*i) - ((int) (x*i+0.5))) > (-fprecision))) { 
		/* if x is close enough up to a little fraction */
	        if ((((int) (x*i+0.5)) < PNOME_FLOAT_TO_FRAC_LEVEL) || (i==1)) {
		    /*print it as a fraction */
		    if (i==1)
			sprintf(*ps, "%d", (int) (x*i+0.5));
		    else
			sprintf(*ps, "%d/%d", (int) (x*i+0.5), i);
		}
		break;
	    }
    }
}

/* void monome_fprint(FILE* fd, Pmonome pm, Pbase pb, bool plus_sign, char* (*variable_name)())
 *  PRIVATE
 *  Outputs to file fd an ASCII form of monomial pm, naming variables
 *  with the "variable-name" function, ordering them with the basis pb.
 *  the "+" sign is printed if plus_sign == true.
 */
void monome_fprint(fd, pm, pb, plus_sign, variable_name)
FILE *fd;
Pmonome pm;
Pbase pb;
bool plus_sign;
char * (*variable_name)(Variable);
{
    char *s = monome_sprint(pm, pb, plus_sign, variable_name);

    fprintf(fd, "%s", s);
    free(s);
}

/* char *monome_sprint(Pmonome pm, Pbase pb, bool plus_sign, char* (*variable_name)())
 *  PRIVATE
 *  Outputs a string representing monomial pm, naming variables
 *  with the "variable-name" function, ordering them with the basis pb.
 *  the "+" sign is printed if plus_sign == true.
 */
char *monome_sprint(pm, pb, plus_sign, variable_name)
Pmonome pm;
Pbase pb;
bool plus_sign;
char * (*variable_name)(Variable);
{
    float x;
    char t[99];
    char *r, *s;
    char *u;

    u = (char *) malloc(99);
    r = t;

    if (MONOME_UNDEFINED_P(pm))
	sprintf(r, "%s", MONOME_UNDEFINED_SYMBOL);
    else {
	x = monome_coeff(pm);
	if (x==0) 
	    sprintf(r, "%s", MONOME_NUL_SYMBOL);
	else {
	    if (x<0) {
		*(r++) = '-';
		if (plus_sign) 
		    *(r++) = ' ';
		x = - x; 
	    }
	    else if (plus_sign) {
		*(r++) = '+';
		*(r++) = ' ';
	    }

	    float_to_frac(x, &u);

	    if (vect_coeff(TCST, monome_term(pm)) == 0) {
		if (x != 1) {
		    sprintf(r, "%s%s", u, MONOME_COEFF_MULTIPLY_SYMBOL);
		    r = strchr(r, '\0');
		}
		s = vect_sprint_as_monome(monome_term(pm), pb,
					  variable_name, MONOME_VAR_MULTIPLY_SYMBOL);
		strcpy(r, s);
		free(s);
	    }
	    else 
		sprintf(r, "%s", u);
	}
    }
    free(u);
    return (char*) strdup((char *) t);
}

/* void polynome_fprint(FILE* fd, Ppolynome pp,
 *                      char* (*variable_name)(), 
 *                      bool (*is_inferior_var)())
 *  Outputs to file fd an ASCII form of polynomial pp, using
 *  the user-provided function variable_name(Variable var) to associate
 *  the "Variable" pointers with the variable names.
 *  is_inferior_var(Variable var1, Variable var2) is also given by the user:
 *  it must return true if var1 must be printed before var2 in monomials.
 *
 *  For the moment, monomials are not sorted.
 *  No "\n" is printed after the polynomial.
 */
void polynome_fprint(fd, pp, variable_name, is_inferior_var)
FILE *fd;
Ppolynome pp;
char * (*variable_name)(Variable);
int (*is_inferior_var)(Pvecteur *, Pvecteur *);
{
    char *s = polynome_sprint(pp, variable_name, is_inferior_var);

    fprintf(fd, "%s", s);
    free(s);
}

/* char *polynome_sprint(Ppolynome pp,
 *                      char* (*variable_name)(), 
 *                      bool (*is_inferior_var)())
 *  Outputs to file fd an ASCII form of polynomial pp, using
 *  the user-provided function variable_name(Variable var) to associate
 *  the "Variable" pointers with the variable names.
 *  is_inferior_var(Variable var1, Variable var2) is also given by the user:
 *  it must return true if var1 must be printed before var2 in monomials.
 *
 *  For the moment, monomials are not sorted.
 *  No "\n" is printed after the polynomial.
 */
char *polynome_sprint(pp, variable_name, is_inferior_var)
Ppolynome pp;
char * (*variable_name)(Variable);
int (*is_inferior_var)(Pvecteur *, Pvecteur *);
{
#define POLYNOME_BUFFER_SIZE 20480
    static char t[POLYNOME_BUFFER_SIZE];
    char *r, *s;

    r = t;

    if (POLYNOME_UNDEFINED_P(pp))
	sprintf(r, "%s", POLYNOME_UNDEFINED_SYMBOL);
    else if (POLYNOME_NUL_P(pp))
	sprintf(r, "%s", POLYNOME_NUL_SYMBOL);
    else {
	Pbase pb = (Pbase) polynome_used_var(pp, is_inferior_var);
	bool print_plus_sign = false;

	/* The following line is added by L.Zhou    Mar. 26, 91 */
	pp = polynome_sort(&pp, is_inferior_var);

	while (!POLYNOME_NUL_P(pp)) {
	    s =	monome_sprint(polynome_monome(pp), pb, 
			      print_plus_sign, variable_name);
	    strcpy(r, s);
	    r = strchr(r, '\0');
	    pp = polynome_succ(pp);
	    print_plus_sign = true;
	    if (!POLYNOME_NUL_P(pp)) *(r++) = ' ';
	    free(s);
	}
    }
    assert( strlen(t) < POLYNOME_BUFFER_SIZE);
    return (char*) strdup((char *) t);
}


/* char *default_variable_name(Variable var)
 *  returns for variable var the name "Vxxxx" where xxxx are
 *  four letters computed from (int) var.

 * I guess that many variables can have the same name since the naming is
 * done modulo 26^4 ? RK. To be fixed...
 */
char *default_variable_name(var)
Variable var;
{
    char *s = (char *) malloc(6);
    int i = (intptr_t) var;
    
    if (var != TCST) {
	sprintf(s, "V%c%c%c%c",
		(char) 93 + (i % 26),
		(char) 93 + ((i / 26) % 26),
		(char) 93 + ((i / 26 / 26) % 26),
		(char) 93 + ((i / 26 / 26 / 26) % 26));
    }
    else 
	sprintf(s, "TCST");
    return(s);
}


/* bool default_is_inferior_var(Variable var1, Variable var2)
 *  return true if var1 is before var2, lexicographically,
 *  according to the "default_variable_name" naming.
 */
int default_is_inferior_var(var1, var2)
Variable var1, var2;
{
    return strcmp(default_variable_name(var1), 
		       default_variable_name(var2));
}

/* bool default_is_inferior_varval(Pvecteur varval1, Pvecteur varval2)
 *  return true if var1 is before var2, lexicographically,
 *  according to the "default_variable_name" naming.
 */
int default_is_inferior_varval(Pvecteur varval1, Pvecteur varval2)
{
    return strcmp(default_variable_name(vecteur_var(varval1)),
		       default_variable_name(vecteur_var(varval2)));
}

/* bool default_is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
 *  return true if var1 is before var2, lexicographically,
 *  according to the "default_variable_name" naming.
 */
int default_is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    return strcmp(default_variable_name(vecteur_var(* pvarval1)),
		       default_variable_name(vecteur_var(* pvarval2)));
}

static void remove_blanks(ps)
char **ps;
{
    char *s = *ps,
         *t = *ps;

    do {
	while (isspace(**ps)) 
	    (*ps)++;
	*t++ = *(*ps)++;
    } while (**ps != '\0');
    *t++ = '\0';
    *ps = s;
}

static float parse_coeff(ps)
char **ps;
{
    float coeff = 0;

    if (isdigit(**ps) || (**ps == '.') || (**ps == '+') || (**ps == '-')) {
	sscanf(*ps, "%f", &coeff);
	if ((coeff == 0) && (**ps == '+')) 
	    coeff = 1;
	if ((coeff == 0) && (**ps == '-')) 
	    coeff = -1;
    }
    else 
	coeff = 1;

    if ((**ps == '-') || (**ps == '+')) 
	(*ps)++;
    while (isdigit(**ps) || (**ps == '.') || (**ps == '*')) 
	(*ps)++;
    
    if (**ps == '/') {  /* handle fractionnal coefficients */
	float denom;
	(*ps)++;
	denom = parse_coeff(ps);
	coeff /= denom;
    }

    return(coeff);
}

static char *parse_var_name(ps)
char **ps;
{
    char *name, *n;

    name = (char *) malloc(MAX_NAME_LENGTH);
    n = name;

    if (isalpha(**ps))                      /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
	do { *n++ = *((*ps)++); }
    while (isalnum(**ps) || (**ps == ':')); /* THE ':' STANDS FOR MODULE_SEP_STRING OF Linear/C3 Library */
                                            /* TO BE ABLE TO HANDLE VARIABLES SUCH AS P:I   */
    *n = '\0';

    return(name);
}

/* Ppolynome polynome_sscanf(char *sp, (*name_to_variable)())
 *  converts into polynomial structure the expression passed
 *  in ASCII form in string sp. (for debug only)
 *  pas vraiment quick mais bien dirty
 */
Ppolynome polynome_sscanf(sp, name_to_variable)
char *sp;
Variable (*name_to_variable)(Variable);
{
    Ppolynome pp = POLYNOME_NUL;
    Pmonome curpm;
    bool constructing_monome = false;
    float coeff = 0.;
    char *varname;
    Value power;
    char *s;

    s = (char*) strdup(sp);
    remove_blanks(&s);

    while (*s != '\0')
    {
	/*fprintf(stderr, "\ns='%s'\n", s);*/
	power = VALUE_ONE;
	if (!constructing_monome) { coeff = parse_coeff(&s);
				}
	varname = parse_var_name(&s);
	if (strlen(varname)!=0) {
	    if (*s == '^') {
		s++;
		power = float_to_value(parse_coeff(&s));
	    }
	    else 
		while ((*s == '.') || (*s == '*')) s++;
	}
	else varname = (char*) strdup("TCST");

	if (constructing_monome) {
	    vect_add_elem(&(monome_term(curpm)), 
			  name_to_variable(varname), 
			  power);
	}
	else {
	    curpm = make_monome(coeff, name_to_variable(varname), power);
            constructing_monome = true;
	}
	/*fprintf(stderr, "au milieu: s='%s'\n", s);*/

	if ((*s == '+') || (*s == '-'))
	{
	    polynome_monome_add(&pp, curpm);
	    monome_rm(&curpm);
            constructing_monome = false;
	}
    }
    if (!MONOME_NUL_P(curpm)) {
	polynome_monome_add(&pp, curpm);
	monome_rm(&curpm);
    }

    return (pp);
}
