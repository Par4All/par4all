/* 
 * This section has been revised and cleaned.
 * The main change is to sort the arguments for 
 * the preconditions print.
 * 
 * Modification : Arnauld LESERVOT
 * Date         : 92 08 27
 * Old version  : prettyprint.old.c
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "constants.h"

#include "misc.h"
#include "properties.h"

#include "transformer.h"
extern void qsort();


/* This function is not declred any where else.
 * It is used to sort arguments preconditions in 
 * arguments_to_string  ( qsort).
 */
static int wordcmp(s1,s2)
char **s1, **s2;
{
    return strcmp(*s1,*s2);
}




string transformer_to_string(tf)
transformer tf;
{
    static char buffer[2048];
    boolean a_la_fortran = get_bool_property("PRETTYPRINT_FOR_FORESYS");

    buffer[0] = '\0';

    if(tf==transformer_undefined)
	(void) strcat(buffer, " TRANSFORMER: TRANSFORMER_UNDEFINED\n");
    else {
	(void) strcat(buffer, "T");
	(void) arguments_to_string(buffer, transformer_arguments(tf));
	if (a_la_fortran)
	    (void) strcat(buffer, ",");
	(void) strcat(buffer, " ");
	(void) relation_to_string(buffer,
				  (Psysteme) predicate_system(transformer_relation(tf)),
				  pips_user_value_name);

    }
    pips_assert("transformer_to_string", strlen(buffer)<2047);
    
    return strdup(buffer);
}



string precondition_to_string(pre)
transformer pre;
{
    static char buffer[2048];
    boolean a_la_fortran = get_bool_property("PRETTYPRINT_FOR_FORESYS");

    buffer[0] = '\0';

    if(pre==transformer_undefined)
	(void) strcat(buffer, " PRECONDITION: TRANSFORMER_UNDEFINED\n");
    else {
	(void) strcat(buffer, "P");
	(void) arguments_to_string(buffer, transformer_arguments(pre));
	if (a_la_fortran)
	    (void) strcat(buffer, ",");
	(void) strcat(buffer, " ");
	(void) relation_to_string(buffer,
				  (Psysteme) predicate_system(transformer_relation(pre)),
				  pips_user_value_name);
    };
    
    pips_assert("precondition_to_string", strlen(buffer)<2047);
    return strdup(buffer);
}


/*
 * Modification : 92 08 26  Arnauld
 *
 * Now, the function also sorts entity_local_name .
 *
 */

string arguments_to_string(s, args)
string s;
cons * args;
{
    int j=0, provi_length = 1;
    char *provi[100];

    strcat(s, "(");
    if(ENDP(args));
    else {
	MAPL(c, {entity e = ENTITY(CAR(c));
		 if (e==entity_undefined) 
			provi[j] = (char*) "entity_undefined";
		 else
			provi[j] = (char*) entity_local_name(e);
		 j++;
	     },
	     args);
	provi_length = j;
    
	qsort(provi, provi_length, sizeof provi[0], wordcmp);
	if ( provi_length > 1 ) {
	    for (j=0; j < provi_length-1; j++) {
		strcat(s, provi[j]);
		strcat(s,",");
	    }
	};
	strcat(s, provi[provi_length-1]);
    }
    strcat(s, ")");
    return s;
}



static string (*transformer_variable_name)(entity) = NULL;

/* The strange argument type is required by qsort(), deep down in the calls */
static int is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    /* The constant term is given the highest weight to push constant
       terms at the end of the constraints and to make those easy
       to compare. If not, constant 0 will be handled differently from
       other constants. However, it would be nice to give constant terms
       the lowest weight to print simple constraints first...

       Either I define two comparison functions, or I cheat somewhere else.
       Let's cheat? */
    int is_equal = 0;
    
    if (term_cst(*pvarval1) && !term_cst(*pvarval2))
	is_equal = 1;
    else if (term_cst(*pvarval1) && term_cst(*pvarval2))
	is_equal = 0;
    else if(term_cst(*pvarval2))
	is_equal = -1;
    else
	is_equal = 
	    strcmp(transformer_variable_name((entity) vecteur_var(*pvarval1)),
		   transformer_variable_name((entity) vecteur_var(*pvarval2)));

    return is_equal; 
}



string relation_to_string(s, ps, variable_name)
string s;
Psysteme ps;
char * (*variable_name)(entity);
{
    Pcontrainte peq;
    boolean a_la_fortran = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    
    transformer_variable_name = variable_name;

    if (ps != NULL) {
	bool first = TRUE;

	sc_lexicographic_sort(ps, is_inferior_pvarval);

	if (!a_la_fortran)
	    (void) strcat(s, "{");

	for (peq = ps->egalites;peq!=NULL; peq=peq->succ) {
	    if(first)
		first = FALSE;
	    else
		switch (a_la_fortran) {
		case FALSE :
		    (void) sprintf(s+strlen(s),", ");
		    break;
		case TRUE : 
		    (void) sprintf(s+strlen(s),".AND.");
		    break;
		}
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),"(");
	    egalite_sprint_format(s,peq,variable_name, a_la_fortran);
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),")");
	}

	for (peq = ps->inegalites;peq!=NULL; peq=peq->succ) {
	    if(first)
		first = FALSE;
	    else
		switch (a_la_fortran) {
		case FALSE :
		    (void) sprintf(s+strlen(s),", ");
		    break;
		case TRUE : 
		    (void) sprintf(s+strlen(s),".AND.");
		    break;
		}
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),"(");
	    inegalite_sprint_format(s,peq,variable_name, a_la_fortran);
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),")");
	}

	if (!a_la_fortran)
	    (void) strcat(s,"}");
    }
    else
	(void) strcat(s,"SC_UNDEFINED");
    
    transformer_variable_name = NULL;

    return s;
}

char * pips_user_value_name(e)
entity e;
{
    if(e == (entity) TCST) {
	return "";
    }
    else {
	(void) gen_check(e, entity_domain);
	return entity_has_values_p(e)? entity_minimal_name(e) :
	    external_value_name(e);
    }
}
