/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1994/03/16 17:22:42 $, ) version $Revision$, 
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*
 * ONE ARRAY REFERENCES MODIFICATIONS
 */


/*
 * TEST GENERATION
 */

/*
 * expression Psysteme_to_expression(Psysteme systeme)
 *
 * From a Psysteme, a logical expression that checks for
 * the constraints is generated.
 */
expression Psysteme_to_expression(systeme)
Psysteme systeme;
{
    entity
	equ = local_name_to_top_level_entity(EQUAL_OPERATOR_NAME),
	leq = local_name_to_top_level_entity(LESS_OR_EQUAL_OPERATOR_NAME);
    list
	conjonction = 
	    gen_nconc
		(Pcontrainte_to_expression_list(sc_egalites(systeme), equ),
		 Pcontrainte_to_expression_list(sc_inegalites(systeme), leq));
    expression
	result = list_to_conjonction(conjonction);

    gen_free_list(conjonction);
    return(result);
}

/*
 * 
 */
list Pcontrainte_to_expression_list(constraint, operator)
Pcontrainte constraint;
entity operator;
{
    list
	result = NIL;
    Pcontrainte
	c = NULL;
    expression
	zero = make_integer_constant_expression(0);

    for(c = constraint;
	!CONTRAINTE_UNDEFINED_P(c);
	c=c->succ)
    {
	result = 
	    CONS(EXPRESSION,
		 MakeBinaryCall(operator,
				make_vecteur_expression(contrainte_vecteur(c)),
				zero),
		 result);
    }

    return(result);
}



/*
 * that's all
 */
