/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* comp_math.c
 *
 * "mathematical" operations on complexities
 *
 *
 * complexity complexity_sigma(comp, index, clower, cupper)
 * complexity_var_subst(comp, var, compsubst)
 * complexity polynome_to_new_complexity(pp)
 * complexity complexity_dup(comp)
 * void complexity_rm(pcomp)
 * complexity make_single_var_complexity(float, var)
 * bool complexity_constant_p(comp)
 * float complexity_TCST(comp)
 * void complexity_scalar_mult(pcomp, f)
 * void complexity_float_add(pcomp, f)
 * void complexity_stats_add(pcomp1, comp2)
 * void complexity_add(pcomp1, comp2)
 * void complexity_sub(pcomp1, comp2)
 * void complexity_mult(pcomp1, comp2)
 * void complexity_polynome_add(pcomp, pp)
 * Ppolynome complexity_polynome(comp)
 * complexity replace_formal_parameters_by_real_ones(comp, mod, args, precond)
 */

/* Modif:
  -- entity_local_name is replaced by module_local_name. LZ 230993
*/

#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "matrice.h"
#include "properties.h"
#include "complexity.h"

/* complexity complexity_sigma(complexity comp, Variable index,
 *                             complexity clower, cupper)
 * return the integration of complexity comp when the index
 * is running between clower and cupper. Based on the polynomial
 * library routine polynome_sigma.
 *   - comp is undefined => result undefined.
 *   - comp is null => result is null whatever the bounds are.
 *   - bound(s) is(are) undefined:
 *       if comp depends on the index, the result is undefined;
 *       else the result is comp * UNKNOWN_RANGE.
 *   - everything is defined: the result is the integration
 *     of the respective polynomials. 
 */
complexity complexity_sigma(comp, index, clower, cupper)
complexity comp;
Variable index;
complexity clower, cupper;
{
    complexity cresult = make_zero_complexity();
    Ppolynome ppsum;
    Ppolynome pplower = POLYNOME_NUL;
    Ppolynome ppupper = POLYNOME_NUL;

    if (COMPLEXITY_UNDEFINED_P(comp))
	pips_internal_error("complexity undefined");

    if (complexity_zero_p(comp))
	return (cresult);
    else if ( COMPLEXITY_UNDEFINED_P(clower) || 
	     COMPLEXITY_UNDEFINED_P(cupper) ) {
	if ( polynome_contains_var(complexity_polynome(comp), 
				   (Variable)index) )
	    return (cresult);
	else {
	    /* FI: Too late to build a meaningful unknown range name! */
	    /*
	    ppsum = make_polynome(1.0, UNKNOWN_RANGE, 1);
	    */
	    entity ur = make_new_scalar_variable_with_prefix
		(UNKNOWN_RANGE_NAME,
		 get_current_module_entity(),
		 MakeBasic(is_basic_int));
        AddEntityToCurrentModule(ur);
	    ppsum = make_polynome(1.0, (Variable) ur, VALUE_ONE);
	    cresult = polynome_to_new_complexity(ppsum); /*stats*/
	    complexity_mult(&cresult, comp);
	    
	    polynome_rm(&ppsum);
	    return(cresult);
	}
    }
    else {
	pplower = complexity_polynome(clower);
	ppupper = complexity_polynome(cupper);
	
	if (false) {
	    fprintf(stderr, "summing ");
	    prp(complexity_polynome(comp));
	    fprintf(stderr, " %s running between ", 
		    module_local_name((entity)index));
	    prp(pplower);
	    fprintf(stderr, " and ");
	    prp(ppupper);
	    fprintf(stderr, "\n");
	}
	
	ppsum = polynome_sigma(complexity_polynome(comp),
			       (Variable) index,
			       pplower, ppupper);
	cresult = polynome_to_new_complexity(ppsum);
	complexity_stats_add(&cresult, comp);
	complexity_stats_add(&cresult, clower);
	complexity_stats_add(&cresult, cupper);
	
	polynome_rm(&ppsum);
	return(cresult);
    }
}

/* complexity complexity_var_subst(comp, var, compsubst)
 * replaces every occurrence of variable var in complexity comp
 * by the polynomial of complexity compsubst. The statistics
 * of compsubst are added to those of comp.
 */
complexity complexity_var_subst(comp, var, compsubst)
complexity comp;
Variable var;
complexity compsubst;
{
    Ppolynome pp, ppsubst, ppresult;
    complexity cresult = make_zero_complexity();

    if (COMPLEXITY_UNDEFINED_P(comp) || COMPLEXITY_UNDEFINED_P(compsubst)) 
	pips_internal_error("complexity undefined");

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"complexity_var_subst, variable name=%s\n",
			variable_name(var) );
    }

    if (complexity_zero_p(comp)) 
	return (cresult);
    else {
	pp = complexity_polynome(comp);
	ppsubst = complexity_polynome(compsubst);

	ppresult = polynome_var_subst(pp, var, ppsubst); /* substitutes */

	cresult = polynome_to_new_complexity(ppresult);
	complexity_stats_add(&cresult, compsubst);

	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    (void) complexity_consistent_p(cresult);
	    fprintf(stderr,"complexity_var_subst, comp    is ");
	    complexity_fprint(stderr, comp, false, true);
	    fprintf(stderr,"complexity_var_subst, compsubst is ");
	    complexity_fprint(stderr, compsubst, false, true);
	    fprintf(stderr,"complexity_var_subst, cresult is ");
	    complexity_fprint(stderr, cresult, false, true);
	}
    }

    return(cresult);
}


/* Create a complexity equal to Ppolynome pp
 * with null statistics. pp IS duplicated.
 */
complexity polynome_to_new_complexity(pp)
Ppolynome pp;
{
    varcount vc = make_varcount(0,0,0,0);
    rangecount rc = make_rangecount(0,0,0,0);
    ifcount ic = make_ifcount(0,0,0);
    Ppolynome ppdup;
    complexity comp;

    pips_assert("polynome_to_new_complexity", !POLYNOME_UNDEFINED_P(pp));

    ppdup = polynome_dup(pp);
    comp = make_complexity(ppdup, vc, rc, ic);
    ifdebug(1) {
	(void) complexity_consistent_p(comp);
    }
    return comp;
}

/* make a complexity "f * var" with null statistics */
complexity make_single_var_complexity(f, var)
float f;
Variable var;
{
    Ppolynome pp = make_polynome(f, var, VALUE_ONE);
    complexity comp = polynome_to_new_complexity(pp);
    polynome_rm(&pp);
    return(comp);
}

/* make a constant complexity "f * TCST" with null statistics */
complexity make_constant_complexity(f)
float f;
{
    return make_single_var_complexity(f, TCST);
}

/* make a zero complexity "0.0000 * TCST" with null statistics */
complexity make_zero_complexity()
{
    return make_constant_complexity(0.0000);
}

/* zero complexity check. Abort if undefined */
bool complexity_zero_p(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_internal_error("undefined complexity");

    if ( POLYNOME_NUL_P((Ppolynome)complexity_eval(comp)) )
	return (true);
    return (false);
}

/* true if comp is constant. Abort if undefined  */
bool complexity_constant_p(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_internal_error("undefined complexity");

    if ( complexity_zero_p(comp) ) 
	return (true);
    else 
	return (polynome_constant_p(complexity_eval(comp)));
}

/* true if comp is unknown. Abort if undefined  */
bool complexity_unknown_p(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_internal_error("undefined complexity");

    /* FI: Management of unknown complexities, when polynomes are in
       fact used to represent the value of a variables or an expression,
       has to be revisited */
    /*
    if ( polynome_contains_var((Ppolynome)complexity_eval(comp), 
			       UNKNOWN_RANGE) ) 
	return (true);
    else 
    */
	return (false);
}

/* return the constant term of comp. Abort if undefined */
float complexity_TCST(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_internal_error("undefined complexity");

    if ( complexity_zero_p(comp) ) 
	return ((float) 0);
    else 
	return (polynome_TCST(complexity_eval(comp)));
}

/* multiply a complexity by a floating-point number. 
 * Abort if undefined.
 */
void complexity_scalar_mult(pcomp, f)
complexity *pcomp;
float f;
{
    if ( COMPLEXITY_UNDEFINED_P(*pcomp) )
	pips_internal_error("complexity undefined");

    if ( f == 0.00 ) {
	complexity_rm(pcomp);	
	*pcomp = make_zero_complexity();
    }
    else if ( !complexity_zero_p(*pcomp) )
	complexity_eval_(*pcomp) = 
	    newgen_Ppolynome
	    (polynome_scalar_multiply(complexity_eval(*pcomp), f));
}

/* Add comp2's statistics to *pcomp1's
 * comp2 keeps unchanged
 */
void complexity_stats_add(pcomp1, comp2)
complexity *pcomp1, comp2;
{
    ifdebug (1) {
	(void) complexity_consistent_p(*pcomp1);
	(void) complexity_consistent_p(comp2);
    }

    if ( COMPLEXITY_UNDEFINED_P(comp2) || COMPLEXITY_UNDEFINED_P(*pcomp1) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(*pcomp1) ) {
	*pcomp1 = complexity_dup(comp2);
	complexity_eval_(*pcomp1) = 
	    newgen_Ppolynome(polynome_free(complexity_eval(*pcomp1)));
    }
    else if ( !complexity_zero_p(comp2) ) {
	varcount   vc1 = complexity_varcount(*pcomp1);
	rangecount rc1 = complexity_rangecount(*pcomp1);
	ifcount    ic1 = complexity_ifcount(*pcomp1);
	varcount   vc2 = complexity_varcount(comp2);
	rangecount rc2 = complexity_rangecount(comp2);
	ifcount    ic2 = complexity_ifcount(comp2);

	varcount_symbolic(vc1) += varcount_symbolic(vc2);
	varcount_guessed(vc1) += varcount_guessed(vc2);
	varcount_bounded(vc1) += varcount_bounded(vc2);
	varcount_unknown(vc1) += varcount_unknown(vc2);

	rangecount_profiled(rc1) += rangecount_profiled(rc2);
	rangecount_guessed(rc1) += rangecount_guessed(rc2);
	rangecount_bounded(rc1) += rangecount_bounded(rc2);
	rangecount_unknown(rc1) += rangecount_unknown(rc2);

	ifcount_profiled(ic1) += ifcount_profiled(ic2);
	ifcount_computed(ic1) += ifcount_computed(ic2);
	ifcount_halfhalf(ic1) += ifcount_halfhalf(ic2);
    }

    ifdebug (1) {
	(void) complexity_consistent_p(*pcomp1);
    }
}    

/* void complexity_add(complexity *pcomp1, comp2)
 *   performs *pcomp1 = *pcomp1 + comp2;
 *   !usage: complexity_add(&comp1, comp2);
 *   comp2 keeps unchanged
 */
void complexity_add(pcomp1, comp2)
complexity *pcomp1, comp2;
{
    if ( COMPLEXITY_UNDEFINED_P(comp2) || COMPLEXITY_UNDEFINED_P(*pcomp1) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(*pcomp1) ) {
	*pcomp1 = complexity_dup(comp2);
    }
    else if ( !complexity_zero_p(comp2) ) {
	complexity_eval_(*pcomp1) = 
	    newgen_Ppolynome(polynome_addition(complexity_eval(*pcomp1),
					       complexity_eval(comp2)));
	complexity_stats_add(pcomp1, comp2);
    }
}

/* void complexity_sub(complexity *pcomp1, comp2)
 *   performs *pcomp1 = *pcomp1 - comp2;
 *   !usage: complexity_sub(&comp1, comp2);
 *   comp2 keeps unchanged
 */
void complexity_sub(pcomp1, comp2)
complexity *pcomp1, comp2;
{
    if ( COMPLEXITY_UNDEFINED_P(comp2) || COMPLEXITY_UNDEFINED_P(*pcomp1) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(*pcomp1) ) 
	*pcomp1 = complexity_dup(comp2);
    else if ( !complexity_zero_p(comp2) ) {
	complexity_eval_(comp2) = 
	    newgen_Ppolynome(polynome_opposed(complexity_eval(comp2)));
	complexity_eval_(*pcomp1) = 
	    newgen_Ppolynome(polynome_addition(complexity_eval(*pcomp1), 
					       complexity_eval(comp2)));
	complexity_stats_add(pcomp1, comp2);
	complexity_eval_(comp2) = 
	    newgen_Ppolynome(polynome_opposed(complexity_eval(comp2)));
    }
}

/* void complexity_mult(complexity *pcomp1, comp2)
 *   performs *pcomp1 = *pcomp1 * comp2;
 *   !usage: complexity_mult(&comp1, comp2);
 */
void complexity_mult(pcomp1, comp2)
complexity *pcomp1, comp2;
{
    if ( COMPLEXITY_UNDEFINED_P(comp2) || COMPLEXITY_UNDEFINED_P(*pcomp1) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(comp2) ) {
	complexity_rm(pcomp1);
	*pcomp1 = make_zero_complexity();
    }
    else if ( !complexity_zero_p(*pcomp1) ) {
	Ppolynome ppmult;

	ppmult = polynome_mult(complexity_eval(*pcomp1),
			       complexity_eval(comp2));
	complexity_eval_(*pcomp1) = 
	    newgen_Ppolynome(polynome_free(complexity_eval(*pcomp1)));
	/* (Ppolynome) complexity_eval(*pcomp1) = (Ppolynome) ppmult; */
	complexity_eval_(*pcomp1) = newgen_Ppolynome(ppmult);
	complexity_stats_add(pcomp1, comp2);
    }
}

/* void complexity_div(complexity *pcomp1, comp2)
 *   performs *pcomp1 = *pcomp1 / comp2;
 *   !usage: complexity_div(&comp1, comp2);
 */
void complexity_div(pcomp1, comp2)
complexity *pcomp1, comp2;
{
    if ( COMPLEXITY_UNDEFINED_P(comp2) || COMPLEXITY_UNDEFINED_P(*pcomp1) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(comp2) ) {
	pips_internal_error("complexity divider is zero");
    }
    else if ( !complexity_zero_p(*pcomp1) ) {
	Ppolynome ppdiv;
	ppdiv = polynome_div(complexity_eval(*pcomp1), complexity_eval(comp2));
/*	polynome_rm(&(complexity_eval(*pcomp1))); */
	complexity_eval_(*pcomp1) = newgen_Ppolynome(ppdiv);
	complexity_stats_add(pcomp1, comp2);
    }
}

void complexity_polynome_add(pcomp, pp)
complexity *pcomp;
Ppolynome pp;
{
    if ( COMPLEXITY_UNDEFINED_P(*pcomp) ) 
	pips_internal_error("complexity undefined");
    else if ( POLYNOME_UNDEFINED_P(pp)  )
	pips_internal_error("polynome undefined");
    else {
	complexity_eval_(*pcomp) = 
	    newgen_Ppolynome(polynome_addition(complexity_eval(*pcomp), pp));
    }
}

/* Add a floating point digit to the complexity
 * May 3, 91
 */
void complexity_float_add(pcomp, f)
complexity *pcomp;
float f;
{
    if ( COMPLEXITY_UNDEFINED_P(*pcomp) ) 
	pips_internal_error("complexity undefined");
    
    if ( complexity_zero_p(*pcomp) ) 
    {
	Ppolynome ppnew = make_polynome(f, TCST, VALUE_ONE);
	*pcomp = polynome_to_new_complexity(ppnew);
	polynome_rm(&ppnew);
    }
    else 
	complexity_eval_(*pcomp) = 
	    newgen_Ppolynome(polynome_scalar_addition(complexity_eval(*pcomp),
						      f));
}


/* Because complexity is composed of two elements,
 * we use this function to get the first element : polynome
 * Usage : complexity_polynome(complexity comp)
 *         you will get a pointer to the polynome
 * May 3, 91        lz
 */
Ppolynome complexity_polynome(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(comp) ) 
	return (POLYNOME_NUL);
    else 
	return ((Ppolynome)complexity_eval(comp));
}

/* transform formal params into real ones (args) in complexity comp */
complexity replace_formal_parameters_by_real_ones(comp, mod, args, precond, effects_list)
complexity comp;
entity mod;
list args;
transformer precond;
list effects_list;
{
    complexity cresult = complexity_dup(comp);
    complexity carg, ctemp;
    list decl = code_declarations(value_code(entity_initial(mod)));
    char *param_name;
    int param_rank;
    list argument;

    pips_assert("replace_formal_parameters_by_real_ones", entity_module_p(mod));

    FOREACH (ENTITY, param,decl) {
        storage st = entity_storage(param);

        /* print out the entity name for debugging purpose */
        if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
            fprintf(stderr,"in REPLACE, entity name is %s\n",
                    entity_name(param));
        }

        if (storage_formal_p(st)) {     
            /* if formal parameter... */
            param_name = entity_name(param);
            param_rank = formal_offset(storage_formal(st));

            argument = gen_nthcdr(param_rank-1,args);/* minus one because offsets start at 1 not 0 */

            if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
                fprintf(stderr,"formal offset=%d, formal name=%s\n",
                        param_rank, param_name);
            }

            carg = expression_to_complexity_polynome(EXPRESSION(CAR(argument)),
                    precond,
                    effects_list,
                    KEEP_SYMBOLS,
                    EXACT_VALUE);

            if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
                fprintf(stderr,"variable name is %s\n", variable_name((Variable)param) );
            }

            ctemp = complexity_var_subst(cresult, (Variable)param, carg);
            complexity_rm(&cresult); 
            cresult = complexity_dup(ctemp);
            complexity_rm(&carg);
        }
    }
   
    return (cresult);
}

/* return a pointer to the (i)th element of the list 
 * if i = 1, the pointer doesn't change at all.
 */
list list_ith_element(thelist, ith)
list thelist;
int ith;
{
    for( ; --ith > 0; thelist = CDR(thelist) ) {
	pips_assert("list_ith_element", thelist != NIL);
    }

    return (thelist);
}
