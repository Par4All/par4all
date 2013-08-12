/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* comp_util.c
 *
 * useful routines for evaluation of the complexity of a program
 *
 * bool complexity_check(comp)
 * void complexity_check_and_warn(function_name, comp)
 * void good_complexity_assert(function_name, comp)
 * void complexity_fprint(fd, comp, print_stats_p, print_local_names_p)
 * char *complexity_sprint(comp, print_stats_p, print_local_names_p)
 * void fprint_statement_complexity(module, stat, hash_statement_to_complexity)
 * void prc(comp) (for dbx)
 * void prp(pp)   (for dbx)
 * void prv(pv)   (for dbx)
 * void fprint_cost_table(fd)
 * void init_cost_table();
 * int intrinsic_cost(name, argstype)
 * bool is_inferior_basic(basic1, basic2)
 * basic simple_basic_dup(b)
 * float constant_entity_to_float(e)
 * void trace_on(va_alist)
 * void trace_off()
 * list entity_list_reverse(l)
 * bool is_linear_unstructured(unstr)
 * void add_formal_parameters_to_hash_table(mod, hash_complexity_params)
 * void remove_formal_parameters_from_hash_table(mod, hash_complexity_params)
 * hash_table fetch_callees_complexities(module_name)
 * hash_table fetch_complexity_parameters(module_name)
 */
/* Modif:
  -- entity_local_name is replaced by module_local_name. LZ 230993
  -- add of missing operators to intrinsic_cost_table. Molka Becher 08.03.2011
*/

/* To have strndup(): */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>      /* getenv */
#include <stdarg.h>

#include "linear.h"

#include "genC.h"
#include "database.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "resources.h"

#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "text-util.h"     /* print_text */
#include "effects-generic.h"
#include "effects-simple.h"
#include "misc.h"
#include "constants.h"     /* IMPLIED_DO_NAME is defined there */
#include "properties.h"    /* get_string_property is defined there */
#include "matrice.h"
#include "polynome.h"
#include "complexity.h"

/* for debugging */
#define INDENT_BLANKS "  "
#define INDENT_VLINE  "| "
#define INDENT_BACK   "-"
#define INDENT_INTERVAL 2

/* return true if allright */
bool complexity_check(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_internal_error("complexity undefined");

    if ( !complexity_zero_p(comp) ) {
	return (polynome_check(complexity_polynome(comp)));
    }
    return (true);
}

void complexity_check_and_warn(s,comp)
const char *s;
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(comp) ) {
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr,"complexity ZERO for %s\n",s);
	}
    }	
    if (!complexity_check(comp))
	user_warning(s,"Bad internal complexity representation!\n");
}    

void good_complexity_assert(char * function, complexity comp)
{
    if (!complexity_check(comp))
	pips_internal_error("bad internal complexity representation");
}

/* duplicates complexity comp */
complexity complexity_dup(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_internal_error("complexity undefined");

    if ( complexity_zero_p(comp) ) 
	return (make_zero_complexity());
    else {
	varcount   vc = complexity_varcount(comp);
	rangecount rc = complexity_rangecount(comp);
	ifcount    ic = complexity_ifcount(comp);

	varcount newvc = make_varcount(varcount_symbolic(vc), 
				       varcount_guessed(vc),
				       varcount_bounded(vc), 
				       varcount_unknown(vc));
	rangecount newrc = make_rangecount(rangecount_profiled(rc), 
					   rangecount_guessed(rc),
					   rangecount_bounded(rc),
					   rangecount_unknown(rc));
	ifcount newic = make_ifcount(ifcount_profiled(ic), 
				     ifcount_computed(ic),
				     ifcount_halfhalf(ic));
    
	Ppolynome ppdup = polynome_dup(complexity_polynome(comp));
	complexity compl = make_complexity(ppdup, newvc, newrc, newic);

	return(compl);
    }
}

/* remove complexity comp */
void complexity_rm(pcomp)
complexity *pcomp;
{
    if ( COMPLEXITY_UNDEFINED_P(*pcomp) )
	pips_internal_error("undefined complexity");

    if ( !complexity_zero_p(*pcomp) ) 
	free_complexity(*pcomp);
    *pcomp = make_zero_complexity();// complexity_undefined;
}

char *complexity_sprint(comp, print_stats_p, print_local_names_p)
complexity comp;
bool print_stats_p, print_local_names_p;
{

    char *s=NULL;

    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_internal_error("complexity undefined");
    else {
        varcount vc   = complexity_varcount(comp);
        rangecount rc = complexity_rangecount(comp);
        ifcount ic    = complexity_ifcount(comp);

        char * p = polynome_sprint(complexity_polynome(comp),
                (print_local_names_p ? variable_local_name 
                 : variable_name),
                is_inferior_pvarval);

        if ( print_stats_p ) {
            asprintf(&s,"[(var:%td/%td/%td/%td)"
                    " (rng:%td/%td/%td/%td)"
                    " (ifs:%td/%td/%td)]  %s",
                    varcount_symbolic(vc),
                    varcount_guessed(vc),
                    varcount_bounded(vc), 
                    varcount_unknown(vc),
                    rangecount_profiled(rc),
                    rangecount_guessed(rc),
                    rangecount_bounded(rc), 
                    rangecount_unknown(rc),
                    ifcount_profiled(ic), 
                    ifcount_computed(ic),
                    ifcount_halfhalf(ic),p);
            free(p);
        }
        else
            s=p;
    }
    return s;
}

void complexity_fprint(fd, comp, print_stats_p, print_local_names_p)
FILE *fd;
complexity comp;
bool print_stats_p, print_local_names_p;
{
    char *s = complexity_sprint(comp, print_stats_p, print_local_names_p);

    fprintf(fd, "%s\n", s);
    free(s);
}

void complexity_dump(complexity comp)
{
    complexity_fprint(stderr, comp, false, true);
}

void prc(comp)   /* for dbxtool: "print complexity" */
complexity comp;
{
    complexity_fprint(stderr, comp, true, true);
}

void prp(pp)     /* for dbxtool: "print polynome" */
Ppolynome pp;
{
    polynome_fprint(stderr, pp, variable_name, is_inferior_pvarval);
    fprintf(stderr, "\n");
}

void prv(pv)     /* for dbxtool: "print vecteur (as a monome)" */
Pvecteur pv;
{
    vect_fprint_as_monome(stderr, pv, BASE_NULLE, variable_name, ".");
    fprintf(stderr, "\n");
}

void fprint_statement_complexity(module, stat, hash_statement_to_complexity)
entity module;
statement stat;
hash_table hash_statement_to_complexity;
{
  text t = text_statement(module, 0, stat, NIL);
    complexity comp;

    comp = ((complexity) hash_get(hash_statement_to_complexity,(char *)stat));
    if (COMPLEXITY_UNDEFINED_P(comp))
	pips_internal_error("undefined complexity");
    else {
	fprintf(stderr, "C -- ");
	complexity_fprint(stderr, comp, DO_PRINT_STATS, PRINT_LOCAL_NAMES);
    }
    print_text(stderr, t);
}

/* The table intrinsic_cost_table[] gathers cost information
 * of each intrinsic's cost; those costs are dynamically loaded
 * from user files. It also returns the "minimum" type
 * of the result of each intrinsic,
 * specified by its basic_tag and number of memory bytes.
 * ("bigger" and "minimum" refer to the order relation
 * defined in the routine "is_inferior_basic"; the tag
 * is_basic_overloaded is used as a don't care tag)
 * (ex: SIN has a type of FLOAT even if its arg is an INT)
 *
 * Modif:
 *  -- LOOP_OVERHEAD and CALL_OVERHEAD are added, 280993 LZ
 *  -- LOOP_OVERHEAD is divided into two: INIT and BRAANCH 081093 LZ
 */

intrinsic_cost_record intrinsic_cost_table[] = {

    { PLUS_OPERATOR_NAME,                      is_basic_int, INT_NBYTES, EMPTY_COST },
    { MINUS_OPERATOR_NAME,                     is_basic_int, INT_NBYTES, EMPTY_COST },
    { MULTIPLY_OPERATOR_NAME,                  is_basic_int, INT_NBYTES, EMPTY_COST },
    { DIVIDE_OPERATOR_NAME,                    is_basic_int, INT_NBYTES, EMPTY_COST },
    { UNARY_MINUS_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { POWER_OPERATOR_NAME,                     is_basic_int, INT_NBYTES, EMPTY_COST },
    { FIELD_OPERATOR_NAME,                     is_basic_int, INT_NBYTES, EMPTY_COST },
    { POINT_TO_OPERATOR_NAME,                  is_basic_int, INT_NBYTES, EMPTY_COST },
    { DEREFERENCING_OPERATOR_NAME,             is_basic_int, INT_NBYTES, EMPTY_COST },
    { POST_INCREMENT_OPERATOR_NAME,            is_basic_int, INT_NBYTES, EMPTY_COST },
    { POST_DECREMENT_OPERATOR_NAME,            is_basic_int, INT_NBYTES, EMPTY_COST },
    { PRE_INCREMENT_OPERATOR_NAME,             is_basic_int, INT_NBYTES, EMPTY_COST },
    { PRE_DECREMENT_OPERATOR_NAME,             is_basic_int, INT_NBYTES, EMPTY_COST },
    { MULTIPLY_UPDATE_OPERATOR_NAME,           is_basic_int, INT_NBYTES, EMPTY_COST },
    { DIVIDE_UPDATE_OPERATOR_NAME,             is_basic_int, INT_NBYTES, EMPTY_COST },
    { PLUS_UPDATE_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { MINUS_UPDATE_OPERATOR_NAME,              is_basic_int, INT_NBYTES, EMPTY_COST },
    { LEFT_SHIFT_UPDATE_OPERATOR_NAME,         is_basic_int, INT_NBYTES, EMPTY_COST },
    { RIGHT_SHIFT_UPDATE_OPERATOR_NAME,        is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_OR_UPDATE_OPERATOR_NAME,         is_basic_int, INT_NBYTES, EMPTY_COST },
    { LEFT_SHIFT_OPERATOR_NAME,                is_basic_int, INT_NBYTES, EMPTY_COST },
    { RIGHT_SHIFT_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { ADDRESS_OF_OPERATOR_NAME,                is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_AND_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_NOT_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_XOR_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { C_AND_OPERATOR_NAME,                     is_basic_int, INT_NBYTES, EMPTY_COST },
    { MODULO_UPDATE_OPERATOR_NAME,             is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_AND_UPDATE_OPERATOR_NAME,        is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_XOR_UPDATE_OPERATOR_NAME,        is_basic_int, INT_NBYTES, EMPTY_COST },
    { CONDITIONAL_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { C_NOT_OPERATOR_NAME,                     is_basic_int, INT_NBYTES, EMPTY_COST },
    { C_NON_EQUAL_OPERATOR_NAME,               is_basic_int, INT_NBYTES, EMPTY_COST },
    { C_MODULO_OPERATOR_NAME,                  is_basic_int, INT_NBYTES, EMPTY_COST },
    { BITWISE_OR_OPERATOR_NAME,                is_basic_int, INT_NBYTES, EMPTY_COST },
    { TYPE_CAST_COST,                          is_basic_int, INT_NBYTES, EMPTY_COST },

	/* intrinsics for integer multiply add/sub */
	{ IMA_OPERATOR_NAME,                  is_basic_int, INT_NBYTES, EMPTY_COST },
	{ IMS_OPERATOR_NAME,                  is_basic_int, INT_NBYTES, EMPTY_COST },

    { LOOP_INIT_OVERHEAD,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { LOOP_BRANCH_OVERHEAD,                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CONDITION_OVERHEAD,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { CALL_ZERO_OVERHEAD,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_ONE_OVERHEAD,  	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_TWO_OVERHEAD,  	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_THREE_OVERHEAD,	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_FOUR_OVERHEAD, 	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_FIVE_OVERHEAD, 	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_SIX_OVERHEAD,  	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_SEVEN_OVERHEAD,	              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { ONE_INDEX_NAME,                         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { TWO_INDEX_NAME,                         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { THREE_INDEX_NAME,                       is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { FOUR_INDEX_NAME,                        is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { FIVE_INDEX_NAME,                        is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { SIX_INDEX_NAME,                         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { SEVEN_INDEX_NAME,                       is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { MEMORY_READ_NAME,                       is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { ASSIGN_OPERATOR_NAME,                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { EQUIV_OPERATOR_NAME,                    is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { NON_EQUIV_OPERATOR_NAME,                is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { OR_OPERATOR_NAME,                       is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { AND_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { LESS_THAN_OPERATOR_NAME,                is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { GREATER_THAN_OPERATOR_NAME,             is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { LESS_OR_EQUAL_OPERATOR_NAME,            is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { GREATER_OR_EQUAL_OPERATOR_NAME,         is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { EQUAL_OPERATOR_NAME,                    is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { NON_EQUAL_OPERATOR_NAME,                is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { CONCATENATION_FUNCTION_NAME,            is_basic_string,  ZERO_BYTE, EMPTY_COST },
    { NOT_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },

    { CONTINUE_FUNCTION_NAME,                 is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { ENDDO_FUNCTION_NAME,                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { RETURN_FUNCTION_NAME,                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { C_RETURN_FUNCTION_NAME,                 is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { STOP_FUNCTION_NAME,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { END_FUNCTION_NAME,                      is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { FORMAT_FUNCTION_NAME,                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { INT_GENERIC_CONVERSION_NAME,            is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { IFIX_GENERIC_CONVERSION_NAME,           is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { IDINT_GENERIC_CONVERSION_NAME,          is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { REAL_GENERIC_CONVERSION_NAME,           is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { FLOAT_GENERIC_CONVERSION_NAME,          is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { SNGL_GENERIC_CONVERSION_NAME,           is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DBLE_GENERIC_CONVERSION_NAME,           is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CMPLX_GENERIC_CONVERSION_NAME,          is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { CHAR_TO_INT_CONVERSION_NAME,            is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { INT_TO_CHAR_CONVERSION_NAME,            is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { AINT_CONVERSION_NAME,                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { DINT_CONVERSION_NAME,                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ANINT_CONVERSION_NAME,                  is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DNINT_CONVERSION_NAME,                  is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { NINT_CONVERSION_NAME,                   is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { IDNINT_CONVERSION_NAME,                 is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { IABS_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { ABS_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DABS_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CABS_OPERATOR_NAME,                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },

    { MODULO_OPERATOR_NAME,                   is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { REAL_MODULO_OPERATOR_NAME,              is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DOUBLE_MODULO_OPERATOR_NAME,            is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ISIGN_OPERATOR_NAME,                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { SIGN_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DSIGN_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { IDIM_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { DIM_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DDIM_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { DPROD_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { MAX_OPERATOR_NAME,                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { MAX0_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { AMAX1_OPERATOR_NAME,                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DMAX1_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { AMAX0_OPERATOR_NAME,                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { MAX1_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { MIN_OPERATOR_NAME,                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { MIN0_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { AMIN1_OPERATOR_NAME,                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DMIN1_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { AMIN0_OPERATOR_NAME,                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { MIN1_OPERATOR_NAME,                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { LENGTH_OPERATOR_NAME,                   is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { INDEX_OPERATOR_NAME,                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { AIMAG_CONVERSION_NAME,                  is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { CONJG_OPERATOR_NAME,                    is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { SQRT_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DSQRT_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CSQRT_OPERATOR_NAME,                    is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },

    { EXP_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DEXP_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CEXP_OPERATOR_NAME,                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { ALOG_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DLOG_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CLOG_OPERATOR_NAME,                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { LOG_OPERATOR_NAME,                      is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { ALOG10_OPERATOR_NAME,                   is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DLOG10_OPERATOR_NAME,                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { LOG10_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { SIN_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DSIN_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CSIN_OPERATOR_NAME,                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { COS_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DCOS_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { CCOS_OPERATOR_NAME,                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { TAN_OPERATOR_NAME,                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DTAN_OPERATOR_NAME,                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ASIN_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DASIN_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ACOS_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DACOS_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ATAN_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DATAN_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { ATAN2_OPERATOR_NAME,                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DATAN2_OPERATOR_NAME,                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { SINH_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DSINH_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { COSH_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DCOSH_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { TANH_OPERATOR_NAME,                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { DTANH_OPERATOR_NAME,                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { LEFT_SHIFT_OPERATOR_NAME,               is_basic_int,     INT_NBYTES,     EMPTY_COST},
    { RIGHT_SHIFT_OPERATOR_NAME,              is_basic_int,     INT_NBYTES,     EMPTY_COST},

    { LGE_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { LGT_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { LLE_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { LLT_OPERATOR_NAME,                      is_basic_logical, ZERO_BYTE, EMPTY_COST },

    { LIST_DIRECTED_FORMAT_NAME,              is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { UNBOUNDED_DIMENSION_NAME,               is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { WRITE_FUNCTION_NAME,                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { REWIND_FUNCTION_NAME,                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { OPEN_FUNCTION_NAME,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CLOSE_FUNCTION_NAME,                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { READ_FUNCTION_NAME,                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { BUFFERIN_FUNCTION_NAME,                 is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { BUFFEROUT_FUNCTION_NAME,                is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { ENDFILE_FUNCTION_NAME,                  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { IMPLIED_DO_NAME,                        is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { NULL,                                   0, ZERO_BYTE, EMPTY_COST },
};


void fprint_cost_table(fd)
FILE *fd;
{
    struct intrinsic_cost_rec *p = intrinsic_cost_table;
    bool skip_one_line = false;
    
    fprintf(fd, "\nIntrinsic cost table:\n\n");
    fprintf(fd, "        Intrinsic name        int    float   double   complex   dcomplex\n");
    fprintf(fd, "------------------------------------------------------------------------\n");
	    
    for(; p->name != NULL ;p++) {
	if (1 ||(p->int_cost      != 0) ||
	    (p->float_cost    != 0) ||
	    (p->double_cost   != 0) ||
	    (p->complex_cost  != 0) ||
	    (p->dcomplex_cost != 0)) {
	    if (skip_one_line) {
		fprintf(fd, "%25s|\n", "");
		skip_one_line = false;
	    }
	    fprintf(fd, "%22.21s   |%6td %6td %7td %8td %8td\n",
		    p->name, p->int_cost, p->float_cost,
		    p->double_cost, p->complex_cost, p->dcomplex_cost);
	}
	else
	    skip_one_line = true;
    }
    fprintf(fd, "\n");
}

/* Completes the intrinsic cost table with the costs read from the files
 * specified in the "COMPLEXITY_COST_TABLE" string property
 * See properties.rc and ~pips/Pips/pipsrc.csh for more information.
 * 
 * L. ZHOU 13/03/91
 *
 * COST_DATA are names of five data files
 */
void init_cost_table()
{
    char *token, *comma, *filename ;
    float file_factor;

    char *cost_data = strdup(COST_DATA);
    char *tmp=NULL;

	for(token = strtok(cost_data, " "); (token != NULL);token = strtok(NULL, " ")) {
		comma = strchr(token, ',');

		if (comma == NULL) {
			tmp=strdup(token);
			file_factor = 1.0;
		}
		else {
			int ii = comma - token;
			tmp=strndup( token, ii);
			sscanf(++comma, "%f", &file_factor);
		}


		filename = strdup(concatenate( COMPLEXITY_COST_TABLES "/", get_string_property("COMPLEXITY_COST_TABLE"), "/", tmp, NULL));

		debug(5,"init_cost_table","file_factor is %f\n", file_factor);
		debug(1,"init_cost_table","cost file is %s\n",filename);

		load_cost_file(fopen_config(filename,NULL,"PIPS_COSTDIR"), file_factor);
		free(tmp);
		free(filename);
	}

}

/* 
 * Load (some) intrinsics costs from file "fd", 
 * multiplying them by "file_factor".
 */
void load_cost_file(fd, file_factor)
FILE *fd;
float file_factor;
{
    char *line = (char*) malloc(199);
    char *intrinsic_name = (char*) malloc(30);
    int int_cost, float_cost, double_cost, complex_cost, dcomplex_cost;
    struct intrinsic_cost_rec *p;
    float scale_factor = 1.0;
    bool recognized;

	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "\nReading cost file ");
	    if (file_factor != 1.0) 
		fprintf(stderr, "(x %.2f)", file_factor);
	}
	
	while (fgets(line, 99, fd) != NULL) {
	    if (*line == '%')
		sscanf(line+1, "%f", &scale_factor);
	    else if ((*line != '#') && (*line != '\n')) {
		sscanf(line, "%s %d %d %d %d %d", intrinsic_name,
		       &int_cost, &float_cost, &double_cost,
		       &complex_cost, &dcomplex_cost);
		recognized = false;
		for (p = intrinsic_cost_table; p->name != NULL; p++) {
		    if (same_string_p(p->name, intrinsic_name)) {
			p->int_cost = (int)
			    (int_cost * scale_factor * file_factor + 0.5);
			p->float_cost = (int)
			    (float_cost * scale_factor * file_factor + 0.5);
			p->double_cost = (int)
			    (double_cost * scale_factor * file_factor + 0.5);
			p->complex_cost = (int)
			    (complex_cost * scale_factor * file_factor + 0.5);
			p->dcomplex_cost = (int)
			    (dcomplex_cost * scale_factor * file_factor + 0.5);
			recognized = true;
			break;
		    }
		}
		if (!recognized)
		    user_warning("load_cost_file",
				 "%s:unrecognized intrinsic\n",intrinsic_name);
	    }
	}
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "\nScale factor is %f\n", scale_factor);
	}
	fclose(fd);

    free(intrinsic_name);
    free(line);
}

/* Return the cost of the intrinsic named s, knowing that
 * the "basic" type of its biggest argument is *pargsbasic.
 * Update *pargsbasic if the intrinsic returns a number
 * of bigger complexity.
 */
int intrinsic_cost(s, pargsbasic)
const char *s;
basic *pargsbasic;
{
  struct intrinsic_cost_rec *p;
  basic b;

  for (p = intrinsic_cost_table; p->name != NULL; p++) {
    if (same_string_p(p->name, s)) {

      /* Inserted by AP, oct 24th 1995 */
      if (same_string_p(p->name, "LOG") || same_string_p(p->name, "LOG10")) {
	user_warning("intrinsic_cost", "LOG or LOG10 functions used\n");
      }

      b = make_basic(p->min_basic_result, (void *) p->min_nbytes_result);
      if (is_inferior_basic(*pargsbasic, b)) {
	free_basic(*pargsbasic);
	*pargsbasic = simple_basic_dup(b);
      }

      switch (basic_tag(*pargsbasic)) {
	case is_basic_int:
	  return(p->int_cost);
	case is_basic_float:
	  return (basic_float(*pargsbasic) <= FLOAT_NBYTES ?
		  p->float_cost : p->double_cost);
	case is_basic_complex:
	  return (basic_complex(*pargsbasic) <= COMPLEX_NBYTES ?
		  p->complex_cost : p->dcomplex_cost);
	case is_basic_string:
	  return (STRING_INTRINSICS_COST);
	case is_basic_logical:
	  return (LOGICAL_INTRINSICS_COST);
	default:
	  pips_internal_error("basic tag is %d", basic_tag(*pargsbasic));
	}
    }
  }
  /* To satisfy cproto . LZ 02 Feb. 93 */
  return (STRING_INTRINSICS_COST);
}


/* Return if possible the value of e in a float.
 * it is supposed to be an int or a float.
 */
float constant_entity_to_float(e)
entity e;
{
    const char *cste = module_local_name(e);
    basic b = entity_basic(e);
    float f;

    if (basic_int_p(b) || basic_float_p(b)) {
	sscanf(cste, "%f", &f);
	return (f);
    }
    else {
	user_warning("constant_entity_to_float",
		     "Basic tag:%d, not 4->9, (entity %s)\n",basic_tag(b),cste);
	return (0.0);
    }
}

/* "trace on" */
static int call_level=0;
void trace_on(char * fmt, ...)
{
    if (get_bool_property("COMPLEXITY_TRACE_CALLS")) {
	va_list args;
	char *indentstring = (char*) malloc(99);
	bool b = (call_level >= 0);
	int i,k=1;

	indentstring[0] = '\0';

	for (i=0; i< (b ? call_level : - call_level); i++) {
	    indentstring = strcat(indentstring,
				  strdup(b ? ( (k>0) ? INDENT_BLANKS
					             : INDENT_VLINE )
					   : INDENT_BACK));
	    k = ( k<INDENT_INTERVAL ? k+1 : 0 );
	}

	fprintf(stderr, "%s>", indentstring);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);

	free(indentstring);
	call_level++;
    }
}

/* "trace off" */
void trace_off()
{
    if (get_bool_property("COMPLEXITY_TRACE_CALLS")) {
	char *indentstring = (char*) malloc(99);
	bool b = (call_level >= 0);
	int i,k=1;

	indentstring[0] = '\0';
	call_level--;
	for (i=0; i< (b ? call_level : - call_level); i++) {
	    indentstring = strcat(indentstring,
				  strdup(b ? ( (k>0) ? INDENT_BLANKS
					             : INDENT_VLINE )
					   : INDENT_BACK));
	    k = ( k<INDENT_INTERVAL ? k+1 : 0 );
	}
	fprintf(stderr, "%s<\n", indentstring);
	free(indentstring);
    }
}

/* return true if unstr is simply a linear 
 * string of controls
 */
bool is_linear_unstructured(unstr)
unstructured unstr;
{
    control current = unstructured_control(unstr);
    control exit = unstructured_exit(unstr);

    while (current != exit) {
	list succs = control_successors(current);

	if (succs == NIL)
	    pips_internal_error("control != exit one,it has no successor");
	if (CDR(succs) != NIL) 
	    return (false);
	current = CONTROL(CAR(succs));
    }

    return(true);
}

list entity_list_reverse(l)
list l;
{
    entity e;

    if ((l == NIL) || (l->cdr == NIL)) 
	return l;
    e = ENTITY(CAR(l));
    return (CONS(ENTITY, e, entity_list_reverse(l->cdr)));
}

void add_formal_parameters_to_hash_table(mod, hash_complexity_params)
entity mod;
hash_table hash_complexity_params;
{
    list decl;

    pips_assert("add_formal_parameters_to_hash_table",
		entity_module_p(mod));
    decl = code_declarations(value_code(entity_initial(mod)));

    MAPL(pe, {
	entity param = ENTITY(CAR(pe));
	if (storage_formal_p(entity_storage(param))) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr,"\nstorage_formal %s\n",
			entity_name(param));
	    }
	    hash_put(hash_complexity_params, (char *) strdup(module_local_name(param)),
		     HASH_FORMAL_PARAM);
        }
    }, decl);
}

void remove_formal_parameters_from_hash_table(mod, hash_complexity_params)
entity mod;
hash_table hash_complexity_params;
{
    list decl;

    pips_assert("remove_formal_parameters_from_hash_table",
		entity_module_p(mod));
    decl = code_declarations(value_code(entity_initial(mod)));

    MAPL(pe, {
	entity param = ENTITY(CAR(pe));
	if (storage_formal_p(entity_storage(param)))
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr,"storage_formal %s to be deleted\n",
			entity_name(param));
	    }
	    hash_del(hash_complexity_params, (char *) module_local_name(param));
    }, decl);
}

hash_table free_callees_complexities(hash_table h)
{
    /* Modified copies of the summary complexities are stored */
    hash_table_clear(h);
    hash_table_free(h);

    return hash_table_undefined;
}

hash_table fetch_callees_complexities(module_name)
char *module_name;
{
    hash_table hash_callees_comp = hash_table_make(hash_pointer, 0);
    callees cl;
    list callees_list;
    complexity callee_comp;

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching callees complexities ...\n");
    }

    cl = (callees)db_get_memory_resource(DBR_CALLEES, module_name, true);
    callees_list = callees_callees(cl);

    if ( callees_list == NIL ) { 
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "Module %s has no callee! Done\n", module_name);
    	}
	return(hash_callees_comp);
    }
 
    MAPL(pc, {
	string callee_name = STRING(CAR(pc));
	entity callee = module_name_to_entity(callee_name);
	type t = entity_type(callee);

	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "%s has callee %s!\n",module_name,callee_name);
	}
	pips_assert("call_to_complexity",
		    type_functional_p(t) || type_void_p(t));

	if (value_code_p(entity_initial(callee))) {
	    complexity new_comp;
	    callee_comp = (complexity)
		db_get_memory_resource(DBR_SUMMARY_COMPLEXITY, 
				       (char *) callee_name,
				       true);

	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "fetched complexity for callee %s",
			         callee_name);
		fprintf(stderr, " of module %s:\n", module_name);
		complexity_fprint(stderr, callee_comp, 
				          DO_PRINT_STATS, 
				          ! PRINT_LOCAL_NAMES);
	    }

	    debug(5,"fetch_callees_complexities","callee_name %s\n",callee_name);

	    /* translate the local name to current module name. LZ 5 Feb.93 */
	    /* i.e. SUB:M -> MAIN:M */
	    /* FI: this seems to be wrong in general because the 
	     * formal parameter and actual argument are assumed to
	     * have the same name; see DemoStd/q and variables IM/IMM;
	     * 3 March 1994
	     */
	    new_comp = translate_complexity_from_local_to_current_name(callee_comp, 
							    callee_name,module_name);

	    hash_put(hash_callees_comp, (char *)callee, (char *)new_comp);
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "translated complexity for callee %s",
			         callee_name);
		fprintf(stderr, " of module %s:\n", module_name);
		complexity_fprint(stderr, new_comp, 
				          DO_PRINT_STATS, 
				          ! PRINT_LOCAL_NAMES);
	    }
	}
    }, callees_list );

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching callees complexities ... done\n");
    }

    return(hash_callees_comp);
}

hash_table fetch_complexity_parameters(module_name)
char *module_name;
{
    hash_table hash_comp_params = hash_table_make(hash_pointer, 0);
    char *parameters = strdup(get_string_property("COMPLEXITY_PARAMETERS"));
    char *sep_chars = strdup(", ");
    char *token = (char*) malloc(30);
    entity e;

    hash_warn_on_redefinition();
   
    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching complexity parameters for module %s:\n",
		module_name);
    }

    token = strtok(parameters, sep_chars);

    while (token != NULL) {
	e = gen_find_tabulated(concatenate(module_name,
					   MODULE_SEP_STRING,
					   token,
					   (char *) NULL),
			       entity_domain);
	if (e != entity_undefined) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "{\t Defined entity  %s }\n", entity_name(e));
	    }
	    hash_put(hash_comp_params,(char *)e,HASH_USER_VARIABLE);
	}
	else {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "{\t Undefined token  %s }\n", token);
	    }
	}
	token = strtok(NULL, sep_chars);
    }

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching complexity parameters: ...done.\n");
    }

    return(hash_comp_params);
}

void add_common_variables_to_hash_table(module, hash_complexity_params)
entity module;
hash_table hash_complexity_params;
{
    const char* module_name = module_local_name(module);
    list sefs_list = list_undefined;
    list ce = list_undefined;

    pips_assert("add_common_variables_to_hash_table",
		entity_module_p(module));

    sefs_list = effects_to_list( (effects)
	db_get_memory_resource(DBR_SUMMARY_EFFECTS, module_name, true));

    ifdebug(5) {
	debug(5, "add_common_variables_to_hash_table",
	      "Effect list for %s\n",
	      module_name);
	print_effects(sefs_list);
    }

    for(ce= sefs_list; !ENDP(ce); POP(ce)) {
	effect obj = EFFECT(CAR(ce));
	reference r = effect_any_reference(obj);
	action ac = effect_action(obj);
	approximation ap = effect_approximation(obj);
	entity e = reference_variable(r);
	storage s = entity_storage(e);

	if ( !storage_formal_p(s) &&
	    action_read_p(ac) && approximation_exact_p(ap) ) {
	    debug(5,"add_common_variables_to_hash_table",
		  "%s added\n", module_local_name(e));
	    hash_put(hash_complexity_params, (char *) module_local_name(e),
		     HASH_COMMON_VARIABLE);
	}
    }
}

void remove_common_variables_from_hash_table(module, hash_complexity_params)
entity module;
hash_table hash_complexity_params;
{
    const char* module_name = module_local_name(module);
    list sefs_list;

    pips_assert("remove_common_variables_from_hash_table",
		entity_module_p(module));

    sefs_list = effects_to_list( (effects)
	db_get_memory_resource(DBR_SUMMARY_EFFECTS, module_name, true));

    MAPL(ce, { 
	effect obj = EFFECT(CAR(ce));
	reference r = effect_any_reference(obj);
	action ac = effect_action(obj);
	approximation ap = effect_approximation(obj);
	entity e = reference_variable(r);

	if ( action_read_p(ac) && approximation_exact_p(ap) ) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "%s deleted\n", module_local_name(e));
	    }
	    hash_del(hash_complexity_params, (char *) module_local_name(e));
	}
    }, sefs_list);
}

bool is_must_be_written_var(effects_list, var_name)
list effects_list;
char *var_name;
{
    MAPL(ce, { 
	effect eff = EFFECT(CAR(ce));

	if(eff == effect_undefined)
	    pips_internal_error("unexpected effect undefined");

	if ( action_write_p(effect_action(eff)) 
	    && approximation_exact_p(effect_approximation(eff)) ) {
	    reference r = effect_any_reference(eff);
	    entity e = reference_variable(r);
/*	    
	    fprintf(stderr, "is_must_be_written_var for entity %s\n", 
		    module_local_name(e) );
*/
	    if ( strcmp(module_local_name(e), var_name) == 0 ) {
		return (true);
	    }
	}
/*
	else {
	    fprintf(stderr, "is_must_be_written_var for NOT entity %s\n", 
		    module_local_name(reference_variable(effect_any_reference(eff))) );
	}
*/
    },effects_list);

    return (false);
}

/*
 * This procedure is used to evaluate the complexity which has been postponed 
 * to be evaluated by is_must_be_writteen.
 * LZ 26 Nov. 92
 */
complexity final_statement_to_complexity_evaluation(comp, precond, effects_list)
complexity comp;
transformer precond;
list effects_list;
{
    complexity final_comp = complexity_dup(comp);
    Ppolynome pp = complexity_polynome(comp);
    Pbase pb = vect_dup(polynome_used_var(pp, default_is_inferior_pvarval));


    fprintf(stderr, "Final evaluation\n");

    for ( ; !VECTEUR_NUL_P(pb); pb = pb->succ) {
	bool mustbewritten;
	char *var = variable_local_name(pb->var);

        fprintf(stderr, "Variable is %s\n", var);

        mustbewritten = is_must_be_written_var(effects_list, var);

        if ( mustbewritten ) {
	    complexity compsubst;
	    fprintf(stderr, "YES once\n");
	    compsubst = evaluate_var_to_complexity((entity)pb->var, 
						   precond, 
						   effects_list, 1);
	    complexity_fprint( stderr, compsubst, false, false);
/*

	    final_comp = complexity_var_subst(comp, pb->var, compsubst);
*/
	}
	comp = complexity_dup(final_comp);
    }

    complexity_fprint( stderr, final_comp, false, false);

    return ( final_comp );
}

/* translate_complexity_from_local_to_current_name(callee_comp,oldname,newname)
 * B:M -> A:M if A calls B
 * 5 Feb. 93 LZ
 *
 * This is not general enough to handle:
 * B:M -> A:N or B:M to A:N+1
 * FI, 3 March 1994
 */
complexity translate_complexity_from_local_to_current_name(callee_comp,oldname,newname)
complexity callee_comp;
string oldname,newname;
{
    Ppolynome pp = complexity_polynome(callee_comp);
    Pbase pb = polynome_used_var(pp, is_inferior_pvarval);
    Pbase pbcur = BASE_UNDEFINED;
    complexity comp = make_zero_complexity();
    complexity old_comp = complexity_dup(callee_comp);

    if(BASE_NULLE_P(pb)) {
	/* constant complexity */
	comp = complexity_dup(callee_comp);
	return comp;
    }

    /* The basis associated to a polynomial includes the constant term! */
    if(base_dimension(pb)==1 && term_cst(pb)) {
	/* constant complexity */
	comp = complexity_dup(callee_comp);
	return comp;
    }

    for (pbcur=pb; pbcur != VECTEUR_NUL ; pbcur = pbcur->succ ) {
	Variable var = pbcur->var;
	char * stmp = strdup(variable_name(var));

	char *s = stmp;
	char *t = strchr(stmp,':');

	if ( t != NULL ) {
	    int length = (int)(t - s);
	    char *cur_name = (char *)malloc(100);

	    (void) strncpy(cur_name,stmp,length);
	    * (cur_name+length) = '\0';

	    if ( 1 || strncmp(cur_name, oldname, length) == 0 ) {
		Variable newvar = name_to_variable(concatenate(strdup(newname),
							       ":",strdup(t+1),NULL));
		if ( newvar != (Variable) chunk_undefined ) {
		    complexity compsubst = make_single_var_complexity(1.0, newvar);

/*
                    polynome_chg_var(&pp, var, newvar);
*/
		    comp = complexity_var_subst(old_comp, var, compsubst);
		    old_comp = complexity_dup(comp);
		}
		else {
		    comp = complexity_dup(old_comp);
		    old_comp = complexity_dup(comp);
		}		    
	    }
	}
    }
    return (comp);
}

bool complexity_is_monomial_p(complexity c)
{
    Ppolynome p = complexity_eval(c);
    bool monomial_p = is_single_monome(p);

    return monomial_p;
}

int complexity_degree(complexity c)
{
    Ppolynome p = complexity_eval(c);
    int degree = polynome_max_degree(p);

    return degree;
}
