/***************************************************************** pnome-error.c
 *
 * "POLYNOME-ERROR" FUNCTION, MONOMIAL AND POLYNOMIAL CHECK
 *
 */

/*LINTLIBRARY*/

#include <stdio.h>
#include <stdarg.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"	

/* void polynome_error(va_dcl va_list): 
 * should be called to terminate execution and to core dump
 * when data structures are corrupted or when an undefined operation is
 * requested (zero divide for instance). 
 * polynome_error should be called as:
 * 
 *   polynome_error(function_name, format, expression-list)
 * 
 * where function_name is a string containing the name of the function
 * calling POLYNOME_ERROR, and where format and expression-list are passed
 * as arguments to vprintf. 
 * POLYNOME_ERROR terminates execution with abort.
 * Ex: polynome_error("polynome_power_n", "negative power: %d\n", p);
 * 
 */

/*VARARGS0*/
void polynome_error(char * name, char * fmt, ...)
{
    va_list args;

    va_start(args, fmt);

    /* print name of function causing error */
    (void) fprintf(stderr, "\npolynome error in %s: ", name);

    /* print out remainder of message */
    (void) vfprintf(stderr, fmt, args);
    va_end(args);

    /* create a core file for debug */
    (void) abort();
}


/* void good_polynome_assert(va_alist)
 *   Check if the second argument is a valid polynomial.
 *   If not, print first argument ((char *) function name) and abort.
 */
void good_polynome_assert(char * function, ...)
{
    va_list args;
    Ppolynome pp;

    va_start(args, function);
    pp = va_arg(args, Ppolynome);

    if (polynome_check(pp)) return;

    fprintf(stderr, "Bad internal polynomial representation in %s\n", function);
    va_end(args);
    abort();
}


/* boolean monome_check(Pmonome pm)
 *   Return TRUE if all's right.
 *   Looks if pm is MONOME_UNDEFINED; if not:
 *     make sure that the coeff is non nul, that the term is non nul,
 *     and checks the (Pvecteur) term.
 *     All this also checks that pm is pointing to a valid address.
 *
 *  Modification:
 *  - MONOME_NUL means 0 monome, and it's a good monome. LZ 10/10/91
 */
boolean monome_check(pm)
Pmonome pm;
{
    if ( MONOME_UNDEFINED_P(pm) )
	return (FALSE); 
    else if (MONOME_NUL_P(pm) )
	return (TRUE);
    else 
	return ((monome_coeff(pm) != 0) && 
		!VECTEUR_NUL_P(monome_term(pm)) &&
		vect_check(monome_term(pm)));
}

/* boolean polynome_check(Ppolynome pp)
 *   Return TRUE if all's right.
 *   Check each monomial, make sure there's no nul or undefined monomial,
 *   then check unicity of each monomial.
 *
 *  Modification:
 *  - POLYNOME_NUL means 0 polynome, and it's a good one. LZ 10/10/91
 */
boolean polynome_check(pp)
Ppolynome pp;
{
    if ( POLYNOME_UNDEFINED_P(pp) )
	return (FALSE);
    if ( POLYNOME_NUL_P(pp) )
	return (TRUE);
    else {
	Ppolynome curpp, curpp2;

	for (curpp = pp; curpp != POLYNOME_NUL; curpp = polynome_succ(curpp)) {
	    if ( !monome_check(polynome_monome(curpp)) ) {
		return (FALSE);
	    }
	    for (curpp2 = polynome_succ(curpp); curpp2 != POLYNOME_NUL;
		 curpp2 = polynome_succ(curpp2))
		if (monome_colin(polynome_monome(curpp),polynome_monome(curpp2))) 
		    return (FALSE);
	}
	return (TRUE);
    }
}

/* boolean is_polynome_a_monome(Ppolynome pp)
 *   Return TRUE if the pp is just a monome.
 *   that means the polynom has only one term
 *   Check each monomial, make sure there's no nul or undefined monomial,
 *   then check unicity of each monomial.
 *
 * LZ 06 Nov. 92
 */
boolean is_polynome_a_monome(pp)
Ppolynome pp;
{
    if ( ! polynome_check(pp) )
	return (FALSE);
    else if ( pp != POLYNOME_NUL && polynome_succ(pp) == POLYNOME_NUL )
	return (TRUE);
    else
	return (FALSE);
}
