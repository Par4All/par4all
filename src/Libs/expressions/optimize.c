/*
 * $Id$
 *
 * $Log: optimize.c,v $
 * Revision 1.4  1998/09/14 12:34:11  coelho
 * added import from eole and substitution in module code.
 *
 * Revision 1.3  1998/09/11 12:18:39  coelho
 * new version thru a reference (suggested by PJ).
 *
 * Revision 1.2  1998/09/11 09:42:49  coelho
 * write equalities...
 *
 * Revision 1.1  1998/04/29 09:07:42  coelho
 * Initial revision
 *
 *
 * expression optimizations by Julien.
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"


#define DEBUG_NAME "TRANSFORMATION_OPTIMIZE_EXPRESSIONS_DEBUG_LEVEL"

/* the list of right hand side expressions.
 */
static list /* of expression */ rhs;

/* rhs expressions of assignments.
 */
static bool call_filter(call c)
{
    if (ENTITY_ASSIGN_P(call_function(c)))
    {
	expression e = EXPRESSION(CAR(CDR(call_arguments(c))));
	rhs = CONS(EXPRESSION, e, rhs);
    }
    return FALSE;
}

/* other expressions may be found in loops and so?
 *//*
static bool expr_filter(expression e)
{
    rhs = CONS(EXPRESSION, e, rhs);
    return FALSE;
}*/

static list /* of expression */ 
get_list_of_rhs(statement s)
{
    list result;

    rhs = NIL;
    gen_multi_recurse(s,
		      expression_domain, gen_false, gen_null,
		      call_domain, call_filter, gen_null,
		      NULL);
    
    result = gen_nreverse(rhs);
    rhs = NIL;
    return result;
}

/* export a list of expression of the current module.
 */
static void write_list_of_rhs(FILE * out, list /* of expression */ le)
{
    reference astuce = make_reference(get_current_module_entity(), le);
    write_reference(out, astuce);
    reference_indices(astuce) = NIL;
    free_reference(astuce);
}

/* export expressions to eole thru the newgen format.
 * both entities and rhs expressions are exported. 
 */
static void write_to_eole(string module, list le, string file_name)
{
    FILE * toeole;

    pips_debug(3, "writing to eole for module %s\n", module);

    toeole = safe_fopen(file_name, "w");
    write_tabulated_entity(toeole);
    write_list_of_rhs(toeole, le);

    safe_fclose(toeole, file_name);
}

/* import expressions from eole.
 */
static list /* of expression */
read_from_eole(string module, string file_name)
{
    FILE * fromeole;
    reference astuce;
    list result;

    pips_debug(3, "reading from eole for module %s\n", module);
    
    fromeole = safe_fopen(file_name, "r");

    /* read entites to create... 
     */

    astuce = read_reference(fromeole);
    result = reference_indices(astuce);
    reference_indices(astuce) = NIL;
    free_reference(astuce);

    return result;
}

/* swap term to term syntax field in expression list, as a side effect...
 */
static void 
swap_syntax_in_expression(
    list /* of expression */ lcode,
    list /* of expression */ lnew)
{
    pips_assert("equal length lists", gen_length(lcode)==gen_length(lnew));

    for(; lcode; lcode=CDR(lcode), lnew=CDR(lnew))
    {
	expression 
	    old = EXPRESSION(CAR(lcode)),
	    new = EXPRESSION(CAR(lnew));
	syntax tmp = expression_syntax(old);
	expression_syntax(old) = expression_syntax(new);
	expression_syntax(new) = tmp;	
    }
}

#define OUT_FILE_NAME 	"/tmp/pips_to_eole"
#define IN_FILE_NAME	"/tmp/eole_to_pips"

#define PIPS_EOLE	"newgen_eole"
#define PIPS_EOLE_FLAGS	"-nfmd"

/* pipsmake interface.
 */
bool optimize_expressions(string module_name)
{
    statement s;
    list /* of expression */ le, ln;
    string in, out, cmd;

    debug_on(DEBUG_NAME);

    /* get needed stuff.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));

    s = get_current_module_statement();

    /* do something here.
     */
    in = safe_new_tmp_file(IN_FILE_NAME);
    out = safe_new_tmp_file(OUT_FILE_NAME);

    le = get_list_of_rhs(s);
    write_to_eole(module_name, le, out);

    /* run eole (Evaluation Optimization for Loops and Expressions) 
     * as a separate process.
     */
    cmd = strdup(concatenate(
	PIPS_EOLE " " PIPS_EOLE_FLAGS " -o ", in, " ", out, NULL));

    pips_debug(2, "executing %s\n", cmd);
    safe_system(cmd);

    ln = read_from_eole(module_name, in);
    swap_syntax_in_expression(le, ln);
    gen_free_list(ln), ln=NIL;

    /* remove temorary files.
     */
    safe_unlink(out);
    safe_unlink(in);
    free(out), out = NULL;
    free(in), in = NULL;
    free(cmd), cmd = NULL;

    /* Should perform more optimizations here...
     */

    /* return result.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, s);

    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

    return TRUE; /* okay! */
}
