/* $RCSfile: simple_atomize.c,v $ ($Revision$)
 * $Date: 1998/04/14 21:49:59 $, 
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"

#include "ri-util.h"
#include "misc.h"
#include "control.h" /* for CONTROL_MAP() */
#include "transformations.h"

/* void atomize_as_required(stat, expr_decide, func_decide, test_decide, new)
 * statement stat;
 * bool (*expr_decide)(ref r, expression e);
 * bool (*func_decide)(ref r, expression e);
 * bool (*test_decide)(expression e);
 * entity (*new)(entity m, tag t);
 *
 * atomizes the given statement as driven by the given functions.
 * ??? does not care of any side effect and so.
 * 
 * - expr_decide tells whether or not to atomize for reference r
 *   and expression indice e of this reference. If yes, expression e
 *   is computed outside.
 * - func_decide tells whether or not to atomize for call c
 *   and argument expression e...
 * - test_decide tells whether expression condition e of a test
 *   should be atomized or not.
 * - new creates a new entity in the current module, the basic type
 *   of which should be t.
 */

static void atomize_object(gen_chunk*);

/* static functions used
 */
static bool (*expr_atomize_decision)(/* ref r, expression e */) = NULL;
static bool (*func_atomize_decision)(/* call c, expression e */)= NULL;
static bool (*test_atomize_decision)(/* expression e */) = NULL;
static entity (*create_new_variable)(/* entity m, tag t */) = NULL;

/* the stack of the encoutered statements is maintained
 * to be able to insert the needed computations just before the
 * very last statement encountered.
 *
 * The use of a stack is not necessary, since there are no effective rewrite.
 * but it could have been... It enables some recursive calls that would have
 * required to save the current statement explicitely otherwise.
 */
DEFINE_LOCAL_STACK(current_statement, statement)
DEFINE_LOCAL_STACK(current_control, control)

static bool cont_filter(c)
control c;
{
    current_control_push(c);
    return(TRUE);
}

static void cont_rewrite(c)
control c;
{
    (void) current_control_pop();
}

static bool stat_filter(s)
statement s;
{
    current_statement_push(s);
    return(TRUE);
}

static void stat_rewrite(s)
statement s;
{
    (void) current_statement_pop();
}

/* s is inserted before the current statement. 
 * if it is a block, it si added at the head of the list,
 * else a block statement is created in place of the current statement.
 */
static void insert_before_current_statement(s)
statement s;
{
    statement   cs = current_statement_head();
    instruction  i = statement_instruction(cs);
    control     cc = current_control_empty_p() ? 
	control_undefined : current_control_head() ;

    if (!control_undefined_p(cc) && control_statement(cc)==cs)
    {
	/* it is in an unstructured, and s is to be inserted properly
	 */
	bool seen;
	control newc;

	debug(8, "insert_before_current_statement", "unstructured\n");

	newc = make_control(s, control_predecessors(cc), 
			    CONS(CONTROL, cc, NIL));
	control_predecessors(cc) = CONS(CONTROL, newc, NIL);

	/*   update the other lists
	 */
	MAPL(c1,
	 {
	     seen=FALSE;
	     
	     MAPL(c2,
	      {
		  if (CONTROL(CAR(c2))==cc) 
		  {
		      CONTROL(CAR(c2))=newc;
		      seen=TRUE;
		  }
	      },
		  control_successors(CONTROL(CAR(c1))));

	     assert(seen);
	 },
	     control_predecessors(newc));

	/*   new current statement, to avoid other control insertions
	 */
	current_statement_replace(s);
	gen_recurse_stop(newc); 
    }
    else
    {
	debug(7, "insert_before_current_statement", "statement\n");

	if (instruction_block_p(i))
	    instruction_block(i) =
		CONS(STATEMENT, s, instruction_block(i));
	else
	    statement_instruction(cs) =
		make_instruction_block(
		 CONS(STATEMENT, s,
		 CONS(STATEMENT, make_stmt_of_instr(statement_instruction(cs)),
		      NIL)));
    }
}

static void compute_before_current_statement(pe)
expression *pe;
{
    entity new_variable;
    statement stat;
    expression rhs;

    assert(!syntax_range_p(expression_syntax(*pe)));

    /*  The index is computed
     */
    new_variable = 
	(*create_new_variable)(get_current_module_entity(), 
			       suggest_basic_for_expression(*pe));

    /*  The normalized field is kept as such, and will be used by
     *  the overlap analysis! ??? just a horrible hack.
     *
     *  stat: Variable = Expression
     *   *pe: Variable
     */
    rhs = make_expression(expression_syntax(*pe), normalized_undefined);
    normalize_all_expressions_of(rhs);

    stat = make_assign_statement(entity_to_expression(new_variable), rhs);

    *pe = make_expression(make_syntax(is_syntax_reference, 
				      make_reference(new_variable, NIL)),
			  expression_normalized(*pe));

    insert_before_current_statement(stat);
}

static bool ref_filter(r)
reference r;
{
    MAPL(ce, 
     {
	 expression *pe = (expression*) REFCAR(ce);
	 
	 if ((*expr_atomize_decision)(r, *pe))
	 {
	     syntax saved = expression_syntax(*pe);

	     compute_before_current_statement(pe);
	     atomize_object(saved);
	 }
     },
	 reference_indices(r));

    return(FALSE);
}

static bool call_filter(c)
call c;
{
    if (same_string_p(entity_local_name(call_function(c)), IMPLIED_DO_NAME))
	return(FALSE); /* (panic mode:-) */

    MAPL(ce, 
     {
	 expression *pe = (expression*) REFCAR(ce);

	 if ((*func_atomize_decision)(c, *pe))
	 {
	     syntax saved = expression_syntax(*pe);

	     compute_before_current_statement(pe);
	     atomize_object(saved);
	 }
     },
	 call_arguments(c));

    return(TRUE);
}

static bool test_filter(t)
test t;
{
    expression *pe = &test_condition(t);

    if ((*test_atomize_decision)(*pe)) 
    {
	syntax saved = expression_syntax(*pe);

	/*   else I have to break the condition
	 *   and to complete the recursion.
	 */
	compute_before_current_statement(pe);
	atomize_object(saved);
    }

    return(TRUE);
}

static void atomize_object(obj)
gen_chunk *obj;
{
    gen_multi_recurse
	(obj,
	 control_domain, cont_filter, cont_rewrite,   /* CONTROL */
	 statement_domain, stat_filter, stat_rewrite, /* STATEMENT */
	 reference_domain, ref_filter, gen_null,      /* REFERENCE */
	 test_domain, test_filter, gen_null,          /* TEST */
         call_domain, call_filter, gen_null,          /* CALL */
	 NULL);
}

void atomize_as_required(stat, expr_decide, func_decide, test_decide, new)
statement stat;
bool (*expr_decide)();
bool (*func_decide)();
bool (*test_decide)();
entity (*new)();
{
    make_current_statement_stack();
    make_current_control_stack();
    expr_atomize_decision = expr_decide;
    func_atomize_decision = func_decide;
    test_atomize_decision = test_decide;
    create_new_variable = new;
    
    atomize_object(stat);

    expr_atomize_decision = NULL;
    func_atomize_decision = NULL;
    test_atomize_decision = NULL;
    create_new_variable = NULL;
    free_current_statement_stack();
    free_current_control_stack();
}

/*  that is all
 */
