/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
   Revision 1.2  1998/12/30 16:54:18  zory
   atomization updated

   Revision 1.1  1998/12/28 15:50:32  coelho
   Initial revision


   Global code motion and Common subexpression elimination for nested
   sequences (a sort of perfect loop nest).
*/


#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"

#include "transformations.h"

#include "eole_private.h"


#define DEBUG_NAME "TRANSFORMATION_ICM_CSE_DEBUG_LEVEL"

#define NARY_ASSIGN EXPRESSION;
#define nary_assign expression;



/************************************************************** NESTING OKAY */

#define CALL_NESTED 1
#define LOOP_NESTED 2
#define NOT_NESTED  0

/* to remember the current statement we're in.
 */
DEFINE_LOCAL_STACK(current_stmt, statement)

/* statement -> int
 */
GENERIC_LOCAL_FUNCTION(is_nested, persistant_statement_to_int)

/* 
   LOOP INSTRUCTION. Check if the loop body as well as the loop indices
   are correct according to nested sequence manipulation 
*/
static void loop_rwt(loop l)
{
  statement current = current_stmt_head();
  
  /* Is the loop body a good candidate for nested sequence manipulation ? */
  int nb = load_is_nested(loop_body(l)); 
  
  /* Check validity for indices ... */

  store_is_nested(current, nb==NOT_NESTED? NOT_NESTED: LOOP_NESTED);
}

/* 
   SEQUENCE INSTRUCTION. Check all statements included in the
   sequence. Loops must be at the end only !  
*/
static void seq_rwt(sequence s)
{
  statement current = current_stmt_head();
  
  int v;
  int result = CALL_NESTED;



  MAP ( STATEMENT, 
	bs,
	{
	  v = load_is_nested(bs);
	  
	  if (result!=NOT_NESTED) {
	    
	    if (result == LOOP_NESTED) {
	      result = NOT_NESTED ;
	      break;
	    }
	    else 
	      result = v;
	  }
	  else 
	    break;
	      
	       
	},
	  sequence_statements(s) );

  store_is_nested(current, result);
}


/* 
   TEST INSTRUCTION. Check if at least one part of the test is empty and
   the other part is ok for nested sequence manipulation 
*/
static void test_rwt(test t)
{
  int result; 
  statement current = current_stmt_head();

  if ( test_true(t)!=statement_undefined ) { /* then part is NOT empty */
    if (test_false(t)==statement_undefined) /* else part is empty */
      result = load_is_nested(test_true(t));
    else /* else part is not empty */
      result = NOT_NESTED ;
  }
  else /* true part is empty (else part not empty) */
    result = load_is_nested(test_false(t));

  store_is_nested(current, result);
}


/* 
   CALL INSTRUCTION. Assign statement or function call. 
*/
static void call_rwt(call c)
{
  statement current = current_stmt_head();

  /* check call validity for nested sequence manipulation  ... */

  store_is_nested(current, CALL_NESTED);
}

static void not_okay(void)
{
  statement current = current_stmt_head();
  store_is_nested(current, NOT_NESTED);
}

/* 
   fill is_nested as a side effect.
*/
static void set_nesting_for_statement(statement s)
{

  make_current_stmt_stack();

  gen_multi_recurse(s,
     statement_domain, current_stmt_filter, current_stmt_rewrite,
		    test_domain, gen_true, test_rwt, 
		    loop_domain, gen_true, loop_rwt,
		    call_domain, gen_true, call_rwt,
		    sequence_domain, gen_true, seq_rwt,
		    unstructured_domain, gen_true, not_okay,
		    whileloop_domain, gen_true, not_okay,
		    expression_domain, gen_false, gen_null, /* no expr call */
		    NULL);

  free_current_stmt_stack();

}


/************************************************************ BUILD SEQUENCE */
#define Nested_Sequence gen_array_t /* of assignement */

static Nested_Sequence nested_sequence;

GENERIC_LOCAL_FUNCTION(atomized, persistant_expression_to_entity)

DEFINE_LOCAL_STACK(current_expr, expression)


/* 
   insert assignement in the nested sequence ... (append) 
*/
static void
add_to_nested_sequence(assignment assign)
{
  gen_array_append(nested_sequence, (assignment)assign);
}

/* 
   make scalar variable of the given type for the current module 
*/ 
static entity
make_scalar_var(basic b)
{
  entity module = get_current_module_entity();

  return make_new_scalar_variable(module,b);
}

/* 
   determine the correct type of the new temporary scalar variable and
   create it afterwards. 
*/
static entity 
make_new_scalar_variable_with_correct_type(expression e)
{

  /* determine type of expression (new object allocated by function) */
  basic b = basic_of_expression(e);

  return make_scalar_var(b);
}


/* 
   Atomization of a call  
*/
static void call_rwte(call c)
{
  entity fun = call_function(c);
  entity tmp;
  list /* entity */ lents = NIL;
  function lhs, rhs;
  
  MAP(EXPRESSION, e,
      lents = CONS(ENTITY, load_atomized(e), lents),
      call_arguments(c));

  lents = gen_nreverse(lents);
  
  rhs = make_function(fun, lents);
  
  tmp = make_new_scalar_variable_with_correct_type(current_expr_head());

  lhs = make_function(tmp, NIL);

  add_to_nested_sequence(make_assignment(lhs, rhs, NIL));

  store_atomized(current_expr_head(), tmp);
}

/* 
   Atomization of a reference - 
*/
static void ref_rwte(reference r)
{
  entity var = reference_variable(r);
  list /* of expression */ lind = reference_indices(r);
  if (lind) 
  {
    /* TMP = VAR ( ... )
     */ 
    function lhs, rhs;
    entity tmp;
    list /* of entity */ lents = NIL;

    MAP(EXPRESSION, e,
	lents = CONS(ENTITY, load_atomized(e), lents),
	lind);
    
    rhs = make_function(var, lents);
    
    tmp = make_new_scalar_variable_with_correct_type(current_expr_head());
    lhs = make_function(tmp, NIL);

    add_to_nested_sequence(make_assignment(lhs, rhs, NIL));
    
    store_atomized(current_expr_head(), tmp);
  }
  else
  {
    store_atomized(current_expr_head(), var);
  }
}


/* expression that is NOT an assignment ! 
 * FONCTIONNEL...
 */
static void atomize_expression_in_sequence(expression e)
{
  gen_multi_recurse(e,
	      expression_domain, current_expr_filter, current_expr_rewrite,
	      call_domain, gen_true, call_rwte,
	      reference_domain, gen_true, ref_rwte,
	      range_domain, gen_true, gen_core,
	      NULL);
}




/* 
   LOOP filter. I = DO(lb,ub,s)
*/
static bool loop_flt(loop l)
{
  entity i, low, up, inc, op;
  range ra;
  list /* of entity */ args = NIL; 
  list /* of entity */ deps = NIL; 
  function rhs, lhs;
  assignment a;

  /* index */
  i = loop_index(l);
  
  lhs = make_function(i,NIL);

  /* fake DO entity */
  op = local_name_to_top_level_entity("IMPLIED_DO_NAME");
  
  pips_assert("IMPLIED DO entity recognized \n", !entity_undefined_p(op) );
  
  /* range */
  ra = loop_range(l);
  
  atomize_expression_in_sequence(range_lower(ra));
  low = load_atomized(range_lower(ra));
  args = CONS(ENTITY,low,args);

  atomize_expression_in_sequence(range_upper(ra));
  up = load_atomized(range_upper(ra));
  args = CONS(ENTITY,up,args);

  atomize_expression_in_sequence(range_increment(ra));
  inc = load_atomized(range_increment(ra));
  args = CONS(ENTITY,inc,args);

  args = gen_nreverse(args);

  rhs = make_function(op, args); 

  /* control dependences ... */
  deps = NIL;

  /* insert in the nested sequence */
  a = make_assignment(lhs, rhs, deps);
  add_to_nested_sequence(a);

  return TRUE;
}

/* 
   TEST FILTER -  X = TEST( condition  )
*/

static bool test_flt(test t)
{
  assignment a;
  entity X, cond, op;
  list /* of entity */ args = NIL;
  list /* of entity */ deps = NIL;
  function lhs, rhs;

  /* please do not free b ... */ 
  int size = 4;
  basic b = make_basic(is_basic_logical,&size);

  /* create a new entity */
  X = make_scalar_var(b); 
  
  lhs = make_function(X,NIL);

  /* */
  atomize_expression_in_sequence(test_condition(t));
  cond = load_atomized(test_condition(t));
  args = CONS(ENTITY,cond,args);

  op = local_name_to_top_level_entity("TEST_NAME");
  pips_assert("TEST entity recognized \n", !entity_undefined_p(op) );
  
  rhs = make_function(op,args);

  /* search control dependencies */ 
  deps = NIL ;

  /* insert in the nested sequence */
  a = make_assignment(lhs, rhs, deps);
  add_to_nested_sequence(a);

  return TRUE;
}


static function 
lhs_call_flt(expression e)
{
  reference r;
  list /* of entity */ args = NIL;
  list /* of expressions */ inds = NIL;
  entity tmp;
  function f;

  pips_assert("lhs is a reference",syntax_reference_p(expression_syntax(e)));
  
  r = syntax_reference(expression_syntax(e));
  inds = reference_indices(r);

  if (inds) /* array reference */
    {
      MAP(EXPRESSION,
	  e,
	  {
	    atomize_expression_in_sequence(e);
	    tmp = load_atomized(e);
	    args = CONS(EXPRESSION, tmp, inds);
	  },
	    inds);
    }
  
  return make_function(reference_variable(r), args); 
}

static function 
rhs_call_flt(expression e)
{
  entity op, tmp;
  list /* of entity */ args = NIL;

  op = local_name_to_top_level_entity("STORE_NAME");
  pips_assert("STORE entity recognized \n", !entity_undefined_p(op) );

  atomize_expression_in_sequence(e);
  tmp = load_atomized(e);

  args == CONS(EXPRESSION, tmp, args); 

  return make_function(op, args);
}

/* 
   CALL FILTER - must be an assignment ! 
*/
static bool call_flt(call c)
{
  list /* of expression */ args = call_arguments(c);
  list /* of entity */ deps = NIL;
  expression e;
  function lhs, rhs;
  assignment a;

  pips_assert("call is an assignement", 
	      entity_local_name(call_function(c))==ASSIGN_OPERATOR_NAME);

  /* lhs */
  e = EXPRESSION(CAR(args));
  lhs = lhs_call_flt(e);
  
  /* rhs */
  e = EXPRESSION(CAR(CDR(args)));
  rhs = rhs_call_flt(e);

  /* search control dependencies */ 
  deps = NIL ;

  /* insert in the nested sequence */
  a = make_assignment(lhs, rhs, deps);
  add_to_nested_sequence(a);
  
  return TRUE; 
}

/* 
   build Nested Sequence form instruction 
*/

static Nested_Sequence
build_sequence(instruction i)
{
  gen_multi_recurse(i, 
		    loop_domain, loop_flt, gen_null, 
		    test_domain, test_flt, gen_null,
		    call_domain, call_flt, gen_null,
		    expression_domain, gen_true, gen_null, 
		    NULL);

  return nested_sequence;
}


/*********************************************** WALK THRU NESTS TO OPTIMIZE */

entity 
new_variable(
    entity module,
    tag t)
{
    return make_new_scalar_variable(module, MakeBasic(t));
}

static bool 
simple_expression_decision(e)
expression e;
{
    syntax s = expression_syntax(e);

    /*  don't atomize A(I)
     */
    if (syntax_reference_p(s)) {
      print_expression(e);
      return(!entity_scalar_p(reference_variable(syntax_reference(s))));
    }

    /*  don't atomize A(1)
     */
    if (expression_constant_p(e)) 
	return(FALSE);

    return(TRUE);
}

static bool
ref_atomization(reference r, expression e)
{
    return(simple_expression_decision(e));
}

static bool
call_atomization(call c, expression e)
{
    entity f = call_function(c);
    syntax s = expression_syntax(e);

    if (ENTITY_ASSIGN_P(f)) return(FALSE); 
    if (value_tag(entity_initial(f)) == is_value_intrinsic ||
	!syntax_reference_p(s))
	return(simple_expression_decision(e));
	
    /* the default is *not* to atomize.
     * should check that the reference is not used as a vector
     * and is not modified?
     */ 
    return(FALSE); 
}

static void 
specific_atomization (statement s) 
{

  pips_debug(2," specific ATOMIZATION applied \n");

  atomize_as_required(s,
		      ref_atomization,
		      call_atomization,
		      gen_true, /* test */
		      gen_true, /* range */
		      gen_true, /* whileloop */
		      /*new_atomizer_create_a_new_entity*/
		      new_variable);
}

static void 
cse(Nested_Sequence ns)
{
  
}

static void 
icm(Nested_Sequence ns)
{
  
}


static instruction 
apply_icm_cse(instruction i) 
{

  instruction new_instruction;

  /* nested_sequence must be before use */
  nested_sequence = gen_array_make(0);  

  /* atomize and create the nested sequence */
  build_sequence(i);

  /* apply icm on nested sequence  */
  icm(nested_sequence);

  /* apply cse on nested sequence   */
  cse(nested_sequence);

  /* create new instruction from nested sequence */
  /* new_instruction = create_instruction_from_nested_sequence(ns); */

  gen_array_free(nested_sequence);

  /* return new_instruction;*/
  return i;
}

static bool do_icm_cse(statement s)
{
  if (load_is_nested(s))
  {    
    /*  BIG transformation here... */
    instruction old = statement_instruction(s);
    instruction new;

    /* hack ....*/
    if (return_statement_p(s))
      return FALSE;

    new = apply_icm_cse(old);
    
    pips_debug(2," nested sequence found !\n");
    print_statement(s);
    
    /* finally update statement_instruction(s)*/
    statement_instruction(s) = new ;
    
    /* free old ?  */
    
    return FALSE; /* no need to enter in the statement ...  */
  }
  return TRUE; /* continue gen_recurse downward */
}

static void apply_icm_cse_on_statement(statement s)
{
  gen_recurse(s, statement_domain, do_icm_cse, gen_null);
}


/*****************************************************************************/

void 
icm_cse_on_sequence(string module_name, 
		    statement s) 
{ 

  pips_debug(2," ICM and CSE optimizations applied ... \n"); 

  /* check consistency before optimizations */
  pips_assert("consistency checking before optimizations",
	      statement_consistent_p(s));


  init_is_nested();  

  set_nesting_for_statement(s);
  
  nested_sequence = gen_array_make(0);

  /* atomization */
  specific_atomization(s);

  /* apply ICM and CSE here */
  /* apply_icm_cse_on_statement(s); */

  /* check consistency after optimizations */
  pips_assert("consistency checking after optimizations",
	      statement_consistent_p(s));
  
  close_is_nested();
} 




