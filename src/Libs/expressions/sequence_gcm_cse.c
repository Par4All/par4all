/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
   Revision 1.7  1999/05/25 13:14:50  zory
   fixes for new atomize_as_required.

   Revision 1.6  1999/05/12 14:46:15  zory
   basic_of_expression replace by please_give_me_a_basic_for_an_expression

   Revision 1.5  1999/05/12 12:24:39  zory
   level of unknow entities changed !

   Revision 1.4  1999/01/08 17:29:45  zory
   level_atomization done

   Revision 1.3  1999/01/04 16:56:32  zory
   atomize_level in progress ...

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

#define max(a,b) (((a)>(b))?(a):(b))

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
  basic b = please_give_me_a_basic_for_an_expression(e);

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

/******************************************* ATOMIZATION WITH LEVELS */

GENERIC_LOCAL_FUNCTION(has_level, entity_int);
DEFINE_LOCAL_STACK(current_statement, statement)

static int depth = 0;

/* 
   A shortcut to create variables for current module.
   The basic is consummed. It must be a fresh one.
*/
entity 
new_variable(entity module, basic b)
{
    return make_new_scalar_variable(module, b);
}

/* 
   Expression is either a call or a reference without args/indices. May be
   scalar reference or constant value or symbolic parameter
*/
static bool
assert_expression_is_scalar(expression e) {
  	
  syntax s = expression_syntax(e);

  if ( syntax_reference_p(s) && /* scalar reference */
       reference_indices(syntax_reference(s))==NIL )
    return TRUE;
  if ( expression_constant_p(e) ) /* constant */
    return TRUE;
  if (syntax_call_p(s) && /* intrinsic parameter */
      symbolic_constant_entity_p(call_function(syntax_call(s)))) 
    return TRUE;
  
  pips_debug(2, " invalid expression for level atomization : ");
  ifdebug(2) 
    print_expression(e);

  return FALSE;
}

/* 
   Assert that all expressions in the list are either a call or a
   reference without args/indices.  
*/
static bool
assert_expressions_are_scalar(list /* of expression */ le) {

  MAP(EXPRESSION, 
      e, 
      {
	if (!assert_expression_is_scalar(e))
	  return FALSE;
      }, le);
  
  return TRUE;

}

/* 
   Lets assume that e is a "scalar expression". i.e. expression is either
   a call or a reference, without args/indices.  

   The idea is the following one : this is a specific case of atomized
   expression where an assignement is something like :

   entity x entity* = entity x entity* 

*/

static entity
get_entity_from_scalar_expression (expression e) 
{
  syntax s = expression_syntax(e);

  pips_assert("expression is scalar", assert_expression_is_scalar(e));

  switch (syntax_tag(s)) { 
  case is_syntax_reference : 
    return reference_variable(syntax_reference(s));
  case is_syntax_call :
    return call_function(syntax_call(s));
  case is_syntax_range :
    pips_user_error(" expression is not scalar !");
  default: 
    pips_user_error(" unexpected type of syntax");
  }
}

/* 
   Insert the statement in a sequence/block of statement or create one.

   A hack from FC... to replace a statement by a list of statement
   containing the initial statement and all the new ones ... 

*/
static void 
insert_new_statement(statement s) {

  /* get current statement from the stack */
  statement   cs = current_statement_head();
  instruction  i = statement_instruction(cs);
  
  if (instruction_block_p(i)) {
    list /* of statement */ block = instruction_block(i);
    
    /* insert <s> before the last one in the block */
    block = gen_insert_before(s, 
			      STATEMENT(CAR(gen_last(block))), 
			      block);
  }
  else {
    list /* of statement */ block = NIL; 

    /* insert old current statement in the block */
    block = CONS(STATEMENT, make_stmt_of_instr(statement_instruction(cs)), NIL);

    /* insert new statement <s> just before */
    block = CONS(STATEMENT, s, block);

    /* replace the old instruction with the new one */
    statement_instruction(cs) = make_instruction_block(block);
  }
}


/*
  create an assignement to a new temporary variable 
*/
static expression
create_new_statement(list /* of expression */ args, 
		     entity operator) 
{
  entity new;
  statement stat;
  expression rhs;
  expression lhs;
  call c;
  basic b;

  pips_assert("only scalar arguments", assert_expressions_are_scalar(args));
  
  /* make rhs */
  c = make_call(operator, args);
  rhs = make_expression(make_syntax(is_syntax_call,c),normalized_undefined);
  normalize_all_expressions_of(rhs);

  /* make lhs */
  b = please_give_me_a_basic_for_an_expression(rhs); /* b is allocated */
  new = make_new_scalar_variable(get_current_module_entity(),b);
  lhs = entity_to_expression(new);

  /* make statement */
  stat = make_assign_statement(lhs, rhs);
  
  /* insert statement in the right place */
  insert_new_statement(stat);

  return lhs;
}

/* 
   create a new assignement and affect level to temporary variable.
*/
static expression
make_new_statement_for_level(entity operator,
			     int level, 
			     list /* of expressions */ le ){ 
  reference r;
  expression new_expr;
  entity new_entity;

  /* create a new statement and insert it in the code */
  new_expr = create_new_statement(le, operator);

  pips_assert("expression is reference", 
	      expression_reference_p(new_expr));

  r = syntax_reference(expression_syntax(new_expr));
  pips_assert(" reference is scalar variable", !reference_indices(r));

  new_entity = reference_variable(r);
  store_has_level(new_entity, level );

  return new_expr;

}

#define LEVEL_OF_UNKNOWN_ENTITY 0

/* 
   entity->int mapping...  

   if entity does not gave a known level, it has been initialized outside
   the loop nest. LEVEL_OF_UNKNOWN_ENTITY is attributed  
*/
static int 
compute_level_for_entity (entity e) {
  
  if ( bound_has_level_p(e) )
    return load_has_level(e);
  else 
    if (entity_constant_p(e) || symbolic_constant_entity_p(e)) 
      return 0;

  store_has_level(e, LEVEL_OF_UNKNOWN_ENTITY);

  pips_user_warning(" entity %s is of unknown level \n",
		    entity_local_name(e));

  return LEVEL_OF_UNKNOWN_ENTITY;
}

/* 
   All expressions in the list are assumed to be "scalar expressions"
   (call of reference without args) !
   
   Return an array with one list of expression for each level. 

   If list is empty, list_undefined_p is used instead of NIL because NIL
   elements in a gen_array_t are not counted as valid elements.  
*/
static gen_array_t /* list of expressions */
group_by_levels(list /* of expressions */ l) {
  
  list /* of expressions */ tmp = NIL;
  gen_array_t /* of list of expressions */ groups = gen_array_make(0);
  entity ent;
  int i, level, size, nitems;

  pips_assert("expressions are scalar",assert_expressions_are_scalar(l));

  pips_debug(3,"making groups for a list with %d expressions \n", gen_length(l));
  
  /* evaluate levels ... */
  MAP(EXPRESSION,
      e,
      {
	/* compute level for this entity */
	ent = get_entity_from_scalar_expression(e);
	level = compute_level_for_entity(ent);

	size  = gen_array_size(groups);
	nitems = gen_array_nitems(groups);
	pips_debug(7,"level : %d - size %d - nitems %d \n", level, size, nitems);
	
	if (level>=nitems) { /* new level */
	  /* create new list with this expression*/
	  tmp = CONS(EXPRESSION,e,NIL);
	  /* insert list for the given level */ 
	  gen_array_addto(groups, level, tmp);

	  /* initialize previous unused levels with undefined list */
	  /* Rem: NIL can not be used because (void *)NULL is not an item
             for gen_array_t */
	  for (i=nitems; i<level; i++) 
	    gen_array_addto(groups, i, list_undefined);
	}
	else {
	  /* get the list for the given level */
	  tmp = (list)gen_array_item(groups, level);
	  
	  if (list_undefined_p(tmp))
	    tmp = CONS(EXPRESSION,e, NIL); /* create new list */
	  else 
	    tmp = CONS(EXPRESSION,e,tmp); /* insert in old list */

	  /* update array */
	  gen_array_addto(groups, level, tmp);
	}
	
      }, l);

  return groups;
}

/* 
   All expressions in the list are assumed to be "scalar expressions"
   (call of reference without args) ! 

   Try to make new temporary assignements with scalar expressions of the
   same level.
   
*/
static list /* of expression */ 
atomize_AC_operator(list /* of expression */ exps, 
		    entity operator ) {
  int i;
  list /* of expression */ result = NIL;
  list /* of expression */ le = NIL;
  list /* of expression */ lnext = NIL;
  int size, levels;
  gen_array_t /* of list of expressions */ groups; 
  expression new_expr; 



  /* empty list of expressions (for instance a symbolic parameter) */
  if (ENDP(exps)) 
    return exps; /* not atomized */

  pips_assert("expressions are scalar",
	      assert_expressions_are_scalar(exps));

  /* group expressions according to their level in the loop nest */
  groups = group_by_levels(exps);
  levels = gen_array_nitems(groups);

  pips_debug(6, " %d different groups have been made \n", levels );

  for (i = 0; i<(levels-1) ; i++) { /* for all groups except the last one */

    /* group i : list of scalar expressions with level =  i */
    le = (list)gen_array_item(groups, i);
    le = list_undefined_p(le)?NIL:le;
    size = gen_length(le);
    
    /* group i+1 : next level */
    lnext = (list)gen_array_item(groups,i+1);
    
    if ( size >= 2 ) {
      pips_debug(3,"making new temporary statement of level %d ...\n", i);
      
      /* make a new statement for this level */
      new_expr = make_new_statement_for_level(operator,i,le);
      
      /* insert the new expression in next level list ... */
      lnext = list_undefined_p(lnext)?NIL:lnext;
      lnext = CONS(EXPRESSION, new_expr, lnext);
    }
    else { /* not enough elements to create new statement */ 

      if (le) {
	/* insert this scalar expression in the next level */
	lnext = list_undefined_p(lnext)?NIL:lnext;
	lnext = CONS(EXPRESSION, EXPRESSION(CAR(le)) , lnext);     
      }
    }
    /* update groups */
    gen_array_addto(groups, i+1, lnext);
  }
  
  /* Last group : expressions with highest level */
  MAP(EXPRESSION, 
      e, 
      {
	result = CONS(EXPRESSION, e, result);
      },
      gen_array_item(groups, levels-1));

  return result;
}

/* List of Associatif commutatif n-ary operators */
char * table_of_AC_operators[] = 
{ EOLE_SUM_OPERATOR_NAME, 
  EOLE_PROD_OPERATOR_NAME,
  MIN_OPERATOR_NAME,
  MAX_OPERATOR_NAME,
  NULL
};

static bool 
Is_Associatif_Commutatif(entity e)
{
  string local_name = entity_local_name(e);
  int i = 0;

  while (table_of_AC_operators[i])
    {
      if (same_string_p(local_name,table_of_AC_operators[i])){
	pips_debug(3," %s is Associatif Commutatif \n", local_name); 
	return TRUE; 
      }
      i++;
    }
  pips_debug(3," %s is NOT Associatif Commutatif \n", local_name); 
  return FALSE;
}

/* 
   If the list only contains "scalar expressions", the level is given by
   the maximum level of scalar expressions (i.e entities). Otherwise, the
   operation applied to this list of expressions is assumed to be of the
   current level (i.e. can't be moved upward) 
*/
static int 
compute_level_from_list(list /* of expression */ l) {

  int level = 0; 
  entity ent; 

  if (assert_expressions_are_scalar(l))
    {
      MAP (EXPRESSION,
	   e, 
	   { 
	     ent = get_entity_from_scalar_expression(e);
	     level = max(level, compute_level_for_entity(ent));
	     
	   }, l);
      return level; 
    }
  else 
    return depth;
}

/* 
   apply level atomization to rhs assignement expression. Return the level
   of the new expression.
*/
static int 
level_atomize_rhs_expression(expression e)
{
  syntax s = expression_syntax(e);

  switch (syntax_tag(s)) {
  case is_syntax_reference :{
    reference r = syntax_reference(s);
    list /* of expressions */ indices = reference_indices(r);
    entity var = reference_variable(r);

    if (indices) /* array */
      return compute_level_from_list(indices);
    else /* scalar */
      return compute_level_for_entity(var);
  }
  case is_syntax_call : {

    call c = syntax_call(s);
    list /* of expressions */ args = call_arguments(c);
    entity function = call_function(c);
  
    /* associatif-commutatif operator */
    if (Is_Associatif_Commutatif(function))
      {
	call_arguments(c) = atomize_AC_operator(args, function);
      }
    
    return compute_level_from_list(call_arguments(c));
  }
  default : /* range included  ... */
    pips_error("atomize_rhs_expression", "unexpected syntax type");
    return 0;
  }
}

/* 
   affect level information to lhs assignement expression. (expression is
   assumed to be a reference !).
*/
static void
affect_level_to_lhs(expression lhs, 
		    int level) { 
  reference r; 
  list /* of expressions */ args = NIL; 
  entity var;

  pips_assert("lhs is a reference ", expression_reference_p(lhs)); 

  r = syntax_reference(expression_syntax(lhs));
  args = reference_indices(r);
  var = reference_variable(r);

  if (args) /* lhs is array reference */
    {
      /* level for array reference must be recomputed for each occurence */
      /* nothing to do */
    }
  else   /* lhs is scalar reference */
    store_has_level(var, level);
}


static bool
call_flt_level (call c) 
{ 
  list /* of expressions */ args = call_arguments(c);
  entity function = call_function(c);

  if (depth<1) /* this call is not in a loop nest */
    return FALSE; /* level_atomization is useless... */

  /* constant expression -> nop */
  if ( entity_constant_p(function) )
    return FALSE;
  
  /* assignment */
  if (same_string_p(entity_local_name(function),ASSIGN_OPERATOR_NAME)) {
    
    expression lhs = EXPRESSION(CAR(args));
    expression rhs = EXPRESSION(CAR(CDR(args)));
    int level = 0;
    
    pips_debug(4," initial expression is :\n");
    ifdebug(4) print_expression(rhs);

    /* atomize rhs */
    level = level_atomize_rhs_expression(rhs);

    pips_debug(4,"atomized expression is :\n");
    ifdebug(4) print_expression(rhs);
    pips_debug(4,"level for rhs is %d \n\n", level);
    
    /* affect level to rhs */
    affect_level_to_lhs(lhs, level);

    return FALSE;  /* stop top-down gen_recursion */
  }
  else 
    {
      /* All "interesting" expressions which were NOT assignements should
	 have been fully atomized - e.g if(a+b+c) -> i0 = a+b+c if(i0)
	 ... */
      /* Thus, nothing is made for all other cases. e.g. subroutine,
         function calls and intrinsics ...  */
      pips_debug(2," This call won't be level_atomized \n");
      pips_debug(2," call is : %s with %d args \n", 
		 entity_local_name(function),
		 gen_length(args));
	
      return TRUE; /* continue top-down gen_recurse */
    }
}

/* 
   Save informations on how loops are nested.
   DO I DO J DO K  -> level of I is 1, level of J is 2 ...
*/
static bool
loop_flt_level (loop l) {

  entity index = loop_index(l);

  store_has_level(index, ++depth);

  return TRUE; /* keep on. */
}


/* 
   In order to improve the efficiency of Invariant code motion techniques,
   arithmetic expressions are atomized according to "level in the loop
   nest" informations.

    For instance : Do I Do J A(I,J) = B(I) + C(J) + D is atomized as
    follow : F0 = B(I) + D A(I,J) = F0 + C(J) ... F0 can be computed
    outside the J loop.

    A classic atomization MUST have been applied before this one. All
    expressions are assumed to contain only one operation (an entity) and
    a list of "scalar operands (call or reference with empty list of args)

*/

static void 
atomization_with_levels (statement s)
{

  pips_debug(2," Level_Atomization running... \n");

  /* initialize data for the current statement */
  depth = 0;
  init_has_level();
  make_current_statement_stack();

  /* gen_recurse on the statement - top down action is applied. */
  gen_multi_recurse
    (s,
     loop_domain, loop_flt_level, gen_true,          /* LOOP */
     call_domain, call_flt_level, gen_true,          /* CALL */
     statement_domain, current_statement_filter, 
                       current_statement_rewrite,    /* STATEMENT */
     NULL);

  /* clean */
  free_current_statement_stack();
  close_has_level();
  
  ifdebug(3) {
    pips_debug(3," statement after level_atomization \n");
    print_statement(s);
  }

}

/*********************************************** CLASSIC ATOMIZATION **********/


/*
  decide wether to atomize expression or not ! 
*/
static bool 
simple_expression_decision(e)
expression e;
{
  syntax s = expression_syntax(e);

  /*  don't atomize A(I)
   */
  if (syntax_reference_p(s)) {
    return(!entity_scalar_p(reference_variable(syntax_reference(s))));
  }
  
  /*  don't atomize neither A(1) 
   */
  if (expression_constant_p(e)) 
    return(FALSE);

  /* don't atomize A(N) when N is a symbolic parameter 
   */
  if (syntax_call_p(s)) {
    return(!symbolic_constant_entity_p(call_function(syntax_call(s))));
  }

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
  
  ifdebug(4) 
    print_expression(e);

  /* do not atomize rhs and lhs in an assignement */
  if (ENTITY_ASSIGN_P(f)) 
    return(FALSE);

  if (value_tag(entity_initial(f)) == is_value_intrinsic ||
      !syntax_reference_p(s)) 
    return(simple_expression_decision(e));

  
  /* the default is *not* to atomize.
   * should check that the reference is not used as a vector
   * and is not modified?
   */ 
  return FALSE; 
}

static bool 
range_atomization (range r, expression e)
{
  return(simple_expression_decision(e));
}

static void 
classic_atomization (statement s) {

  pips_debug(2," classic ATOMIZATION on progress... \n");
  
  /* classic atomization */  
  atomize_as_required(s,
		      ref_atomization,
		      call_atomization,
		      gen_true, /* test */
		      range_atomization, /* range */
		      gen_true, /* whileloop */
		      new_variable);
  
  ifdebug(3) {
    pips_debug(3," statement after classic atomization  : \n");
    print_statement(s);
  }


}

/*************************************************************** ICM CSE *****/

static void 
specific_atomization (statement s) 
{

  /* classic atomization */
  classic_atomization(s);
 
  /* atomize expression with respect to level of variables in the loop nest */
  atomization_with_levels(s);

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

      /* hack to avoid problem with RETURN statements ....*/
      if (return_statement_p(s))
	return FALSE;

      ifdebug(2) {
	pips_debug(2," INITIAL NESTED SEQUENCE !\n");
	print_statement(s);
      }
      /* apply atomization */
      specific_atomization(s);

      ifdebug(3) {
	pips_debug(3," ATOMIZED NESTED SEQUENCE !\n");
	print_statement(s);
      }

      /* apply icm - cse */
      /* new = apply_icm_cse(old);*/
      
      /* finally update statement_instruction(s)*/
      /* statement_instruction(s) = new ; */
      
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

  /* apply ICM and CSE here */
  apply_icm_cse_on_statement(s);

  /* check consistency after optimizations */
  pips_assert("consistency checking after optimizations",
	      statement_consistent_p(s));
  
  close_is_nested();
} 




