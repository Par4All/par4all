/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
   Revision 1.11  1999/05/27 16:51:01  ancourt
   also ICM direct expressions such as conditions and bounds.

   Revision 1.10  1999/05/27 16:09:44  ancourt
   code cleaned up...

   Revision 1.9  1999/05/27 14:47:26  ancourt
   working combined association/atomization for ICM.

   Revision 1.8  1999/05/26 14:25:42  coelho
   emprunt...

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

/* option: whether to push the procedure statement.
 */
#define PUSH_BODY

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

#include "effects-generic.h"
#include "effects-simple.h"

#include "transformations.h"

#include "eole_private.h"

extern char * itoa(int); /* in newgen */

/***************************************** COMMUTATIVE ASSOCIATIVE OPERATORS */

/* List of Associatif commutatif n-ary operators */
static char * table_of_AC_operators[] = 
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

/*********************************************************** ICM ASSOCIATION */

/* assumes:
   - cumulated effects (as a dependence abstraction).
   - proper effects (?)
 */

/* current nesting of code, bottom-up order, to determine level.
 */
static list nesting = NIL;

/* statement stack to current statement.
 */
DEFINE_LOCAL_STACK(current_statement, statement)

/* statements to be inserted as a sequence.
 */
GENERIC_LOCAL_FUNCTION(inserted, persistant_statement_to_statement)

/* keep track of nesting.
 */
static void push_nesting(statement s)
{
  nesting = CONS(STATEMENT, s, nesting);
}

static void pop_nesting(statement s)
{
  list old = nesting;
  pips_assert("same ", nesting && (s == STATEMENT(CAR(nesting))));
  nesting = CDR(nesting);
  CDR(old) = NIL;
  gen_free_list(old);
}

/* there is a side effect if there is a W effect in the expression.
 */
static bool side_effects_p(expression e)
{
  effects ef = load_expr_prw_effects(e);
  list /* of effect */ le = effects_effects(ef);
  MAP(EFFECT, ef, if (effect_write_p(ef)) return TRUE, le);
  return FALSE;
}

/* some effect in les interfere with var.
 */
static bool interference_on(entity var, list /* of effect */ les)
{
  MAP(EFFECT, ef, 
      if (effect_write_p(ef) &&
	  entity_conflict_p(var, reference_variable(effect_reference(ef))))
        return TRUE,
      les);
  return FALSE;
}

/* whether sg with effects le can be moved up to s.
 */
static bool moveable_to(list /* of effects */ le, statement s)
{
  list les = load_cumulated_rw_effects_list(s);
  MAP(EFFECT, ef,
      if (interference_on(reference_variable(effect_reference(ef)), les))
        return FALSE,
      le);
  return TRUE;
}

/* return the level of this expression, using the current nesting list.
 * 0: before any statement!
 * n: outside nth loop.
 * and so on.
 */
static int level_of(list /* of effects */ le)
{
  list /* of statement */ up_nesting = gen_nreverse(gen_copy_seq(nesting));
  int level = 0;
  MAP(STATEMENT, s,
      if (moveable_to(le, s))
      {
	gen_free_list(up_nesting);
	return level;
      }
      else
        level++,
      up_nesting);
  
  gen_free_list(up_nesting);
  return level;
}

/* the level can be queried for a sub expression. */
static int expr_level_of(expression e)
{
  list le;
  if (!bound_expr_prw_effects_p(e)) return -1; /* assigns... */
  le = effects_effects(load_expr_prw_effects(e));
  return level_of(le);
}

/* or for a statement. */
static int stat_level_of(statement s)
{
  list le = load_proper_rw_effects_list(s);
  return level_of(le);
}

/* returns the statement of the specified level
   should returns current_statement_head() to avoid ICM directly.
 */
static statement statement_of_level(int level)
{
  int n = gen_length(nesting)-1-level;
  
  if (n>=0)
    return STATEMENT(gen_nth(n, nesting));
  else
    return current_statement_head();
}

static int current_level(void)
{
  return gen_length(nesting);
}

static bool currently_nested_p(void)
{
#if defined(PUSH_BODY)
  return current_level()>1;
#else
  return current_level()>0;
#endif
}

static void insert_before_statement(statement news, statement s)
{
  if (!bound_inserted_p(s))
    {
      store_inserted(s, make_block_statement(CONS(STATEMENT, news, NIL)));
    }
  else
    {
      statement sb = load_inserted(s);
      instruction i = statement_instruction(sb);
      pips_assert("inserted in block", statement_block_p(sb));

      /* statements are stored in reverse order...
	 this will have to be fixed latter on.
       */
      instruction_block(i) = CONS(STATEMENT, news, instruction_block(i));
    }
}

/* atomizable if some computation.
 */
static bool atomizable_sub_expression_p(expression e)
{
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_range:
    case is_syntax_reference:
      return FALSE;
    case is_syntax_call:
      return !entity_constant_p(call_function(syntax_call(s)));
    default:
      pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
      return FALSE;
    }
}

extern entity hpfc_new_variable(entity, basic);

static gen_array_t /* of list of expressions */
group_expr_by_level(int nlevels, list le)
{
  gen_array_t result = gen_array_make(nlevels+1);
  int i, lastlevel;
  bool first_alone;

  /* initialize chunks. */
  for (i=0; i<=nlevels; i++)
    gen_array_addto(result, i, list_undefined);
  
  /* put expressions in chunks. */
  MAP(EXPRESSION, e,
  {
    int elevel = expr_level_of(e);
    list eatlevel;
    pips_assert("coherent level", elevel>=0 && elevel<=nlevels);
    
    if (side_effects_p(e))
      elevel = nlevels;
    
    eatlevel = (list) gen_array_item(result, elevel);
    if (eatlevel == list_undefined)
      eatlevel = CONS(EXPRESSION, e, NIL);
    else
      eatlevel = CONS(EXPRESSION, e, eatlevel);
    gen_array_addto(result, elevel, eatlevel);
  },
      le);

  for (i=0; i<=nlevels; i++)
    if (((list)gen_array_item(result, i)) != list_undefined)
      lastlevel = i;

  /* fix expressions by useful levels, with some operations.
   */
  first_alone = TRUE;
  for (i=0; i<nlevels && first_alone; i++)
    {
      list lei = (list) gen_array_item(result, i);
      if (lei != list_undefined)
	{
	  int lenlei = gen_length(lei);
	  if (lenlei == 1)
	    {
	      list next = (list) gen_array_item(result, i+1);
	      if (next == list_undefined)
		next = lei;
	      else
		next = gen_nconc(next, lei);
	      gen_array_addto(result, i+1, next);
	      gen_array_addto(result, i, list_undefined);
	    }
	  else
	    first_alone = FALSE;
	}
    }

  return result;
}

/* atomize sub expressions with 
   - lower level
   - not simple (references or constants)
   - no side effects.
*/
static void do_atomize_if_different_level(expression e, int level)
{
  int elevel = expr_level_of(e);

  /* fprintf(stderr, "ATOM %d/%d\n", elevel, level); print_expression(e); */

  if (elevel!=-1 && 
      elevel<level &&
      atomizable_sub_expression_p(e) &&
      !side_effects_p(e))
    {
      statement atom = atomize_this_expression(hpfc_new_variable, e);
      if (atom)
	insert_before_statement(atom, statement_of_level(elevel));
    }
}

static void atomize_call(call c, int level)
{
  list /* of expression */ args;
  int lenargs;
  
  args = call_arguments(c);
  lenargs = gen_length(args);

  if (lenargs>=2)
    {
      MAP(EXPRESSION, sube, do_atomize_if_different_level(sube, level), args);
    }
}

static void atomize_or_associate_for_level(expression e, int level)
{
  syntax syn;
  call c;
  entity func;
  list /* of expression */ args;
  int lenargs, exprlevel;

  /* some depth, otherwise no ICM needed!
   * should be fixed if root statement is pushed?
   */
  if (!currently_nested_p()) return;
  
  /* only a scalar expression can be atomized.
   */



  /* only do something with calls.
   */
  syn = expression_syntax(e);
  if (!syntax_call_p(syn))
    return;

  /* something to icm 
   */
  c = syntax_call(syn);
  func = call_function(c);
  args = call_arguments(c);
  lenargs = gen_length(args);
  exprlevel = expr_level_of(e);

  if (Is_Associatif_Commutatif(func) && lenargs>2)
    {
      /* reassociation + atomization maybe needed.
	 code taken from JZ.
       */
      int i, nlevels = current_level();
      gen_array_t groups = group_expr_by_level(nlevels, args);
      list lenl;

      /* note: the last level of an expression MUST NOT be moved!
	 indeed, there may be a side effects in another part of the expr.
       */
      for (i=0; i<nlevels; i++)
	{
	  list lei = (list) gen_array_item(groups, i);
	  if (lei!=list_undefined)
	    {
	      int j;
	      bool satom_inserted = FALSE;
	      syntax satom = make_syntax(is_syntax_call, make_call(func, lei));
	      expression eatom = expression_undefined;
	      statement atom;

	      /* insert expression in upward expression, eventually in e!
	       */
	      for (j=i+1; j<=nlevels && !satom_inserted; j++)
		{
		  list lej = (list) gen_array_item(groups, j);
		  if (lej!=list_undefined)
		    {
		      eatom = make_expression(satom, normalized_undefined);
		      lej = CONS(EXPRESSION, eatom, lej);
		      gen_array_addto(groups, j, lej);
		      satom_inserted = TRUE;
		    }
		}
	      if (!satom_inserted)
		{
		  expression_syntax(e) = satom;
		  eatom = e;
		}
	      
	      atom = atomize_this_expression(hpfc_new_variable, eatom);
	      insert_before_statement(atom, statement_of_level(i));
	    }
	}

      /* fix last level if necessary.
       */
      lenl = (list) gen_array_item(groups, nlevels);
      if (lenl != list_undefined)
	  call_arguments(c) = lenl;
    }
  else
    {
      atomize_call(c, level);
    }
}

/* maybe I could consider moving the call as a whole?
 */
static void atomize_instruction(instruction i)
{
  if (!currently_nested_p()) return;
  if (!instruction_call_p(i)) return;
  /* stat_level_of(current_statement_head())); */
  atomize_call(instruction_call(i), current_level());
}

static void atomize_test(test t)
{
  if (!currently_nested_p()) return;
  do_atomize_if_different_level(test_condition(t), current_level());
}

static void atomize_whileloop(whileloop w)
{
  if (!currently_nested_p()) return;
  do_atomize_if_different_level(whileloop_condition(w), current_level());
}

static void atomize_or_associate(expression e)
{
  atomize_or_associate_for_level(e, expr_level_of(e));
}

static bool loop_flt(loop l)
{
  statement sofl = current_statement_head();
  pips_assert("statement of loop", 
	      instruction_loop(statement_instruction(sofl))==l)
  push_nesting(sofl);
  return TRUE;
}

static void loop_rwt(loop l)
{
  range bounds;
  int level;
  statement sofl = current_statement_head();
  pop_nesting(sofl);

  if (!currently_nested_p()) return;
  bounds = loop_range(l);
  level = current_level();
  do_atomize_if_different_level(range_lower(bounds), level);
  do_atomize_if_different_level(range_upper(bounds), level);
  do_atomize_if_different_level(range_increment(bounds), level);
}

/* insert in front if some inserted.
 */
static void insert_rwt(statement s)
{
  if (bound_inserted_p(s))
    {
      statement sblock = load_inserted(s);
      instruction i = statement_instruction(sblock);
      sequence seq;
      pips_assert("it is a sequence", instruction_sequence_p(i));

      /* reverse list */
      seq = instruction_sequence(i);
      sequence_statements(seq) = gen_nreverse(sequence_statements(seq));

      /* insert */
      sequence_statements(seq) = 
	gen_append(sequence_statements(seq),
		   CONS(STATEMENT,
			instruction_to_statement(statement_instruction(s)),
			NIL));

      statement_instruction(s) = i;
    }
}

/* perform ICM and association on operators.
   this is kind of an atomization.
 */
void 
perform_icm_association(
    string name, /* of the module */
    statement s  /* of the module */)
{
  pips_assert("clean static structures on entry",
	      (get_current_statement_stack() == stack_undefined) &&
	      inserted_undefined_p() &&
	      (nesting==NIL));

  /* set full (expr and statements) PROPER EFFECTS
   */
  full_simple_proper_effects(name, s);

  /* GET CUMULATED EFFECTS
   */
  set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, name, TRUE));
  
  init_inserted();
  make_current_statement_stack();

#if defined(PUSH_BODY)
  push_nesting(s);
#endif

  /* ATOMIZE and REASSOCIATE by level.
   */
  gen_multi_recurse(s,
      statement_domain, current_statement_filter, current_statement_rewrite,
      instruction_domain, gen_true, atomize_instruction,
      test_domain, gen_true, atomize_test,
      loop_domain, loop_flt, loop_rwt,
      whileloop_domain, gen_true, atomize_whileloop,
      /* could also push while loops... */
      expression_domain, gen_true, atomize_or_associate,
      /* do not atomize index computations at the time... */     
      reference_domain, gen_false, gen_null,
		    NULL);

  /* insert moved code. */
  gen_multi_recurse(s, statement_domain, gen_true, insert_rwt, NULL);

#if defined(PUSH_BODY)
  pop_nesting(s);
#endif

  pips_assert("clean static structure on exit",
	      (nesting==NIL) &&
	      (current_statement_size()==0));

  free_current_statement_stack();
  close_inserted();

  reset_cumulated_rw_effects();

  close_expr_prw_effects();  /* no memory leaks? */
  close_proper_rw_effects();
}
