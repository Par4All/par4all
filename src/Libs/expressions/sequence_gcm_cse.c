/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
   Revision 1.16  2000/07/03 12:51:16  coelho
   typo fixed.

   Revision 1.15  2000/07/03 12:26:06  phamdinh
   Pour changer repertoire a travailer

   Revision 1.14  2000/06/28 14:16:17  coelho
   CSE not inverted?

   Revision 1.13  1999/07/15 20:35:46  coelho
   temporary working version of AC-CSE...

   Revision 1.12  1999/05/28 09:15:37  coelho
   cleaner and more comments.

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

/* option: 
   - whether to push the procedure statement (default no).
   - whether to do ICM or not directly (default yes).
 */
/* #define PUSH_BODY */
/* #define NO_ICM */

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

extern char * itoa(int); /* somewhere in newgen */

/******************************************************************* FLATTEN */

static void flatten_sequence(sequence sq)
{
  list /* of statement */ nsl = NIL;
  MAP(STATEMENT, s,
  {
    instruction i = statement_instruction(s);
    if (instruction_sequence_p(i))
      {
	sequence included = instruction_sequence(i);
	nsl = gen_nconc(nsl, sequence_statements(included));
	sequence_statements(included) = NIL;
	free_statement(s);
      }
    else
      {
	nsl = gen_nconc(nsl, CONS(STATEMENT, s, NIL));
      }
  },
      sequence_statements(sq));

  gen_free_list(sequence_statements(sq));
  sequence_statements(sq) = nsl;
}

void flatten_sequences(statement s)
{
  gen_recurse(s, sequence_domain, gen_true, flatten_sequence);
}

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
   - proper effects for statements AND sub-expressions...
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
  /* Just for testing nesting */
  fprintf(stderr,"Level before push: %d\n", gen_length(nesting));
  print_statement(s);

  nesting = CONS(STATEMENT, s, nesting);
}

static void pop_nesting(statement s)
{
  /* Just for testing nesting */
  //fprintf(stderr,"Level before pop: %d\n", gen_length(nesting));
  //print_statement(s);

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
/*
static int stat_level_of(statement s)
{
  list le = load_proper_rw_effects_list(s);
  return level_of(le);
}
*/

static int current_level(void)
{
  return gen_length(nesting);
}

/* returns the statement of the specified level
   should returns current_statement_head() to avoid ICM directly.
 */
static statement statement_of_level(int level)
{
#if !defined(NO_ICM)
  int n = current_level()-1-level;

  if (n>=0)
    return STATEMENT(gen_nth(n, nesting));
  else
#endif
    return current_statement_head();
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
	 this will have to be fixed latter on. see #1#.
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
      {
	entity called = call_function(syntax_call(s));
	/* missing cases? user functions? I/O? */
	return !entity_constant_p(called) && !ENTITY_IMPLIEDDO_P(called);
      }
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

  fprintf(stderr, "ATOM %d/%d\n", elevel, level); print_expression(e);

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

  if (Is_Associatif_Commutatif(func) && lenargs>=2)
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

/* don't go into I/O calls... */
static bool icm_atom_call_flt(call c)
{
  entity called = call_function(c);
  return !(io_intrinsic_p(called) || ENTITY_IMPLIEDDO_P(called));
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
  fprintf(stderr,"\n---Atomize or associate---");
  print_expression(e);

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

  /* deal with loop bound expressions
   */
  if (!currently_nested_p()) return;
  bounds = loop_range(l);
  level = current_level();
  do_atomize_if_different_level(range_lower(bounds), level);
  do_atomize_if_different_level(range_upper(bounds), level);
  do_atomize_if_different_level(range_increment(bounds), level);
}

static bool insert_reverse_order = TRUE;

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

      /* reverse list of inserted statements (#1#) */
      seq = instruction_sequence(i);
      if (insert_reverse_order)
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
   many side effects: modifies the code, uses simple effects
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
  /*
  simple_cumulated_effects(name, s);
  set_cumulated_rw_effects(get_rw_effects());
  */

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
      loop_domain, loop_flt, loop_rwt,
      test_domain, gen_true, atomize_test,
      whileloop_domain, gen_true, atomize_whileloop,
      /* could also push while loops... */
      expression_domain, gen_true, atomize_or_associate,
      /* do not atomize index computations at the time... */     
      reference_domain, gen_false, gen_null,
      call_domain, icm_atom_call_flt, gen_null, /* skip IO calls */
		    NULL);

  /* insert moved code in statement. */
  insert_reverse_order = TRUE;
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

  /* some clean up */
  flatten_sequences(s);
}

/********************************************************** EXTRACT ENTITIES */

static list seen = NIL;

static void add_ref_ent(reference r)
{ seen = gen_once(reference_variable(r), seen); }

static void add_call_ent(call c)
{ seen = gen_once(call_function(c), seen); }

static list get_all_entities(expression e)
{
  list result;
  seen = NIL;

  gen_multi_recurse(e,
		    reference_domain, gen_true, add_ref_ent,
		    call_domain, gen_true, add_call_ent,
		    NULL);

  result = seen;
  seen = NIL;
  return result;
}

/******************************************************************** AC-CSE */
/* performs 
   Associative-Commutative Common Subexpression Elimination
   on sequences.

   reimplements an atomization once more?
   should not atomize I/O...
*/

typedef struct
{
  entity scalar; /* the entity which holds the value? or partial? */
  entity operator; /* NULL for references */
  list /* of entity */ depends; /* W effects on these invalidates the scalar */
  statement container; /* statement in which it appears (for atomization) */
  expression contents; /* call or reference... could be a syntax? */
  list available_contents; /* part of which is available */
}
  available_scalar_t, * available_scalar_pt;

static void dump_aspt(available_scalar_pt aspt)
{
  syntax s = expression_syntax(aspt->contents);
  fprintf(stderr, 
	  "DUMP ASPT\n"
	  "%s [%s] len=%d, avail=%d\n",
	  entity_name(aspt->scalar), 
	  aspt->operator? entity_name(aspt->operator): "NOP",
	  syntax_call_p(s)? gen_length(call_arguments(syntax_call(s))): -1,
	  gen_length(aspt->available_contents));
}

static list current_availables = NIL;
static statement current_statement = statement_undefined;

/* whether to get in here, whether to atomize... */
static bool expr_cse_flt(expression e)
{
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      return !IO_CALL_P(syntax_call(s));
    case is_syntax_reference:
      return entity_scalar_p(reference_variable(syntax_reference(s)));
    case is_syntax_range:
    default:
      return FALSE;
    }
}

#define NO_SIMILARITY (0)
#define MAX_SIMILARITY (1000)

/* return the entity stored by this function.
   should be a scalar reference or a constant...
   or a call to an inverse operator.
 */
static entity entity_of_expression(expression e, bool * inverted, entity inv)
{
  syntax s = expression_syntax(e);
  *inverted = FALSE;
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      {
	call c = syntax_call(s);
	if (call_function(c)==inv)
	  {
	    *inverted = TRUE;
	    return entity_of_expression(EXPRESSION(CAR(call_arguments(c))),
					inverted, NULL);
	  }
	else
	  return call_function(c);
      }
    case is_syntax_reference:
      {
	reference r = syntax_reference(s);
	if (reference_indices(r))
	  return NULL;
	else
	  return reference_variable(r);
      }
    case is_syntax_range:
    default:
    }
  return NULL;
}

static bool expression_in_list_p(expression e, list seen)
{
  MAP(EXPRESSION, f, if (e==f) return TRUE, seen);
  return FALSE;
}

static expression 
find_equal_expression_not_in_list(expression e, list avails, list seen)
{
  MAP(EXPRESSION, f,
      if (expression_equal_p(e, f) && !expression_in_list_p(f, seen))
        return f,
      avails);
  return NULL;
}

static list /* of expression */ 
common_expressions(list args, list avails)
{
  list already_seen = NIL;

  MAP(EXPRESSION, e, 
  {
    expression n = find_equal_expression_not_in_list(e, avails, already_seen);
    if (n) already_seen = CONS(EXPRESSION, n, already_seen);
  },
      args);

  return already_seen;
}

static int similarity(expression e, available_scalar_pt aspt)
{
  syntax s = expression_syntax(e), sa = expression_syntax(aspt->contents);

  /*
  fprintf(stderr, "similarity on %s\n", entity_name(aspt->scalar));
  print_expression(e);
  dump_aspt(aspt);
  */

  if (syntax_tag(s)!=syntax_tag(sa)) return NO_SIMILARITY;
  
  if (syntax_reference_p(s))
    {
      reference r = syntax_reference(s), ra = syntax_reference(sa);
      if (reference_equal_p(r, ra)) return MAX_SIMILARITY;
    }

  if (syntax_call_p(s))
    {
      call c = syntax_call(s), ca = syntax_call(sa);
      entity cf = call_function(c);
      if (cf!=call_function(ca)) return NO_SIMILARITY;

      /* same function...
       */
      if (Is_Associatif_Commutatif(cf))
	{
	  /* similarity is the number of args in common.
	     inversion is not tested at the time.
	   */
	  list com = common_expressions(call_arguments(c),
					aspt->available_contents);
	  int n = gen_length(com);
	  gen_free_list(com);
	  if (n<=1) return NO_SIMILARITY;
	  return (n == gen_length(call_arguments(ca)) &&
	          n == gen_length(call_arguments(c))) ? MAX_SIMILARITY: n;
	}
      else
	{
	  /* any call: must be equal
	   */
	  list l = call_arguments(c), la = call_arguments(ca);
	  pips_assert("same length", gen_length(l)==gen_length(la));
	  for (; l; l = CDR(l), la = CDR(la))
	    {
	      expression el = EXPRESSION(CAR(l)), ela = EXPRESSION(CAR(la));
	      if (!expression_equal_p(el, ela))
		return NO_SIMILARITY;
	    }
	  return MAX_SIMILARITY;
	}
    }

  return NO_SIMILARITY;
}

static available_scalar_pt 
best_similar_expression(expression e, int * best_quality)
{
  available_scalar_pt best = NULL;
  (*best_quality) = 0;

  MAPL(caspt,
  {
    available_scalar_pt aspt = (available_scalar_pt) STRING(CAR(caspt));
    int quality = similarity(e, aspt);
    if (quality==MAX_SIMILARITY) 
    {
	(*best_quality) = quality;
      return aspt;
    }
    if (quality>0 && (*best_quality)<quality)
      {
	best = aspt;
	(*best_quality) = quality;
      }
  },
       current_availables);

  return best;
}

static available_scalar_pt 
make_available_scalar(
    entity scalar,
    statement container,
    expression contents)
{
  syntax s = expression_syntax(contents);

  available_scalar_pt aspt = 
    (available_scalar_pt) malloc(sizeof(available_scalar_t));

  aspt->scalar = scalar;
  aspt->container = container;
  aspt->contents = contents;

  aspt->depends = get_all_entities(contents);

  if (syntax_call_p(s))
    {
      call c = syntax_call(s);
      aspt->operator = call_function(c);
      if (Is_Associatif_Commutatif(aspt->operator))
	aspt->available_contents = gen_copy_seq(call_arguments(c));
      else
	aspt->available_contents = NIL;
    }
  else
    {
      aspt->operator = NULL;
      aspt->available_contents = NIL;
    }

  return aspt;
}

static bool expression_eq_in_list_p(expression e, list l, expression *f)
{
  MAP(EXPRESSION, et,
      if (expression_equal_p(e, et))
      {
	*f = et;
	return TRUE;
      },
      l);
  return FALSE;
}

/* returns an allocated l1-l2 with expression_equal_p
   l2 included in l1.
 */
static list /* of expression */ list_diff(list l1, list l2)
{
  list diff = NIL, l2bis = gen_copy_seq(l2);
  expression found;

  MAP(EXPRESSION, e,
      if (expression_eq_in_list_p(e, l2bis, &found))
      {
	gen_remove(&l2bis, found);
      }
      else
      {
	diff = CONS(EXPRESSION, e, diff);
      },
      l1);

  pips_assert("list should be empty...", !l2bis);
  /* if (l2bis) gen_free_list(l2bis); */

  return diff;
}

static bool simple_reference_p(expression e)
{
  syntax s = expression_syntax(e);
  return syntax_reference_p(s) && !reference_indices(syntax_reference(s));
}

/* remove some inpropriate ones...
 */
static void clean_current_availables(void)
{
  
}

static void atom_cse_expression(expression e)
{
  /* statement head = current_statement_head(); */
  syntax s = expression_syntax(e);
  int quality;
  available_scalar_pt aspt;
  
  /* fprintf(stderr, "[atom_cse_expression]\n"); */

  if (syntax_call_p(s) && ENTITY_ASSIGN_P(call_function(syntax_call(s))))
    return;

  do { /* extract every possible common subexpression */
    aspt = best_similar_expression(e, &quality);
    if (aspt) /* some common expression found. */ {

      /*
      fprintf(stderr, "some similar expression found (%s: %d)\n", 
	      entity_name(aspt->scalar), quality);
      print_expression(aspt->contents);
      */

      switch (syntax_tag(s))
	{
	case is_syntax_reference:
	  syntax_reference(s) = make_reference(aspt->scalar, NIL);
	  return;
	case is_syntax_call:
	  {
	    if (quality==MAX_SIMILARITY)
	      {
		/* identicals, just make a reference to the scalar.
		   whatever the stuff stored inside.
		 */
		/* fprintf(stderr, "AC-CSE is equal...\n"); */

		syntax_tag(s) = is_syntax_reference;
		syntax_reference(s) = make_reference(aspt->scalar, NIL);
		return;
	      }
	    else /* partial common expression... */
	      {
		available_scalar_pt naspt;
		call c = syntax_call(s), ca;
		syntax sa = expression_syntax(aspt->contents);
		entity op = call_function(c), scalar;
		expression cse, cse2;
		statement scse;
		list /* of expression */ in_common, linit, lo1, lo2, old;

		pips_assert("AC operator", 
			 Is_Associatif_Commutatif(op) && op==aspt->operator);
		pips_assert("contents is a call", syntax_call_p(sa));
		ca = syntax_call(sa);

		linit = call_arguments(ca);

		/*
		   there is a common part to build,
		   and a CSE to substitute in both...
		*/
		in_common = common_expressions(call_arguments(c),
					       aspt->available_contents);

		if (gen_length(linit)==gen_length(in_common))
		{
		  /* fprintf(stderr, "AC-CSE is included...\n"); */
		  /* just substitute lo1, don't build a new aspt. */
		  lo1 = list_diff(call_arguments(c), in_common);
		  free_arguments(call_arguments(c));
		  call_arguments(c) = 
		    gen_nconc(lo1, 
			      CONS(EXPRESSION,
				   entity_to_expression(aspt->scalar), NIL));
		}
		else
		{
		  /* fprintf(stderr, "AC-CSE is shared...\n"); */

		lo1 = list_diff(call_arguments(c), in_common);
		lo2 = list_diff(linit, in_common);
		
		cse = call_to_expression(make_call(aspt->operator, in_common));
		scse = atomize_this_expression(hpfc_new_variable, cse);
		/* now cse is a reference to the newly created scalar. */
		pips_assert("a reference...",
			    syntax_reference_p(expression_syntax(cse)));
		scalar = reference_variable(syntax_reference
					    (expression_syntax(cse)));
		insert_before_statement(scse, aspt->container);

		/* don't visit it later. */
		gen_recurse_stop(scse);
		
		cse2 = copy_expression(cse);
		
		/* update both expressions... */
		old = call_arguments(c); /* in code */
		call_arguments(c) = CONS(EXPRESSION, cse, lo1);
		gen_free_list(old);

		old = call_arguments(ca);
		call_arguments(ca) = CONS(EXPRESSION, cse2, lo2);
		gen_free_list(old);

		/* updates... */
		aspt->depends = CONS(ENTITY, scalar, aspt->depends);
		old = aspt->available_contents; 
		aspt->available_contents = list_diff(old, in_common);
		gen_free_list(old);

		/* add the new scalar as an available CSE. */
		naspt = make_available_scalar(scalar,
					      aspt->container,
		   EXPRESSION(CAR(CDR(call_arguments(instruction_call
			      (statement_instruction(scse)))))));
		}
	      }
	    break;
	  }
	case is_syntax_range:
	default:
	  /* nothing */
	  return;
	}
    }
  } while(aspt);

  if (!simple_reference_p(e))
    {
      statement atom;

      /* fprintf(stderr, "atomizing..."); print_expression(e); */
      /* create a new atom... */
      atom = atomize_this_expression(hpfc_new_variable, e);
      insert_before_statement(atom, current_statement);
      /* don't visit it later, just in case... */
      gen_recurse_stop(atom);
      {
	entity scalar;
	call ic;
	syntax s = expression_syntax(e);
	instruction i = statement_instruction(atom);
	pips_assert("it is a reference", syntax_reference_p(s));
	pips_assert("instruction is an assign", 
		    instruction_call_p(i)) /* ??? */
		    
	scalar = reference_variable(syntax_reference(s));
	ic = instruction_call(i);
	
	/* if there are variants in the expression it cannot be moved,
	   and it is not available. Otherwise I should move it?
	 */
	aspt = make_available_scalar(scalar, 
				     current_statement,
				     EXPRESSION(CAR(CDR(call_arguments(ic)))));
	
	current_availables = CONS(STRING, (char*)aspt, current_availables);
      }
    }
}

static bool loop_stop(loop l)
{ gen_recurse_stop(loop_body(l)); return TRUE; }

static bool test_stop(test t)
{ gen_recurse_stop(test_true(t));
  gen_recurse_stop(test_false(t)); return TRUE; }

static bool while_stop(whileloop wl)
{ gen_recurse_stop(whileloop_body(wl)); return TRUE; }

static bool cse_atom_call_flt(call c)
{
  entity called = call_function(c);

  /* should avoid any W effect... */
  if (ENTITY_ASSIGN_P(called))
    gen_recurse_stop(EXPRESSION(CAR(call_arguments(c))));

  return !(io_intrinsic_p(called) || ENTITY_IMPLIEDDO_P(called));
}

/* side effects: use current_availables and current_statement
 */
static list /* of available_scalar_pt */
atomize_cse_this_statement_expressions(statement s, list availables)
{
  current_availables = availables;
  current_statement = s;

  /* fprintf(stderr, "[] BEFORE\n"); print_statement(s); */

  /* scan expressions in s; 
     atomize/cse them; 
     update availables;
  */
  gen_multi_recurse(s,
		    /* statement_domain, current_statement_filter, 
		       current_statement_rewrite, */

		    /* don't go inside these... */
		    loop_domain, loop_stop, gen_null,
		    test_domain, test_stop, gen_null,
		    unstructured_domain, gen_false, gen_null,
		    whileloop_domain, while_stop, gen_null,
		    
		    /* . */
		    call_domain, cse_atom_call_flt, gen_null,
		    
		    /* do the job on the found expression. */
		    expression_domain, expr_cse_flt, atom_cse_expression,
		    NULL);

  /* fprintf(stderr, "[] AFTER\n"); print_statement(s); */
  /* free_current_statement_stack(); */
  availables = current_availables;
  
  current_statement = statement_undefined;
  current_availables = NIL;

  return availables;
}

/* top down. */
static bool seq_flt(sequence s)
{
  list availables = NIL;

  /* fprintf(stderr, "considering sequence %d\n", 
     gen_length(sequence_statements(s))); */

  MAP(STATEMENT, ss,
      /* should clean availables with effects...
       */
      availables = atomize_cse_this_statement_expressions(ss, availables),
      sequence_statements(s));

  return TRUE;
}

void perform_ac_cse(string name, statement s)
{
  /* they have to be recomputed, because if ICM before. */

  /* set full (expr and statements) PROPER EFFECTS
     well, they are computed twice here...
     looks rather temporary.
   */
  /*
  full_simple_proper_effects(name, s);
  simple_cumulated_effects(name, s);
  */

  /* make_current_statement_stack(); */
  init_inserted();

  gen_recurse(s, sequence_domain, seq_flt, gen_null);

  /* insert moved code in statement. */
  insert_reverse_order = TRUE;
  gen_multi_recurse(s, statement_domain, gen_true, insert_rwt, NULL);

  close_inserted();
  /*
  close_expr_prw_effects();
  close_proper_rw_effects();
  close_rw_effects();
  */
}








