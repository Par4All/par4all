/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
   Revision 1.25  2000/08/19 14:03:03  phamdinh
   Add comments in code!

   Revision 1.24  2000/07/29 17:05:24  phamdinh
   W effects is resolved!

   Revision 1.23  2000/07/28 14:12:20  phamdinh
   	* Number of use of every new variable is counted (using the field
   	'comments' of the statement defining it. This value is modified
   	often, it may be increased but also be reduced!
   	ex:
   		x = a + b*c + d
   		y = b*c + e

   	==>	F_0 = b*c (Number of use = 2)
   		x = F_0 + a + d
   		y = F_0 + e

   	But:
   		x = a + b*c + d
   		y = a + b*c + e

   	==>	F_0 = b*c 	(Number of use F_0 = 1)
   		F_1 = a + F_0	(Number of use F_1 = 2)
   		x = F_1 + d
   		y = F_1 + e
   	Note: 	Before the creation of F_1, Number of use F_0 = 2. After
   		the appaire of F_1, #F_0 = 1 because F_0 is contained in F_1

   	==>> Code optimal:
   		F_1 = a + b*c	(F_0 is removed)
   		x = F_1 + d
   		y = F_1 + e

   *******	Summary: Code is more optimal!!!

   Revision 1.22  2000/07/27 14:33:04  phamdinh
   Debug: Constant, Unary minus, and "y = x + ..."

   	x = a+b
   	y = a+b+c

   	==> 	x = a+b
   		y = x+c
   ======

   	x = a-b+5
   	y = a-b+5+c

   	==>	x = a-b+5
   		y = x+c

   Revision 1.21  2000/07/24 16:25:46  phamdinh
   CSE: Remove statements temporel...

   Revision 1.20  2000/07/21 16:06:14  phamdinh
   CSE: Remove statements temporel

   Revision 1.19  2000/07/21 08:26:56  phamdinh
   Modification of the update aspt->available_contents

   Revision 1.18  2000/07/20 16:47:27  phamdinh
   New statements are inserted in the correct position in hash_table

   Revision 1.17  2000/07/03 12:52:07  coelho
   headers fixed.

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

#include "transformations.h" /* out? */
#include "expressions.h"

#include "eole_private.h"

extern char * itoa(int); /* somewhere in newgen */

/******************************************************************* FLATTEN */

static void 
flatten_sequence(sequence sq)
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

/* For CSE */
static list current_availables = NIL;
static statement current_statement = statement_undefined;

/* PDSon: w_effect store all the variables modified 
 * in the sequence of statement
 */
static list *w_effects;

/* Keep track of nesting.
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

/* There is a side effect if there is a W effect in the expression.
 */
static bool 
side_effects_p(expression e)
{
  effects ef = load_expr_prw_effects(e);
  list /* of effect */ le = effects_effects(ef);
  MAP(EFFECT, ef, if (effect_write_p(ef)) return TRUE, le);
  return FALSE;
}

/* Some effect in les interfere with var.
 */
static bool 
interference_on(entity var, list /* of effect */ les)
{
  MAP(EFFECT, ef, 
  {
      if (effect_write_p(ef) &&
	  entity_conflict_p(var, reference_variable(effect_reference(ef))))
      {
	return TRUE;
      }
  },
      les);
  return FALSE;
}

/* Whether sg with effects le can be moved up to s.
 */
static bool 
moveable_to(list /* of effects */ le, statement s)
{
  list les = load_cumulated_rw_effects_list(s);
  MAP(EFFECT, ef,
      if (interference_on(reference_variable(effect_reference(ef)), les))
        return FALSE,
      le);
  return TRUE;
}

/* Return the level of this expression, using the current nesting list.
 * 0: before any statement!
 * n: outside nth loop.
 * and so on.
 */
static int 
level_of(list /* of effects */ le)
{
  list /* of statement */ up_nesting = gen_nreverse(gen_copy_seq(nesting));
  int level = 0;
  MAP(STATEMENT, s,
  {
      if (moveable_to(le, s))
      {
	gen_free_list(up_nesting);
	return level;
      }
      else
      {
        level++;
      }
  },
      up_nesting);
  
  gen_free_list(up_nesting);
  return level;
}

/* The level can be queried for a sub expression. */
static int 
expr_level_of(expression e)
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

/* Returns the statement of the specified level
   should returns current_statement_head() to avoid ICM directly.
 */
static statement 
statement_of_level(int level)
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

static expression
right_side_of_assign_statement(statement stat)
{
  instruction i;
  call assign;
  expression right_side;

  i = statement_instruction(stat);

  pips_assert("Instruction is a call", instruction_call_p(i));
  assign = instruction_call(i);

  pips_assert("Call is an assignment!", 
	      ENTITY_ASSIGN_P(call_function(assign)));

  right_side = EXPRESSION(CAR(CDR(call_arguments(assign))));
  
  return right_side;
}

/* Verify if entity ent is an argument in the right expression 
 * of the assign statement stat
 */
static bool 
entity_as_arguments(entity ent, statement stat)
{
  expression right_side;
  call right_call;

  right_side = right_side_of_assign_statement(stat);

  pips_assert("Right expression is a call!", 
	      syntax_call_p(expression_syntax(right_side)));
  right_call = syntax_call(expression_syntax(right_side));

  MAP(EXPRESSION, e,
  {
    syntax s = expression_syntax(e);
    /* Argument is maybe a call to a constant */
    if (syntax_reference_p(s) &&
	ent == reference_variable(syntax_reference(s)))
    {
      return TRUE;
    }
  }, call_arguments(right_call));

  return FALSE;
}

static bool
assign_statement_p(statement s)
{
  instruction i = statement_instruction(s);
  if (instruction_call_p(i))
  {
    call c = instruction_call(i);
    if(ENTITY_ASSIGN_P(call_function(c)))
    {
      return TRUE;
    }
  }
  return FALSE;
}

/* Return the expression in the left side of an assign statement 
 */
static expression
expr_left_side_of_assign_statement(statement stat)
{
  if (assign_statement_p(stat))
  {
    call assign = instruction_call(statement_instruction(stat));
    return EXPRESSION(CAR(call_arguments(assign)));
  }

  pips_internal_error("It is not an assign statement !");

  return NULL;
}

/* Return the entity in left side of an assign statement 
 */
static entity 
left_side_of_assign_statement(statement stat)
{
  expression left_side = expr_left_side_of_assign_statement(stat);

  pips_assert("Left side is not a reference!", 
	      syntax_reference_p(expression_syntax(left_side)));

  return reference_variable(syntax_reference(expression_syntax(left_side)));
}

/* Insert statement s in the list of statement l
 */
static list
insertion_statement_in_correct_position(statement news, list l)
{
  entity ent = left_side_of_assign_statement(news);
  statement s = STATEMENT(CAR(l));

  if (entity_as_arguments(ent, s) || CDR(l) == NIL)
  {
    return CONS(STATEMENT, s, CONS(STATEMENT, news, CDR(l)));
  }
  return CONS(STATEMENT, s,
	      insertion_statement_in_correct_position(news, CDR(l)));
}

/* Just for test 
static void
dump_list_of_statement(list l)
{
  fprintf(stderr, "\n===== Dump List: \n");
  MAP(STATEMENT, ss,
  {
    print_statement(ss);
  }, l);
  fprintf(stderr, "\n END dumpt List!!! \n");
}
*/

static void 
insert_before_statement(statement news, statement s, bool last)
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
    
    /* Statements are stored in reverse order...
       this will have to be fixed latter on. see #1#.
    */
    
    /* Insert in the front of list
     * ===========================
     */
    if (last)
    {
      instruction_block(i) = CONS(STATEMENT, news, instruction_block(i));
    }
    /* Insert just before the appropriate statement of list
     * ====================================================
     */
    else
    {
       instruction_block(i) = 
	 insertion_statement_in_correct_position(news, instruction_block(i));
    }  
  }
}

/* Atomizable if some computation.
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
      /* Missing cases? user functions? I/O? */
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
    {
      elevel = nlevels;
    }
    
    eatlevel = (list) gen_array_item(result, elevel);
    if (eatlevel == list_undefined)
    {
      eatlevel = CONS(EXPRESSION, e, NIL);
    }
    else
    {
      eatlevel = CONS(EXPRESSION, e, eatlevel);
    }
    gen_array_addto(result, elevel, eatlevel);
  },
      le);

  for (i=0; i<=nlevels; i++)
  {
    if (((list)gen_array_item(result, i)) != list_undefined)
      lastlevel = i;
  }

  /* Fix expressions by useful levels, with some operations.
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

static void 
print_group_expr(gen_array_t /* array of group of expressions */ g
		 , int nombre_grp)
{
  int i;
  for(i=0; i<= nombre_grp; i++)
  {
    list l = (list)gen_array_item(g, i);
    fprintf(stderr,"\n\n***GROUP LEVEL %d:\n", i);
    if (l)
    {
      MAP(EXPRESSION, e,
      {
	fprintf(stderr,"; ");
	print_syntax(expression_syntax(e));
      },
	  l);
    }
  }
}

/* Atomize sub expressions with 
   - lower level
   - not simple (references or constants)
   - no side effects.
*/
static void 
do_atomize_if_different_level(expression e, int level)
{
  int elevel = expr_level_of(e);

  if (elevel!=-1 && 
      elevel<level &&
      atomizable_sub_expression_p(e) &&
      !side_effects_p(e))
    {
      statement atom = atomize_this_expression(hpfc_new_variable, e);
      if (atom)
	insert_before_statement(atom, statement_of_level(elevel), TRUE);
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

static void 
atomize_or_associate_for_level(expression e, int level)
{
  syntax syn;
  call c;
  entity func;
  list /* of expression */ args;
  int lenargs, exprlevel;

  /* Some depth, otherwise no ICM needed!
   * should be fixed if root statement is pushed?
   */
  if (!currently_nested_p()) return;
  
  /* Only a scalar expression can be atomized.
   */



  /* Only do something with calls.
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
    /* Reassociation + atomization maybe needed.
     * code taken from JZ.
     */
      int i, nlevels = current_level();
      gen_array_t groups = group_expr_by_level(nlevels, args);
      list lenl;

      /* Note: the last level of an expression MUST NOT be moved!
       * indeed, there may be a side effects in another part of the expr.
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
	  
	  /* Insert expression in upward expression, eventually in e!
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
	  insert_before_statement(atom, statement_of_level(i), TRUE);
	}
      }
      
      /* Fix last level if necessary.
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

/* Don't go into I/O calls... */
static bool icm_atom_call_flt(call c)
{
  entity called = call_function(c);
  return !(io_intrinsic_p(called) || ENTITY_IMPLIEDDO_P(called));
}

/* Maybe I could consider moving the call as a whole?
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

  /* Deal with loop bound expressions
   */
  if (!currently_nested_p()) return;
  bounds = loop_range(l);
  level = current_level();
  do_atomize_if_different_level(range_lower(bounds), level);
  do_atomize_if_different_level(range_upper(bounds), level);
  do_atomize_if_different_level(range_increment(bounds), level);
}

/* PDSon: I use the field 'comments' of statement for counting its number of 
 * use. Raison: The field 'comments' of new statement added is always empty!
 * This function verifies if number of use is greater than 1 or not.
 */
static bool
number_of_use_greater_1(statement s)
{
  char* comment = statement_comments(s);
  int number = 0;

  if(empty_comments_p(comment))
  {
    return FALSE;
  }

  sscanf((const char*)comment,"%d", &number);
  return (number > 1);
}

/* Update the field 'comments' of a statement 
 */
static void
set_comment_of_statement(statement s, char *new_comment)
{
  if (empty_comments_p(statement_comments(s)))
  {
    statement_comments(s) = new_comment;
  }
  else
  {
    free(statement_comments(s));
    statement_comments(s) = new_comment;
  }
}

/* Update the number of use of statement defining 'ent' which is a member 
 * of list lst_stat with step up_down (up_down = 1 | 0 | -1).
 * If ent is a variable of original code (ex: x, y), not new variable added
 * (ex: F_0, F_1,..), there is not any statement in lst_stat defining ent.
 * Return: statement updated
 */
static statement
update_number_of_use(entity ent, list lst_stat, int up_down)
{
  MAP(STATEMENT, s,
  {
    entity left_side = left_side_of_assign_statement(s);
    if(ent == left_side) // s defines ent
    {
      if (up_down == 0)
      {
	return s;
      }

      /* First time, no value */
      if (empty_comments_p(statement_comments(s)))
      {
	if (up_down == -1)
	{
	  pips_internal_error("Number of use of '%s' < 0 !!!\n", 
			      entity_name(ent));
	}
	statement_comments(s) = strdup("1");
      }
      /* Update old value */
      else
      {
	char new[20];
	int number_use = 0;
	char* comment = statement_comments(s);
	sscanf((const char*)comment, "%d", &number_use);
	
	number_use += up_down;
	sprintf(new, "%d", number_use);
	set_comment_of_statement(s, strdup(new));
      }
      return s;
    }
  }, lst_stat);

  return NULL;
}

/* Increase number of use of variable ent by one */
static void
increase_number_of_use_by_1(entity ent, statement container)
{
  if (bound_inserted_p(container))
  {
    statement updated, sblock = load_inserted(container);
    instruction i = statement_instruction(sblock);
    sequence seq;
    int step;
    pips_assert("it is a sequence", instruction_sequence_p(i));
    
    seq = instruction_sequence(i);

    step = +1;
    if(assign_statement_p(container))
    {
      if (ent == left_side_of_assign_statement(container))
      {
	step = 0;
      }
    }

    if (!(updated = update_number_of_use(ent, sequence_statements(seq), step)))
    {
      pips_internal_error("No statement defines '%s'\n", entity_name(ent));    
    }

    /* Reduce by 1 number of use of variables contained by statement Updated */
    {
      expression exp = right_side_of_assign_statement(updated);
      if (syntax_call_p(expression_syntax(exp)))
      {
	list args = call_arguments(syntax_call(expression_syntax(exp)));
	MAP(EXPRESSION, arg,
	{
	  syntax syn = expression_syntax(arg);
	  if(syntax_reference_p(syn))
	  {
	    entity en = reference_variable(syntax_reference(syn));
	    update_number_of_use(en, sequence_statements(seq), -1);
	  }
	}, args);
      }
    }
  }
  else
  {
    pips_internal_error("No statement inserted!");    
  }
}

static void
remove_statement_redundant(statement s, list* inserted);

static bool
cse_expression_flt(expression e, list* inserted)
{
  entity scala;

  if(!syntax_reference_p(expression_syntax(e)))
  {
    /* Go down  */
    return TRUE;
  }

  scala = reference_variable(syntax_reference(expression_syntax(e)));
  MAP(STATEMENT, s,
  {
    entity ent = left_side_of_assign_statement(s);
    if (scala == ent)
    {
      if (number_of_use_greater_1(s))
      {
	/* This statement is a real one. But it maybe contains a variable
	 * temporal -> Continue remove redundant statement from this one 
	 */
	remove_statement_redundant(s, inserted);

	/* Remove its comment for pretty look: It is visited! Not do again! */
	set_comment_of_statement(s, string_undefined);

	/* Go up */
	return FALSE;
      }
      else if (string_undefined_p(statement_comments(s)))
      {
	/* This statement is already visited! Not do again! Go up */
	return FALSE;
      }
      else
      {
	/* s is a redundant statement. Replace e by the right side of 
	 * the assign statement s 
	 */
	expression exp = right_side_of_assign_statement(s);
	expression_syntax(e) = expression_syntax(exp);
	expression_syntax(exp) = NULL;
	
	/* Remove s from list
	 */
	gen_remove_once(inserted, s);
	
	free_statement(s);

	/* Continue go down the new expression */
	return TRUE;
      }
    }
  }, *inserted);

  /* Nothing is done! Go up */
  return FALSE;  
}

/* Avoid  visit the left side of assign statement
 */
static bool
cse_call_flt(call c, list* inserted)
{
  if(ENTITY_ASSIGN_P(call_function(c)))
  {
    expression left = EXPRESSION(CAR(call_arguments(c)));
    gen_recurse_stop(left);    
  }
  /* Go down! */
  return TRUE;
}

/* Remove statement redundant inserted before statement s
 */
static void
remove_statement_redundant(statement s, list* inserted)
{
  gen_context_multi_recurse(s, inserted,
			    call_domain, cse_call_flt, gen_null,
			    expression_domain, cse_expression_flt, gen_null,
			    NULL);  
}

static bool insert_reverse_order = TRUE;

/* Insert the new statements created
 */
static void insert_rwt(statement s)
{  
  if (bound_inserted_p(s))
  {
    statement sblock = load_inserted(s);
    instruction i = statement_instruction(sblock);
    sequence seq;
    
    pips_assert("it is a sequence", instruction_sequence_p(i));
    
    /* Reverse list of inserted statements (#1#) */
    seq = instruction_sequence(i);
    
    /* Remove statements redundant */
    remove_statement_redundant(s, &sequence_statements(seq));


    if (insert_reverse_order)
    {
      sequence_statements(seq) = gen_nreverse(sequence_statements(seq));
    }

      /* insert */
    sequence_statements(seq) = 
      gen_append(sequence_statements(seq),
		 CONS(STATEMENT,
		      instruction_to_statement(statement_instruction(s)),
		      NIL));
    
    statement_instruction(s) = i;
  }
}

/* Perform ICM and association on operators.
   this is kind of an atomization.
   many side effects: modifies the code, uses simple effects
 */
void 
perform_icm_association(string name, /* of the module */
			statement s  /* of the module */)
{
  pips_assert("clean static structures on entry",
	      (get_current_statement_stack() == stack_undefined) &&
	      inserted_undefined_p() &&
	      (nesting==NIL));

  /* Set full (expr and statements) PROPER EFFECTS
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

  /* Added by PDSon:
     This list stores variables modified. It is used to avoid of creating 
     expression common containing variable modified.
     It shoud remove membre 'depends'!
   */
  list *w_effects; /* list of expression */ 
}
  available_scalar_t, * available_scalar_pt;

/*
static void dump_aspt(available_scalar_pt aspt)
{
  syntax s;

  if (!aspt)
  {
    fprintf(stderr, "DUMP ASPT\n ASPT = NULL !!!\n");
    return;
  }

  s = expression_syntax(aspt->contents);
  fprintf(stderr, 
	  "DUMP ASPT\n"
	  "Scalar: %s \t Operator:[%s] len=%d, avail=%d\n",
	  entity_name(aspt->scalar), 
	  aspt->operator? entity_name(aspt->operator): "NOP",
	  syntax_call_p(s)? gen_length(call_arguments(syntax_call(s))): -1,
	  gen_length(aspt->available_contents));
  fprintf(stderr, "\n===Container:\n");
  print_statement(aspt->container);
  fprintf(stderr, "\n===Contents:\n");
  print_expression(aspt->contents);
  fprintf(stderr, "\n===Available_contents:\n");
  MAP(EXPRESSION, e,
  {
    print_expression(e);    
  }, aspt->available_contents);
}
*/

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

static bool 
expression_in_list_p(expression e, list seen)
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

/*
static void 
dump_list_of_exp(list l)
{
  fprintf(stderr,"\n======\nDUMP LIST OF EXPRESSIONS \n");
  fprintf(stderr,"Length: %d\n", gen_length(l));
  MAP(EXPRESSION, e, 
  {
    print_expression(e);
  },
      l);

  fprintf(stderr,"\n======\nEND DUMP\n");
}

static void
dump_expresison_nary(expression e)
{
  syntax s = expression_syntax(e);
  fprintf(stderr,"\n===== DUMP EXPRESSION: \n");
  if (syntax_reference_p(s))
  {
    fprintf(stderr,"\n Reference!!");
  }
  else if (syntax_call_p(s))
  {
    fprintf(stderr,"\n Call '%s' with %d arguments \n",
	    entity_local_name(call_function(syntax_call(s))),
	    gen_length(call_arguments(syntax_call(s))));
  }
  fprintf(stderr,"\n===== END DUMP EXPRESSION!! \n");  
}
*/

static list /* of expression */
list_diff(list l1, list l2); /* Define forward */

static bool  /* Define forward */
expression_eq_in_list_p(expression e, list l, expression *f);

/* Find the commun sub-expression between e & aspt. 
 * Attention: Commun expression does't contain any expression in w_effects
 */
static int 
similarity(expression e, available_scalar_pt aspt)
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
    if (reference_equal_p(r, ra)) 
    {
      return MAX_SIMILARITY;
    }
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
      list com = common_expressions(list_diff(call_arguments(c), 
					      *(aspt->w_effects)),
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
	{
	  return NO_SIMILARITY;
	}
	
	if (expression_eq_in_list_p(el, *(aspt->w_effects), &el))
	{
	  /* A variable is already modified ! */
	  return NO_SIMILARITY;
	}
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
  }, current_availables);

  return best;
}

static available_scalar_pt 
make_available_scalar(entity scalar, statement container, expression contents)
{
  syntax s = expression_syntax(contents);

  available_scalar_pt aspt = 
    (available_scalar_pt) malloc(sizeof(available_scalar_t));

  aspt->scalar = scalar;
  aspt->container = container;
  aspt->contents = contents;

  aspt->depends = get_all_entities(contents);

  aspt->w_effects = w_effects;

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

static bool 
expression_eq_in_list_p(expression e, list l, expression *f)
{
  MAP(EXPRESSION, et,
  {
    if (expression_equal_p(e, et))
    {
      *f = et;
      return TRUE;
    }
  }, l);

  return FALSE;
}

/* returns an allocated l1-l2 with expression_equal_p
   l2 included in l1.
 */
static list /* of expression */ 
list_diff(list l1, list l2)
{
  list diff = NIL, l2bis = gen_copy_seq(l2);
  expression found;

  MAP(EXPRESSION, e,
  {
    if (expression_eq_in_list_p(e, l2bis, &found))
    {
      gen_remove(&l2bis, found);
    }
    else
    {
      diff = CONS(EXPRESSION, e, diff);
    }
  }, l1);

  if (l2bis) gen_free_list(l2bis);

  return diff;
}

static bool 
simple_reference_p(expression e)
{
  syntax s = expression_syntax(e);
  return syntax_reference_p(s) && !reference_indices(syntax_reference(s));
}

static bool
expression_constant_p(expression e)
{
  syntax s = expression_syntax(e);
  if(syntax_call_p(s))
  {
    call c = syntax_call(s);
    entity en = call_function(c);
    return entity_constant_p(en);
  }
  return FALSE;
}

static bool
call_unary_minus_p(expression e)
{
  syntax s = expression_syntax(e);
  if(syntax_call_p(s))
  {
    call c = syntax_call(s);
    return ENTITY_UNARY_MINUS_P(call_function(c));
  }
  return FALSE;
}

/* Remove some inpropriate ones...
static void clean_current_availables(void)
{
  
}
*/

static void atom_cse_expression(expression e)
{
  /* statement head = current_statement_head(); */
  syntax s = expression_syntax(e);
  int quality;
  available_scalar_pt aspt;

  if (syntax_call_p(s) && ENTITY_ASSIGN_P(call_function(syntax_call(s))))
    return;

  do { /* extract every possible common subexpression */
    aspt = best_similar_expression(e, &quality);

    if (aspt) /* some common expression found. */ 
    {
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
	    
	    syntax_tag(s) = is_syntax_reference;
	    syntax_reference(s) = make_reference(aspt->scalar, NIL);
	    
	    /* Set the statement to status REAL */
	    increase_number_of_use_by_1(aspt->scalar, aspt->container);
	    
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
	    in_common = common_expressions(list_diff(call_arguments(c),
						     *(aspt->w_effects)),
					   aspt->available_contents);

	    /* Case: in_common == aspt->contents 
	     * =================================
	     */
	    if (gen_length(linit)==gen_length(in_common))
	    {
	      /* just substitute lo1, don't build a new aspt. */
	      lo1 = list_diff(call_arguments(c), in_common);
	      free_arguments(call_arguments(c));
	      call_arguments(c) = 
		gen_nconc(lo1, 
			  CONS(EXPRESSION,
			       entity_to_expression(aspt->scalar), NIL));
	      
	      /* Set the statement to status REAL */
	      increase_number_of_use_by_1(aspt->scalar, aspt->container);
	      
	    }
	    else
	    {
	      lo1 = list_diff(call_arguments(c), in_common);
	      lo2 = list_diff(linit, in_common);
	      
	      cse = call_to_expression(make_call(aspt->operator, 
						 in_common));
	      scse = atomize_this_expression(hpfc_new_variable, cse);
	      
	      /* now cse is a reference to the newly created scalar. */
	      pips_assert("a reference...",
			  syntax_reference_p(expression_syntax(cse)));
	      scalar = reference_variable(syntax_reference
					  (expression_syntax(cse)));
	      cse2 = copy_expression(cse);
	      
	      /* update both expressions... */
	      old = call_arguments(c); /* in code */
	      if (gen_length(call_arguments(c))==gen_length(in_common))
	      {
		expression_syntax(e) = expression_syntax(cse);
		expression_normalized(e) = expression_normalized(cse);
	      }
	      else
	      {
		call_arguments(c) = CONS(EXPRESSION, cse, lo1);
	      }
	      gen_free_list(old);
	      
	      old = call_arguments(ca);
	      call_arguments(ca) = CONS(EXPRESSION, cse2, lo2);
	      gen_free_list(old);
	      
	      set_comment_of_statement(scse, strdup("1"));
	      insert_before_statement(scse, aspt->container, FALSE);
	      increase_number_of_use_by_1(scalar, aspt->container);
	      
	      /* don't visit it later. */
	      gen_recurse_stop(scse);		  		  
	      
	      /* updates available contents */
	      aspt->depends = CONS(ENTITY, scalar, aspt->depends);
	      old = aspt->available_contents; 
	      aspt->available_contents = CONS(EXPRESSION, cse,
					      list_diff(old, in_common));
	      gen_free_list(old);
	      
	      /* add the new scalar as an available CSE. */
	      naspt = make_available_scalar(scalar,
					    aspt->container,
		   EXPRESSION(CAR(CDR(call_arguments(
		       instruction_call(statement_instruction(scse)))))));
	      current_availables = CONS(STRING, (char*)naspt, 
					current_availables);
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

  /* PDSon: Do not atomize: 
   *   - simple reference (ex: a, x)
   *   - constant (ex: 2.6 , 5)
   *   - Unary minus (ex: -a , -5)
   */
  if (!simple_reference_p(e) && 
      !expression_constant_p(e) &&
      !call_unary_minus_p(e))
  {
    expression exp = NULL;
    instruction ins = statement_instruction(current_statement);
    if (instruction_call_p(ins))
    {
      call c_assign = instruction_call(ins);

      if (ENTITY_ASSIGN_P(call_function(c_assign)))
      {
	exp = EXPRESSION(CAR(CDR(call_arguments(c_assign))));
      }
    }

    if (exp != e)
    {
      statement atom;

      /* create a new atom... */
      atom = atomize_this_expression(hpfc_new_variable, e);
      /* At fisrt, this statement is set "virtual" */
      set_comment_of_statement(atom,strdup("1"));
      insert_before_statement(atom, current_statement, TRUE);

      /* don't visit it later, just in case... */
      gen_recurse_stop(atom);
      {
	entity scalar;
	call ic;
	syntax s = expression_syntax(e);
	instruction i = statement_instruction(atom);
	pips_assert("it is a reference", syntax_reference_p(s));
	pips_assert("instruction is an assign", 
		    instruction_call_p(i)); /* ??? */
		    
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
    else /* exp == e == <right expression of assignment> 
	  * => Atomize this expression without using temporel variable;
	  *    we use the left side variable of assignment in order to reduce
	  *    the number of the variable temporel for the comprehensive of
	  *    the code.
	  */
    {
      entity scalar = left_side_of_assign_statement(current_statement);
      expression rhs;
      statement assign;
      syntax ref;
      
      rhs = make_expression(expression_syntax(e), normalized_undefined);
      normalize_all_expressions_of(rhs);
      
      ref = make_syntax(is_syntax_reference, make_reference(scalar, NIL));
      
      assign = make_assign_statement(make_expression(copy_syntax(ref), 
						     normalized_undefined), 
				     rhs);
      expression_syntax(e) = ref;
      
      set_comment_of_statement(assign,strdup("1"));
      insert_before_statement(assign, current_statement, TRUE);

      aspt = make_available_scalar(scalar, current_statement, rhs);	
      current_availables = CONS(STRING, (char*)aspt, current_availables);
    }
  }
}

static bool loop_stop(loop l)
{ 
  gen_recurse_stop(loop_body(l)); 
  return TRUE; 
}

static bool test_stop(test t)
{ 
  gen_recurse_stop(test_true(t));
  gen_recurse_stop(test_false(t)); 
  return TRUE; 
}

static bool while_stop(whileloop wl)
{ 
  gen_recurse_stop(whileloop_body(wl)); 
  return TRUE; 
}

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

  /* free_current_statement_stack(); */
  availables = current_availables;
  
  current_statement = statement_undefined;
  current_availables = NIL;

  /* Update w_effects: add the variable modified by the current statement */
  if (assign_statement_p(s))
  {
    /* Update contents */
    expression var_defined = 
      copy_expression(expr_left_side_of_assign_statement(s));
    *w_effects = CONS(EXPRESSION, var_defined, NIL);

    /* Update address: w_effects always points to the end of the list of
     * variable modified which is filled after each statement
     */
    w_effects = &CDR(*w_effects);
  }

  return availables;
}

/* top down. */
static bool seq_flt(sequence s)
{
  list availables = NIL;
  list *top_of_w_effects;

  /* At first, w_effects is empty list BUT not list points to NIL!!!*/
  w_effects = &CDR(CONS(EXPRESSION, NIL, NIL));

  /* top_of_w_effects points to the top of list, 
   * It is used to free memory later 
   */
  top_of_w_effects = w_effects;

  MAP(STATEMENT, ss,
  {
    availables = atomize_cse_this_statement_expressions(ss, availables);

  }, sequence_statements(s));

  /* Free top_of_w_effects and availables */
  gen_full_free_list(*top_of_w_effects);
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
