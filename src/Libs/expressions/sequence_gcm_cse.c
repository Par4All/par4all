/* 
   $Id$

   $Log: sequence_gcm_cse.c,v $
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

#define DEBUG_NAME "TRANSFORMATION_GCM_CSE_DEBUG_LEVEL"

#define ELEMENT EXPRESSION ;
#define element expression ;

typedef list /* of elements */ Nested_Sequence;

#define NARY_ASSIGN EXPRESSION;
#define nary_assign expression;

typedef list /* of nary_assign */ Transformed_Sequence;

/************************************************************** NESTING OKAY */

#define CALL_NESTED 1
#define LOOP_NESTED 2
#define NOT_NESTED  0

/* to remember the current statement we're in.
 */
DEFINE_LOCAL_STACK(current_stmt, statement)

/* statement -> int
 */
GENERIC_LOCAL_FUNCTION(is_nested, statement_int)

static void loop_rwt(loop l)
{
  statement current = get_current_stmt();
  int nb = load_is_nested(loop_body(l)); // ...
  store_is_nested(current, nb==NOT_NESTED? NOT_NESTED: LOOP_NESTED);
}

static void seq_rwt(sequence s)
{
  statement current = get_current_stmt();
  ...; // v = load_is_nested(st)
  store_is_nested(current, ...);
}

static void not_okay(void)
{
  statement current = get_current_stmt();
  store_is_nested(current, NOTNESTED);
}

/* initialize is_nested as a side effect.
 */
static void set_nesting_for_statement(statement s)
{
  // init_is_nested();
  init_current_stmt_stack();

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

  close_current_stmt_stack();
  // close_is_nested();
}

/*********************************************** WALK THRU NESTS TO OPTIMIZE */

static bool do_gcm_cse(statement s)
{
  if (load_is_nested(s))
  {
    // BIG transformation here...
    // finally update statement_instruction(s)
  }
}

static void apply_gcm_cse(statement s)
{
  gen_recurse(s, statement_domain, do_gcm_cse, gen_null);
}


/************************************************************ BUILD SEQUENCE */

static list my_current_sequence = NIL;

static bool loop_flt(loop l)
{
  my_current_sequence = ...;
  return TRUE;
}

static void build_sequence(instruction i)
{
  gen_multi_recurse(i, 
		    loop_domain, loop_flt, gen_null, 
		    test_domain, test_flt, gen_null,
		    call_domain, call_flt, gen_null,
		    expression_domain, gen_true, gen_null, 
		    NULL)
}

/*****************************************************************************/


static Transformed_Sequence
recursive_atomization(element e)
{
  

}





static Transformed_Sequence 
nary_atomization (Nested_Sequence ns)
{
  element e;
  Transformed_Sequence tmp;

  /* for all elements in the nested sequence */
  MAP(ELEMENT, e,
      {
	/* atomize this element */
	tmp = recursive_atomization(e);
	
	/* insert the new nested sequence in the right place */
	replace_element_by_list_of_elements(ns, e, tmp);
      }
	, ns);
  
}

static void 
create_new_nested_sequence (Nested_Sequence ns)
{
  
  
  

  /* atomization */
  
  
  /* global code motion - sort */


  /* common subexpression elimination */

}


static void
apply_gcm_cse_on_nested_sequence (Nested_Sequence ns)
{
  
  /* create new nested_sequence */

  /* swap old and new nested_sequences */

  /* free old and new nested sequence ? */

}






static void 
apply_gcm_cse_on_statement( string module_name, 
			    statement s)
{

  Nested_Sequence ns; 

  /* search all nested_sequences (i.e. perfect loop nest) for this
     statement */
  
  
  /* for all nested sequences */
  for ( ; ; ) {
    /* apply gcm + cse on the current nested sequence */
    apply_gcm_cse_on_nested_sequence(ns);

    /* jump on next nested_sequence */

  }

}

bool gcm_cse_on_sequence(string module_name)
{

    statement s;

    debug_on(DEBUG_NAME);

    /* get needed stuff.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_current_optimization_strategy();

    s = get_current_module_statement();

    /* check consistency before optimizations */
    pips_assert("consistency checking before optimizations",
		statement_consistent_p(s));


    /* apply GCM and CSE here */
    apply_gcm_cse_on_statement(module_name, s);

    /* check consistency after optimizations */
    pips_assert("consistency checking after optimizations",
		statement_consistent_p(s));

    

    /* return result to pipsdbm
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, s);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_current_optimization_strategy();

    debug_off();

    return TRUE; /* okay ! */
}


