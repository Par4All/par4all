/* Name     :   loop_normalize.c
 * Package  : 	loop_normalize 
 * Author   :   Arnauld LESERVOT & Alexis PLATONOFF
 * Date     :	27 04 93
 * Modified :   moved to Lib/transformations, AP, sep 95
 * Documents:	"Implementation du Data Flow Graph dans Pips"
 * Comments :	
 *
 * Functions of normalization of DO loops. Normalization consists in renaming
 * the loop index so as to have something like:
 *      DO NLC = 0, UPPER, 1
 *
 * If the old DO loops was:
 *      DO I = lower, upper, incre
 * then : UPPER = (upper - lower + incre)/incre - 1
 *
 * The normalization is done only if "incre" is a constant number.
 * The normalization produces two statements. One assignment of the old
 * loop index (in the exemple I) to its value function of the new index;
 * the formula is: I = incre*NLC + lower
 * and one assignment of the old index at the end of the loop for its final
 * value: I = incre * MAX(UPPER+1, 0) + lower
 *
 * So, for exemple:
 *      DO I = 2, 10, 4
 *        INST
 *      ENDDO
 * is normalized in:
 *      DO NLC = 0, 2, 1
 *        I = 4*NLC + 2
 *        INST
 *      ENDDO
 *      I = 14
 *
 * Or:
 *      DO I = 2, 1, 4
 *        INST
 *      ENDDO
 * is normalized in:
 *      DO NLC = 0, -1, 1
 *        I = 4*NLC + 2
 *        INST
 *      ENDDO
 *      I = 2
 *
 * The "NLC" variables are the Normalized Loop Counters. Their implicit type
 * is INTEGER.
 *
 * If a loop has a label, it is removed. For example:
 *      DO 10 I = 1, 10, 1
 *        INST
 * 10   CONTINUE
 *
 * is modified in:
 *      DO i = 1, 10, 1
 *        INST
 *      ENDDO
 */

/* Ansi includes	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "linear.h"
#include "ri.h"

#include "paf_ri.h"
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
#include "graph.h"
#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "paf-util.h"
#include "static_controlize.h"
#include "transformations.h"

#define MAX_OPERATOR_NAME "MAX"

/*============================================================================*/
/* void loop_normalize(string mod_name): Apply the loop normalization upon
 * a module of name "mod_name".
 *
 * You may note the reorder of the module body. It is necessary if you want to
 * make others analyses upon the generated CODE.
 */
bool loop_normalize(mod_name)
char *mod_name;
{
  statement mod_stat;
  int Gcount_nlc;
  hash_table 	Gforward_substitute_table;
  list		Genclosing_loops;
  list		Genclosing_tests;
  list 		Gscalar_written_forward;

  debug_on("LOOP_NORMALIZE_DEBUG_LEVEL");

  if (get_debug_level() > 1) {
    user_log("\n\n *** LOOP_NORMALIZE for %s\n", mod_name);
  }

  /* Sets the current module to "mod_name". */
  set_current_module_entity(local_name_to_top_level_entity(mod_name));

  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
 
  /* Initialization of the NLC counters. It numbers the NLC variables
   * created during the computation.
   */
  Gcount_nlc = 0;

  /* Initialize the lists and the hash table */
  Genclosing_loops      	= (list) NIL;
  Genclosing_tests      	= (list) NIL;
  Gscalar_written_forward = (list) NIL;
  Gforward_substitute_table = hash_table_make( hash_pointer, 0 );
 
  /* Compute the loops normalization of the module. */

  /* No longer assumption on a global unstructured around... */
  ln_of_statement(mod_stat,
		  Gforward_substitute_table, &Genclosing_loops,
		  &Genclosing_tests, &Gscalar_written_forward,
		  &Gcount_nlc);
  
  debug_off();

  /* Reorder the module. It is necessary because new statements have been
   * generated.
   */
  module_reorder(mod_stat);

  /* Save the new generated CODE. */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

  if (get_debug_level() > 1) {
    user_log("\n\n *** LOOP_NORMALIZE done\n");
  }

  reset_current_module_entity();

  return TRUE;
}


/*==================================================================*/
/* list ln_of_loop((loop) l, hash_table fst, list *ell, *etl,
 * *swfl, int *Gcount_nlc) AL 04/93
 */
list ln_of_loop( l, fst, ell, etl, swfl, Gcount_nlc)
loop l;
hash_table fst; /* forward substitute table */
list *ell,  /* enclosing loops list */
     *etl,  /* enclosing tests list */
     *swfl; /* scalar written forward list */
int *Gcount_nlc;
{
  entity          index, nlc_ent, max_ent;
  expression      rl, ru, ri, nub, nlc_exp, exp_plus;
  expression	nub2, nub3, index_exp, new_index_exp;
  expression      exp_max = expression_undefined;
  range           lr;
  statement       before_stmt = statement_undefined;
  statement       end_stmt = statement_undefined;
  int             incre;
  list            stmt_list = NIL;
  
  debug(4, "ln_of_loop", "begin LOOP\n");

  loop_label( l ) = entity_empty_label();
  loop_body( l ) = make_block_with_stmt( loop_body( l ));
  index = loop_index( l );

  /* If it is not a constant step, we just normalize the loop body */
  if(!normalizable_loop_p(l)) {
    ADD_ELEMENT_TO_LIST(*swfl, ENTITY, index);
    ln_of_statement(loop_body( l ), fst, ell,
		    etl, swfl, Gcount_nlc);
    return( make_undefined_list() );
  }

  lr = loop_range( l );
  rl = range_lower( lr );
  ru = range_upper( lr );
  ri = range_increment( lr );
  incre = expression_to_int( ri );

  nub =   make_op_exp(DIVIDE_OPERATOR_NAME,
		      make_op_exp(PLUS_OPERATOR_NAME,
				  make_op_exp(MINUS_OPERATOR_NAME,
					      expression_dup(ru),
					      expression_dup(rl)),
				  expression_dup(ri)),
		      expression_dup(ri));
  nub2 = expression_dup(nub);

  ADD_ELEMENT_TO_LIST( stmt_list, STATEMENT, before_stmt );

  nlc_ent = make_nlc_entity(Gcount_nlc);
  ADD_ELEMENT_TO_LIST(*swfl, ENTITY, nlc_ent);
  nlc_exp = make_entity_expression( nlc_ent, NIL);
  loop_index( l ) = nlc_ent;
  range_lower( lr ) = make_integer_constant_expression( 1 );
  range_upper( lr ) = nub2;
  range_increment( lr ) = make_integer_constant_expression( 1 );

  new_index_exp = make_op_exp(PLUS_OPERATOR_NAME,
			      make_op_exp(MINUS_OPERATOR_NAME,
					  make_op_exp(MULTIPLY_OPERATOR_NAME,
						      expression_dup(ri),
						      nlc_exp),
					  expression_dup(ri)),
			      expression_dup(rl));
  hash_put(fst, (char*) index, (char*) new_index_exp);

  nub3 = expression_dup( nub );
  if ( expression_constant_p( nub3 )) {
    int upper = expression_to_int( nub3 );
    if ( upper > 0 )
      exp_max = make_integer_constant_expression( upper );
  }
  else {
    max_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						      MAX_OPERATOR_NAME),
				 entity_domain);
    exp_max = make_max_exp(max_ent, expression_dup( nub ),
			   make_integer_constant_expression( 0 ));
  }
  if ( exp_max == expression_undefined )
    exp_plus = expression_dup( rl );
  else
    exp_plus = make_op_exp(PLUS_OPERATOR_NAME,
			   make_op_exp( MULTIPLY_OPERATOR_NAME,
				       expression_dup( ri ),
				       exp_max),
			   expression_dup( rl ));
  index_exp = make_entity_expression( index, NIL );
  end_stmt = make_assign_statement( expression_dup(index_exp), exp_plus );
  ADD_ELEMENT_TO_LIST( stmt_list, STATEMENT, end_stmt);

  ln_of_statement(loop_body(l), fst , ell, etl, swfl, Gcount_nlc);

  hash_del(fst, (char*) index );
  debug( 4, "ln_of_loop", "end LOOP\n");
  return( stmt_list );
}


/*==================================================================*/
/* list ln_of_statement(statement s, hash_table fst, list
 * *ell, *etl, *swfl, int *Gcount_nlc): Normalization of a statement.
 *
 * Before walking down the statements, we forward-substitute the 
 * new-loop-counters on each type of statements.
 * We then return a list of two statements to be put 
 * before and after statement 's'. These two new statements
 * are generated by loops when they are treated. 
 * See document for more detail, section : "Normalisation des boucles".
 */
list ln_of_statement(s, fst, ell, etl, swfl, Gcount_nlc)
statement s;
hash_table fst; /* forward substitute table */
list *ell,  /* enclosing loops list */
     *etl,  /* enclosing tests list */
     *swfl; /* scalar written forward list */
int *Gcount_nlc;
{
instruction 	inst = statement_instruction(s);
list		return_list = NIL;


debug(3, "ln_of_statement", "begin STATEMENT\n");
return_list = make_undefined_list();

switch(instruction_tag(inst))
  {
  case is_instruction_block :
    {
    list tail, head;

    tail = instruction_block(inst);
    for(head = NIL ; tail != NIL; )
      {
      statement stmt, before_stmt, after_stmt;
      list 	insert_stmt;

      stmt = STATEMENT(CAR(tail));
      insert_stmt = ln_of_statement(stmt, fst, ell,
						etl, swfl, Gcount_nlc);
      before_stmt = STATEMENT(CAR( insert_stmt ));
      after_stmt  = STATEMENT(CAR(CDR( insert_stmt )));

      if( before_stmt != statement_undefined)
      		ADD_ELEMENT_TO_LIST( head, STATEMENT, before_stmt);
      ADD_ELEMENT_TO_LIST( head, STATEMENT, stmt );
      if ( after_stmt != statement_undefined ) 
		ADD_ELEMENT_TO_LIST( head, STATEMENT, after_stmt );

      tail = CDR(tail);
      }
    instruction_block(inst) = head;
    break;
    }
  case is_instruction_test :
    {
    test t = instruction_test(inst);
    (void) forward_substitute_in_exp(&test_condition(t), fst);
    ADD_ELEMENT_TO_LIST(*etl, EXPRESSION, test_condition(t) );
    test_true(t) = make_block_with_stmt( test_true(t) );
    test_false(t) = make_block_with_stmt( test_false(t) );
    ln_of_statement(test_true(t), fst, ell,
				etl, swfl, Gcount_nlc);
    ln_of_statement(test_false(t), fst, ell,
				etl, swfl, Gcount_nlc);
    gen_remove(etl, (gen_chunk*) test_condition( t ));
    break;
    }
  case is_instruction_loop :
    {
    (void) forward_substitute_in_loop(&instruction_loop(inst), fst);
    ADD_ELEMENT_TO_LIST(*ell, LOOP, instruction_loop( inst ));  
    return_list = ln_of_loop(instruction_loop(inst), fst, ell,
					 etl, swfl, Gcount_nlc);
    gen_remove(ell, (gen_chunk*) instruction_loop( inst ));
    break;
    }
  case is_instruction_call : 
    {
    (void) forward_substitute_in_call(&instruction_call(inst), fst);
    scalar_written_in_call( instruction_call( inst ), ell,
			   etl, swfl);
    break;
    }
  case is_instruction_goto : break;
  case is_instruction_unstructured :
    {
    ln_of_unstructured(instruction_unstructured(inst), fst, ell,
				   etl, swfl, Gcount_nlc);
    break;
    }
  default : pips_error("ln_of_statement", "Bad instruction tag");
  }
debug(3, "ln_of_statement", "end STATEMENT\n");
return( return_list );
}


/*==================================================================*/
/* void ln_of_unstructured(unstructured u, fst): Normalization
 * of an unstructured instruction.
 */
void ln_of_unstructured(u, fst, ell, etl, swfl, Gcount_nlc)
unstructured u;
hash_table fst; /* forward substitute table */
list *ell,  /* enclosing loops list */
     *etl,  /* enclosing tests list */
     *swfl; /* scalar written forward list */
int *Gcount_nlc;
{
	list blocs = NIL, lc;
	list insert_stmts;

	debug(2, "ln_of_unstructured", "begin UNSTRUCTURED\n");
	control_map_get_blocs(unstructured_control(u), &blocs ) ;
	blocs = gen_nreverse( blocs ) ;

	for(lc = blocs; lc != NIL; lc = CDR(lc)) {
		list   		head = NIL;
		control 	ctl;
		statement 	before_stmt, after_stmt, stmt;

		ctl 		= CONTROL(CAR( lc ));
		stmt 		= control_statement( ctl );
		insert_stmts 	= ln_of_statement(stmt, fst,
							      ell, etl,
							      swfl,
							      Gcount_nlc);
		before_stmt 	= STATEMENT(CAR( insert_stmts ));
		after_stmt 	= STATEMENT(CAR(CDR( insert_stmts )));

	   if (!undefined_statement_list_p( insert_stmts )) {
		if( before_stmt != statement_undefined)
			ADD_ELEMENT_TO_LIST( head, STATEMENT, before_stmt );
		ADD_ELEMENT_TO_LIST( head, STATEMENT, stmt ); 
  		if ( after_stmt != statement_undefined )
			ADD_ELEMENT_TO_LIST( head, STATEMENT, after_stmt );
		stmt = make_block_with_stmt( stmt );
		instruction_block(statement_instruction( stmt )) = head;
		head = NIL;
	   }
	}

	gen_free_list(blocs);
	debug(2, "ln_of_unstructured", "end UNSTRUCTURED\n");
}

/*==================================================================*/
