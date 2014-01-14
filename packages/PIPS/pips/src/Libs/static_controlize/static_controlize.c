/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* Name      :	static_controlize.c
 * package   :	static_controlize
 * Author    :	Arnauld LESERVOT
 * Date      :	May 93
 * Modified  :
 * Documents :	"Implementation du Data Flow Graph dans Pips"
 * Comments  :
 */

/* Ansi includes	*/
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"
#include "boolean.h"

/* C3 includes		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "properties.h"
#include "matrix.h"

/* Pips includes	*/
#include "ri.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
typedef void * arc_label;
typedef void * vertex_label;
#include "graph.h"
#include "paf_ri.h"
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

/* Global Variables	*/
list			Gstructure_parameters;
static list			Genclosing_loops;
static list			Genclosing_tests;
static list			Gscalar_written_forward;
static hash_table		Gforward_substitute_table;
//static set io_functions_set;
statement_mapping	Gstatic_control_map;


/*=================================================================*/
/* static_control static_controlize_call( (call) c )		AL 05/93
 * It computes now just a static-control assignement if it is
 * an ASSIGN statement.
 */
static_control static_controlize_call(call c)
{
 	static_control sc;

	pips_debug( 3, "begin CALL\n");
	pips_debug( 7,
		    "call : %s\n", entity_local_name( call_function(c) ) );
	entity e = call_function(c);
	tag tt;
	    if ((tt = value_intrinsic_p(entity_initial(e))))
	      {
		sc = make_static_control(false,
					 sc_list_of_entity_dup( Gstructure_parameters ),
					 /* _list_of_loop_dup(
					    Genclosing_loops ),*/
					 copy_loops( Genclosing_loops ),
					 sc_list_of_exp_dup( Genclosing_tests ));
		static_control_yes( sc ) = splc_linear_access_to_arrays_p
		  (call_arguments(c),&Genclosing_loops);
	      } 
	    else
	      {
		bool b = splc_linear_access_to_arrays_p(call_arguments(c),
						   &Genclosing_loops);
		
		sc = make_static_control(b,
					 sc_list_of_entity_dup( Gstructure_parameters ),
					 /* _list_of_loop_dup(
					    Genclosing_loops ),*/
					 copy_loops( Genclosing_loops ),
					 sc_list_of_exp_dup( Genclosing_tests ));
	      }
	    pips_debug( 3, "end CALL\n");
	    return sc;
}

/*==================================================================*/
/* static_control static_controlize_loop( (loop) l )
 * It computes the loop's static_control.
 */
static_control static_controlize_loop(loop l)
{
  static_control	sc;
	expression	low, up;

	pips_debug(3, "begin LOOP\n");

	low = range_lower( loop_range( l ) );
	up  = range_upper( loop_range( l ) );
	ADD_ELEMENT_TO_LIST(Genclosing_loops, LOOP, l);
	sc = static_controlize_statement( loop_body( l ) );
	if (	   !constant_step_loop_p(l)
		|| !splc_feautrier_expression_p(low, &Genclosing_loops)
		|| !splc_feautrier_expression_p(up, &Genclosing_loops)  ) {
		static_control_yes( sc ) = false;
	}
	SET_STATEMENT_MAPPING(Gstatic_control_map, loop_body(l), sc);
	gen_remove( &Genclosing_loops, l );

	pips_debug(3, "end LOOP\n");
	return( sc );
}




/*==================================================================*/
/* static_control static_controlize_forloop( (forloop) fl )
 * It computes the forloop's static_control.
 */
static_control static_controlize_forloop(forloop fl)
{
  static_control	sc;

  pips_debug(3, "begin FORLOOP\n");


  sc = static_controlize_statement( forloop_body( fl ) );
  static_control_yes( sc ) = false;

  SET_STATEMENT_MAPPING(Gstatic_control_map, forloop_body(fl), sc);

  pips_debug(3, "end FORLOOP\n");
  return( sc );
}

/*==================================================================*/
/* static_control static_controlize_whileloop( (whileloop) l )
 * It computes the whileloop's static_control.
 */
static_control static_controlize_whileloop(whileloop wl)
{
  static_control	sc;

  pips_debug(3, "begin WHILELOOP\n");


  sc = static_controlize_statement( whileloop_body( wl ) );
  static_control_yes( sc ) = false;

  SET_STATEMENT_MAPPING(Gstatic_control_map, whileloop_body(wl), sc);

  pips_debug(3, "end WHILELOOP\n");
  return( sc );
}

/*==================================================================*/
/* static_control static_controlize_statement((statement) s)	AL 05/93
 * Computes s's static_control
 */

// Global variables added in order to avoid changing the definition of several functions
list assigned_var = NIL;

static_control static_controlize_statement(statement s)
{
  // Update of the list containing the variables assigned directly or indirectly by an array
  get_reference_assignments(s, &assigned_var);
  
  bool		is_static = true, static_test = false;
  instruction	inst = statement_instruction(s);
  static_control  sc, sc1, sc2;
  expression	exp, exp1, exp2;

  pips_debug(3, "begin STATEMENT\n");
  pips_debug(7, "statement_ordering = %zd \n", statement_ordering(s));

  switch(instruction_tag(inst))
    {
    case is_instruction_block :
      {
	FOREACH(STATEMENT, local_stmt, instruction_block(inst)) {
	  sc1 = static_controlize_statement( local_stmt );
	  SET_STATEMENT_MAPPING( Gstatic_control_map, local_stmt, sc1);
	  //printf("s=%p, sc=%p\n", local_stmt, sc1);
	  is_static = ( is_static && static_control_yes( sc1 ));
	}
	break;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	forward_substitute_in_exp( &test_condition( t ),
				   Gforward_substitute_table);
	
	/* We put condition under a normal disjunctive form */
	if ((exp = sc_conditional(test_condition(t), &Genclosing_loops)) !=
	    expression_undefined || sp_feautrier_expression_p(test_condition(t))) {
	  //DK, don't change the structure of conditions
	  //test_condition( t ) = exp ;
	  static_test = true;
	}
	ADD_ELEMENT_TO_LIST( Genclosing_tests, EXPRESSION, test_condition(t) );
	sc1 = static_controlize_statement(test_true(t));
	SET_STATEMENT_MAPPING( Gstatic_control_map, test_true( t ), sc1);
	gen_remove( &Genclosing_tests, (void*) test_condition( t ));

	exp1 = MakeUnaryCall( ENTITY_NOT, copy_expression( test_condition(t) ));
	if ( (exp2 = sc_conditional(exp1, &Genclosing_loops)) ==
	     expression_undefined) {
	  exp2 = exp1;
	}
	ADD_ELEMENT_TO_LIST( Genclosing_tests, EXPRESSION, exp2 );
	sc2 = static_controlize_statement(test_false(t));
	SET_STATEMENT_MAPPING( Gstatic_control_map, test_false( t ), sc2);
	gen_remove( &Genclosing_tests, (void *) exp2 );
	is_static = (  static_control_yes( sc1 )
		       && static_control_yes( sc2 )
		       && static_test );
	break;
      }
    case is_instruction_loop :
      {
	loop the_loop = instruction_loop( inst );
	forward_substitute_in_anyloop( instruction_loop( inst ),
				       Gforward_substitute_table);
	sc1 = static_controlize_loop( the_loop );
	is_static = static_control_yes( sc1 );
	break;
      }
    case is_instruction_forloop :
      {
	forloop forl = instruction_forloop( inst );
	forward_substitute_in_anyloop( instruction_forloop( inst ),
				       Gforward_substitute_table);
	sc1 = static_controlize_forloop( forl );
	is_static = static_control_yes( sc1 );
	break;
      }
    case  is_instruction_whileloop :
      {
	whileloop whilel = instruction_whileloop( inst );
	forward_substitute_in_anyloop( instruction_whileloop( inst ),
				       Gforward_substitute_table);
	sc1 = static_controlize_whileloop( whilel );
	is_static = static_control_yes( sc1 );
	break;
      }
    case is_instruction_call :
      {
	if (!(/*continue_statement_p(s) ||*/ declaration_statement_p(s)))
	  {
	    call the_call = instruction_call( inst);
	    forward_substitute_in_call( &the_call, Gforward_substitute_table );
	    is_static = static_control_yes(static_controlize_call( the_call ));
	  }
	else
	  is_static = false;
	if (!get_bool_property("STATIC_CONTROLIZE_ACROSS_USER_CALLS"))
	  {
	    if (statement_contains_user_call_p(s)
		|| io_intrinsic_p((call_function(instruction_call( inst)))))
	      is_static = false;
	  }
	break;
      }

    case is_instruction_goto :
      {
	pips_internal_error("No goto in code ressource\n");
	is_static = false;
	break;
      }
    case is_instruction_expression :
      {
	is_static = true;
	break;
      }
    case is_instruction_unstructured :
      {
	unstructured local_un = instruction_unstructured( inst );
	is_static =
	  static_control_yes(static_controlize_unstructured( local_un ));
	break;
      }
    default :
      {
	pips_error("static_controlize_statement", "Bad instruction tag");
      }
    }
  //if (!(continue_statement_p(s) || declaration_statement_p(s)))
    sc = make_static_control(is_static,
			     sc_list_of_entity_dup(Gstructure_parameters),
			     sc_list_of_loop_dup(Genclosing_loops),
			     sc_list_of_exp_dup(Genclosing_tests) );

  pips_debug(7,
	     " Returning static_control : \n bool   : %s \n params : %s \n loops  : %zd \n tests  : %zd \n ",
	     ((is_static)?"TRUE":"FALSE"),
	     print_structurals( Gstructure_parameters ),
	     gen_length( Genclosing_loops ),
	     gen_length( Genclosing_tests ) );
  pips_debug(3, "end STATEMENT\n");
           
  return sc;

}


/*==================================================================*/
/* static_control static_controlize_unstructered((unstructured) u)  AL 05/93
 * Computes an unstructured's static_control
 */
static_control static_controlize_unstructured(u)
unstructured u;
{
	bool		is_static = true;
	list		blocs = NIL;
	static_control	sc, ret_sc = static_control_undefined;

	pips_debug(3, "begin UNSTRUCTURED\n");
	control_map_get_blocs(unstructured_control(u), &blocs ) ;
	blocs = gen_nreverse( blocs ) ;

	FOREACH(CONTROL, ctl, blocs) {
	  statement stmt = control_statement(ctl);
	  sc	= static_controlize_statement( stmt );
	  pips_assert("stmt is consistent", statement_consistent_p(stmt));
	  SET_STATEMENT_MAPPING( Gstatic_control_map, stmt, sc );
	  //fprintf(stdout, "stmt=%p, sharing=%p\n", stmt, sc);
	  is_static = is_static && static_control_yes( sc );
	}

	gen_free_list(blocs);
	ret_sc = make_static_control( is_static,
                                sc_list_of_entity_dup(Gstructure_parameters),
                                sc_list_of_loop_dup(Genclosing_loops),
				sc_list_of_exp_dup(Genclosing_tests) );

	pips_debug(7,
"\n Returning static_control : \n bool   : %s \n params : %s \n loops  : %zd \n tests  : %zd \n ",
                 ((is_static)?"TRUE":"FALSE"),
                 print_structurals( Gstructure_parameters ),
                 gen_length( Genclosing_loops ),
		 gen_length( Genclosing_tests ) );

	pips_debug(3, "end UNSTRUCTURED\n");
	return( ret_sc );
}



 /* the following data structure describes an io intrinsic function: its
    name */

typedef struct IOIntrinsicDescriptor {
    string name;
} IOIntrinsicDescriptor;

static IOIntrinsicDescriptor IOIntrinsicDescriptorTable[] = {
    {"WRITE"},
    {"REWIND"},
    {"BACKSPACE"},
    {"OPEN"},
    {"CLOSE"},
    {"READ"},
    {"BUFFERIN"},
    {"BUFFEROUT"},
    {"ENDFILE"},
    {"INQUIRE"},
    {IMPLIED_DO_NAME},

    {NULL}
};


/*=======================================================================*/
/* bool io_filter(st)
 *
 * Tests if the statement is a call to an IO function. In such a case, we
 * put the statement inside a comment.
 *
 * AP, oct 9th 1995
 */

static bool io_filter(st)
statement st;
{
  bool res_b;
  instruction ins;

  res_b = true;
  ins = statement_instruction(st);

  if(instruction_tag(ins) == is_instruction_call) {
    call ca = instruction_call(ins);
    entity e = call_function(ca);

    /* There can be no statement inside a call, so gen_recurse() do not
       need to go further. */
    res_b = false;

    if(value_tag(entity_initial(e)) == is_value_intrinsic) {
      const char* s = entity_local_name(e);
      IOIntrinsicDescriptor *pid = IOIntrinsicDescriptorTable;
      bool found = false;

      while ((pid->name != NULL) && (!found)) {
	if (strcmp(pid->name, s) == 0) {
	  char         *comment;
	  statement    stat;
	  list lstat;

	  comment = (char*) malloc(64);
	  sprintf(comment, "C  ");
	  /* FI: I'm not sure about the fourth argument of
	     words_call() */
	  sprintf(comment, "%s %s", comment,
		  words_to_string(words_call(ca, 0, true, true, NIL)));
	  sprintf(comment, "%s\n", comment);

	  pips_assert("no buffer overflow", strlen(comment)<64);

	  stat = make_statement(entity_empty_label(),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				comment,
				make_instruction_block(NIL),
				NIL, // No local declarations
				NULL, // null or empty string...
				empty_extensions (), make_synchronization_none());
	  lstat = CONS(STATEMENT, stat, NIL);

	  statement_instruction(st) = make_instruction_block(lstat);
	  /* Do not forget to move forbidden information associated with
	     block: */
	  fix_sequence_statement_attributes(st);

	  found = true;
	}
	pid += 1;
      }
    }
  }

  return(res_b);
}


/*==================================================================*/
/* void static_controlize((char*) mod_name)			AL 05/93
 * Computes the static_control attached to module-name : mod_name.
 */
bool static_controlize(string mod_name)
{
	statement	mod_stat;
	instruction	mod_inst;
	entity		ent;
	list		formal_integers;
	static_control	sc;

	debug_on("STATIC_CONTROLIZE_DEBUG_LEVEL");

	ifdebug(2)
	  user_log("\n\n *** STATIC CONTROLIZE CODE for %s\n", mod_name);

	/*
	 * Set the current language
	 */
	entity module = module_name_to_entity(mod_name);


	value mv = entity_initial(module);
	if(value_code_p(mv)) {
	  code c = value_code(mv);
	  set_prettyprint_language_from_property(language_tag(code_language(c)));
	}

	Gcount_nsp = 0;
	Gcount_nub = 0;
	Gstructure_parameters	= (list) NIL;
	Genclosing_loops	= (list) NIL;
	Genclosing_tests	= (list) NIL;
	Gscalar_written_forward = (list) NIL;
	Gforward_substitute_table = hash_table_make( hash_pointer, 0 );
	hash_warn_on_redefinition();
	Gstatic_control_map = MAKE_STATEMENT_MAPPING();

	ent = local_name_to_top_level_entity(mod_name);
	set_current_module_entity(ent);
	formal_integers = sc_entity_to_formal_integer_parameters( ent );
	pips_debug(7, "\n formal integers : %s \n",
		print_structurals( formal_integers ));

	mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

	mod_inst = statement_instruction( mod_stat );

	/* HAS TO BE REMOVED AS SOON AS POSSIBLE: as the IOs are not
	   treated as they would, all the instructions containing IOs are
	   put inside comments. AP, oct 9th 1995 */
	gen_recurse(mod_inst, statement_domain, io_filter, gen_null);
	/* Normalization of all loops */
	/* Modification: loop_normalize_of_statement is used instead of */
	/* loop_normalize_of_unstructured since we cannot ensure that */
	/* mod_inst is an unstructured --11th Dec 1995, DB */

	/***
	   D.K desactivate  the normalization of loops to preserve the
	   original code
	*/

	/*loop_normalize_of_statement(mod_stat,
				       Gforward_substitute_table,
				       &Genclosing_loops,
				       &Genclosing_tests,
				       &Gscalar_written_forward,
				       &Gcount_nlc);
	*/




	/* The code has been modified, so the orderings are recomputed. */
	module_reorder( mod_stat );
	verify_structural_parameters(formal_integers,
				     &Gscalar_written_forward);

	Genclosing_loops      = (list) NIL;
	Genclosing_tests      = (list) NIL;

	/* We compute the static control infos for each instruction. */
	/* Same remark as before --DB */

	sc = static_controlize_statement(mod_stat);
	pips_assert("controlized statement mod_stat is consistent",
		    statement_consistent_p(mod_stat));
	pips_assert("static_control sc is consistent",
		    static_control_consistent_p(sc));

	stco_renumber_code( mod_stat, 0 );

	SET_STATEMENT_MAPPING( Gstatic_control_map, mod_stat, sc );

	DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), (char*) mod_stat);
	DB_PUT_MEMORY_RESOURCE(DBR_STATIC_CONTROL, strdup(mod_name),
			(char*) Gstatic_control_map);

	ifdebug(2)
	  user_log("\n\n *** STATIC CONTROLIZE CODE done\n");

	reset_current_module_entity();

	debug_off();
	// Reset of the assigned_var list after the phase is complete
	gen_free_list(assigned_var);
	assigned_var = NIL;

	return(true);
}

/*==================================================================*/





/*==================================================================*/
/* list loop_normalize_of_loop((loop) l, hash_table fst, list *ell, *etl,
 * *swfl, int *Gcount_nlc) AL 04/93
 *
 * FI/SG Question: why change the loop index when it is not necessary?
 * To have a uniform behavior when it is necessary?
 */
list loop_normalize_of_loop( l, fst, ell, etl, swfl, Gcount_nlc)
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
  statement       before_stmt = make_continue_statement(entity_empty_label());
  statement       end_stmt = statement_undefined;
  list            stmt_list = NIL;

  pips_debug(4, "begin LOOP\n");

  loop_label( l ) = entity_empty_label();
  /* loop_body( l ) = make_block_with_stmt( loop_body( l ));*/
  loop_body( l ) = make_block_with_stmt_if_not_already( loop_body( l ));
  index = loop_index( l );

  /* If it is not a constant step, we just normalize the loop body */
  if(!normalizable_loop_p(l)) {
    ADD_ELEMENT_TO_LIST(*swfl, ENTITY, index);
    (void) loop_normalize_of_statement(loop_body( l ), fst, ell,
		    etl, swfl, Gcount_nlc);
    return( make_undefined_list() );
  }

  lr = loop_range( l );
  rl = range_lower( lr );
  ru = range_upper( lr );
  ri = range_increment( lr );

  /* new upper bound, or at least iteration count */
  nub =   make_op_exp(DIVIDE_OPERATOR_NAME,
		      make_op_exp(PLUS_OPERATOR_NAME,
				  make_op_exp(MINUS_OPERATOR_NAME,
					      copy_expression(ru),
					      copy_expression(rl)),
				  copy_expression(ri)),
		      copy_expression(ri));
  nub2 = copy_expression(nub);

  ADD_ELEMENT_TO_LIST( stmt_list, STATEMENT, before_stmt );

  /* Generate the new loop index and the new loop bounds */
  nlc_ent = make_nlc_entity(Gcount_nlc);
  ADD_ELEMENT_TO_LIST(*swfl, ENTITY, nlc_ent);
  nlc_exp = make_entity_expression( nlc_ent, NIL);
  loop_index( l ) = nlc_ent;
  if(fortran_module_p(get_current_module_entity())) {
    range_lower( lr ) = int_to_expression( 1 );
    range_upper( lr ) = nub2;
  }
  else {
    /* assume C */
    range_lower( lr ) = int_to_expression( 0 );
    range_upper( lr ) = make_op_exp(MINUS_OPERATOR_NAME, nub2,
				    int_to_expression(1));
  }
  range_increment( lr ) = int_to_expression( 1 );

  /* Generate the change of basis expression: the new index starts at
     0 in C and 1 in Fortran:

     old_index = rl + (new_index * ri) // C

     old_index = rl + (new_index * ri) - ri // Fortran

 */
  new_index_exp = make_op_exp(MULTIPLY_OPERATOR_NAME,
			      copy_expression(ri),
			      nlc_exp);
  if(fortran_module_p(get_current_module_entity())) {
    new_index_exp = make_op_exp(MINUS_OPERATOR_NAME,
				new_index_exp,
				copy_expression(ri));
  }
  new_index_exp = make_op_exp(PLUS_OPERATOR_NAME,
			      new_index_exp,
			      copy_expression(rl));
  hash_put(fst, (char*) index, (char*) new_index_exp);

  /* Compute the value of the index when the loop is exited: exp_max */
  nub3 = copy_expression( nub );
  if ( expression_constant_p( nub3 )) {
    int upper = expression_to_int( nub3 );
    if ( upper > 0 )
      /* nub3 is not used any longer */
      exp_max = int_to_expression( upper );
  }
  else {
    max_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						      MAX_OPERATOR_NAME),
				 entity_domain);
    /* FI: Why copy nub? it does not seem used anywhere else. */
    exp_max = make_max_exp(max_ent, copy_expression( nub ),
			   int_to_expression( 0 ));
  }
  if ( exp_max == expression_undefined )
    exp_plus = copy_expression( rl );
  else
    exp_plus = make_op_exp(PLUS_OPERATOR_NAME,
			   make_op_exp( MULTIPLY_OPERATOR_NAME,
				       copy_expression( ri ),
				       exp_max),
			   copy_expression( rl ));
  index_exp = make_entity_expression( index, NIL );
  end_stmt = make_assign_statement(copy_expression(index_exp), exp_plus );
  ADD_ELEMENT_TO_LIST( stmt_list, STATEMENT, end_stmt);

  loop_normalize_of_statement(loop_body(l), fst , ell, etl, swfl, Gcount_nlc);

  hash_del(fst, (char*) index );
  pips_debug(4, "end LOOP\n");
  return( stmt_list );
}


/*==================================================================*/
/* list loop_normalize_of_statement(statement s, hash_table fst, list
 * *ell, *etl, *swfl, int *Gcount_nlc): Normalization of a statement.
 *
 * Before walking down the statements, we forward-substitute the
 * new-loop-counters on each type of statements.
 * We then return a list of two statements to be put
 * before and after statement 's'. These two new statements
 * are generated by loops when they are treated.
 * See document for more detail, section : "Normalisation des boucles".
 */
list loop_normalize_of_statement(s, fst, ell, etl, swfl, Gcount_nlc)
statement s;
hash_table fst; /* forward substitute table */
list *ell,  /* enclosing loops list */
     *etl,  /* enclosing tests list */
     *swfl; /* scalar written forward list */
int *Gcount_nlc;
{
  instruction	inst = statement_instruction(s);
  list		return_list = NIL;

  pips_assert("statement s is consistent\n", statement_consistent_p(s));

  pips_debug(3, "begin STATEMENT\n");
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
	    list	insert_stmt;

	    stmt = STATEMENT(CAR(tail));
	    insert_stmt = loop_normalize_of_statement(stmt, fst, ell,
						      etl, swfl, Gcount_nlc);
	    before_stmt = STATEMENT(CAR( insert_stmt ));
	    after_stmt  = STATEMENT(CAR(CDR( insert_stmt )));

	    if( before_stmt != statement_undefined)
	      ADD_ELEMENT_TO_LIST( head, STATEMENT, before_stmt);
	    ADD_ELEMENT_TO_LIST( head, STATEMENT, stmt );
	    if (after_stmt != statement_undefined )
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
	/*test_true(t) = make_block_with_stmt( test_true(t) );*/
	test_true(t) = make_block_with_stmt_if_not_already(test_true(t));
	/* test_false(t) = make_block_with_stmt( test_false(t) );*/
	test_false(t) = make_block_with_stmt_if_not_already( test_false(t) );
	loop_normalize_of_statement(test_true(t), fst, ell,
				    etl, swfl, Gcount_nlc);
	loop_normalize_of_statement(test_false(t), fst, ell,
				    etl, swfl, Gcount_nlc);
	gen_remove(etl, (gen_chunk*) test_condition( t ));
	break;
      }
    case is_instruction_loop :
      {
	(void) forward_substitute_in_anyloop(instruction_loop(inst), fst);
	ADD_ELEMENT_TO_LIST(*ell, LOOP, instruction_loop( inst ));
	return_list = loop_normalize_of_loop(instruction_loop(inst), fst, ell,
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
	loop_normalize_of_unstructured(instruction_unstructured(inst), fst, ell,
				       etl, swfl, Gcount_nlc);
	break;
      }
    default : pips_internal_error("Bad instruction tag");
    }
  pips_debug(3, "end STATEMENT\n");
  return( return_list );
}


/*==================================================================*/
/* void loop_normalize_of_unstructured(unstructured u, fst): Normalization
 * of an unstructured instruction.
 */
void loop_normalize_of_unstructured(u, fst, ell, etl, swfl, Gcount_nlc)
unstructured u;
hash_table fst; /* forward substitute table */
list *ell,  /* enclosing loops list */
     *etl,  /* enclosing tests list */
     *swfl; /* scalar written forward list */
int *Gcount_nlc;
{
	list blocs = NIL, lc;
	list insert_stmts;

	pips_debug(2, "begin UNSTRUCTURED\n");
	control_map_get_blocs(unstructured_control(u), &blocs ) ;
	blocs = gen_nreverse( blocs ) ;

	for(lc = blocs; lc != NIL; lc = CDR(lc)) {
		list   		head = NIL;
		control 	ctl;
		statement	before_stmt, after_stmt, stmt;

		ctl		= CONTROL(CAR( lc ));
		stmt		= control_statement( ctl );
		insert_stmts	= loop_normalize_of_statement(stmt, fst,
							      ell, etl,
							      swfl,
							      Gcount_nlc);
		before_stmt	= STATEMENT(CAR( insert_stmts ));
		after_stmt	= STATEMENT(CAR(CDR( insert_stmts )));

	   if (!undefined_statement_list_p( insert_stmts )) {
		if( before_stmt != statement_undefined)
			ADD_ELEMENT_TO_LIST( head, STATEMENT, before_stmt );
		ADD_ELEMENT_TO_LIST( head, STATEMENT, stmt );
		if (after_stmt != statement_undefined )
			ADD_ELEMENT_TO_LIST( head, STATEMENT, after_stmt );
		/*stmt = make_block_with_stmt( stmt );*/
		stmt = make_block_with_stmt_if_not_already(stmt);
		instruction_block(statement_instruction( stmt )) = head;
		head = NIL;
	   }
	}

	gen_free_list(blocs);
	pips_debug(2, "end UNSTRUCTURED\n");
}

