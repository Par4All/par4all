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
#include <malloc.h>
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
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
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
int			Gcount_nsp;
int			Gcount_nub;
list			Gstructure_parameters;
static list			Genclosing_loops;
static list			Genclosing_tests;
static list			Gscalar_written_forward;
static hash_table 		Gforward_substitute_table;
statement_mapping	Gstatic_control_map;


/*=================================================================*/
/* static_control static_controlize_call( (call) c )		AL 05/93
 * It computes now just a static-control assignement if it is      
 * an ASSIGN statement.                                           
 */
static_control static_controlize_call( c )
call c;
{
	static_control 	sc;
	bool		b;
	
	debug( 3, "static_controlize_call", "begin CALL\n");
	debug( 7, "static_controlize_call", 
		"call : %s\n", entity_local_name( call_function(c) ) );

	b = splc_linear_access_to_arrays_p(call_arguments(c),
					   &Genclosing_loops);

	sc = make_static_control( b,
			sc_list_of_entity_dup( Gstructure_parameters ),
			sc_list_of_loop_dup( Genclosing_loops ),
			sc_list_of_exp_dup( Genclosing_tests ));
	debug( 3, "static_controlize_call", "end CALL\n");
	return( sc );	
}

/*==================================================================*/
/* static_control static_controlize_loop( (loop) l )
 * It computes the loop's static_control.
 */
static_control static_controlize_loop(l)
loop l;
{
	static_control 	sc;  
	expression	low, up;

	debug(3, "static_controlize_loop", "begin LOOP\n");

	low = range_lower( loop_range( l ) );
	up  = range_upper( loop_range( l ) );
	ADD_ELEMENT_TO_LIST(Genclosing_loops, LOOP, l); 
	sc = static_controlize_statement( loop_body( l ) );
	if (	   !normalizable_loop_p(l) 
		|| !splc_feautrier_expression_p(low, &Genclosing_loops)
		|| !splc_feautrier_expression_p(up, &Genclosing_loops)  ) {
		static_control_yes( sc ) = FALSE;
	}
	SET_STATEMENT_MAPPING(Gstatic_control_map, loop_body(l), sc);
	gen_remove( &Genclosing_loops, l ); 

	debug(3, "static_controlize_loop", "end LOOP\n");
	return( sc );
}



/*==================================================================*/
/* static_control static_controlize_statement((statement) s)	AL 05/93
 * Computes s's static_control
 */
static_control static_controlize_statement(s)
statement s;
{
bool		is_static = TRUE, static_test = FALSE;
instruction 	inst = statement_instruction(s);
static_control  sc, sc1, sc2;
expression	exp, exp1, exp2;

debug(3, "static_controlize_statement", "begin STATEMENT\n");
debug(7, "static_controlize_statement", 
	"statement_ordering = %d \n", statement_ordering( s ));

switch(instruction_tag(inst))
  {
  case is_instruction_block :
    {
    MAPL( stmt_ptr,
	{
	statement local_stmt = STATEMENT(CAR( stmt_ptr ));
	sc1 = static_controlize_statement( local_stmt );
	SET_STATEMENT_MAPPING( Gstatic_control_map, local_stmt, sc1);
	is_static = ( is_static && static_control_yes( sc1 ));
	},
	instruction_block( inst ) );
    break;
    }
  case is_instruction_test :
    {
    test t = instruction_test(inst);
    forward_substitute_in_exp( &test_condition( t ),
			      Gforward_substitute_table);
    /* We put condition under a normal disjunctive form */
    if ((exp = sc_conditional(test_condition(t), &Genclosing_loops)) !=
	expression_undefined) {
      test_condition( t ) = exp ;
      static_test = TRUE;
    } 
    ADD_ELEMENT_TO_LIST( Genclosing_tests, EXPRESSION, test_condition(t) );
    sc1 = static_controlize_statement(test_true(t));
    SET_STATEMENT_MAPPING( Gstatic_control_map, test_true( t ), sc1);
    gen_remove( &Genclosing_tests, (chunk*) test_condition( t ));

    exp1 = MakeUnaryCall( ENTITY_NOT, expression_dup( test_condition(t) ));
    if ( (exp2 = sc_conditional(exp1, &Genclosing_loops)) ==
	expression_undefined) {
	exp2 = exp1;
    }
    ADD_ELEMENT_TO_LIST( Genclosing_tests, EXPRESSION, exp2 );
    sc2 = static_controlize_statement(test_false(t));
    SET_STATEMENT_MAPPING( Gstatic_control_map, test_false( t ), sc2);
    gen_remove( &Genclosing_tests, (chunk*) exp2 );
    is_static = (  static_control_yes( sc1 ) 
		&& static_control_yes( sc2 )
		&& static_test );
    break;
    }
  case is_instruction_loop :
    {
    loop the_loop = instruction_loop( inst );
    forward_substitute_in_loop( &instruction_loop( inst ),
			       Gforward_substitute_table);
    sc1 = static_controlize_loop( the_loop );
    is_static = static_control_yes( sc1 );
    break;
    }
  case is_instruction_call : 
    {
    call the_call = instruction_call( inst );
    forward_substitute_in_call( &the_call, Gforward_substitute_table );
    /* If it is a redefinition of a SP, substitute it with a new one */ 
    if (get_sp_of_call_p(the_call, Gforward_substitute_table,
			 &Gscalar_written_forward)) {
	/* We are in an assign call case */
	forward_substitute_in_exp(&(EXPRESSION(CAR(call_arguments(the_call)))),
				  Gforward_substitute_table);
	is_static = TRUE; }
    else 
	is_static = static_control_yes(static_controlize_call( the_call ));

    break;
    }
  case is_instruction_goto :
    {
    is_static = FALSE;
    break;
    } 
  case is_instruction_unstructured :
    {
    unstructured local_un = instruction_unstructured( inst );
    is_static = static_control_yes(
			static_controlize_unstructured( local_un ));
    break;
    }
  default : pips_error("static_controlize_statement", "Bad instruction tag");
  }

    sc = make_static_control(is_static, 
                                sc_list_of_entity_dup(Gstructure_parameters),
                                sc_list_of_loop_dup(Genclosing_loops),
				sc_list_of_exp_dup(Genclosing_tests) );

    debug(7, "static_controlize_statement", 
"\n Returning static_control : \n bool   : %s \n params : %s \n loops  : %d \n tests  : %d \n ",
		 ((is_static)?"TRUE":"FALSE"), 
		 print_structurals( Gstructure_parameters ),
		 gen_length( Genclosing_loops ),
		 gen_length( Genclosing_tests ) );
    debug(3, "static_controlize_statement", "end STATEMENT\n");
    return( sc );
}


/*==================================================================*/
/* static_control static_controlize_unstructered((unstructured) u)  AL 05/93
 * Computes an unstructured's static_control
 */
static_control static_controlize_unstructured(u)
unstructured u;
{
	bool		is_static = TRUE;
	list 		blocs = NIL;
	static_control 	sc, ret_sc = static_control_undefined;

	debug(3, "static_controlize_unstructured", "begin UNSTRUCTURED\n");
	control_map_get_blocs(unstructured_control(u), &blocs ) ;
	blocs = gen_nreverse( blocs ) ;

	MAPL( ctl_ptr, 	{
		statement stmt = control_statement(CONTROL(CAR( ctl_ptr )));
		sc 	= static_controlize_statement( stmt );
		SET_STATEMENT_MAPPING( Gstatic_control_map, stmt, sc );
		is_static = is_static && static_control_yes( sc );
		},
	      blocs);

	gen_free_list(blocs);
	ret_sc = make_static_control( is_static, 
                                sc_list_of_entity_dup(Gstructure_parameters),
                                sc_list_of_loop_dup(Genclosing_loops),
				sc_list_of_exp_dup(Genclosing_tests) );

	debug(7, "static_controlize_unstructured",
"\n Returning static_control : \n bool   : %s \n params : %s \n loops  : %d \n tests  : %d \n ",
                 ((is_static)?"TRUE":"FALSE"),
                 print_structurals( Gstructure_parameters ),
                 gen_length( Genclosing_loops ),
		 gen_length( Genclosing_tests ) );

	debug(3, "static_controlize_unstructured", "end UNSTRUCTURED\n");
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

  res_b = TRUE;
  ins = statement_instruction(st);

  if(instruction_tag(ins) == is_instruction_call) {
    call ca = instruction_call(ins);
    entity e = call_function(ca);

    /* There can be no statement inside a call, so gen_recurse() do not
       need to go further. */
    res_b = FALSE;

    if(value_tag(entity_initial(e)) == is_value_intrinsic) {
      string s = entity_local_name(e);
      IOIntrinsicDescriptor *pid = IOIntrinsicDescriptorTable;
      bool found = FALSE;

      while ((pid->name != NULL) && (!found)) {
        if (strcmp(pid->name, s) == 0) {
	  char         *comment;
	  statement    stat;
	  list lstat;

	  comment = (char*) malloc(64);
	  sprintf(comment, "C  ");
	  sprintf(comment, "%s %s", comment,
		  words_to_string(words_call(ca, 0, TRUE)));
	  sprintf(comment, "%s\n", comment);
 
	  stat = make_statement(entity_empty_label(),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				comment, 
				make_instruction_block(NIL));
	  lstat = CONS(STATEMENT, stat, NIL);
 
	  statement_instruction(st) = make_instruction_block(lstat);
	  /* Do not forget to move forbidden information associated with
	     block: */
	  fix_sequence_statement_attributes(st);

	  found = TRUE;
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
boolean static_controlize(mod_name)
char* mod_name;
{
	statement 	mod_stat;
	instruction 	mod_inst;
	entity		ent;
	list		formal_integers;
	static_control	sc;
	int Gcount_nlc;

	debug_on("STATIC_CONTROLIZE_DEBUG_LEVEL");

	if (get_debug_level() > 1) 
        	user_log("\n\n *** STATIC CONTROLIZE CODE for %s\n", mod_name);

	Gcount_nlc = 0;
        Gcount_nsp = 0;
	Gcount_nub = 0;
	Gstructure_parameters 	= (list) NIL;
	Genclosing_loops      	= (list) NIL;
	Genclosing_tests      	= (list) NIL;
	Gscalar_written_forward = (list) NIL;
        Gforward_substitute_table = hash_table_make( hash_pointer, 0 );
        hash_warn_on_redefinition();
	Gstatic_control_map = MAKE_STATEMENT_MAPPING();

	ent = local_name_to_top_level_entity(mod_name);
	set_current_module_entity(ent);
	formal_integers = sc_entity_to_formal_integer_parameters( ent );
	debug(7, "static_controlize", "\n formal integers : %s \n",
		print_structurals( formal_integers ));

	mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
	mod_inst = statement_instruction( mod_stat );

	/* HAS TO BE REMOVED AS SOON AS POSSIBLE: as the IOs are not
	   treated as they would, all the instructions containing IOs are
	   put inside comments. AP, oct 9th 1995 */
	gen_recurse(mod_inst, statement_domain, io_filter, gen_null);

	/* Normalization of all loops */
	/* Modification: loop_normalize_of_statement is used instead of */
	/* loop_normalize_of_unstructured since we cannot ensure that */
	/* mod_inst is an unstructured --11th Dec 1995, DB */
        loop_normalize_of_statement(mod_stat,
				       Gforward_substitute_table,
				       &Genclosing_loops,
				       &Genclosing_tests,
				       &Gscalar_written_forward,
				       &Gcount_nlc);

	/* The code has been modified, so the orderings are recomputed. */
 	module_reorder( mod_stat );  
	verify_structural_parameters(formal_integers,
				     &Gscalar_written_forward);

	Genclosing_loops      = (list) NIL;
	Genclosing_tests      = (list) NIL;

	/* We compute the static control infos for each instruction. */
	/* Same remark as before --DB */
	sc =
	  static_controlize_statement(mod_stat);

	/* Renumber the statements. */
	stco_renumber_code( mod_stat, 0 );

	SET_STATEMENT_MAPPING( Gstatic_control_map, mod_stat, sc );

	DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), (char*) mod_stat);
	DB_PUT_MEMORY_RESOURCE(DBR_STATIC_CONTROL, strdup(mod_name),
			(char*) Gstatic_control_map);

	if (get_debug_level() > 1) 
	        user_log("\n\n *** STATIC CONTROLIZE CODE done\n");

	reset_current_module_entity();
	
	debug_off();

	return(TRUE);
}

/*==================================================================*/





