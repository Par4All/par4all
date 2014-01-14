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
/* -- atomizer.c
 *
 * package atomizer :  Alexis Platonoff, juin 91
 * --
 *
 * These functions produce atomic instructions.
 *
 * An atomic instruction is an instruction that whether, loads a variable
 * into a temporary variable, stores a variable from a temporary variable
 * or computes numericals operations upon temporary variables.
 *
 * The scalar variables have a special treatment. Indeed, these variables
 * are loaded in temporaries and kept in them while they are used or defined.
 * It is only when they are no more used that they are stored back.
 * This treatment is done with the dependence graph.
 *
 * Note : in the following, we'll distinguish two kinds of variables: the
 *        memory variables and the temporary variables. The firsts will be
 *        called variables, the latter temporaries. The temporaries should
 *        not appear in the dependence graph; ie. that there can not be any
 *        dependences upon the temporaries. The temporaries can also be
 *        called registers.
 *
 * This phase produces variables and temporaries. The variables produced are
 * prefixed : "AUX". The temporaries produced are prefixed : "TMP".
 * Another kind of entities can be encounter, the NLCs. They have the same
 * status as the temporaries, ie. they should not appear in the dependence
 * graph. More, they have another property : an expression that contains only
 * NLCs and is integer linear, is not decomposed by the atomizer, ie. it is
 * considered as a constant.
 */

#include "local.h"
#include "expressions.h"

/* Gives the size of the hash table named "MemToTmp". */
#define MEM_TO_TMP_SIZE 100

/* Useful for atomizer_of_expression(). It tells if the function can return
 * a variable (MEM_VAR) or if it must return a temporary.
 */
#define MEM_VAR 1
#define NO_MEM_VAR 0

/* FI: the following global variables are not declared STATIC because
 * they also are used in codegen.c
 */

/* This global variable is used for the modification of the control graph,
 * see commentaries of atomizer_of_unstructured() in this file.
 */
list l_inst = list_undefined;

/* These lists memorize all the new created entities of each type. They
 * are used for the declarations of the temporaries and the auxiliaries.
 */
list integer_entities = list_undefined;
list real_entities = list_undefined; 
list complex_entities = list_undefined;
list logical_entities = list_undefined;
list double_entities = list_undefined;
list char_entities = list_undefined;

/* A hash table to map temporary variables (entities)
 * to memory variables (entities).
 */
hash_table MemToTmp = hash_table_undefined;

/* Dependence graph of the current module. */
static graph mod_dg = graph_undefined;

/*============================================================================*/
/* static void initialize_global_variables(char *mod_name) : Initializes the
 * global variables used through out all the computation of atomizer.
 *
 */
static void initialize_global_variables(mod_name)
char *mod_name;
{
    /* set_current_module_entity(local_name_to_top_level_entity(mod_name)); */

    /* The last argument says if the resource is to be modified or not :
     * - false : it is not modified
     * - true  : it is modified
     */
    set_rw_effects((statement_effects)
		   db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, false));

    mod_dg = (graph) db_get_memory_resource(DBR_DG, mod_name, true);

    MemToTmp = hash_table_make(hash_pointer, MEM_TO_TMP_SIZE);

    reset_rw_effects();

    integer_entities = NIL;
    real_entities = NIL;
    logical_entities = NIL;
    complex_entities = NIL;
    double_entities = NIL;
    char_entities = NIL;
}

static void reset_global_variables()
{
    /* reset_current_module_entity(); */

    mod_dg = graph_undefined;

    MemToTmp = hash_table_undefined;

    integer_entities = list_undefined;
    real_entities = list_undefined;
    logical_entities = list_undefined;
    complex_entities = list_undefined;
    double_entities = list_undefined;
    char_entities = list_undefined;
}

static  entity build_new_variable(entity module, basic b)
{
    entity ent = make_new_module_variable(module,0);
    AddEntityToDeclarations( ent,module); 
    return ent;
}

static bool  indirection_test(reference ref, expression expr)
{
    return(!normalized_linear_p(NORMALIZE_EXPRESSION(expr)));
}

void normalize_wp65_code(statement stat)
{
    normalize_all_expressions_of(stat);
    atomize_as_required(stat,
			indirection_test,    /* reference test */
			(bool (*)(call,expression))     gen_false,           /* function call test */
			(bool (*)(test,expression))     gen_false,           /* test condition test */
			(bool (*)(range,expression))    gen_false,           /* range arguments test */
			(bool (*)(whileloop,expression))gen_false,           /* whileloop condition test */
			build_new_variable); /* new variable */
}

static void rm_db_block(statement stat)
{
    instruction inst1 = statement_instruction(stat);
    if ( instruction_block_p(inst1)) {
	list lt =  instruction_block(inst1);
	list newl = NIL;
	MAPL (lt2, {
	    if (instruction_block_p(statement_instruction(STATEMENT(CAR(lt2)))))
	    {
		MAPL(lt3, {
		    newl=gen_nconc(newl,CONS(STATEMENT,STATEMENT(CAR(lt3)),NIL));
		},
		     instruction_block(statement_instruction(STATEMENT(CAR(lt2)))));
	    }
	    else newl = gen_nconc(newl,CONS(STATEMENT,STATEMENT(CAR(lt2)),NIL));
	},
	      lt);
	instruction_block(inst1) = newl;
	ifdebug(8) {
	    entity module =get_current_module_entity();
	    fprintf(stderr,"statement without db blocks \n");
	    print_text(stderr,
		       text_statement(module,0,stat,NIL));
	}
    }
}

static statement rm_block_block_statement(statement stat)
{
    gen_recurse(stat, statement_domain, gen_true, rm_db_block);

    return(stat);
}



/*============================================================================*/
/* void atomizer(const char* module_name): computes the translation of Fortran
 * instructions into Three Adresses Code instructions.
 *
 * This translation is done after two pre-computations :
 *        _ The first one puts all the integer linear expressions into a
 *          normal pattern (see norm_exp.c).
 *        _ The second one removes all the defs with no def-use dependence.
 *
 * Also, after the atomization, the module statement is reordered, and
 * declarations are made for the new variables and temporaries.
 *
 * The atomization uses the CODE, CUMULATED_EFFECTS and DG (dependence graph)
 * resources.
 *
 * Called functions:
 *       _ module_body_reorder() : control/control.c
 */
bool atomizer(mod_name)
char *mod_name;
{
    statement mod_stat;
    entity module;
    pips_user_warning("this transformation is being obsoleted by SIMD_ATOMIZER\nIt is no longer maintained and is likely to crash soon\n");

    debug_on("ATOMIZER_DEBUG_LEVEL");

    if(get_debug_level() > 0)
	user_log("\n\n *** ATOMIZER for %s\n", mod_name);

    mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

    

    set_current_module_statement(mod_stat);
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,mod_name,true));

    module = local_name_to_top_level_entity(mod_name);
    set_current_module_entity(module);

    if (get_bool_property("ATOMIZE_INDIRECT_REF_ONLY"))
    { 
	normalize_wp65_code(mod_stat);
	mod_stat = rm_block_block_statement(mod_stat); 
	module_reorder(mod_stat);
    } 
    else {


	initialize_global_variables(mod_name);

	/* COMPUTATION */

	/* All the expressions are put into a normal pattern : the NLCs in the
	 * innermost parenthesis.
	 */
	normal_expression_of_statement(mod_stat);

	/* All the defs with no def-use dependence are removed. */
	defs_elim_of_statement(mod_stat, mod_dg);

	/* Module is atomized. */
	atomizer_of_statement(mod_stat, (Block *) NULL);

	/* We reorder the module. It is necessary because new statements have been
	 * generated. The reordering will permit to reuse the generated code for
	 * further analysis.
	 */
	module_reorder(mod_stat);

	/* We declare the new variables and the new temporaries. */
	/* insert_new_declarations(mod_name); */
	discard_module_declaration_text(module);

	reset_global_variables();

    }
    /* We save the new CODE. */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();

    if(get_debug_level() > 0)
	user_log("\n\n *** ATOMIZER done\n");

    debug_off();

    return(true);
}



/*============================================================================*/
/* void atomizer_of_unstructured(unstructured u): Computes the transformation
 * of the unstructured "u".
 *
 * This unstructured contains the control graph.
 * In the control graph, some nodes (of Newgen type "control") have
 * a statement that is not a "block" of statements. As we need, almost
 * always, to add new instructions before the IFs and DOs, we have to change
 * the control graph by adding nodes :
 *        For each node V with a "control_statement" that is not a "block",
 *        we generate a node Vnew that contain a "block" with no instruction.
 *        This node Vnew has the same predecessors as V, and has V as
 *        successor; the node V keeps the same successors, but his
 *        predecessor became Vnew.
 *
 * So, for each such node V, when the translation produces instructions,
 * we need to know the corresponding Vnew, in which we put the
 * new instructions.
 *
 * That is why we use a global variable "l_inst" and a code of CONTROL_MAP
 * with some modifications (see below).
 * The list "l_inst" contains the instructions for which we create Vnew.
 * The modified code of CONTROL_MAP uses a function atom_get_blocs() that
 * creates new nodes when needed, and updates the list "l_inst".
 */
void atomizer_of_unstructured( u)
unstructured u ;
{
    extern list l_inst;

    list blocs = NIL ;
    l_inst = NIL;

    debug(1, "atomizer_of_unstructured", "begin UNSTRUCTURED\n");

    /* The following code is a modification of CONTROL_MAP,
     * from control/control.h. This code uses atom_get_blocs() which is
     * a modification of get_blocs() (in control/control.h).
     */
{
    cons *_cm_list = blocs ;
    if( _cm_list == NIL ) 
    {
	atom_get_blocs(unstructured_control(u), &_cm_list ) ;
	_cm_list = gen_nreverse( _cm_list ) ;
    }
    MAPL( _cm_ctls, {control ctl = CONTROL( CAR( _cm_ctls )) ;
			(void) find_control_block(ctl);
			hash_table_clear(MemToTmp);
			atomizer_of_statement(control_statement(ctl),
					      (Block *) NULL) ;},
	 _cm_list ) ;
    if( blocs == NIL )
        blocs = _cm_list ;
}

    gen_free_list( blocs ) ;

    debug(1, "atomizer_of_unstructured", "end UNSTRUCTURED\n");
}



/*============================================================================*/
/* void atomizer_of_statement(statement stmt, Block *cb): computes the
 * translation from Fortran to Three Addresses Code (ATOMIZER) of a statement
 * ("stmt").
 *
 * This function can be called in two different cases:
 *    _ If "cb" is NULL, then "stmt" is taken from a control C of the control
 *      graph. In such case, we have to find out where is the block in which
 *      we'll put the statements created by the translation of "stmt".
 *          . if "stmt" is a block, then this is it.
 *          . else, it is the predecessor of C. This control is obtained by
 *            the function find_control_block().
 *    _ Else, "stmt" is one of the statements of a block, its creations are
 *      put in this block.
 *
 * The "cb" variable refers to the current block of statements where "stmt" is.
 */
void atomizer_of_statement(stmt, cb)
statement stmt;
Block *cb;
{
    instruction inst;
    bool stmt_with_remote_control_block = false;
    control c = control_undefined;

    debug(2, "atomizer_of_statement", "begin STATEMENT\n");

    inst = statement_instruction(stmt);

    /* Initialisation of "cb", if it is a NULL pointer. */
    if(cb == (Block *) NULL)
	if (instruction_tag(inst) != is_instruction_block)
	{
	    /* The control in which the created statements are put is not the same
	     * as the control of "stmt".
	     */
	    stmt_with_remote_control_block = true;

	    /* We get the control in which we'll put our new statements. */
	    c = find_control_block(control_undefined);

	    /* We create the structure that will keep the created statements
	     * during the computation, before putting them in the control "c".
	     */
	    cb = (Block *) malloc(sizeof(Block));
	    if (cb == (Block *) NULL)
		user_error("atomizer_of_statement", "Block malloc: no memory left");
	    cb->first = NIL;
	    cb->last = CONS(STATEMENT, stmt, NIL);
	    cb->stmt_generated = false;
	}

    /* Computation of "stmt". */
    switch(instruction_tag(inst))
    {
    case is_instruction_block : { atomizer_of_block(inst); break; }
    case is_instruction_test : { atomizer_of_test(instruction_test(inst), cb);
				   break; }
    case is_instruction_loop : { atomizer_of_loop(instruction_loop(inst), cb);
				   break; }
    case is_instruction_call : { atomizer_of_call(instruction_call(inst), cb);
				   break; }
    case is_instruction_goto : break;
    case is_instruction_unstructured :
    { atomizer_of_unstructured(instruction_unstructured(inst));
	break; }
    default : pips_internal_error("Bad instruction tag");
    }

    /* Updates of the control graph, if the generated statements are not put in
     * the same control as "stmt".
     */
    if(stmt_with_remote_control_block)
    {
	/* The created statements are put in the control just before the control
	 * of the statement that created them.
	 */
	instruction_block(statement_instruction(control_statement(c))) = cb->first;

	/* Memory deallocation */
	if (cb != (Block *) NULL)
	{
	    cb->first = NIL;
	    cb->last = NIL;
	    free((char *) cb);
	}
    }
    debug(2, "atomizer_of_statement", "end STATEMENT\n");
}



/*============================================================================*/
/* void atomizer_of_block(instruction inst): Applies the translation on all
 * the statements of the block of the instruction given in argument.
 *
 * Note: the "instruction_tag" of "inst" must be a block, otherwise, it's a
 *       user error.
 *
 * We enter a new block of statements, so we generate a new variable of
 * type "Block".
 * "last" representes the list of statements not translated yet.
 * "first" representes the list of statements that are translated, plus the
 * statements that were generated by them.
 * The statement being translated (current statement) is the first of the
 * "last" list.
 * When the translation of a statement is done, it is put at the end of the
 * "first" list.
 */
void atomizer_of_block(i)
instruction i;
{
    Block *cb;

    debug(2, "atomizer_of_block", "begin BLOCK\n");

    if (instruction_tag(i) != is_instruction_block)
	user_error("atomizer_of_block", "Instruction is not a block");

    /* Nothing to do!! */
    if(instruction_block(i) == NIL)
	return;

    /* Initialization of the new "Block". */
    cb = (Block *) malloc(sizeof(Block));
    if (cb == (Block *) NULL)
	user_error("atomizer_of_block", "Block malloc: no memory left");
    cb->last = instruction_block(i);
    cb->first = NIL;

    /* "cb->last" is the list of the statements not yet visited */
    for(; cb->last != NIL; cb->last = CDR(cb->last) )
    {
	/* Gets the current statement. */
	statement s = STATEMENT(CAR(cb->last));

	/* This current statement has not yet generated another statement. */
	cb->stmt_generated = false; 

	/* Translation of the current statement. */
	atomizer_of_statement(s, cb);

	/* The current statement is put at the end of the "first" list. */
	cb->first = gen_nconc(cb->first, CONS(STATEMENT, s, NIL));
    }

    /* Since there could have been creations of statements in the list "cb->first"
     * we have to update the block of the instruction.
     */
    instruction_block(i) = cb->first;

    /* Memory deallocation */
    cb->first = NIL;
    free( (char *) cb);

    debug(2, "atomizer_of_block", "end BLOCK\n");
}



/*============================================================================*/
/* void atomizer_of_test(test t, Block *cb): Applies the translation on an 
 * instruction test.
 *
 * It consists in translating the three arguments of the test instruction :
 * the condition expression and the two conditional statements (true, false).
 *
 * The condition argument is an expression which may contain a logical
 * intrinsic operator : "x op y" or "op x", with "op" in (<, =, >=, etc).
 * In such case, the tranlation does not assign a temporary for the call
 * expression associated with the operator, it only translates the arguments
 * of the logical operator.
 *
 * The variable "cb" memorises the information about the block of statements
 * that contains the test statement.
 *
 * Called functions :
 *       _  make_block_with_stmt() : loop_normalize/utils.c
 */
void atomizer_of_test(t, cb)
test t;
Block *cb;
{
    expression cond = test_condition(t);

    debug(2, "atomizer_of_test", "begin TEST\n");

    if (expression_intrinsic_operation_p(cond))
	/* If the expression is a call to a intrinsic operation, 
	 * only its arguments
	 * are translated.
	 * Note : it is not tested that the intrinsic is a logical
	 * operator. In fact,
	 * Fortran requires it.
	 */
    {
	call c = syntax_call(expression_syntax(cond));
	call_arguments(c) = (list) atomizer_of_expressions(call_arguments(c), cb);

	debug(3, "atomizer_of_test", "CONDITION: %s\n",
	      entity_local_name(call_function(c)));
    }
    else
	/* Else, the conditional expression is translated, 
	 * and the returned expression
	 * must be a temporary (NO_MEM_VAR).
	 */
	test_condition(t) = atomizer_of_expression(cond, cb, NO_MEM_VAR);

    /* Afterwards, the two conditional statements are translated. If one of these
     * statements is not a block of statements, then it is put inside one (the
     * resulting block contains only one statement !!).
     */
    debug(2, "atomizer_of_test", "begin TEST IF\n");
    test_true(t) = make_block_with_stmt_if_not_already(test_true(t));
    atomizer_of_statement(test_true(t), cb);

    debug(2, "atomizer_of_test", "begin TEST ELSE\n");
    test_false(t) = make_block_with_stmt_if_not_already(test_false(t));
    atomizer_of_statement(test_false(t), cb);

    debug(2, "atomizer_of_test", "end   TEST\n");
}



/*============================================================================*/
/* void atomizer_of_loop(loop l, Block *cb): Applies the translation on an 
 * instruction loop.
 *
 * All written variables of the loop are removed from MemToTmp.
 *
 * The variable "cb" memorizes the information about the block of statements
 * that contains the loop statement.
 *
 * Called functions:
 *       _ entity_scalar_p() : ri-util/entity.c
 *       _ make_block_with_stmt() : loop_normalize/utils.c
 */
void atomizer_of_loop(l, cb)
loop l;
Block *cb;
{

    list cumu_effs, lce;
    statement stmt;
    range r;

    debug(2, "atomizer_of_loop", "begin LOOP: %s\n",
	  entity_local_name(loop_index(l)));

    /* We have to remove from MemToTmp all the (scalars) variables that are
     * written in this loop.
     */
    stmt = STATEMENT(CAR(cb->last));
    cumu_effs = load_rw_effects_list(stmt);
    for(lce = cumu_effs; lce != NIL; lce = CDR(lce))
    {
	effect eff = EFFECT(CAR(lce));
	entity eff_e = reference_variable(effect_any_reference(eff));
	if( entity_scalar_p(eff_e) &&
	   (action_tag(effect_action(eff)) == is_action_write) )
	{
	    (void) hash_del(MemToTmp, (char *) eff_e);
	}
    }

    /* Translation of the three expressions of the loop range. */
    r = loop_range(l);
    range_lower(r) = atomizer_of_expression(range_lower(r), cb, NO_MEM_VAR);
    range_upper(r) = atomizer_of_expression(range_upper(r), cb, NO_MEM_VAR);
    range_increment(r) = atomizer_of_expression(range_increment(r), cb, NO_MEM_VAR);

    /* Afterwards, the body statement of the loop is translated. If this
     * statement is not a block of statements, then it is put inside one (the
     * resulting block contains only one statement !!).
     */
    loop_body(l) = make_block_with_stmt_if_not_already(loop_body(l));
    atomizer_of_statement(loop_body(l), cb);

    debug(2, "atomizer_of_loop", "end LOOP\n");
}



/*============================================================================*/
/* void atomizer_of_call(call c, Block *cb): Applies the translation on an 
 * instruction call.
 *
 * The two main kinds of call are:
 *    _ external call, ie user_defined function or subroutine.
 *    _ intrinsic call.
 *
 * The variable "cb" memorises the information about the block of statements
 * that contains the call statement.
 */
void atomizer_of_call(c, cb)
call c;
Block *cb;
{
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    string n = entity_name(e);

    switch (t)
    {
    case is_value_code: { atomizer_of_external(c, cb); break; }
    case is_value_intrinsic: {
	if (call_arguments(c) == NIL)
	    /* Evite d'atomiser un parame`tre inexistant ( :-) )
	       d'un intrinsic sans argument, du style * dans :
	       write ... FMT=*...
	       RK, 24/02/1994. */
	    break;
	atomizer_of_intrinsic(c, cb);
	break;
			     }
    case is_value_symbolic: break;
    case is_value_constant: break;
    case is_value_unknown:
	    pips_internal_error("unknown function %s", n);
	    break;
    default: pips_internal_error("unknown tag %d", t);
    }
}



/*============================================================================*/
/* void atomizer_of_intrinsic(call c, block *cb): translates the arguments of
 * the intrinsic function. It treats two cases: assign call, and others.
 *
 * Assign calls are treated differently because the first argument is the
 * left-hand-side, so this argument is translated with the MEM_VAR option.
 */
void atomizer_of_intrinsic(c, cb)
call c;
Block *cb;
{
    if (ENTITY_ASSIGN_P(call_function(c)))
    {
	entity lhs_entity;
	expression lhs, rhs;

	/* Assign expressions. */
	lhs = EXPRESSION(CAR(call_arguments(c)));
	rhs = EXPRESSION(CAR(CDR(call_arguments(c))));

	debug(4, "atomizer_of_intrinsic", "ASSIGN CALL: %s = %s\n",
	      words_to_string(words_expression(lhs, NIL)),
	      words_to_string(words_expression(rhs, NIL)));

	/* If the rhs expression is integer linear and exclusively composed of NLC
	 * variables, it is considered like a constant, ie not translated.
	 * Otherwise, it is translated normaly.
	 */
	if(! nlc_linear_expression_p(rhs))
	    rhs = atomizer_of_expression(rhs, cb, NO_MEM_VAR);

	/* Translation of the lhs expression. We keep the memory variable. */
	lhs = atomizer_of_expression(lhs, cb, MEM_VAR);

	/* The lhs variable is stored, so it is delete from MemToTmp. */
	lhs_entity = reference_variable(syntax_reference(expression_syntax(lhs)));
	(void) hash_del(MemToTmp, (char *) lhs_entity);

	call_arguments(c) = CONS(EXPRESSION, lhs, CONS(EXPRESSION, rhs, NIL));
    }
    else
	/* The call is not an assignment, then each arguments is translated. */
	call_arguments(c) = atomizer_of_expressions(call_arguments(c), cb);
}



/*============================================================================*/
/* void atomizer_of_external(call c, block *cb): Translates the arguments
 * of the call to an external function.
 *
 * In fact, these arguments are kept as memory variable. When, the argument
 * is an expression containing a "call" (a real call to a function and not a
 * call to a constant), this expression is assigned to a new auxiliary
 * variable which takes its place in the list of arguments. It is a variable,
 * not a temporary (see the introduction at the beginning of this file).
 *
 * Called functions :
 *       _ make_entity_expression() : ri-util/util.c
 *       _ make_assign_statement() : ri-util/statement.c
 */
void atomizer_of_external(c, cb)
call c;
Block *cb;
{
    list args, new_args;

    args = call_arguments(c);
    new_args = NIL;

    /* All the argument expressions are scanned. If an expression is not a
     * reference (then, it is a call), we create an auxiliary variable.
     * The expression is assigned to this new variable which is given in
     * argument to the function, instead of the expression.
     * If it is a reference, the corresponfding key (if it exists) in the
     * hash table MemToTmp is deleted.
     */
    for(; args != NIL; args = CDR(args))
    {
	expression ae = EXPRESSION(CAR(args));
	if(syntax_tag(expression_syntax(ae)) == is_syntax_reference)
	{
	    reference ar = syntax_reference(expression_syntax(ae));
	    (void) hash_del(MemToTmp, (char *) reference_variable(ar));
	    reference_indices(ar) = 
		atomizer_of_expressions(reference_indices(ar), cb);
	    new_args = gen_nconc(new_args, CONS(EXPRESSION, ae, NIL));
	}
	else
	{
	    call arg_call = syntax_call(expression_syntax(ae));
	    if(call_constant_p(arg_call))
		new_args = gen_nconc(new_args, CONS(EXPRESSION, ae, NIL));
	    else
	    {
		expression tmp = atomizer_of_expression(ae, cb, MEM_VAR);
		basic aux_basic = basic_of_expression(tmp);
		entity aux_ent = make_new_entity(aux_basic, AUX_ENT);
		expression aux = make_entity_expression(aux_ent, NIL);
		put_stmt_in_Block(make_assign_statement(aux, tmp), cb);
		new_args = gen_nconc(new_args, CONS(EXPRESSION, aux, NIL));
	    }
	}
    }

    call_arguments(c) = new_args;
}



/*============================================================================*/
/* list atomizer_of_expressions(list expl, Block *cb): Applies the
 * translation on a list of expressions.
 *
 * The variable "cb" memorises the information about the block of statements
 * that contains the expressions.
 *
 * Returns a list of expressions containing the translated expressions.
 */
list atomizer_of_expressions(expl, cb)
list expl;
Block *cb;
{
    list newl;
    expression exp, trad_exp;
    for (newl = NIL; expl != NIL; expl = CDR(expl))
    {
	exp = EXPRESSION(CAR(expl));
	trad_exp = atomizer_of_expression(exp, cb, NO_MEM_VAR);
	newl = gen_nconc(newl, CONS(EXPRESSION, trad_exp, NIL)); 
    }
    return newl;
}



/*============================================================================*/
/* expression atomizer_of_expression(expression exp, Block *cb, int mem_var):
 * Applies the translation on an expression.
 *
 * It consists in assigning a new temporary variable to the expression, while
 * the sub-expressions that it may contain are recursively translated.
 *
 * If "exp" is only composed of NLCs variables (eventually a constant term),
 * then the expression is treated like a constant, ie unchanged.
 * NLC means Normalized Loop Counter.
 *
 * The translation of expressions that reference a variable is not done only
 * if the flag "mem_var" has the value "MEM_VAR". Otherwise, the variable is
 * assigned to a temporary variable.
 * The function "atomizer_of_array_indices()" translates the list of indices
 * (when not empty) of the variable given in argument.
 * Note: a scalar variable is an array with an empty list of indices.
 *
 * The variable "cb" memorises the information about the block of statements
 * that contains the expression.
 */
expression atomizer_of_expression(exp, cb, mem_var)
expression exp;
Block *cb;
int mem_var;
{
    expression ret_exp = expression_undefined;
    syntax sy = expression_syntax(exp);
    bool IS_NLC_LINEAR = false;

    debug(5, "atomizer_of_expression", "begin : %s\n",
	  words_to_string(words_expression(exp, NIL)));

    /* An expression that is integer linear and is exclusively composed of NLCs
     * is considered like a constant, ie not translated.
     */
    if(nlc_linear_expression_p(exp))
	IS_NLC_LINEAR = true;

    switch(syntax_tag(sy))
    {
    case is_syntax_reference:
    {
	atomizer_of_array_indices(exp, cb);

	if ( (mem_var == MEM_VAR) || IS_NLC_LINEAR )
	    ret_exp = exp;
	else
	    ret_exp = assign_tmp_to_exp(exp, cb);
	break;
    }
    case is_syntax_call:
    {
	call c = syntax_call(sy);

	/* Two cases : _ a "real" call, ie an intrinsic or external function
	 *             _ a constant
	 */
	if( ! call_constant_p(c) )
	{
	    if ((call_arguments(c) == NIL)
		&& value_intrinsic_p(entity_initial(call_function(c)))) {
		/* Evite d'atomiser un parame`tre inexistant ( :-) )
		   d'un intrinsic sans argument, du style * dans :
		   write ... FMT=*...
		   RK, 24/02/1994. */
		ret_exp = exp;
		break;
	    }
	    else {
		/* Translates the arguments of the call. */
		if( ! IS_NLC_LINEAR )
		    atomizer_of_call(c, cb);

		/* Generates the assign statement, and put it into the current block. */
		ret_exp = assign_tmp_to_exp(exp, cb);
	    }}
	else				/* Constant value. */
	    ret_exp = exp;
	break;
    }
    case is_syntax_range:
    {
	debug(6, "atomizer_of_expression", " Expression RANGE\n");
	ret_exp = exp;;
	break;
    }
    default : pips_internal_error("Bad syntax tag");
    }
    debug(5, "atomizer_of_expression", "end   : %s\n",
	  words_to_string(words_expression(ret_exp, NIL)));

    return(ret_exp);
}



/*============================================================================*/
/* void atomizer_of_array_indices(expression exp, Block *cb): Applies the
 * translation on an array expression.
 *
 * Only the indices of the array are translated, in order to have a list
 * of temporary variables.
 */
void atomizer_of_array_indices(exp, cb)
expression exp;
Block *cb;
{
    reference array_ref = syntax_reference(expression_syntax(exp));
    list inds = reference_indices(array_ref);

    debug(6, "atomizer_of_array_indices", "begin : %s\n",
	  words_to_string(words_expression(exp, NIL)));

    /* We translate all the indices expressions. */
    reference_indices(array_ref) = atomizer_of_expressions(inds, cb);

    debug(6, "atomizer_of_array_indices", "end : %s\n",
	  words_to_string(words_expression(exp, NIL)));
}
