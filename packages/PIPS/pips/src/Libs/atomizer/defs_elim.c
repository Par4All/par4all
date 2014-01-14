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
/* -- defs_elim.c
 *
 * package atomizer :  Alexis Platonoff, aout 91
 * --
 *
 * Those functions remove the definition instructions with no def-use
 * dependence.
 */

#include "local.h"



/*============================================================================*/
/* static vertex get_vertex_of_statement(graph dg, statement stmt): returns
 * the vertex of "dg" corresponding to "stmt".
 *
 * We scan all the "dg" until we find the "vertex" with the same "ordering"
 * as "stmt".
 */
static vertex 
get_vertex_of_statement(dg, stmt)
graph dg;
statement stmt;
{
    vertex rv = vertex_undefined;
    list dg_vertices = graph_vertices(dg);
    bool not_found = true;

    for(; (dg_vertices != NIL) && not_found ; dg_vertices = CDR(dg_vertices))
    {
	vertex v = VERTEX(CAR(dg_vertices));
	if( vertex_ordering(v) == statement_ordering(stmt) )
	{
	    rv = v;
	    not_found = false;
	}
    }
    return(rv);
}


/*============================================================================*/
/* bool true_dependence_with_entity_p(conflict conf, entity e): returns TRUE
 * if the conflict "conf" is a true dependence upon the entity "e".
 *
 * A true dependence is a conflict with a Write at the "source" and a Read
 * at the "sink".
 *
 * called functions :
 *       _ effect_entity() : ri-util/util.c
 *       _ same_entity_p() : ri-util/util.c
 */
bool 
true_dependence_with_entity_p(conf, e)
conflict conf;
entity e;
{
    effect source_eff, sink_eff;
    entity source_ent, sink_ent;

    source_eff = conflict_source(conf);
    sink_eff = conflict_sink(conf);
    source_ent = effect_entity(source_eff);
    sink_ent = effect_entity(sink_eff);

    debug(6, "true_dependence_with_entity_p", "  CONFLICT : %s --> %s\n",
	  effect_to_string(source_eff),
	  effect_to_string(sink_eff));

    if(! same_entity_p(source_ent, sink_ent))
	pips_internal_error("Source and sink entities must be equal");

    return( same_entity_p(e, source_ent)                               &&
	    (action_tag(effect_action(source_eff)) == is_action_write) &&
	    (action_tag(effect_action(sink_eff)) == is_action_read)       );
}


/*============================================================================*/
/* static bool entity_dynamic_p(entity e): returns true if "e" is a local
 * variable, ie an entity with a storage DYNAMIC.
 *
 * Called_functions :
 *       _ dynamic_area_p() : ri-util/util.c
 */
static bool 
entity_dynamic_p(e)
entity e;
{
    ram r;
    storage s = entity_storage(e);

    if(storage_tag(s) != is_storage_ram)
	return(false);
    r = storage_ram(s);
    if(dynamic_area_p(ram_section(r)))
	return(true);
    return(false);
}


/*============================================================================*/
/* bool defs_elim_of_assign_call(statement assign_stmt, graph dg): returns
 * true if "assign_stmt" is to be eliminated.
 * It is eliminated if the lhs of this assignment verifies two conditions :
 *    1. it is a local variable
 *    2. it is not at the source of a def-use dependence, ie true dependence.
 */
bool 
defs_elim_of_assign_call(assign_stmt, dg)
statement assign_stmt;
graph dg;
{
    call assign_call;
    expression lhs_exp;
    entity lhs_ent;
    vertex stmt_vertex;
    list succs;
    bool true_dep_found = false;

    if(instruction_tag(statement_instruction(assign_stmt)) != is_instruction_call)
	pips_internal_error("Statement must be a CALL");

    assign_call = instruction_call(statement_instruction(assign_stmt));
    if(! ENTITY_ASSIGN_P(call_function(assign_call)))
	pips_internal_error("Call must be an ASSIGN");

    pips_debug(5, "begin ASSIGN : %s\n",
	       words_to_string(words_call(assign_call, 0, true, true, NIL)));

    lhs_exp = EXPRESSION(CAR(call_arguments(assign_call)));
    if(syntax_tag(expression_syntax(lhs_exp)) != is_syntax_reference)
	pips_internal_error("Lhs must be a REFERENCE");

    lhs_ent = reference_variable(syntax_reference(expression_syntax(lhs_exp)));

/* Definitions upon non local (non dynamic) variables are always kept. */
    if(! entity_dynamic_p(lhs_ent) )
	return(false);

/* Gets the vertex of the dependence graph that gives all the edges of
 * which the assign statement is the source.
 */
    stmt_vertex = get_vertex_of_statement(dg, assign_stmt);

/* We scan all the dependences of the assign statement. If at least one
 * true dependence is found, the statement is not removed.
 */
    if(stmt_vertex != vertex_undefined)
    {
	list confs;
	dg_arc_label dal;
	succs = vertex_successors(stmt_vertex);
	for(; (succs != NIL) && (! true_dep_found) ; succs = CDR(succs))
	{
	    dal = (dg_arc_label) successor_arc_label(SUCCESSOR(CAR(succs)));
	    confs = dg_arc_label_conflicts(dal);
	    for(; (confs != NIL) && (! true_dep_found) ; confs = CDR(confs))
		if( true_dependence_with_entity_p(CONFLICT(CAR(confs)), lhs_ent) )
		    true_dep_found = true;
	}
    }
    else
	user_warning("defs_elim_of_assign_call",
		     "Vertex of assign stmt should not be undefined\n");

    debug(5, "defs_elim_of_assign_call", "end ASSIGN , true dep : %s\n",
	  bool_to_string(true_dep_found));

    return(! true_dep_found);
}



/*============================================================================*/
/* bool defs_elim_of_statement(statement s, graph dg): returns true if "s"
 * is to be eliminated.
 * As we eliminate assign statements, only statement with call to the
 * assign function may be eliminated.
 *
 * Called_functions :
 *       _ make_empty_statement() : ri-util/statement.c
 */
bool 
defs_elim_of_statement(s, dg)
statement s;
graph dg;
{
    bool elim = false;
    instruction inst = statement_instruction(s);

    debug(4, "defs_elim_of_statement", "begin STATEMENT\n");

    switch(instruction_tag(inst))
    {
	/* We scan all the statements of the block, and we build in the same time
	 * a new block where the statements to delete do not appear.
	 */
    case is_instruction_block :
    {
	list new_block = NIL,
	    block = instruction_block(inst);
	for(; block != NIL ; block = CDR(block))
	{
	    statement stmt = STATEMENT(CAR(block));
	    if(! defs_elim_of_statement(stmt, dg) )
		new_block = gen_nconc(new_block, CONS(STATEMENT, stmt, NIL));
	}
	instruction_block(inst) = new_block;
	break;
    }
    case is_instruction_test :
    {
	test t = instruction_test(inst);
	if( defs_elim_of_statement(test_true(t), dg) )
	    test_true(t) = make_empty_statement();
	if( defs_elim_of_statement(test_false(t), dg) )
	    test_false(t) = make_empty_statement();
	break;
    }
    case is_instruction_loop :
    {
	loop l = instruction_loop(inst);
	if( defs_elim_of_statement(loop_body(l), dg) )
	    loop_body(l) = make_empty_statement();
	break;
    }
    case is_instruction_call : 
    {
	call c = instruction_call(inst);

	debug(4, "defs_elim_of_statement", "Stmt CALL: %s\n",
	      entity_local_name(call_function(c)));

	if(ENTITY_ASSIGN_P(call_function(c)))
	    elim = defs_elim_of_assign_call(s, dg);
	break;
    }
    case is_instruction_goto : break;
    case is_instruction_unstructured :
    {
	defs_elim_of_unstructured(instruction_unstructured(inst), dg);
	break;
    }
    default : pips_internal_error("Bad instruction tag");
    }
    debug(4, "defs_elim_of_statement", "end STATEMENT\n");

    return(elim);
}


/*============================================================================*/
/* void defs_elim_of_unstructured(unstructured, graph dg): computes the
 * elimination of all the definitions with no def-use dependence of an
 * unstructured instruction.
 *
 * If the statement of the control of a node of the control graph has to
 * be eliminated, it is replaced by an empty block of statement.
 *
 * Called_functions :
 *       _ make_empty_statement() : ri-util/statement.c
 */
void 
defs_elim_of_unstructured(u, dg)
unstructured u;
graph dg;
{
    list blocs = NIL;

    debug(3, "defs_elim_of_unstructured", "begin UNSTRUCTURED\n");

    CONTROL_MAP(c, { bool elim = defs_elim_of_statement(control_statement(c), dg);
    if(elim) { control_statement(c) = make_empty_statement();};},
    unstructured_control( u ), blocs);

    gen_free_list(blocs);

    debug(3, "defs_elim_of_unstructured", "end UNSTRUCTURED\n");
}

