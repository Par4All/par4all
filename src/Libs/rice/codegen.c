/*
 * $Id$
 */

#include <stdio.h>
#include <string.h>
#include <values.h>

#include "genC.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ri.h"
#include "graph.h"
#include "dg.h"

#include "misc.h"
#include "ri-util.h"
#include "text-util.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "constants.h"

#include "ricedg.h"
#include "rice.h"

extern current_module_entity;

/* instantiation of the dependence graph */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;


/* This function checks if conflict c between vertices v1 and v2 should
be ignored at level l. 

A conflict is to be ignored if the variable that creates the conflict is
local to one of the enclosing loops.

Attension: The loops around every statement got by load_statement_loops(statement) here are just these after
taking off the loops on which the Kennedy's algo. can't be applied. (YY) 
*/

bool ignore_this_conflict(v1, v2, c, l)
vertex v1, v2;
conflict c;
int l;
{
    extern int enclosing;
    effect e1 = conflict_source(c);
    reference r1 = effect_reference(e1) ;
    entity var1 = reference_variable(r1);
    statement s1 = vertex_to_statement(v1);
    list loops1 = load_statement_enclosing_loops(s1);

    effect e2 = conflict_sink(c);
    reference r2 = effect_reference( e2 ) ;
    entity var2 = reference_variable(r2);
    statement s2 = vertex_to_statement(v2);
    list loops2 = load_statement_enclosing_loops(s2);
   register int i;

    if (var1 != var2) {
	/* equivalences do not deserve more cpu cycles */
	return(FALSE);
    }

    for (i = 1; i < l-enclosing; i++) {
	if( !ENDP(loops1)) {
	    loops1 = CDR(loops1);
	}
	if( !ENDP(loops2)) {
	    loops2 = CDR(loops2);
	}
    }

    ifdebug(8) {
	fprintf(stderr, "\n[ignore_this_conflict] verifing the following conflit at level %d: \n",l);
	fprintf(stderr, "\t%02d --> %02d ", statement_number(s1), statement_number(s2));
	fprintf(stderr, "\t\tfrom ");
	print_words(stderr, words_effect(conflict_source(c)));

	fprintf(stderr, " to ");
	print_words(stderr, words_effect(conflict_sink(c)));
	fprintf(stderr, "\n");
    }

    MAPL(pl, {
	statement st = STATEMENT(CAR(pl));
	list l = loop_locals(instruction_loop(statement_instruction(st)));
	ifdebug(8) {
	  print_statement(st);
	    fprintf(stderr,"The list of prived variables : \n");
	    MAPL(v, {entity e = ENTITY (CAR(v));
		     fprintf(stderr," %s", entity_local_name(e));
		 }, l);
	    fprintf(stderr,"\n");
	}

	if (gen_find_eq(var1, l) != entity_undefined) {
	    return(TRUE);
	}
    }, loops1);
    
    MAPL(pl, {
	statement st = STATEMENT(CAR(pl));
	list l = loop_locals(instruction_loop(statement_instruction(st)));
	ifdebug(8) {
	  print_statement(st);
	    fprintf(stderr,"The list of prived variables : \n");
	    MAPL(v, {entity e = ENTITY (CAR(v));
		     fprintf(stderr," %s", entity_local_name(e));
		 }, l);
	    fprintf(stderr,"\n");
	}
	if (gen_find_eq(var1, l) != entity_undefined)  {
	    return(TRUE);
	}
    }, loops2);
    return(FALSE);
}



/* s is a strongly connected component s which is being analyzed at
level l. its vertices are enclosed in at least l loops. this gives us a
solution to retrieve the level l loop enclosing a scc: to take its first
vertex and retrieve the l th loop aroung this vertex.  */

statement find_level_l_loop_statement(s, l)
scc s;
int l;
{
    vertex v = VERTEX(CAR(scc_vertices(s)));
    statement st = vertex_to_statement(v);
    list loops = load_statement_enclosing_loops(st);
    
    if (l > 0)
	MAPL(pl, {
	    if (l-- == 1)
		return(STATEMENT(CAR(pl)));
	}, loops);
    return( statement_undefined ) ;
}


set scc_region(scc s)
{
    set region = set_make(set_pointer);
    MAPL(pv, {
	set_add_element(region, region, 
			(char *) vertex_to_statement(VERTEX(CAR(pv))));
    }, scc_vertices(s));
    return(region);
}


/* s is a strongly connected component for which a DO loop is being
produced.  this function returns FALSE if s contains no dependences at
level l. in this case, the loop will be a DOALL loop. */

bool contains_level_l_dependence(s, region, level)
scc s;
set region;
int level;
{
    MAPL(pv, {
	vertex v = VERTEX(CAR(pv));
	statement s1 = vertex_to_statement(v);
	MAPL(ps, {
	    successor su = SUCCESSOR(CAR(ps));
	    vertex vs = successor_vertex(su);
	    statement s2 = vertex_to_statement(vs);

	    if (! ignore_this_vertex(region, vs)) {
		dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
		MAPL(pc, {
		    conflict c = CONFLICT(CAR(pc));
		    if (! ignore_this_conflict(v, vs, c, level)) {
			if (conflict_cone(c) != cone_undefined){
			    MAPL(pi, {
				if (INT(CAR(pi)) == level) {
				    ifdebug(7) {
					fprintf(stderr, 
						"\n[contains_level_l_dependence] containing conflit at level %d: ",level);  
					fprintf(stderr, "\t%02d --> %02d ", 
						statement_number(s1), statement_number(s2));
					fprintf(stderr, "\t\tfrom ");
					print_words(stderr, words_effect(conflict_source(c)));
					fprintf(stderr, " to ");
					print_words(stderr, words_effect(conflict_sink(c)));
					fprintf(stderr, "\n");
				    }
				    return(TRUE);
				}
			    }, cone_levels(conflict_cone(c)));
			}
		    }
		}, dg_arc_label_conflicts(dal));
	    }
	}, vertex_successors(v));
    }, scc_vertices(s));

    return(FALSE);
}



/* this function returns TRUE if scc s is stronly connected at level l,
i.e. dependence arcs at level l or greater form at least one cycle */

bool strongly_connected_p(s, l)
scc s;
int l;
{
    cons *pv = scc_vertices(s);
    vertex v = VERTEX(CAR(pv));

    /* if s contains more than one vertex, it is strongly connected */
    if (CDR(pv) != NIL)
	return(TRUE);

    /* if there is a dependence from v to v, s is strongly connected */
    MAPL(ps, {
	successor s = SUCCESSOR(CAR(ps));
	if (!ignore_this_level((dg_arc_label) successor_arc_label(s), l) && 
	    successor_vertex(s) == v)
	    return(TRUE);
    }, vertex_successors(v));

    /* s is not strongly connected */
    return(FALSE);
}



/* this function creates a nest of parallel loops around an isolated
statement whose iterations may execute in parallel. 

loops is the loop nest that was around body in the original program. l
is the current level; it tells us how many loops have already been
processed. */

statement MakeNestOfParallelLoops(l, loops, body, task_parallelize_p)
int l;
cons *loops;
statement body;
bool task_parallelize_p;
{
    statement s;
    debug(3, " MakeNestOfParallelLoops"," at level %d ...\n",l);

    if (loops == NIL)
	s = body;
    else if( l > 0 )
	s = MakeNestOfParallelLoops(l-1, 
				    CDR(loops), body, task_parallelize_p) ;
    else {
      statement slo = STATEMENT(CAR(loops));
	loop lo = statement_loop(slo);
	tag  seq_or_par = ((CDR(loops) == NIL || task_parallelize_p)
			   && index_private_p(lo)) ? 
			       is_execution_parallel : is_execution_sequential;
	
	/* At most one outer loop parallel */
	bool task_parallelize_inner = 
	  (seq_or_par == is_execution_parallel
	   && ! get_bool_property("GENERATE_NESTED_PARALLEL_LOOPS") ) ?  
	  FALSE:task_parallelize_p;
	
	s = MakeLoopAs(slo, seq_or_par,
		       MakeNestOfParallelLoops(0, CDR(loops), body,
					       task_parallelize_inner));
    }
    return(s);
}




/* this function implements Kennedy's algorithm. */
/* bb: task_parallelize_p is TRUE when we want to parallelize the loop, 
FALSE when we only want to vectorize it */
statement CodeGenerate(g, region, l, task_parallelize_p)
graph g;
set region;
int l;
bool task_parallelize_p;
{
    list lst =NIL;
    list loops = NIL; 
    cons *lsccs, *ps;
    cons *block = NIL, *eblock = NIL;
    statement stat = statement_undefined;
    statement statb = statement_undefined;
    statement rst = statement_undefined;
    int nbl = 0;
    extern int enclosing;

   debug_on("RICE_DEBUG_LEVEL");

    debug(9, "CodeGenerate", "Begin: starting at level %d ...\n", l); 
    ifdebug(9)
	print_statement_set(stderr, region);
 
    debug(9, "CodeGenerate", "finding and top-sorting sccs ...\n");
    lsccs = FindAndTopSortSccs(g, region, l);

    debug(9, "CodeGenerate", "generating code ...\n");
    for (ps = lsccs; ps != NULL; ps = CDR(ps)) {
	scc s = SCC(CAR(ps));
	stat = statement_undefined;
	
	if (get_bool_property("PARTIAL_DISTRIBUTION")) {
	    /* statements that are independent are gathered 
	       into the same doall loop */
	    if (! strongly_connected_p(s, l)) {
		set inner_region = scc_region(s);
		if (contains_level_l_dependence(s,inner_region,l)) {
		    stat = IsolatedStatement(s, l, task_parallelize_p);
		    debug(9, "CodeGenerate", 
			  "isolated comp.that contains dep. at Level %d\n",
			  l);
		}
		else  { 
		    vertex v = VERTEX(CAR(scc_vertices(s)));
		    statement st = vertex_to_statement(v); 
		    instruction sbody = statement_instruction(st);
		    if (instruction_call_p(sbody) &&  
			(strcmp(entity_local_name(call_function(instruction_call(sbody))), 
				"CONTINUE") != 0))  { 
			ifdebug(9) {
			    debug(9,"CodeGenerate",
				  "isolated comp. that does not contain"
				  " dep.at Lev %d\n",
					   l);
			    print_statement(st);
			}
			lst = gen_nconc(lst, CONS(STATEMENT, st, NIL));
			loops = load_statement_enclosing_loops(st);
			nbl =gen_length(loops);
		    }
		}
	    }
	    else stat = ConnectedStatements(g, s, l, task_parallelize_p);
	}
	else {
	    /* if s contains a single vertex and if this vertex is not 
	       dependent upon itself, we generate a doall loop for it */
	    if (! strongly_connected_p(s, l)) 
		stat = IsolatedStatement(s, l, task_parallelize_p);
	    else stat = ConnectedStatements(g, s, l, task_parallelize_p);
	}
	if (stat != statement_undefined) {
	    /* In order to preserve the dependences, statements that have 
	       been collected should be generated before the isolated statement 
	       that has just been detected */
	    if (nbl) { 
		if (nbl>=l) {
		    if (gen_length(lst)== 1)  rst = (STATEMENT(CAR(lst)));
		    else rst = make_block_statement(lst);
		    statb = MakeNestOfParallelLoops(l-1-enclosing,loops,rst, 
						    task_parallelize_p);
		    INSERT_AT_END(block, eblock, CONS(STATEMENT, statb, NIL));
		}
		else  eblock =  block =  lst; 
		nbl = 0; 
	    }
	    INSERT_AT_END(block, eblock, CONS(STATEMENT, stat, NIL));
	    ifdebug(9) {
		debug(9, "CodeGenerate", "generated statement:\n") ;
		print_statement(stat);
	    }
	}
    }
       
    if (nbl) {  
	if (nbl>=l) { 
	    if (gen_length(lst)== 1)
		rst = (STATEMENT(CAR(lst)));
	    else 
		rst = make_block_statement(lst);
	    statb = MakeNestOfParallelLoops(l-1-enclosing, 
					    loops, 
					    rst, 
					    task_parallelize_p);
	    INSERT_AT_END(block, eblock, CONS(STATEMENT, statb, NIL));
	}
	else
	    eblock = block = lst;
	nbl = 0;
    }

    switch(gen_length(block)) {
    case 0:
      rst = statement_undefined ;
      break;
    case 1:
      rst = STATEMENT(CAR(block));
      gen_free_list(block);
      break;
    default:
      rst = make_block_statement(block);
    }

    ifdebug(8) { 
	debug(8, "CodeGenerate", "Result:\n") ;

	if (rst==statement_undefined)  
	    debug(8, "CodeGenerate", "No code to generate\n") ;
	else
	  print_statement(rst);

	debug(8, "CodeGenerate", "Result:\n") ;
    }
    debug_off();
    return(rst);
}



/* this function creates a new loop whose characteristics (index,
 * bounds, ...) are similar to those of old_loop. The body and the
 * execution type are different between the old and the new loop.
 * fixed bug about private variable without effects, FC 22/09/93
 */

statement MakeLoopAs(old_loop_statement, seq_or_par, body)
statement old_loop_statement;
tag seq_or_par;
statement body;
{
  loop old_loop = statement_loop(old_loop_statement);
    loop new_loop;
    statement new_loop_s;
    list new_locals = NewLoopLocals(body, loop_locals(old_loop));
    instruction ibody = statement_instruction(body);  
    entity module = get_current_module_entity();
    char * module_name = entity_module_name(module);

    if (instruction_loop_p(ibody))
	body = make_block_statement(CONS(STATEMENT,body,NIL));
    ibody = statement_instruction(body);
  
    if (rice_distribute_only)
	seq_or_par = is_execution_sequential;

    if (get_bool_property("PARTIAL_DISTRIBUTION")) {
	entity looplabel =   make_new_label(module_name);
	statement cs = make_continue_statement(looplabel);
	if(instruction_block_p(ibody))
	    (void) gen_nconc(instruction_block(ibody),
			     CONS(STATEMENT, cs, NIL));
	else {
	    body = make_block_statement(CONS(STATEMENT, body,
					     CONS(STATEMENT, cs, NIL)));
	}
	new_loop = make_loop(loop_index(old_loop), 
			    loop_range(old_loop),
			    body,
			    looplabel, 
			    make_execution(seq_or_par, UU), 
			    new_locals);
    }
    else {  
	new_loop = make_loop(loop_index(old_loop), 
			     loop_range(old_loop), 
			     body,
			     entity_empty_label(),
			     make_execution(seq_or_par, UU), 
			     new_locals);
    }
   
    new_loop_s = make_statement(entity_empty_label(), 
			  statement_number(old_loop_statement),
			  STATEMENT_ORDERING_UNDEFINED,
			  string_undefined,
			  make_instruction(is_instruction_loop, new_loop));

    ifdebug(8) {
      debug(8, "MakeLoopAs", "New loop\n");
      print_statement(new_loop_s);
    }
    
    return(new_loop_s);
}

/*
 * list NewLoopLocals(body, locals)
 *
 * the private variable of the new loop are the old one one's
 * on which there are effects. 
 * ??? the effects shouldn't be computed once more...
 *
 * FC 22/09/93
 */
list NewLoopLocals(body, locals)
statement body;
list locals;
{
    list
	body_effects = statement_to_effects(body),
	result = NIL;

    ifdebug(8) {
      debug(8, "NewLoopLocals", "body is:\n");
      print_statement(body);
      debug(8, "NewLoopLocals", "effects considered are:\n");
      print_effects(body_effects);
    }

    MAPL(ce,
     {  entity
	     private_variable = ENTITY(CAR(ce));

	 debug(8, "NewLoopLocals", 
	       "considering entity %s with given effects\n",
	       entity_local_name(private_variable));

	 if (effects_read_or_write_entity_p(body_effects, private_variable))
	     result = CONS(ENTITY, private_variable, result);
     },
	 locals);

    gen_free_list(body_effects); 

    debug(8, "NewLoopLocals", 
	  "%d private variables kept out of %d\n",
	  gen_length(result),
	  gen_length(locals));

    return(result);
}

statement IsolatedStatement(s, l, task_parallelize_p)
scc s;
int l;
bool task_parallelize_p;
{
    vertex v = VERTEX(CAR(scc_vertices(s)));
    statement st = vertex_to_statement(v), rst;
    list loops = load_statement_enclosing_loops(st);
    instruction sbody = statement_instruction(st);
    extern int enclosing ;

    debug(8, "IsolatedStatement", "statement %d\n", statement_number(st));

    /* continue statements are ignored. */
    if (!instruction_call_p(sbody) 
	|| (strcmp(entity_local_name(call_function(instruction_call(sbody))), 
	       "CONTINUE") == 0))
	return (statement_undefined);

    rst = MakeNestOfParallelLoops(l-1-enclosing,
				  loops, st, task_parallelize_p);
    return(rst);
}



/* bb: ConnectedStatements() is called when s contains more than one
vertex or one vertex dependent upon itself. Thus, vectorization can't
occure */
statement ConnectedStatements(g, s, l, task_parallelize_p)
graph g;
scc s;
int l;
bool task_parallelize_p;
{
    extern int enclosing ;
    statement slo = find_level_l_loop_statement(s, l-enclosing);
    loop lo = statement_loop(slo);
    statement inner_stat;
    set inner_region;
    tag seq_or_par;
    bool task_parallelize_inner;

    debug(8, "ConnectedStatements", "at level %d:",l);
    ifdebug(8)	PrintScc(s);
	
    inner_region = scc_region(s);
    seq_or_par = (!task_parallelize_p 
		  || contains_level_l_dependence(s, inner_region, l) 
		  || ! index_private_p(lo)) ?
		      is_execution_sequential : is_execution_parallel;

    /* At most one outer loop parallel */
    task_parallelize_inner = 
	(seq_or_par == is_execution_parallel
	 && ! get_bool_property("GENERATE_NESTED_PARALLEL_LOOPS") ) ?
	FALSE : task_parallelize_p;

    inner_stat = CodeGenerate(g, inner_region, l+1, task_parallelize_inner);

    set_free(inner_region);

    if (statement_undefined_p(inner_stat))
	return inner_stat;
    else 
	return MakeLoopAs(slo, seq_or_par, inner_stat);
}
