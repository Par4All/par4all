/*
 * $Id$
 *
 * Olivier Albiez
 */


#include "local.h"
#include "effects-generic.h"
#include "effects-simple.h"


#define FLOW_DEPENDANCE 1
#define ANTI_DEPENDANCE 2
#define OUTPUT_DEPENDANCE 4
#define INPUT_DEPENDANCE 8
#define ALL_DEPENDANCES (FLOW_DEPENDANCE|ANTI_DEPENDANCE|OUTPUT_DEPENDANCE)

/*
 * Definition of local variables
 */
GENERIC_LOCAL_FUNCTION(has_level, persistant_statement_to_int)
GENERIC_LOCAL_MAPPING(has_indices, list, statement)

static list indices = NIL;    
static int depth = 0;
static int reference_level=0;
static string current_module_name; /* Bad hack to avoid an argument 
				      in icm_codegen.... */




/*
 * Utilities function
 */


void 
dump_sef(statement_effects se)
{
    fprintf(stderr, "\n");
    STATEMENT_EFFECTS_MAP(s, e, {
	fprintf(stderr, 
		"%02d(%X) -> (%d) : ", 
		statement_number(s), 
		s,
		gen_length(effects_effects(e)));

	MAP(EFFECT, e, { 
	    print_words(stderr, words_effect(e));
	    fprintf(stderr, "(%X), ", e);
	}, effects_effects(e));
	
	fprintf(stderr, "\n");
    }, se);
}


static void 
prettyprint_conflict(FILE *fd, conflict c)
{
    fprintf(fd, "\t\tfrom ");
    print_words(fd, words_effect(conflict_source(c)));

    fprintf(fd, " to ");
    print_words(fd, words_effect(conflict_sink(c)));

    fprintf(fd, " at levels ");
    MAP(INT, level, {
	fprintf(fd, " %d", level);
    }, cone_levels(conflict_cone(c)));
    fprintf(fd, "\n");
}


static void 
prettyprint_successor(FILE *fd, successor su)
{
    vertex v = successor_vertex(su);
    fprintf(fd, 
	    "link with %02d :\n", 
	    statement_number(vertex_to_statement(v)));

    MAP(CONFLICT, c, { 
	prettyprint_conflict(fd, c); 
    }, dg_arc_label_conflicts((dg_arc_label) successor_arc_label(su)));
}


static void 
prettyprint_vertex(FILE *fd, vertex v)
{
    fprintf(fd, 
	    "vertex %02d :\n", 
	    statement_number(vertex_to_statement(v)));
    MAP(SUCCESSOR, s, { 
	prettyprint_successor(fd, s); 
    }, vertex_successors(v));
}



static bool
action_dependance_p(action s, action k, int dependance_type) 
{
    return (((dependance_type & FLOW_DEPENDANCE) && 
	     ((action_write_p(s) && action_read_p(k)))) ||
	    ((dependance_type & ANTI_DEPENDANCE) && 
	     ((action_read_p(s) && action_write_p(k)))) ||
	    ((dependance_type & OUTPUT_DEPENDANCE) && 
	     ((action_write_p(s) && action_write_p(k)))) ||
	    ((dependance_type & INPUT_DEPENDANCE) && 
	     ((action_read_p(s) && action_read_p(k)))));
}


static set /* of statement */
verticies_to_statements(list /* of vertex */ vl, 
			set /* of statement */ss)
{
    MAP(VERTEX, v, {
	ss = set_add_element(ss, ss, (char *) vertex_to_statement(v));
    }, vl);

    return ss;
}



/* This fonction test the existance of a dependances between 
 * two verticies v1, v2
 *  
 *  dependance_type is a bitfield which contains the kind of dependances
 *  level is the minimum level wanted
 */
static bool
dependance_verticies_p(vertex v1, vertex v2, int dependance_type, int level)
{    
    ifdebug(9) {
	debug(9, "dependance_verticies_p", "");
	
	if (dependance_type & FLOW_DEPENDANCE) fprintf(stderr, "F ");
	if (dependance_type & ANTI_DEPENDANCE) fprintf(stderr, "A ");
	if (dependance_type & OUTPUT_DEPENDANCE) fprintf(stderr, "O ");
	if (dependance_type & INPUT_DEPENDANCE) fprintf(stderr, "I ");
	fprintf(stderr, "dep. at level >= %d ", level);
	fprintf(stderr, 
		"between %02d and %02d ? ", 
		statement_number(vertex_to_statement(v1)), 
		statement_number(vertex_to_statement(v2))); 
    }


    MAP(SUCCESSOR, su, {
	vertex s = successor_vertex(su);
	if (s == v2) {
	    MAP(CONFLICT, c, {
		if (conflict_cone(c) != cone_undefined) {
		    MAP(INT, l, {
			if (l >= level) {
			    action s = effect_action(conflict_source(c)) ;
			    action k = effect_action(conflict_sink(c)) ;

			    if (action_dependance_p(s, k, dependance_type))
			    {
				ifdebug(9) { fprintf(stderr, "yes\n"); }
				return TRUE;
			    }
			}
		    }, cone_levels(conflict_cone(c)));
		}
	    }, dg_arc_label_conflicts(successor_arc_label(su)));
	}
    }, vertex_successors(v1));

    ifdebug(9) { fprintf(stderr, "no\n"); }
    return FALSE;
}


static list /* of level */
remove_dependance_from_levels(list /* of level */ levels, 
			      int level_min, 
			      int level_max)
{
    list /* of level */ new_levels = NIL;

    MAP(INT, level, {
	if ((level < level_min) || (level > level_max)) {
	    new_levels = gen_nconc(new_levels, CONS(INT, level, NIL));
	}
    }, levels);

    /* supprime la liste *SANS* supprimer les element de la liste!!!!!!!!
     * si la fonction gen_free_list() libere les elements, alors on va 
     * droit au plantage.
     
     gen_free_list(levels);

     */

    return new_levels;
}


static list /* of conflict */
remove_dependance_from_conflicts(list /* of conflict */ conflicts, 
				 int dependance_type, 
				 int level_min, 
				 int level_max)
{
    list /* of conflict */ new_conflicts = NIL;

    ifdebug(7) 
    {
	fprintf(stderr, "Old conflicts :\n");
	MAP(CONFLICT, c, { 
	    prettyprint_conflict(stderr, c); 
	}, conflicts);
    }

    MAP(CONFLICT, c, {
	if(conflict_cone(c) != cone_undefined) {
	    action s = effect_action (conflict_source(c)) ;
	    action k = effect_action (conflict_sink(c)) ;

	    if (action_dependance_p(s, k, dependance_type))
	    {
		list /* of level */ levels = cone_levels(conflict_cone(c));

		list /* of level */ new_levels = 
		    remove_dependance_from_levels(levels, 
						  level_min, 
						  level_max);

		if (new_levels != NIL) {
		    cone_levels(conflict_cone(c)) = new_levels;
		    new_conflicts = CONS(CONFLICT, c, new_conflicts);
		}

		/* Liberer l'ancienne liste...

		 gen_free_list(levels);

		 */
	    }
	    else {
		new_conflicts = CONS(CONFLICT, c, new_conflicts);
	    }
	}
	else {
	    new_conflicts = CONS(CONFLICT, c, new_conflicts);
	}

    }, conflicts);


    /* supprime la liste *SANS* supprimer les element de la liste!!!!!!!!
     * si la fonction gen_free_list() libere les elements, alors on va 
     * droit au plantage

    gen_free_list(conflicts);

     */

    ifdebug(7) 
    {
	fprintf(stderr, "New conflicts :\n");
	MAP(CONFLICT, c, { 
	    prettyprint_conflict(stderr, c); 
	}, new_conflicts);
    }

    return new_conflicts;
}


static list /* of successor */
remove_dependances_from_successors(successors, v2, dependance_type, level_min, level_max)
list successors;
vertex v2;
int dependance_type;
int level_min;
int level_max;
{
    list /* of successor */ new_successors = NIL;

    ifdebug(7) 
    {
	fprintf(stderr, "Old successors :\n");
	MAP(SUCCESSOR, s, { 
	    prettyprint_successor(stderr, s); 
	}, successors);
    }

    MAP(SUCCESSOR, su, {
	vertex v = successor_vertex(su);
	
	if (v == v2) {
	    list /* of conflict */ lc = 
		dg_arc_label_conflicts(successor_arc_label(su));

	    list /* of conflict */ new_lc = 
		remove_dependance_from_conflicts(lc,
						 dependance_type, 
						 level_min, 
						 level_max);

	    if (new_lc != NIL) {
		dg_arc_label_conflicts(successor_arc_label(su)) = new_lc;
		new_successors = CONS(SUCCESSOR, su, new_successors);
	    }

	    /* Liberer l'ancienne liste...

	       gen_free_list(lc);

	     */
	}
	else {
	    new_successors = CONS(SUCCESSOR, su, new_successors);
	}

    }, successors);

    /* supprime la liste *SANS* supprimer les element de la liste!!!!!!!!
     * si la fonction gen_free_list() libere les elements, alors on va 
     * droit au plantage

     gen_free_list(successors);

     */

    ifdebug(7) 
    {
	fprintf(stderr, "New successors :\n");
	MAP(SUCCESSOR, s, { 
	    prettyprint_successor(stderr, s); 
	}, successors);
    }

    return new_successors;
}


static void
remove_dependance(vertex v1, /* The list of this vertex sucessors is updated */
		  vertex v2, 
		  int dependance_type, 
		  int level_min, 
		  int level_max)
{
    list /* of successor */ v1_successors = vertex_successors(v1);

    ifdebug(3)
    {
	debug(3, "remove_dependance", "Remove ");
	
	if (dependance_type & FLOW_DEPENDANCE) fprintf(stderr, "F ");
	if (dependance_type & ANTI_DEPENDANCE) fprintf(stderr, "A ");
	if (dependance_type & OUTPUT_DEPENDANCE) fprintf(stderr, "O ");
	if (dependance_type & INPUT_DEPENDANCE) fprintf(stderr, "I ");

	fprintf(stderr, 
		"dep. at level >= %d and level <= %d ", 
		level_min, 
		level_max);

	fprintf(stderr, 
		"between %02d and %02d.\n", 
		statement_number(vertex_to_statement(v1)), 
		statement_number(vertex_to_statement(v2))); 
    } 

    vertex_successors(v1) = 
	remove_dependances_from_successors(v1_successors, 
					   v2, 
					   dependance_type, 
					   level_min, 
					   level_max);

    ifdebug(7) {
	prettyprint_vertex(stderr, v1);
    }
}


static bool 
common_ignore_this_vertex(set /* of statement */ region, vertex v)
{
    return(!set_belong_p(region, (char *) vertex_to_statement(v)));
}


static bool 
icm_ignore_this_successor(vertex v, 
			  set /* of statement */ region, 
			  successor su, 
			  int level)
{
    if (common_ignore_this_vertex(region, successor_vertex(su)))
	return(TRUE);

    if (!dependance_verticies_p(v, 
				successor_vertex(su), 
				ALL_DEPENDANCES, 
				level))
	return (TRUE);

  return FALSE;
}


static bool 
invariant_ignore_this_successor(vertex v, 
				set /* of statement */ region, 
				successor su, 
				int level)
{
    if (common_ignore_this_vertex(region, successor_vertex(su)))
	return(TRUE);

    /* Keep only flow dependances. */
    if (!dependance_verticies_p(v, 
				successor_vertex(su), 
				FLOW_DEPENDANCE, 
				level))
	return (TRUE);

    return (FALSE);
}


void
print_list_entities(list /* of entity */ l)
{
    MAP(ENTITY, e, { 
	fprintf(stderr, "%s ", entity_name(e));
    }, l);
}


static bool
statement_depend_of_indices_p(statement st, 
			      list /* of entity */ indices, 
			      int level)
{
    list /* of eftfect */ le = load_proper_rw_effects_list(st);

    /* Ignore the first indicies... */
    for(; level > reference_level; --level) {
	indices = CDR(indices);
    }

    MAP(ENTITY, index, {
	MAP(EFFECT, ef, {
	    entity e = reference_variable(effect_reference(ef));
	    if (e == index)
		return TRUE;
	}, le);
    }, indices);

    return FALSE;
}


bool
vertex_variant_p(vertex v, 
		 graph g, 
		 int level, 
		 set /* of statement */ variant)
{
    statement st = vertex_to_statement(v);

    ifdebug(6) {
    debug(6, "vertex_variant_p", "test ");
    fprintf(stderr, "statement %02d : ", statement_number(st));
    }

    /* Test if the statement is depandant of ALL loop indexes >= level */
    if (statement_depend_of_indices_p(st,
				      load_statement_has_indices(st),        
				      level)) {
	ifdebug(6) { 
	    fprintf(stderr, "variant (depend of indices)\n"); 
	}
	return TRUE;
    }

    /* If there is a flow dependance from v to v, then v is variant */
    if (dependance_verticies_p(v, v, FLOW_DEPENDANCE, level)) {
	ifdebug(6) { 
	    fprintf(stderr, "variant (self flow dep)\n"); 
	}
	return TRUE;
    }

    /* If there is a flow dependance from y to v and if y is variant, 
       then v is variant */
    MAP(VERTEX, y, {
	if (dependance_verticies_p(y, v, FLOW_DEPENDANCE, level)) {
	    statement st = vertex_to_statement(y);

	    if (set_belong_p(variant, (char *) st)) {

		ifdebug(6) { 
		    fprintf(stderr, 
			    "depend of variant stat %02d\n", 
			    statement_number(st)); 
		}
	    
		return TRUE;
	    }
	}

    },  graph_vertices(g));

    ifdebug(6) { 
	fprintf(stderr, "invariant\n"); 
    }

    return FALSE;
}


graph
SimplifyGraph(graph g, 
	      set /* of statement */ region, 
	      int level)
{  
    list /* of scc */ lsccs;
    
    /* Find sccs */
    set_sccs_drivers(&common_ignore_this_vertex, 
		     &icm_ignore_this_successor);
    lsccs = FindAndTopSortSccs(g, region, level);
    reset_sccs_drivers();

    MAP(SCC, elmt, {
	list /* of vertex */ lv = scc_vertices(elmt);

	if (gen_length(lv) > 1) {
	    set new_region = set_make(set_pointer);
	    new_region = verticies_to_statements(lv, new_region);
	    g = SupressDependances(g, new_region, level);
	    set_free(new_region);
	}
    }, lsccs); 

    return g;
}





/* We looking for all vertex y which satisfy :
 * output(v -> v, <= level)
 * flow(v -> y, infiny)
 * anti(y -> v, <= level)
 */ 

static set /* of vertex */
gather_matching_vertices(graph g, 
			 vertex v,
			 set /* of statement */ region, 
			 int level,
			 set /* of vertex */ result)
{
    if (dependance_verticies_p(v, v, OUTPUT_DEPENDANCE, level)) {
	MAP(SUCCESSOR, su, {
	    vertex y = successor_vertex(su);
	    if (!common_ignore_this_vertex(region, y)) {
		if (dependance_verticies_p(v, y, FLOW_DEPENDANCE, level) && 
		    dependance_verticies_p(y, v, ANTI_DEPENDANCE, level)) {
		    result = set_add_element(result, result, (char *) y);
		}
	    }
	}, vertex_successors(v));
    }

    return result;
}




graph
SupressDependances(graph g, 
		   set /* of statement */ region, 
		   int level)
{
    list /* of scc */ lsccs;
    set /* of statement */ variant = set_make(set_pointer);

    /* Find sccs considering only flow dependances */
    set_sccs_drivers(&common_ignore_this_vertex, 
		     &invariant_ignore_this_successor);
    lsccs = FindAndTopSortSccs(g, region, level);
    reset_sccs_drivers();

    printf("Parcours des sccs\n");

    MAP(SCC, s, {
	list /* of vertex */ lv = scc_vertices(s);

	if (gen_length(lv) > 1) {
	    /* Group of vertices : all are variants */
	    variant = verticies_to_statements(lv, variant);
	}
	else {
	    /* One statement... */
	    vertex v = VERTEX(CAR(lv));
	    statement st = vertex_to_statement(v);

	    if (vertex_variant_p(v, g, level, variant)) {
		/* which is variant : added to the list */
		variant = set_add_element(variant, variant, (char *) st);
	    }
	    else {
		/* which is invariant */
		set /* of vertex */ matching_vertices = set_make(set_pointer);

		matching_vertices = 
		    gather_matching_vertices(g,  v,
					     region, 
					     level,
					     matching_vertices);

		if (!set_empty_p(matching_vertices)) {
		    remove_dependance(v, v, 
				      OUTPUT_DEPENDANCE, 
				      0, load_has_level(st));

		    SET_MAP(y, {
			remove_dependance((vertex) y, v, 
					  ANTI_DEPENDANCE, 
					  0, load_has_level(st));
		    }, matching_vertices);
		}

		set_free(matching_vertices);
	    }
	}
    }, lsccs); 

    set_free(variant);

    return SimplifyGraph(g, region, level+1);
}


static bool
statement_mark(statement s)
{
     store_has_level(s, depth);
     store_statement_has_indices(s, gen_nreverse(gen_copy_seq(indices)));
     
     return TRUE;
}


static bool
loop_level_in (loop l) 
{
    entity index = loop_index(l);
    indices = CONS(ENTITY, index, indices);
    
    ++depth;

    return TRUE;
}


static bool
loop_level_out (loop l) 
{
    list new_indices = CDR(indices);
    free(indices);
    indices = new_indices;

    --depth;

    return TRUE;
}

/******************************************************** REMOVE DUMMY LOOPS */

DEFINE_LOCAL_STACK(stmt, statement)

static list /* of entity */ depending_indices;
static bool it_depends;

/* set whether s depends from enclosing indices
 */
static void does_it_depend(statement s)
{
  it_depends |= statement_depend_of_indices_p(s, depending_indices, 0);
  if (it_depends) gen_recurse_stop(NULL);
}

static bool push_depending_index(loop l)
{
  depending_indices = CONS(ENTITY, loop_index(l), depending_indices);
  return TRUE;
}

static void pop_depending_index(loop l)
{
  list tmp = depending_indices;
  pips_assert("current loop index is poped", 
	      loop_index(l)==ENTITY(CAR(depending_indices)));
  depending_indices = CDR(depending_indices);
  CDR(tmp) = NIL;
  free(tmp);
}

static bool stmt_filter_and_check(statement s)
{
  stmt_push(s);
  does_it_depend(s);
  return TRUE;
}

static bool drop_it(loop l)
{
  if (execution_parallel_p(loop_execution(l)))
  {
    depending_indices = NIL;
    it_depends = FALSE;
    gen_multi_recurse(l,
		      statement_domain, stmt_filter_and_check, stmt_rewrite,
		      loop_domain, push_depending_index, pop_depending_index,
		      NULL);
    depending_indices = NIL; /* assert? */
    return it_depends;
  }
  
  return FALSE;
}

static void loop_rwt(loop l)
{
  if (drop_it(l))
  {
    statement head = stmt_head();
    statement_instruction(head) = statement_instruction(loop_body(l));
    /* memory leak... */
  }
}

void drop_dummy_loops(statement s)
{
  make_stmt_stack();

  gen_multi_recurse(s,
		    statement_domain, stmt_filter, stmt_rewrite,
		    loop_domain, gen_true, loop_rwt,
		    NULL);

  free_stmt_stack();
}

/*********************************************************** REGENERATE CODE */

statement
icm_codegen(statement stat, 
	    graph g, 
	    set /* of statement */ region, 
	    int level, 
	    bool task_parallelize_p)
{
    statement result = statement_undefined;
    graph simplyfied_graph = graph_undefined;

    reference_level = level;

    debug_on("ICM_DEBUG_LEVEL");
    debug(9, "icm_codegen", "start\n");


    set_proper_rw_effects((statement_effects) 
			  db_get_memory_resource(DBR_PROPER_EFFECTS, 
						 current_module_name, 
						 TRUE));

    /* Compute has_level hash and has_indices tables */

    init_has_level();
    make_has_indices_map();

    indices = NIL;

    gen_multi_recurse
	(stat,
	 loop_domain, loop_level_in, loop_level_out, /* LOOP */	 
	 statement_domain, statement_mark, gen_true, /* STATEMENT */
	 NULL);

    gen_free_list(indices);

    /* Simplify the dependance graph */

    simplyfied_graph = copy_graph(g);
    simplyfied_graph = SimplifyGraph(simplyfied_graph, 
				     region, 
				     level);

    ifdebug(4) {
	prettyprint_dependence_graph(stderr, 
				     statement_undefined, 
				     simplyfied_graph);    
    }

    close_has_level();
    free_has_indices_map();
    reset_proper_rw_effects(); /* CodeGenerate reload the
				  proper_rw_effects table, so we must
				  reset before... */

    debug(9, "icm_codegen", "stop\n");
    debug_off();

    /* Generate the code (CodeGenerate don't use the first
       parameter...) */

    result =  CodeGenerate(/* big hack */ statement_undefined,
			   simplyfied_graph, 
			   region, 
			   level, 
			   task_parallelize_p); 

    free_graph(simplyfied_graph);

    /* Remove dummy loops. */
    drop_dummy_loops(result);

    /* Print debugginf information : statement */
    print_statement(result);

    return result;
}


/*
  Entry point for the zory invariant code motion :)
  OA
*/ 
bool
invariant_code_motion(string module_name)
{
    entity module = local_name_to_top_level_entity(module_name);
    statement mod_stat = statement_undefined;
    set_current_module_entity(module);

    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE );
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE );

    set_current_module_statement((statement) 
				 db_get_memory_resource(DBR_CODE, 
							module_name, 
							TRUE));

    mod_stat = get_current_module_statement();
    
    current_module_name = module_name;

    debug_on("ICM_DEBUG_LEVEL");

    ifdebug(7)
    {
	fprintf(stderr, "\nTesting NewGen consistency for initial code %s:\n",
		module_name);
	if (statement_consistent_p((statement)mod_stat))
	    fprintf(stderr," NewGen consistent statement\n");
    }

    ifdebug(1) {
	debug(1, "do_it", "original sequential code:\n\n");
	print_statement(mod_stat);
    }

    if (graph_undefined_p(dg)) {
	dg = (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
    }
    else {
	pips_error("do_it", "dg should be undefined\n");
    }

    enclosing = 0;
    rice_statement(mod_stat, 1, &icm_codegen);   

    ifdebug(7)
    {
	fprintf(stderr, "\nparallelized code %s:",module_name);
	if (statement_consistent_p((statement)mod_stat))
	    fprintf(stderr," gen consistent ");
    }

    debug_off();


    /* 
     * Je pense qu'il serait souhaitable de ranger le resultat dans la 
     * base... Mais comment et ou ?
     */

//    DB_PUT_MEMORY_RESOURCE(what, module_name, (char*) mod_stat);


    dg = graph_undefined;
    reset_current_module_statement();
    reset_current_module_entity();
    return TRUE;
}


