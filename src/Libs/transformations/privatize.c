/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* -- privatize.c

   This algorithm introduces local definitions into loops that are
   kennedizable. The privatization is only performed on dynamic scalar
   variables (could be extended to STACK variables, but why would you
   allocate a variable in the PIPS *STACK* area). It is based on the
   dependence levels. The variable can only be private to loops
   containing all dependence arcs related to the variable. This should
   fail in C when a dynamic variable is initialized.

 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "misc.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "dg.h"
#include "control.h"
#include "pipsdbm.h"
#include "properties.h"
#include "transformations.h"

#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "properties.h"

/* instantiation of the dependence graph */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

typedef struct {
  entity e;
  bool loop_index_p;
  bool used_through_external_calls;
} privatizable_ctxt;

static bool loop_in(loop l, privatizable_ctxt *ctxt)
{
  if (loop_index(l) == ctxt->e)
    ctxt->loop_index_p = true;
  return ctxt->loop_index_p;
}

static bool call_in(call c, privatizable_ctxt *ctxt)
{
  entity func = call_function(c);
  const char* func_name = module_local_name(func);
  type uet = ultimate_type(entity_type(func));

  pips_debug(4, "begin for %s\n", entity_local_name(func));

  if(type_functional_p(uet))
    {
      if (value_code_p(entity_initial(func)))
	{
	  pips_debug(4, "external function\n");
	  /* Get the summary effects of "func". */
	  list func_eff = (*db_get_summary_rw_effects_func)(func_name);
	  /* tests if the function may refer to the global variable */
	  list l_conflicts = effects_entities_which_may_conflict_with_scalar_entity(func_eff, ctxt->e);
	  if (!ENDP(l_conflicts))
	    {
	      ctxt->used_through_external_calls = true;
	      gen_free_list(l_conflicts);
	    }
	}
      // else
      // nothing to do
    }
  else if(type_variable_p(uet))
    {
      pips_debug(4, "function called through pointer -> assume worst case\n");
      ctxt->used_through_external_calls = true;
    }

  return (!ctxt->used_through_external_calls);
}

/* privatizable() checks whether the entity e is privatizable in statement s. */
bool entity_privatizable_in_loop_statement_p(entity e, statement stmt, bool even_globals)
{
    storage s = entity_storage( e ) ;
    bool result = entity_scalar_p(e);
    loop l = statement_loop(stmt);

    ifdebug(4)
      {
	pips_debug(4, "begin for statement: \n");
	print_statement(stmt);
      }

    /* For C, it should be checked that e has no initial value because
       there is no dependence arc between the initialization in t he
       declaration and the other references. This is not very smart,
       because it all depends on where e is declared.

       FI: OK, I removed this safety test because declarations now
       have effects and are part of the use-def chains

       Also, stack_area_p() would be OK for a privatization.
    */

    pips_debug(3, "checking entity %s, with storage %s \n", entity_name(e), storage_to_string(s));

    /* Since there is currently no variable liveness analysis, we
       cannot privatize global variables.  However, we consider that
       inner loop indices are privatizable to allow correct
       parallelization of outer loops
    */
    if (result && even_globals
	&& storage_ram_p( s ) && static_area_p( ram_section( storage_ram( s )))
	&& top_level_entity_p(e) )
      {
	pips_debug(3, "global variable\n");

	/* check that the value of e is not re-used outside the loop */
	list l_live_out = load_live_out_paths_list(stmt);
	list l_conflicts_out = effects_entities_which_may_conflict_with_scalar_entity(l_live_out, e);
	list l_live_in = load_live_in_paths_list(stmt);
	list l_conflicts_in = effects_entities_which_may_conflict_with_scalar_entity(l_live_in, e);

	if (ENDP(l_conflicts_out) && ENDP(l_conflicts_in))
	  {
	    privatizable_ctxt ctxt = {e, false, false};
	    /* check that e is not used through called functions */
	    /* It may be passed to a function as a parameter,
	       but it must not be used as a global variable */
	    gen_context_multi_recurse(loop_body(l), &ctxt, call_domain, call_in, gen_null, NULL);
	    result = !ctxt.used_through_external_calls;
	  }
	else
	  {
	    gen_free_list(l_conflicts_out);
	    gen_free_list(l_conflicts_in);
	    result = true;
	  }
      }

    /* Here is the old behavior, to be removed when the
       previous if branch has been sufficiently tested */
    else if (result && !even_globals && c_module_p(get_current_module_entity())
	&& storage_ram_p( s ) && static_area_p( ram_section( storage_ram( s )))
	&& top_level_entity_p( e) /* this test may be removed since we check that e is used as a loop index */)
      {
	privatizable_ctxt ctxt = {e, false, false};
	/* check if e is an internal loop index */
	gen_context_recurse(loop_body(l), &ctxt, loop_domain, loop_in, gen_null);
	result = ctxt.loop_index_p;
      }
    else
      {
	result = result &&
	  ((storage_formal_p( s ) && parameter_passing_by_value_p(get_current_module_entity()) )||
	   (storage_ram_p( s ) && dynamic_area_p( ram_section( storage_ram( s ))))) ;
      }
    pips_debug(3, "returning %s\n", bool_to_string(result));
    return(result);
}

/* SCAN_STATEMENT gathers the list of enclosing LOOPS of statement S.
   Moreover, the locals of loops are initialized to all possible
   private entities. */

static void scan_unstructured(unstructured u, list loops, bool even_globals) ;

static void scan_statement(statement s, list loops, bool even_globals)
{
    instruction i = statement_instruction(s);

    if (get_enclosing_loops_map() == hash_table_undefined) {
        set_enclosing_loops_map( MAKE_STATEMENT_MAPPING() );
    }
    store_statement_enclosing_loops(s, loops);

    switch(instruction_tag(i)) {
    case is_instruction_block:
        MAPL(ps, {scan_statement(STATEMENT(CAR(ps)), loops, even_globals);},
             instruction_block(i));
        break ;
    case is_instruction_loop: {
        loop l = instruction_loop(i);
        statement b = loop_body(l);
        list new_loops =
                gen_nconc(gen_copy_seq(loops), CONS(STATEMENT, s, NIL)) ;
        list locals = NIL ;

        FOREACH(EFFECT, f, load_cumulated_rw_effects_list(b)) {
            entity e = effect_entity( f ) ;

            if(!anywhere_effect_p(f)
               && action_write_p( effect_action( f ))
               &&  entity_privatizable_in_loop_statement_p( e, s, even_globals)
               &&  gen_find_eq( e, locals ) == entity_undefined ) {
                locals = CONS( ENTITY, e, locals ) ;
            }
        }

        /* Add the loop index if it's privatizable because it does not have to be taken
           into account for parallelization. */
	//if (entity_privatizable_in_loop_statement_p( loop_index(l), s, even_globals))
	  loop_locals( l ) = CONS( ENTITY, loop_index( l ), locals ) ;

        /* FI: add the local variables of the loop body at least, but
           they might have to be added recursively for all enclosed
           loops. Note: their dependency pattern should lead to
           privatization, but they are eliminated from the body
           effect and not taken into consideration. */
        loop_locals(l) = gen_nconc(loop_locals(l),
                                   gen_copy_seq(statement_declarations(b)));

        scan_statement( b, new_loops, even_globals ) ;
        hash_del(get_enclosing_loops_map(), (char *) s) ;
        store_statement_enclosing_loops(s, new_loops);
        break;
    }
    case is_instruction_test: {
        test t = instruction_test( i ) ;

        scan_statement( test_true( t ), loops, even_globals ) ;
        scan_statement( test_false( t ), loops, even_globals ) ;
        break ;
    }
    case is_instruction_whileloop: {
        whileloop l = instruction_whileloop(i);
        statement b = whileloop_body(l);
        scan_statement(b, loops, even_globals ) ;
        break;
    }
    case is_instruction_forloop: {
        forloop l = instruction_forloop(i);
        statement b = forloop_body(l);
        scan_statement(b, loops, even_globals ) ;
        break;
    }
   case is_instruction_unstructured:
        scan_unstructured( instruction_unstructured( i ), loops, even_globals ) ;
        break ;
    case is_instruction_call:
    case is_instruction_expression:
    case is_instruction_goto:
        break ;
    default:
        pips_internal_error("unexpected tag %d", instruction_tag(i));
    }
}

static void scan_unstructured(unstructured u, list loops, bool even_globals)
{
    list blocs = NIL ;

    CONTROL_MAP( c, {scan_statement( control_statement( c ), loops, even_globals );},
                 unstructured_control( u ), blocs ) ;
    gen_free_list( blocs ) ;
}

/* LOOP_PREFIX returns the common list prefix of lists L1 and L2. */

list loop_prefix(list l1, list l2)
{
    statement st ;

    if( ENDP( l1 )) {
        return( NIL ) ;
    }
    else if( ENDP( l2 )) {
        return( NIL ) ;
    }
    else if( (st=STATEMENT( CAR( l1 ))) == STATEMENT( CAR( l2 ))) {
        return( CONS( STATEMENT, st, loop_prefix( CDR( l1 ), CDR( l2 )))) ;
    }
    else {
        return( NIL ) ;
    }
}

/* UPDATE_LOCALS removes the entity E from the locals of loops in LS that
   are not in common with the PREFIX. */

static void update_locals(list prefix, list ls, entity e)
{
  pips_debug(1, "Begin\n");

  if( ENDP( prefix )) {
    if(!ENDP(ls)) {
      ifdebug(1) {
        pips_debug(1, "Removing %s from locals of ", entity_name( e )) ;
        FOREACH(STATEMENT, st, ls) {
          pips_debug(1, "%td ", statement_number( st )) ;
        }
        pips_debug(1, "\n" ) ;
      }
      FOREACH(STATEMENT, st, ls) {
        instruction i = statement_instruction( st ) ;

        pips_assert( "instruction i is a loop", instruction_loop_p( i )) ;
        gen_remove( &loop_locals( instruction_loop( i )), e );
        pips_debug(1, "Variable %s is removed from locals of statement %td\n",
                   entity_name(e), statement_number(st));
      }
    }
    else {
      pips_debug(1, "ls is empty, end of recursion\n");
    }
  }
  else {
    pips_assert( "The first statements in prefix and in ls are the same statement", 
                 STATEMENT( CAR( prefix )) == STATEMENT( CAR( ls ))) ;

    pips_debug(1, "Recurse on common prefix\n");

    update_locals( CDR( prefix ), CDR( ls ), e ) ;
  }

  pips_debug(1, "End\n");
}

/* expression_implied_do_index_p
   return true if the given entity is the index of an implied do
   contained in the given expression. --DB
*/
static bool expression_implied_do_index_p(expression exp, entity e)
{
  bool li=false;
  bool dep=false;

  if (expression_implied_do_p(exp)) {
    list args = call_arguments(syntax_call(expression_syntax(exp)));
    expression arg1 = EXPRESSION(CAR(args)); /* loop index */
    expression arg2 = EXPRESSION(CAR(CDR(args))); /* loop range */
    entity index = reference_variable(syntax_reference(expression_syntax(arg1)));
    range r = syntax_range(expression_syntax(arg2));
    list range_effects;

    pips_debug(5, "begin\n");
    pips_debug(7, "%s implied do index ? index: %s\n",
               entity_name(e),entity_name(index));

    range_effects = proper_effects_of_range(r);

    FOREACH(EFFECT, eff, range_effects) {
      if (reference_variable(effect_any_reference(eff)) == e &&
          action_read_p(effect_action(eff))) {
          pips_debug(7, "index read in range expressions\n");
          dep=true;
      }
      free_effect(eff);
    }
    gen_free_list(range_effects);
  
    if (!dep) {
      if (same_entity_p(e,index))
        li=true;
      else {
        FOREACH(EXPRESSION,expr, CDR(CDR(args))) {
          syntax s = expression_syntax(expr);
          if(syntax_call_p(s)) {
            pips_debug(5,"Nested implied do\n");
            if (expression_implied_do_index_p(expr,e))
              li=true;
          }
        }
      }
    }
    pips_debug(5,"end\n");
  }
  return li;
}

/* is_implied_do_index
   returns true if the given entity is the index of one of the
   implied do loops of the given instruction. --DB
*/
bool is_implied_do_index(entity e, instruction ins)
{
  bool li = false;

  debug(5,"is_implied_do_index","entity name: %s ", entity_name( e )) ;

  if (instruction_call_p(ins))
    MAP(EXPRESSION,exp,{
      if (expression_implied_do_index_p(exp,e)) li=true;
    },call_arguments( instruction_call( ins ) ));

  ifdebug(5)
    fprintf(stderr, "%s\n", bool_to_string(li));

  return li;
}

/* TRY_PRIVATIZE knows that the effect F on entity E is performed in
   the statement ST of the vertex V of the dependency graph. Arrays
   are not privatized. */

static void try_privatize(vertex v, statement st, effect f, entity e)
{
  list ls ;

  /* BC : really dirty : overrides problems in the computation of
     effects for C programs; Should be fixed later. */
  if (anywhere_effect_p(f))
    {return;}

  /* Only scalar entities can be privatized */
  if( !entity_scalar_p( e )) {
    return ;
  }

  /* Only program variables can be privatized. This test may not be
     strong enough to guarantee that e is a program variable */
  if(!entity_variable_p(e)) {
    return;
  }

  ls = load_statement_enclosing_loops(st);

  ifdebug(1) {
    if(statement_loop_p(st)) {
      pips_debug(1, "Trying to privatize %s in loop statement %td (ordering %03zd) with local(s) ",
                 entity_local_name( e ), statement_number( st ), statement_ordering(st)) ;
      print_arguments(loop_locals(statement_loop(st)));
    }
    else {
      pips_debug(1, "Trying to privatize %s in statement %td\n",
                 entity_local_name( e ), statement_number( st )) ;
    }
  }

  FOREACH(SUCCESSOR, succ, vertex_successors(v)) {
    vertex succ_v = successor_vertex( succ ) ;
    dg_vertex_label succ_l =
      (dg_vertex_label)vertex_vertex_label( succ_v ) ;
    dg_arc_label arc_l =
      (dg_arc_label)successor_arc_label( succ ) ;
    statement succ_st =
      ordering_to_statement(dg_vertex_label_statement(succ_l));
    instruction succ_i = statement_instruction( succ_st ) ;
    list succ_ls = load_statement_enclosing_loops( succ_st ) ;

    /* this portion of code induced the erroneous privatization of
       non-private variables, for instance in :

       DO I = 1,10
       J = J +1
       a(I) = J
       ENDDO

       so I comment it out. But some loops are not parallelized
       anymore for instance Semantics/choles, Ricedg/private and
       Prettyprint/matmul.  (see ticket #152). BC.

       In fact, J is privatizable in the above loop if J is not
       initialized. We have to remember that PIPS maintains the
       semantics of well-defined code, but correctly output weird
       results when it does not matter. We also have to remember that
       a PIPS phase that has been used for almost 20 years is unlikely
       to be buggy with Fortran code. For C code, careful extensions
       might be needed: in this case, dynamic variables can be
       initialized and the initialization is not taken into account by
       the dependence graph (for the time being). Hence function
       "privatizable" was made a bit mor restrictive.

       The test commented out is correct, but not explained. SO I
       chose to change the behavior at a lower level, where it is
       easier to understand. Ticket #152 should probably be
       closed. FI.
    */
    /*
      if( v == succ_v) {
      continue ;
      }
    */

    FOREACH(CONFLICT, c, dg_arc_label_conflicts(arc_l)) {
      effect sc = conflict_source( c ) ;
      effect sk = conflict_sink( c ) ;

      if(store_effect_p(sc)) {
        /* Only store effects are considered for privatization */
        list prefix = list_undefined;

        pips_assert("Both effects sc and sk are of the same kind",
                    store_effect_p(sk));

        /* Take into account def-def and use-def arcs only */
        if(!entities_may_conflict_p( e, effect_entity( sc )) ||
           !entities_may_conflict_p( e, effect_entity( sk )) ||
           action_write_p( effect_action( sk))) {
          continue ;
        }
        /* PC dependance and the sink is a loop index */
        if(action_read_p( effect_action( sk )) &&
           (instruction_loop_p( succ_i) ||
            is_implied_do_index( e, succ_i))) {
          continue ;
        }
        pips_debug(5,"Conflict for %s between statements %td and %td\n",
                   entity_local_name(e),
                   statement_number(st),
                   statement_number(succ_st));

        if (v==succ_v) {
          /* No decision can be made from this couple of effects alone */
          ;
          //pips_debug(5,"remove %s from locals in all enclosing loops\n",
          //           entity_local_name(e));
          //update_locals( NIL, ls, e ); /* remove e from all enclosing loops */
        }
        else {
          pips_debug(5,"remove %s from locals in non common enclosing loops\n",
                     entity_local_name(e));
          prefix = loop_prefix( ls, succ_ls ) ;
          /* e cannot be a local variable at a lower level than
             the common prefix because of this dependence
             arc. */
          update_locals( prefix, ls, e ) ;
          update_locals( prefix, succ_ls, e ) ;
          gen_free_list( prefix ) ;
        }
      }
    }
  }

  pips_debug(1, "End\n");
}

/* PRIVATIZE_DG looks for definition of entities that are locals to the loops
   in the dependency graph G for the control graph U. */

bool generic_privatize_module(char *mod_name, bool even_globals)
{
    entity module;
    statement mod_stat;
    graph mod_graph;

    set_current_module_entity(module_name_to_entity(mod_name) );
    module = get_current_module_entity();

    set_current_module_statement( (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true) );
    mod_stat = get_current_module_statement();

    set_proper_rw_effects((statement_effects)
        db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

    set_cumulated_rw_effects((statement_effects)
        db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, true) );

    if (even_globals)
      {
	set_constant_paths_p(true);
	set_pointer_info_kind(with_no_pointer_info);
	set_methods_for_simple_effects();
	set_methods_for_live_paths(mod_name);
	set_live_out_paths((*db_get_live_out_paths_func)(mod_name));
  	set_live_in_paths((*db_get_live_in_paths_func)(mod_name));
    }

    mod_graph = (graph)
        db_get_memory_resource(DBR_CHAINS, mod_name, true);

    debug_on("PRIVATIZE_DEBUG_LEVEL");
    pips_debug(1, "\n begin for module %s\n\n", mod_name);
    set_ordering_to_statement(mod_stat);

    /* Set the prettyprint language for debug */
    value mv = entity_initial(module);
    if(value_code_p(mv)) {
      code c = value_code(mv);
      set_prettyprint_language_from_property(language_tag(code_language(c)));
    } else {
      /* Should never arise */
      set_prettyprint_language_from_property(is_language_fortran);
    }

    /* Build maximal lists of private variables in loop locals */
    /* scan_unstructured(instruction_unstructured(mod_inst), NIL); */
    scan_statement(mod_stat, NIL, even_globals);

    /* remove non private variables from locals */
    FOREACH(VERTEX, v, graph_vertices( mod_graph )) {
        dg_vertex_label vl = (dg_vertex_label) vertex_vertex_label( v ) ;
        statement st =
            ordering_to_statement(dg_vertex_label_statement(vl));

        pips_debug(1, "Entering statement %03zd :\n", statement_ordering(st));
        ifdebug(4) {
          print_statement(st);
        }

        FOREACH(EFFECT, f, load_proper_rw_effects_list( st )) {
            entity e = effect_entity( f ) ;
            ifdebug(4) {
              pips_debug(1, "effect :");
              print_effect(f);
            }
            if( action_write_p( effect_action( f ))) {
                try_privatize( v, st, f, e ) ;
            }
        }
    }

    /* sort locals
     */
    sort_all_loop_locals(mod_stat);

    debug_off();
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stat);
    DB_PUT_FILE_RESOURCE(DBR_PRIVATIZED, mod_name, strdup(""));

    reset_current_module_entity();
    reset_current_module_statement();
    reset_proper_rw_effects();
    reset_cumulated_rw_effects();
    if (even_globals)
      {
	reset_live_out_paths();
	reset_live_in_paths();
      }

    reset_ordering_to_statement();
    clean_enclosing_loops();

    return true;
}

bool privatize_module(char *mod_name)
{
  return generic_privatize_module(mod_name, false);
}

bool privatize_module_even_globals(char *mod_name)
{
  return generic_privatize_module(mod_name, true);
}



/**
 * @name localize declaration
 * @{ */

static void do_gather_loop_indices(loop l, set s) {
        set_add_element(s,s,loop_index(l));
}

static set gather_loop_indices(void *v) {
        set s = set_make(set_pointer);
        gen_context_recurse(v,s,loop_domain,gen_true,do_gather_loop_indices);
        return s;
}

/**
 * gen_recurse context for localize declaration, keep track of the scope !
 */
#define MAX_DEPTH 100
typedef struct {
  int depth;
  int scope_numbers[MAX_DEPTH];
  hash_table old_entity_to_new;
} localize_ctx;

/**
 * Create a context
 */
static localize_ctx make_localize_ctx() {
  localize_ctx ctx;
  ctx.depth= -1;
  memset(ctx.scope_numbers, -1, MAX_DEPTH*sizeof(ctx.scope_numbers[0]));
  ctx.old_entity_to_new = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);

  return ctx;
}

/**
 * Keep track of the scope during recursion
 */
static bool localize_track_scope_in(statement s, localize_ctx *ctx) {
  if(statement_block_p(s)) {
    pips_assert("depth is in acceptable range\n", ctx->depth<MAX_DEPTH);
    ctx->depth++;
    ctx->scope_numbers[ctx->depth]++;
  }
  return true;
}

/**
 * Keep track of the scope during recursion
 */
static void localize_track_scope_out(statement s, localize_ctx *ctx) {
  if(statement_block_p(s)) {
    ctx->depth--;
  }
}


/**
 * Create an (unique) entity taking into account the current scope
 */
static entity make_localized_entity(entity e, localize_ctx *ctx) {
  const char * module_name = get_current_module_name();
  string build_localized_name = strdup("");
  for(int i=1;i<=ctx->depth;i++) {
    string new_name;
    asprintf(&new_name,"%s%d" BLOCK_SEP_STRING,build_localized_name,ctx->scope_numbers[i]);
    free(build_localized_name);
    build_localized_name = new_name;
  }
  string localized_name;
  asprintf(&localized_name,"%s%s",build_localized_name, entity_user_name(e));
  free(build_localized_name);

  string unique_name = strdup(localized_name);
  int count = 0;
  while(!entity_undefined_p(FindEntity(module_name,unique_name))) {
    free(unique_name);
    asprintf(&unique_name,"%s%d",localized_name, ++count);
    pips_assert("infinite loop ?",count<10000);
  }
  free(localized_name);

  entity new_ent =  FindOrCreateEntity(module_name,unique_name);
  entity_type(new_ent)=copy_type(entity_type(e));
  entity_initial(new_ent) = make_value_unknown();

  entity f = local_name_to_top_level_entity(module_name);
  entity a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);

  type pvt      = ultimate_type(entity_type(new_ent));
  variable pvtv = type_variable(pvt);
  basic pvb     = variable_basic(pvtv);

  int offset = 0;
  if (c_module_p(get_current_module_entity()))
    offset = (basic_tag(pvb)!=is_basic_overloaded)?
      (add_C_variable_to_area(a, e)):(0);
  else
    offset = (basic_tag(pvb)!=is_basic_overloaded)?
      (add_variable_to_area(a, e)):(0);
  
  entity_storage(new_ent) = make_storage(is_storage_ram,
					 make_ram(f, a, offset, NIL));

  return new_ent;
}


/**
 * @brief Create a statement block around the statement if it is a do-loop with
 * local/private variable
 *
 * It creates statement_blocks where needed to hold further declarations later
 * And then performs localization based on the locals field
 *
 * @param s concerned statement
 */
static void localize_declaration_walker(statement s, localize_ctx *ctx) {
  if(statement_loop_p(s)) {
    instruction i = statement_instruction(s);
    loop l = instruction_loop(i);

    /* create a new statement to hold the future private declaration SG:
       as a side effect, pragmas on the loop are moved to the enclosing
       block __this_is_usefull__ at least to me ^^
    */
    if(!ENDP(loop_locals(l))) {
        /* Put the loop in a new statement block if there are loop-private variable(s): */
        statement new_statement = instruction_to_statement(i);
        instruction iblock = make_instruction_block(CONS(STATEMENT,new_statement,NIL));
        statement_instruction(s) = iblock;
        /* keep comments and extensions attached to the loop */
        string stmp = statement_comments(s);
        statement_comments(s)=statement_comments(new_statement);
        statement_comments(new_statement)=stmp;

        extensions ex = statement_extensions(new_statement);
        statement_extensions(new_statement) = statement_extensions(s);
        statement_extensions(s) = ex;

        entity etmp = statement_label(s);
        statement_label(s) = statement_label(new_statement);
        statement_label(new_statement) = etmp;

        intptr_t itmp = statement_number(s);
        statement_number(s) = statement_number(new_statement);
        statement_number(new_statement) = itmp;

        /* now add declarations to the created block */
        list locals = gen_copy_seq(loop_locals(l));
        list sd = statement_to_declarations(s);
        set li = gather_loop_indices(s);
        bool skip_loop_indices = get_bool_property("LOCALIZE_DECLARATION_SKIP_LOOP_INDICES");

        // take into account that we are in a new block
        localize_track_scope_in(s,ctx);
        pips_debug(1,"Handling a loop with scope (%d,%d)",ctx->depth,ctx->scope_numbers[ctx->depth]);

        FOREACH(ENTITY,e,locals)
        {
          if(!entity_in_list_p(e,sd)
	     && !same_entity_p(e,loop_index(l)) // do not localize this one, or initialize its value!
	     && !(skip_loop_indices && set_belong_p(li,e))) {
            /* create a new name for the local entity */
            entity new_entity = make_localized_entity(e,ctx);
            pips_debug(1,"Creating localized entity : %s (from %s)\n",entity_name(new_entity), entity_name(e));
            /* get the new entity and initialize it and register it*/
            /* add the variable to the loop body if it's not an index */
            AddLocalEntityToDeclarations(new_entity,get_current_module_entity(), loop_body(l));

            list previous_replacements = hash_get(ctx->old_entity_to_new,e);
            if( previous_replacements == HASH_UNDEFINED_VALUE )
                previous_replacements = CONS(ENTITY,new_entity,NIL);
            previous_replacements=gen_nconc(previous_replacements,CONS(ENTITY,e,NIL));
            hash_put(ctx->old_entity_to_new,e,previous_replacements);
            FOREACH(ENTITY,prev,previous_replacements)  {
                replace_entity(s,prev,new_entity);
            }
          }
        }

        // take into account that we exit a block
        localize_track_scope_out(s,ctx);

        set_free(li);
        gen_free_list(locals);
        gen_free_list(sd);
    }
  } else {
    localize_track_scope_out(s, ctx);
  }

}

/**
 * @brief make loop local variables declared in the innermost statement
 *
 * @param mod_name name of the module being processed
 *
 * @return
 */
bool localize_declaration(char *mod_name) {
        /* prelude */
        debug_on("LOCALIZE_DECLARATION_DEBUG_LEVEL");
        pips_debug(1,"begin localize_declaration ...\n");
        set_current_module_entity(module_name_to_entity(mod_name) );
        set_current_module_statement( (statement) db_get_memory_resource(DBR_CODE, mod_name, true) );

        /* Propagate local informations to loop statements */

        // To keep track of what has been done:
        ifdebug(1) {
          pips_debug(1,"The statement before we create block statement:\n");
          print_statement(get_current_module_statement());
        }

        // Create the statement_block where needed and perform localization
        clean_up_sequences(get_current_module_statement());

        // Context for recursion
        localize_ctx ctx = make_localize_ctx();

        gen_context_recurse(get_current_module_statement(),&ctx,
                            statement_domain,localize_track_scope_in,localize_declaration_walker);
        ifdebug(1) {
          pips_debug(1,"The statement before we convert loop_locals to local declarations:\n");
          print_statement(get_current_module_statement());
        }
        // Use loop_locals data to fill local declarations:
        clean_up_sequences(get_current_module_statement());
        hash_table_free(ctx.old_entity_to_new);

        ifdebug(1) {
          pips_debug(1,"The statement after conversion:\n");
          print_statement(get_current_module_statement());
        }

        /* Renumber the statement with a new ordering */
        module_reorder(get_current_module_statement());

        /* postlude */
        debug_off();

        /* Apply clean declarations ! */
        debug_on("CLEAN_DECLARATIONS_DEBUG_LEVEL");
        set_cumulated_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                      mod_name,
                                                      true));
        module_clean_declarations(get_current_module_entity(),
                                  get_current_module_statement());
        reset_cumulated_rw_effects();
        debug_off();

        DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, get_current_module_statement());

        reset_current_module_entity();
        reset_current_module_statement();
        pips_debug(1,"end localize_declaration\n");
        return true;
}

/**
    @brief update the input loop loop_locals by removing entities
           with no corresponding effects in loop body (e.g. entities
           already private in inner loops and not used in other
           statements of the current loop body).
    @param l is the loop to operate one
    @param changed track if the list have been changed
 */
static void update_loop_locals(loop l, bool *changed)
{
  statement body = loop_body(l);
  list body_effects = load_rw_effects_list(body);
  ifdebug(1) {
    fprintf(stderr, "new body effects:\n");
    print_effects(body_effects);
  }
  list new_loop_locals = NIL;
  FOREACH(ENTITY, private_variable, loop_locals(l)) {
    pips_debug(1, "considering entity %s\n",
               entity_local_name(private_variable));
    if(effects_may_read_or_write_memory_paths_from_entity_p(body_effects,
                                                            private_variable)) {
      pips_debug(1, "keeping entity\n");
      new_loop_locals = CONS(ENTITY, private_variable, new_loop_locals);
    } else {
      // This is a change :)
      *changed = true;
    }
  }

  gen_free_list(loop_locals(l));
  loop_locals(l) = new_loop_locals;
  ifdebug(1) {
    print_entities(loop_locals(l));
    fprintf(stderr, "\n");
  }
}


/**
   @brief update loop_locals found by privatize_module by taking parallel loops
   into account
 */
bool update_loops_locals(const char* module_name, statement module_stat)
{
  init_proper_rw_effects();
  init_rw_effects();
  init_invariant_rw_effects();
  set_current_module_entity(module_name_to_entity(module_name));

  debug_on("EFFECTS_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
    set_constant_paths_p(false);
  else
    set_constant_paths_p(true);
  set_pointer_info_kind(with_no_pointer_info); /* should use current pointer information
                                                  according to current effects active phase
                                                */

  set_methods_for_proper_simple_effects();
  proper_effects_of_module_statement(module_stat);
  set_methods_for_simple_effects();
  rw_effects_of_module_statement(module_stat);

  bool changed = false; // Keep track of a change
  gen_context_recurse(module_stat,&changed,
              loop_domain, gen_true, update_loop_locals);

  pips_debug(1, "end, %s\n",(changed) ? "There was at least a change": "There was no change at all");
  debug_off();

  /* Hope that these close actually free the effects in the mappings */
  close_proper_rw_effects();
  close_rw_effects();
  close_invariant_rw_effects();
  generic_effects_reset_all_methods();
  reset_current_module_entity();

  return changed;
}



/**  @} */
