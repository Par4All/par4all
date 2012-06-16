/*
 Copyright 2012 MINES ParisTech
 Copyright 2012 Silkan

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

/**
 * @file starpu_pragma_generation.c
 * Task generation
 * @author Mehdi Amini <mehdi.amini@silkan.com>
 */

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects-util.h"
#include "gpu.h"
#include "accel-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects.h"
#include "effects-simple.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "expressions.h"
#include "text-util.h"
#include "parser_private.h"

static statement get_statement_pragma_register(entity e, bool address_of) {
  statement pragma = make_plain_continue_statement();
  string str;
  if(address_of)
    asprintf(&str,"starpu register &%s sizeof(%s)",entity_user_name(e),entity_user_name(e));
  else
    asprintf(&str,"starpu register %s",entity_user_name(e));

  add_pragma_str_to_statement(pragma,str,false);
  return pragma;
}

static statement get_statement_pragma_init() {
  statement pragma = make_plain_continue_statement();
  add_pragma_str_to_statement(pragma,"starpu initialize",true);
  return pragma;
}
static statement get_statement_pragma_shutdown() {
  statement pragma = make_plain_continue_statement();
  add_pragma_str_to_statement(pragma,"starpu shutdown",true);
  return pragma;
}

struct context_address_of_entity {
  bool found;
  entity e;
};

bool address_of_entity_p(call c, struct context_address_of_entity *context) {
  if(ENTITY_ADDRESS_OF_P(call_function(c))) {
    expression arg = EXPRESSION(CAR(call_arguments(c)));
    if(expression_to_entity(arg)==context->e) {
      context->found=true;
    }
  }
  return !context->found;
}

static bool address_of_variable_is_taken(void *start, entity e) {
  struct context_address_of_entity c = { false, e };
  gen_context_recurse(start,&c,call_domain,address_of_entity_p,gen_true);
  return c.found;
}

static void scalar_to_array(reference r, entity e) {
  if(reference_variable(r)==e) {
    pips_assert("no index for scalar",ENDP(reference_indices(r)));
    expression zero = entity_to_expression(make_integer_constant_entity(0));
    reference_indices(r) = CONS(expression,zero,NIL);
  }
}

static bool add_pragma_to_sequence(sequence seq) {
  list stmts = sequence_statements(seq);
  list registered_entities = NIL;
  list last_stmt = NIL;
  for( ; !ENDP(stmts) ;  POP(stmts) ) {
    statement s = STATEMENT(CAR(stmts));
    last_stmt = stmts;
    if(declaration_statement_p(s)) {
      FOREACH(entity,e,statement_declarations(s)) {
        if(entity_array_p(e)) {
          statement pragma = get_statement_pragma_register(e,false);
          CDR(stmts) = CONS(statement, pragma, CDR(stmts) );
          POP(stmts);
          registered_entities = CONS(entity,e,registered_entities);
        } else if(address_of_variable_is_taken(seq,e)) {
          type t = entity_type(e);
          pips_assert("is a variable",type_variable_p(t));
          variable v = type_variable(t);
          pips_assert("scalar !", ENDP(variable_dimensions(v)));
          dimension d = make_dimension(entity_to_expression(make_integer_constant_entity(0)),
                                       entity_to_expression(make_integer_constant_entity(0)));
          variable_dimensions(v) = CONS(dimension,d,NIL);
          gen_context_recurse(seq,e,reference_domain,gen_true,scalar_to_array);
          statement pragma = get_statement_pragma_register(e,false);
          CDR(stmts) = CONS(statement, pragma, CDR(stmts) );
          POP(stmts);

          registered_entities = CONS(entity,e,registered_entities);
        }
      }
    }
  }
  FOREACH(entity,e,registered_entities) {
    statement pragma = make_plain_continue_statement();
    string str;
    asprintf(&str,"starpu unregister %s",entity_user_name(e));
    add_pragma_str_to_statement(pragma,str,false);
    CDR(last_stmt) = CONS(statement, pragma, NIL );
    POP(last_stmt);
  }
  return true;
}

bool array_bounded_p(entity e) {
  type array_t = ultimate_type(entity_type(e));
  if(array_type_p(array_t)) {
    variable v = type_variable(array_t);
    FOREACH(dimension,d,variable_dimensions(v)) {
      if(unbounded_dimension_p(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}


bool add_shutdown_pragma_to_return(statement s) {
  if(return_statement_p(s)) {
    add_pragma_str_to_statement(s,"starpu shutdown",true);
  }
  return true;
}

bool generate_starpu_pragma(char * module_name) {

  statement module_stat = (statement)db_get_memory_resource(DBR_CODE,
                                                            module_name,
                                                            true);
  set_current_module_statement(module_stat);

  set_current_module_entity(local_name_to_top_level_entity(module_name));

  debug_on("GENERATE_STARPU_PRAGMA_DEBUG_LEVEL");


  /* regions */
//  set_proper_rw_effects((statement_effects) db_get_memory_resource(DBR_REGIONS, module_name, true));
  set_proper_rw_effects((statement_effects) db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
  set_cumulated_rw_effects((statement_effects) db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                      module_name,
                                                                      true));


  /* Initialize set for each statement */
  gen_recurse(module_stat, sequence_domain,
      add_pragma_to_sequence,
      gen_identity);



  // If module is main, then register global variables
  if(entity_main_module_p(get_current_module_entity())) {
    list regs = load_cumulated_rw_effects_list(module_stat);
    ifdebug(2) {
      pips_debug(0,"Regions for main module : ");
      print_regions(regs);
    }
    // Do not use regions currently because of anywhere effects !
 //   FOREACH(effect, reg, regs) {
 //     entity e = region_entity(reg);
    list entities = gen_filter_tabulated(gen_true, entity_domain);
    FOREACH(entity, e, entities ) {
      pips_debug(6,"Considering entity %s\n",entity_name(e));
      if(top_level_entity_p(e)
          && array_bounded_p(e)) {
        pips_debug(2,"Registering global array %s in main\n",entity_user_name(e));
        statement pragma = get_statement_pragma_register(e,false);
        insert_statement(module_stat,pragma,true);
      }
    }
    insert_statement(module_stat,get_statement_pragma_init(),true);
    insert_statement(module_stat,get_statement_pragma_shutdown(),false);
    gen_recurse(module_stat, statement_domain, add_shutdown_pragma_to_return, gen_true);
  }


  module_reorder(get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
      module_name,
      module_stat);

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
       compute_callees(get_current_module_statement()));


  debug_off();

  reset_proper_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_cumulated_rw_effects();

  return true;
}

