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
/*
 * replace reductions with atomic op !
 * debug driven by REDUCTIONS_DEBUG_LEVEL
 *
 * Mehdi Amini, April 2011.
 */

#include "local-header.h"
#include "properties.h"
#include "syntax.h"
#include "reductions.h"
#include "reduction.h"
#include "callgraph.h"

GENERIC_LOCAL_FUNCTION(statement_reductions, pstatement_reductions)


/** Profiles for supported atomic operations
 * FIXME should be newgen tag
 */
enum atomic_op { UNKNOWN_OP, ADD, SUB, INC, DEC, AND, OR, XOR };
enum atomic_type { UNKNOWN_TYPE, DOUBLE, FLOAT, INT, LONG_INT };


/* Define an atomic operation
 * FIXME : should be moved under newgen management
 */
typedef struct atomic_operation {
  enum atomic_op op;
  enum atomic_type type;
} atomic_operation;


/* Stucture that define the list of atomic_operations */
typedef struct atomic_profile {
  atomic_operation *profile;
  int profile_size;
} atomic_profile;

static atomic_operation cuda[] = {
                           { ADD, INT },
                           { ADD, LONG_INT },
                           { INC, INT },
                           { INC, LONG_INT },
                           { DEC, INT },
                           { DEC, LONG_INT },
                           { SUB, INT },
                           { SUB, LONG_INT },
                           { OR, INT },
                           { OR, LONG_INT },
                           { AND, INT },
                           { AND, LONG_INT },
                           { XOR, INT },
                           { XOR, LONG_INT },
};


static atomic_profile load_atomic_profile() {
  struct atomic_profile profile = { NULL, 0 };
  const char* s_profile = get_string_property("ATOMIC_OPERATION_PROFILE");
  if(same_string_p(s_profile,"cuda")) {
    profile.profile = cuda;
    profile.profile_size = sizeof(cuda)/sizeof(atomic_operation);
  } else {
    pips_user_error("Unknown profile for atomic ops : '%s'\n",profile);
  }
  return profile;
}


typedef enum {
  DO_REPLACE,
  DO_NOT_REPLACE,
  USE_OPENMP
} process_reductions_mode;
/**
 * Keep track of what is done in replace recursion
 */
struct replace_ctx {
  bool replace;
  bool replaced;
  bool unsupported;
  atomic_profile *profile;
  list *replaced_statements;
};

/**
 * Get an atomic function that replace a reduction operator
 */
static entity atomic_function_of_operation(atomic_operation op) {

  type t = type_undefined;
  string type_suffix = "";
  switch(op.type) {
    case DOUBLE:
      t = MakeDoubleprecisionResult();
      type_suffix = "Double";
      break;
    case FLOAT:
      t = MakeRealResult();
      type_suffix = "Float";
      break;
    case INT:
      t = MakeIntegerResult();
      type_suffix = "Int";
      break;
    case LONG_INT:
      t = MakeLongIntegerResult();
      type_suffix = "LongInt";
      break;
    default:
      pips_internal_error("Unsupported type\n");
  }

  string name;
  switch(op.op) {
    case ADD:
      name = "Add";
      break;
    case SUB:
      name = "Sub";
      break;
    case INC:
      name = "Inc";
      break;
    case DEC:
      name = "Dec";
      break;
    case AND:
      name = "And";
      break;
    case OR:
      name = "Or";
      break;
    case XOR:
      name = "Xor";
      break;
    default:
      pips_internal_error("Unsupported op tag\n");
  }

  string prefix = "atomic"; // FIXME property

  name = strdup(concatenate(prefix,name,type_suffix,NULL));
  entity atomic_func = make_empty_function(name, t,make_language_c());
  free(name);
  return atomic_func;

}

static atomic_operation reduction_to_atomic_operation(reduction r) {
  atomic_operation aop = { UNKNOWN_OP, UNKNOWN_TYPE };
  reduction_operator ro = reduction_op(r);
  switch(reduction_operator_tag(ro)) {
    case is_reduction_operator_sum: aop.op = ADD; break;
    case is_reduction_operator_and: aop.op = AND; break;
    case is_reduction_operator_or: aop.op = OR; break;
    case is_reduction_operator_csum: ;
    case is_reduction_operator_none: ;
    case is_reduction_operator_prod:        ;
    case is_reduction_operator_min:         ;
    case is_reduction_operator_max:         ;
    case is_reduction_operator_bitwise_or:  ;
    case is_reduction_operator_bitwise_xor: ;
    case is_reduction_operator_bitwise_and: ;
    case is_reduction_operator_eqv:         ;
    case is_reduction_operator_neqv:        ;
    default: pips_user_warning("No atomic operation for this reduction operator !\n");
  }

  //reference rr = reduction_reference(r);
  // FIXME : get type of reference
  aop.type = INT;

  return aop;

}



static bool supported_atomic_operator_p(atomic_operation op,
                                        atomic_profile *profile) {
  bool found_p = false;
  for(int i=0; i<profile->profile_size; i++) {
    // Loop over supported operation and look is we support requested op
    if(profile->profile[i].op == op.op
       && profile->profile[i].type == op.type ) {
      found_p = true;
      break;
    }
  }
  return found_p;
}

/* Replace a reduction in a statement by a pattern matching way */
static bool replace_reductions_in_statement( statement s, struct replace_ctx *ctx ) {
  reductions rs = (reductions)load_statement_reductions(s);
  if(gen_in_list_p(s,*ctx->replaced_statements)) {
    pips_debug(1,"Already handled statement !\n");
    ctx->replaced = true;
  } else {
    if(rs && reductions_list(rs) && statement_call_p(s)) {
      ifdebug(4) {
        pips_debug(4,"Statement has a reduction ");
        print_statement(s);
      }
      if(gen_length(reductions_list(rs)) > 1) {
        pips_user_warning("Don't know how to handle multiple reductions on a "
            "statement ! Abort...\n");
        return false;
      }

      reduction r = REDUCTION_CAST(CAR(reductions_list(rs)));
      atomic_operation op = reduction_to_atomic_operation(r);
      if(!supported_atomic_operator_p(op, ctx->profile)) {
        ifdebug(1) {
          pips_debug(1,"Unsupported reduction by atomic operation profile : ");
          print_reduction(r);
          fprintf(stderr, "\n");
        }
      } else {
        expression atomic_param =
            get_complement_expression(s, reduction_reference(r));
        if(expression_undefined_p(atomic_param)) {
          ifdebug(1) {
            pips_debug(1,"Didn't manage to get complement expression :(\n");
          }
        } else {
          entity atomic_fun = atomic_function_of_operation(op);
          pips_assert("have a valid function",!entity_undefined_p(atomic_fun));
          if(ctx->replace) {
            reference ref = copy_reference(reduction_reference(r));
            expression addr_of_ref =
              MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                            reference_to_expression(ref));
            list args = CONS(EXPRESSION,
                             addr_of_ref,
                             CONS(EXPRESSION,
                                  int_to_expression(1),
                                  NULL));
            statement_instruction(s) = make_call_instruction(atomic_fun, args);
            ifdebug(4) {
              pips_debug(4,"Atomized statement : ");
              print_statement(s);
            }
          }
          ctx->replaced = true;
          *ctx->replaced_statements = CONS(statement,s,*ctx->replaced_statements);
        }
      }
    }
  }

  return true;
}


static bool process_reductions_in_loop(statement loop_stat,
                                       atomic_profile *profile,
                                       list *replaced_statements,
                                       bool replace) {
  bool ret = true;
  if(statement_loop_p(loop_stat) && !loop_parallel_p(statement_loop(loop_stat))) {
    pips_debug(1,"Loop is parallel only with reduction, fetching reductions...\n");
    loop l = statement_loop(loop_stat);
//    reduction_reference(r);
    reductions rs = (reductions)load_statement_reductions(loop_stat);
    if(rs && !ENDP(reductions_list(rs)) ) { 
      pips_debug(1,"Loop has a reduction ! Let's replace reductions inside\n");
      struct replace_ctx ctx = { replace, false, false, profile, replaced_statements };
      gen_context_recurse(l,
                          &ctx,
                          statement_domain,
                          replace_reductions_in_statement,
                          gen_null);
      if(!ctx.replaced) {
        pips_debug(1,"No reduction replaced for this loop, make it sequential "
                   "to be safe :-(\n");
        execution_tag(loop_execution(l)) = is_execution_sequential;
      } else if(ctx.unsupported) {
        pips_debug(1,"Unsupported reduction found ! Make it sequential "
                           "to be safe :-(\n");
        execution_tag(loop_execution(l)) = is_execution_sequential;
      } else {
        execution_tag(loop_execution(l)) = is_execution_parallel;
        ret = false;
      }
    }
  }
  return ret;
}

static bool process_reductions_in_openmp_loop(statement st) {
  pips_assert("in a loop", statement_loop_p(st));
  loop l = statement_loop(st);
  bool ret = false;
  if (!(ret=omp_pragma_expr_for_reduction (l, st, false)))
    ret=omp_pragma_expr_for (l, st);
  if(ret)
      execution_tag(loop_execution(l))=is_execution_parallel;
  return ret;
}


static bool process_reduced_loops(const char *mod_name, process_reductions_mode mode) {
  atomic_profile current_profile = load_atomic_profile();
  if(!current_profile.profile && current_profile.profile_size) {
    return false;
  }

  set_current_module_entity(local_name_to_top_level_entity(mod_name));
  statement module_stat = (statement)db_get_memory_resource(DBR_CODE, mod_name, true);
  set_current_module_statement(module_stat);

  // Load detected reductions
  set_statement_reductions(
      (pstatement_reductions)db_get_memory_resource(DBR_CUMULATED_REDUCTIONS,
                                                    mod_name,
                                                    true));
  set_printed_reductions(
      (pstatement_reductions)db_get_memory_resource(DBR_CUMULATED_REDUCTIONS,
                                                    mod_name,
                                                    true));

  // Load targeted loops, those that need reductions to be parallel !
  reduced_loops loops = (reduced_loops)db_get_memory_resource(DBR_REDUCTION_PARALLEL_LOOPS,
                                                           mod_name,
                                                           true);
  list replaced_statements = NIL;
  set_ordering_to_statement(module_stat);
  FOREACH(int, ordering, reduced_loops_ordering(loops)) {
    statement s = ordering_to_statement(ordering);
      switch(mode) {
        case DO_REPLACE:
          process_reductions_in_loop(s,&current_profile,&replaced_statements, true);
          break;
        case DO_NOT_REPLACE:
          process_reductions_in_loop(s,&current_profile,&replaced_statements, false);
          break;
        case USE_OPENMP:
          process_reductions_in_openmp_loop(s);
          break;
      }
  }
  reset_ordering_to_statement();

  if (!statement_consistent_p(module_stat)) {
    pips_internal_error("Statement is inconsistent ! No choice but abort...\n");
  }


  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
                         mod_name,
                         (char*) module_stat);

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name,
       compute_callees(get_current_module_statement()));

  reset_statement_reductions();
  reset_printed_reductions();
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}


/**
 * Replace reduction with atomic operations
 */
bool replace_reduction_with_atomic( string mod_name) {
  debug_on("REPLACE_REDUCTION_WITH_ATOMIC_DEBUG_LEVEL");

  bool result = process_reduced_loops(mod_name, DO_REPLACE);

  debug_off();

  return result;
}

/**
 * Flag loop as parallel when replacement with atomic is possible without doing
 * the replacement
 */
bool flag_parallel_reduced_loops_with_atomic( string mod_name) {
  debug_on("FLAG_PARALLEL_REDUCED_LOOPS_WITH_ATOMIC_DEBUG_LEVEL");

  bool result = process_reduced_loops(mod_name, DO_NOT_REPLACE);

  debug_off();

  return result;

}

/**
 * Flag loop as parallel with OpenMP directives taking reductions into
 * account
 */
bool flag_parallel_reduced_loops_with_openmp_directives(const char * mod_name) {
  debug_on("FLAG_PARALLEL_REDUCED_LOOPS_WITH_ATOMIC_DEBUG_LEVEL");

  bool result = process_reduced_loops(mod_name, USE_OPENMP);

  debug_off();

  return result;

}
