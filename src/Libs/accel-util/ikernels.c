/*
 Copyright 1989-2010 MINES ParisTech
 Copyright 2010 HPC Project

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
 * @file ikernels.c
 * kernels manipulation
 * @author Mehdi Amini <mehdi.amini@hpc-project.com>
 */

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "kernel_memory_mapping.h"
#include "effects.h"
#include "ri-util.h"
#include "semantics.h"
#include "effects-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "preprocessor.h"
#include "transformer.h"
#include "accel-util.h"

/**
 *
 */
bool user_call_p(call c) {
  entity f = call_function(c);
  value v = entity_initial(f);
  return value_code_p(v);
}


/** Make a sizeof expression
 *
 */
static expression makeTransfertSizeExpression( entity e ) {
  type array_type = entity_type(e);
  pips_assert("Must be a variable type", type_variable_p(array_type));
  variable v_type = type_variable(array_type);


  // FIXME free memory
  type element_type = make_type_variable(
            make_variable(
                copy_basic(variable_basic(v_type)),
                NIL,NIL)
            );

  expression transfer_size = SizeOfDimensions(variable_dimensions(v_type));
  transfer_size=MakeBinaryCall(
          entity_intrinsic(MULTIPLY_OPERATOR_NAME),
          make_expression(
              make_syntax_sizeofexpression(make_sizeofexpression_type(element_type)),
              normalized_undefined
              ),
          transfer_size);

  return transfer_size;
}

/**
 * converts a region_to_dma_switch to corresponding dma name
 * according to properties
 */
static const char* get_dma_name(enum region_to_dma_switch m, int d) {
  char *seeds[] = { "KERNEL_LOAD_STORE_LOAD_FUNCTION",
                    "KERNEL_LOAD_STORE_STORE_FUNCTION",
                    "KERNEL_LOAD_STORE_ALLOCATE_FUNCTION",
                    "KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION" };
  const char * propname = seeds[(int)m];
  /* If the DMA is not scalar, the DMA function name is in the property
   of the form KERNEL_LOAD_STORE_LOAD/STORE_FUNCTION_dD: */
  char * apropname=NULL;
  if(d > 0 && (int)m < 2)
    asprintf(&apropname, "%s_%dD", seeds[(int)m], (int)d);
  const char* dmaname = get_string_property(propname);
  if(apropname)
    free(apropname);
  pips_debug(6,"Get dma name : %s\n",dmaname);
  return dmaname;
}

static statement make_dma_transfert(entity e, enum region_to_dma_switch dma) {

  const char* function_name = get_dma_name(dma, 0);
  entity mcpy = module_name_to_entity(function_name);
  if(entity_undefined_p(mcpy)) {
    mcpy
        = make_empty_subroutine(function_name,
                                copy_language(module_language(get_current_module_entity())));
    pips_user_warning("Cannot find \"%s\" method. Are you sure you have set\n"
        "KERNEL_LOAD_STORE_..._FUNCTION "
        "to a defined entity and added the correct .c file?\n",function_name);
  } else {
    AddEntityToModuleCompilationUnit(mcpy, get_current_module_entity());
  }
  list args = NIL;
  /*
  expression entity_exp = entity_to_expression(e);
  if(entity_pointer_p(e)) {
    entity_exp = MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
       entity_to_expression(e));
  }

  expression size_of_e = MakeSizeofExpression(entity_exp);*/
  expression size_of_e = makeTransfertSizeExpression(e);

  args = make_expression_list(entity_to_expression(e),size_of_e);

  call c = make_call(mcpy, args);
  return instruction_to_statement(make_instruction_call(c));
}

#define INIT_STATEMENT_SIZE 20
#define INIT_ENTITY_SIZE 10

/* Macro to create set */
#define MAKE_SET() (set_make( set_pointer ))

/* copy_from_in maps each statement to the "copy from" data that are in-coming
 * the statement
 * */
static hash_table copies_from_in;
#define COPY_FROM_IN(st) ((set)hash_get(copies_from_in, (char *)st))
/*
set COPY_FROM_IN(statement st) {
  set res = (set)hash_get(copies_from_in, (char *)st);
  if(res == HASH_UNDEFINED_VALUE) {
    res = MAKE_SET();
    hash_put(copies_from_in, (char *)res, (char *)MAKE_SET());
  }
  return res;
}
*/
/* copy_from_out maps each statement to the "copy from" data that are out-going
 * from the statement
 * */
static hash_table copies_from_out;
#define COPY_FROM_OUT(st) ((set)hash_get(copies_from_out, (char *)st))
/*
set COPY_FROM_OUT(statement st) {
  set res = (set)hash_get(copies_from_out, (char *)st);
  if(res == HASH_UNDEFINED_VALUE) {
    res = MAKE_SET();
    hash_put(copies_from_out, (char *)res, (char *)MAKE_SET());
  }
  return res;
}
*/

/* copy_to_in maps each statement to the "copy to" data that are in-coming the
 * statement */
static hash_table copies_to_in;
#define COPY_TO_IN(st) ((set)hash_get(copies_to_in, (char *)st))
/*
set COPY_TO_IN(statement st) {
  set res = (set)hash_get(copies_to_in, (char *)st);
  if(res == HASH_UNDEFINED_VALUE) {
    res = MAKE_SET();
    hash_put(copies_to_in, (char *)res, (char *)MAKE_SET());
  }
  return res;
}
*/

/* copy_to_out maps each statement to the "copy to" data that are out-going from
 * the statement */
static hash_table copies_to_out;
#define COPY_TO_OUT(st) ((set)hash_get(copies_to_out, (char *)st))
/*
set COPY_TO_OUT(statement st) {
  set res = (set)hash_get(copies_to_out, (char *)st);
  if(res == HASH_UNDEFINED_VALUE) {
    res = MAKE_SET();
    hash_put(copies_to_out, (char *)res, (char *)MAKE_SET());
  }
  return res;

}
*/

void dump_entity_set(set s) {
  SET_FOREACH(entity, e, s) {
    fprintf(stderr,"%s,",entity_name(e));
  }
  fprintf(stderr,"\n");
}

/**
 * Should be useful, but for some bullshit reason the array dimensions even
 * for simple case like a[512][512] might be previously normalized "COMPLEX"
 * This occurs in some special cases...
 */
static bool force_renormalize(expression e) {
  if(!normalized_undefined_p(expression_normalized(e))) {
    // Remove a warning
    free_normalized(expression_normalized(e));
    expression_normalized(e) = normalized_undefined;
  }
  expression_normalized(e) = NormalizeExpression(e);
  return true;
}

/**
 * Check that an array is fully written by an exact W region in a list
 */
static bool array_must_fully_written_by_regions_p( entity e_used, list lreg) {
  // Ugly hack :-(
  effect array = entity_whole_region(e_used,make_action_write_memory());

  // Force re-normalization of expression, this is a hack !
  gen_recurse(e_used,expression_domain,force_renormalize,force_renormalize);
  if(type_variable_p(entity_type(e_used))) {
    // I don't understand why the gen_recurse doesn't work, so here is the hack for the hack...
    FOREACH(dimension, d, variable_dimensions(type_variable(entity_type(e_used)))) {
      force_renormalize(dimension_upper(d));
      force_renormalize(dimension_lower(d));
    }
  }

  //
  Psysteme p = descriptor_convex(effect_descriptor(array));
  descriptor_convex(effect_descriptor(array)) = sc_safe_append(p,
                         sc_safe_normalize(entity_declaration_sc(e_used)));
  effect_approximation(array) = make_approximation_exact();

  ifdebug(6) {
    pips_debug(6,"Array region is : ");
    print_region(array);
  }
  FOREACH(effect, reg, lreg) {
    if(!descriptor_convex_p(effect_descriptor(reg))) continue;
    // Proj region over PHI to remove precondition variables
    // FIXME : Can be done by difference between the 2 bases
    list proj = base_to_list(sc_base(descriptor_convex(effect_descriptor(reg))));
    list l_phi = NIL;
    FOREACH(expression,e, reference_indices(effect_any_reference(reg))) {
      pips_assert("expression_reference_p", expression_reference_p(e));
      l_phi = CONS(entity,reference_variable(expression_reference(e)),l_phi);
      gen_remove(&proj,reference_variable(expression_reference(e)));
    }


    region_exact_projection_along_variables(reg, proj);


    ifdebug(6) {
      pips_debug(6,"Evaluating region : ");
      print_region(reg);
    }
    if(effect_write_p(reg) && store_effect_p(reg) && region_exact_p(reg)) {
      list ldiff = region_sup_difference(array,reg);
      bool empty=true;
      FOREACH(effect, diff, ldiff) {
        if( !region_empty_p(diff)) {
          ifdebug(6) {
            pips_debug(6,"Diff is not empty :");
            print_region(diff);
          }
          empty = false;
          break;
        } else {
          ifdebug(6) {
            pips_debug(6,"Diff is empty !\n");
          }
        }
      }
      if(empty) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Translate the interprocedural summary for a function into a set of local
 * corresponding parameters (entity only)
 */
static set interprocedural_mapping(string res, call c) {
  entity callee = call_function(c);
  const char* func_name = entity_local_name(callee);
  memory_mapping mem_map = (memory_mapping)db_get_memory_resource(res,
                                                                  func_name,
                                                                  true);
  set summary = MAKE_SET();
  set_assign(summary,memory_mapping_map(mem_map));


  list /* of entity */l_formals = module_formal_parameters(callee);
  int arg_num, n_formals = gen_length(l_formals);
  gen_free_list(l_formals);

  /* if there are more actuals than formals, they are skipped.
   */
  set at_call_site = MAKE_SET();
  list real_args = call_arguments(c);
  for (arg_num = 1; !ENDP(real_args) && arg_num <= n_formals; real_args
      = CDR(real_args), arg_num++) {
    entity formal_ent = find_ith_formal_parameter(callee, arg_num);
    bool found = false;
    SET_FOREACH(entity, e, summary) {
      if(same_entity_p(e, formal_ent)) {
        set_del_element(summary,summary,e);
        found = true;
        break;
      }
    }
    if(found) {
      expression real_exp = EXPRESSION(CAR(real_args));

      if(expression_reference_p(real_exp)) {
        reference r = syntax_reference(expression_syntax(real_exp));
        set_add_element(at_call_site, at_call_site, reference_variable(r));
      } else {
        pips_user_warning(
            "Interprocedural mapping on %s map on a non ref expression ",
            entity_name(formal_ent));
        print_syntax(expression_syntax(real_exp));
        fprintf(stderr, "\n");
      }
    }
  }
  SET_FOREACH(entity, e, summary) {
    pips_debug(4,"%s not mapped on formal\n",entity_name(e));
    if(top_level_entity_p(e)) {
      set_add_element(at_call_site,at_call_site,e);
    } else {
      pips_internal_error("Interprocedural error : entity %s is not a formal"
          " not a global variable !\n",entity_name(e));
    }
  }
  return at_call_site;
}



bool is_a_kernel(const char* func_name) {
  callees kernels = (callees)db_get_memory_resource(DBR_KERNELS, "", true);
  bool found = false;
  FOREACH(STRING,kernel_name,callees_callees(kernels)) {
    if((found = (same_string_p(kernel_name,func_name))))
      break;
  }
  return found;
}

static void init_one_statement(statement st) {
  if(COPY_FROM_IN( st ) == (set)HASH_UNDEFINED_VALUE) {
    /* First initialization (normal use) */
    /* If the statement has never been seen, allocate new sets: */
    hash_put(copies_from_in, (char *)st, (char *)MAKE_SET());
    hash_put(copies_from_out, (char *)st, (char *)MAKE_SET());
    hash_put(copies_to_in, (char *)st, (char *)MAKE_SET());
    hash_put(copies_to_out, (char *)st, (char *)MAKE_SET());
  }

}

static void copy_from_statement(statement st);

static void copy_from_block(statement st, cons *sts) {
  /* The initial in for the block */
  set copy_from_in = COPY_FROM_IN( st );
  /* loop over statements inside the block */
  FOREACH( statement, one, sts ) {
    /* propagate "in" sets */
    set_assign(COPY_FROM_IN( one ), copy_from_in);

    /* Compute the "copy from" for this statement */
    copy_from_statement(one);

    /* Get the ins for next statement */
    copy_from_in = COPY_FROM_OUT( one );
  }

  /* The outs for the whole block */
  set_assign(COPY_FROM_OUT( st ), copy_from_in);

}
static void copy_from_test(statement st, test t) {
  /* Compute true branch */
  set_assign(COPY_FROM_IN( test_true(t) ), COPY_FROM_IN( st ));
  copy_from_statement(test_true(t));

  /* Compute false branch */
  set_assign(COPY_FROM_IN( test_false(t) ), COPY_FROM_IN( st ));
  copy_from_statement(test_false(t));


  /* Compute "out" sets for the test */
  set copy_from_out = COPY_FROM_OUT( st );

  set_intersection(copy_from_out, COPY_FROM_OUT( test_true(t)), COPY_FROM_OUT( test_false(t)));


  ifdebug(6) {
    pips_debug(6,"Handling copy from for test expression : ");
    print_expression(test_condition(t));
    fprintf(stderr,"\n");
    pips_debug(6,"Copy from is : ");
    print_entities(set_to_list(copy_from_out));
    fprintf(stderr,"\n");
  }

  // We remove from "copy from" what is used by this statement
  FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
    entity e_used = reference_variable(effect_any_reference(eff));
    if(entity_array_p(e_used)) {
      pips_debug(6,"Removing %s from copy_out\n",entity_name(e_used));
      set_del_element(copy_from_out, copy_from_out, e_used);
    }
  }


}

static void copy_from_all_kind_of_loop(statement st, statement body) {
  /* Compute "in" sets for the loop body */
  set_assign(COPY_FROM_IN( body ), COPY_FROM_IN( st ));

  /* Compute loop body */
  copy_from_statement(body);

  /* Fix point  */
  /*
  if(!set_equal_p(COPY_FROM_IN( body ),set_intersection(COPY_FROM_IN( st ),
                                                        COPY_FROM_OUT( body ),
                                                        COPY_FROM_IN( st )))) {
    return copy_from_all_kind_of_loop(st,body);
  }
  */

  /* Compute "out" sets for the loop */
  if(true) { //loop_executed_once_p(st, l)) { FIXME !!!!
    pips_debug(1,"Loop executed once !\n");
    /* Body is always done at least one time */
    set_assign(COPY_FROM_OUT( st ), COPY_FROM_OUT( body ));
  } else {
    pips_debug(1,"Loop not executed once :-( !\n");
    /* Body is not always done */
    set_intersection(COPY_FROM_OUT( st ),
                     COPY_FROM_OUT( body ),
                     COPY_FROM_IN( st ));
  }

}

static void copy_from_loop(statement st, loop l) {
  statement body = loop_body(l);
  copy_from_all_kind_of_loop(st, body);

  /*
   * For further "already_transfered" optimization to be correct !
   * We have to force to transfert from the accelerator in the body of the loop
   * if we can't know that the data won't be used on the CPU at the begin of the
   * next iteration
   * see validation/Gpu/Comm_optimization.sub/loop01.c
   */
  set to_delete = MAKE_SET();
  SET_FOREACH(entity,e,COPY_FROM_OUT(st)) {
    if(!set_belong_p(COPY_TO_IN(st),e)) {
      //insert_statement(body,make_dma_transfert(e,dma_store),false);
      pips_debug(1,"Removing %s from loop copy-out !\n",entity_name(e));
      set_add_element(to_delete,to_delete,e);
    }
  }
  set_difference(COPY_FROM_OUT(st),COPY_FROM_OUT(st),to_delete);
  set_difference(COPY_FROM_OUT(body),COPY_FROM_OUT(body),to_delete);
  set_free(to_delete);

}
static void copy_from_whileloop(statement st, whileloop l) {
  statement body = whileloop_body(l);
  copy_from_all_kind_of_loop(st, body);
}
static void copy_from_forloop(statement st, forloop l) {
  statement body = forloop_body(l);
  copy_from_all_kind_of_loop(st, body);
}

static void copy_from_call(statement st, call c) {
  entity callee = call_function(c);
  const char* func_name = entity_local_name(callee);

  /* The initial in */
  set copy_from_in = COPY_FROM_IN( st );
  set copy_from_out = COPY_FROM_OUT( st );

  /* The outs is initially the in */
  set_assign(copy_from_out, copy_from_in);
  if(is_a_kernel(func_name)) {
    pips_debug(3,"%s is a Kernel !\n",
        func_name);
    // it's a kernel call !!
    // Let's add the "region out" to copy_from set !
    list l_out = load_statement_out_regions(st);
    FOREACH(EFFECT, eff, l_out )
    {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        copy_from_out = set_add_element(copy_from_out, copy_from_out, e_used);
      }
    }
  } else {
    // Standard call (not a kernel)
    pips_debug(3,"%s is a simple call\n",
        func_name);


    // We get the inter-procedural results for copy in
    set at_call_site;
    if(user_call_p(c)) { // Ignoring intrinsics
      at_call_site = interprocedural_mapping(DBR_KERNEL_COPY_IN,c);
    } else
      at_call_site = set_make(set_pointer);

    // We remove from "copy from" what is used by this statement but not what's
    // in copy-in
    FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used) && !set_belong_p(at_call_site,e_used)) {
        pips_debug(6,"Removing %s from copy_out\n",entity_name(e_used));
        copy_from_out = set_del_element(copy_from_out, copy_from_out, e_used);
      }
    }

    // No longer used
    set_free(at_call_site);

    // We add the inter-procedural results for copy out, but not for intrinsics
    if(user_call_p(c)) { // Ignoring intrinsics
      at_call_site = interprocedural_mapping(DBR_KERNEL_COPY_OUT,c);
      copy_from_out = set_union(copy_from_out, at_call_site, copy_from_out);
      ifdebug(6) {
        pips_debug(6,"Adding interprocedural summary : ");
        print_entities(set_to_list(at_call_site));
        fprintf(stderr,"\n");
      }
      set_free(at_call_site);
    }
  }
}

static void copy_from_unstructured(statement st, unstructured u) {
  pips_user_warning("Unimplemented !! Results may be wrong...\n");
}

static void copy_from_statement(statement st) {
  instruction i;

  /* Compute "copy from" sets for the instruction : recursion */
  switch(instruction_tag( i = statement_instruction( st ))) {
    case is_instruction_block:
      copy_from_block(st, instruction_block( i ));
      break;
    case is_instruction_test:
      copy_from_test(st, instruction_test( i ));
      break;
    case is_instruction_loop:
      copy_from_loop(st, instruction_loop( i ));
      break;
    case is_instruction_whileloop:
      copy_from_whileloop(st, instruction_whileloop( i ));
      break;
    case is_instruction_forloop:
      copy_from_forloop(st, instruction_forloop( i ));
      break;
    case is_instruction_call:
      copy_from_call(st, instruction_call( i ));
      break;
    case is_instruction_expression: {
      expression e = instruction_expression( i );
      if(expression_call_p(e)) {
        copy_from_call(st, expression_call(e));
      } else {
        pips_user_warning("Unsupported stmt expression : ");
        print_expression(e);
        fprintf(stderr,"\n");
        // Just propagate then...
        set copy_from_in = COPY_FROM_IN( st );
        set copy_from_out = COPY_FROM_OUT( st );
        set_assign(copy_from_out, copy_from_in);
      }
      break;
    }
    case is_instruction_goto:
      pips_error("inout_statement", "Unexpected tag %d\n", i);
      break;
    case is_instruction_unstructured:
      copy_from_unstructured(st, instruction_unstructured( i ));
      break;
    default:
      pips_internal_error("Unknown tag %d\n", instruction_tag(i) );
  }

  ifdebug(4) {
    fprintf(stderr,
            "** stmt (%d) associated copy from : **\n",
            (int)statement_ordering(st));
    fprintf(stderr, "*\n* OUT : ");
    {
      string str = concatenate("Copy from out : ", NULL);
      list t_from = set_to_sorted_list(COPY_FROM_OUT(st),(gen_cmp_func_t)compare_entities);
      FOREACH(entity, e, t_from) {
        fprintf(stderr, "%s ", entity_name(e));
        str = concatenate(str, " ", entity_local_name(e), NULL);
      }
      str = concatenate(str, "\n", NULL);
      if(!statement_block_p(st) || ! empty_statement_or_continue_without_comment_p(st) )
        insert_comments_to_statement(st, strdup(str));
    }
    fprintf(stderr, "*\n**********************************\n");

    fprintf(stderr, "* IN  : ");
    {
      string str = concatenate("Copy from in : ", NULL);
      list t_from = set_to_sorted_list(COPY_FROM_IN(st),(gen_cmp_func_t)compare_entities);
      FOREACH(entity, e, t_from) {
        fprintf(stderr, "%s ", entity_name(e));
        str = concatenate(str, " ", entity_local_name(e), NULL);
      }
      str = concatenate(str, "\n", NULL);
      if(!statement_block_p(st) || ! empty_statement_or_continue_without_comment_p(st) )
        insert_comments_to_statement(st, strdup(str));
    }
    fprintf(stderr, "*\n**********************************\n");
  }

}

/****************************************************************
 * COPY TO
 */

static void copy_to_statement(statement st);

static void copy_to_block(statement st, list sts) {
  /* The initial out for the block */
  set copy_to_out = MAKE_SET();
  set_assign(copy_to_out, COPY_TO_OUT( st ));
  list sts_reverse = gen_nreverse(gen_copy_seq(sts));
  /* loop over statements inside the block */
  FOREACH( statement, one, sts_reverse ) {
    /* propagate "out" sets */
    set_assign(COPY_TO_OUT( one ), copy_to_out);

    /* Compute the "copy to" for this statement */
    copy_to_statement(one);

    /* Get the outs for next statement */
    set_assign(copy_to_out, COPY_TO_IN( one ));

    /* This is a non necessary optimisation, the set is just smaller with */
//    set_difference(copy_to_out, copy_to_out, COPY_FROM_IN(one));
  }

  /* The outs for the whole block */
  set_assign(COPY_TO_IN( st ), copy_to_out);

  /* FIXME : free sts_reverse list && copy_to_out*/

}
static void copy_to_test(statement st, test t) {
  /* Compute true branch */
  set_assign(COPY_TO_OUT( test_true(t) ), COPY_TO_OUT( st ));
  copy_to_statement(test_true(t));

  /* Compute false branch */
  set_assign(COPY_TO_OUT( test_false(t) ), COPY_TO_OUT( st ));
  copy_to_statement(test_false(t));


  /* Compute "in" sets for the test */
  set copy_to_in = COPY_TO_IN( st );

  set_intersection(copy_to_in, COPY_TO_IN( test_true(t)), COPY_TO_IN( test_false(t)));

  ifdebug(6) {
    pips_debug(6,"Handling copy to for test expression : ");
    print_expression(test_condition(t));
    fprintf(stderr,"\n");
  }

  // We remove from "copy to" what is used by the test condition
  FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
    entity e_used = reference_variable(effect_any_reference(eff));
    if(entity_array_p(e_used)) {
      pips_debug(6,"Removing %s from copy_in\n",entity_name(e_used));
      set_del_element(copy_to_in, copy_to_in, e_used);
    }
  }



}
static void copy_to_all_kind_of_loop(statement st, statement body) {
  /* Compute "out" sets for the loop body */
  set body_out = COPY_TO_OUT(body);
  set_union(body_out,
            COPY_TO_OUT( st ),
            COPY_TO_IN( body ));

  /* Save result */
  set tmp = MAKE_SET();
  set_assign(tmp,body_out);

  /* Compute loop body */
  copy_to_statement(body);

  /* Fix point  */
  set_union(body_out ,
            COPY_TO_OUT( st ),
            COPY_TO_IN( body ));
  if(!set_equal_p(tmp,body_out)) {
    set_free(tmp);
    return copy_to_all_kind_of_loop(st,body);
  }
  set_free(tmp);


  /* Compute "in" sets for the loop */
  if(true) { //loop_executed_once_p(st, l)) {
    pips_debug(1,"Loop executed once !\n");
    /* Body is always done at least one time */
    set_assign(COPY_TO_IN( st ), COPY_TO_IN( body ));
  } else {
    pips_debug(1,"Loop not executed once :-( !\n");
    /* Body is not always done */
    set_intersection(COPY_TO_IN( st ), COPY_TO_IN( body ), COPY_TO_OUT( st ));
  }




}
static void copy_to_loop(statement st, loop l) {
  statement body = loop_body(l);
  copy_to_all_kind_of_loop(st, body);
}
static void copy_to_whileloop(statement st, whileloop l) {
  statement body = whileloop_body(l);
  copy_to_all_kind_of_loop(st, body);
}
static void copy_to_forloop(statement st, forloop l) {
  statement body = forloop_body(l);
  copy_to_all_kind_of_loop(st, body);
}
static void copy_to_call(statement st, call c) {
  entity callee = call_function(c);
  const char* func_name = entity_local_name(callee);

  /* The initial */
  set copy_to_in = COPY_TO_IN( st );
  set copy_to_out = COPY_TO_OUT( st );

  /* The ins are initially the out minus CopyFrom*/
  set_assign(copy_to_in, copy_to_out);
  if(is_a_kernel(func_name)) {
    pips_debug(3,"%s is a Kernel !\n",
        func_name);
    // it's a kernel call !!
    // Let's add the "region in" and "region out" to copy_to set !
    list l_in = load_statement_in_regions(st);
    FOREACH(EFFECT, eff, l_in )
    {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        pips_debug(6,"Adding in %s into copy_in\n",entity_name(e_used));
        copy_to_in = set_add_element(copy_to_in, copy_to_in, e_used);
      }
    }
    list l_out = load_statement_out_regions(st);
    FOREACH(EFFECT, eff, l_out )
    {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        list lreg = load_proper_rw_effects_list(st);

        if(!array_must_fully_written_by_regions_p(e_used,lreg)) {
          pips_debug(6,"Adding written %s into copy_in\n",entity_name(e_used));
          copy_to_in = set_add_element(copy_to_in, copy_to_in, e_used);
        } else {
          pips_debug(6,"Do not add %s into copy_in because fully overwritten\n",
                     entity_name(e_used));
        }
      }
    }
  } else {
    // Standard call (not a kernel)
    pips_debug(3,"%s is a simple call\n",
        func_name);
    // We remove from "copy to" what is written by this statement
    FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
//      if(effect_write_p(eff)) { BUGGY at that time ! it breaks hoisting transfers out of loop... the scheme is broken by design !
        entity e_used = reference_variable(effect_any_reference(eff));
        ifdebug(6) {
          if(entity_array_p(e_used)) {
            pips_debug(6,"Removing %s from copy_in\n",entity_name(e_used));
          }
        }
        copy_to_in = set_del_element(copy_to_in, copy_to_in, e_used);
//      }
    }


    // We add the inter-procedural results for copy in, but not for intrinsics
    if(user_call_p(c)) { // Ignoring intrinsics
      set at_call_site = interprocedural_mapping(DBR_KERNEL_COPY_IN,c);
      copy_to_in = set_union(copy_to_in, at_call_site, copy_to_in);
      set_free(at_call_site);
    }

  }
  //  copy_to_in = set_difference(copy_to_in, copy_to_in, COPY_FROM_IN(st));
}

static void copy_to_unstructured(statement st, unstructured u) {
  pips_user_warning("Unimplemented !! Results may be wrong...\n");
}

static void get_written_entities(statement s, set written_entities) {
  FOREACH(EFFECT, eff, load_proper_rw_effects_list(s) ) {
    if(effect_write_p(eff) && store_effect_p(eff)) {
      entity e_used = reference_variable(effect_any_reference(eff));
      set_add_element(written_entities,written_entities,e_used);
    }
  }
}

static void copy_to_statement(statement st) {
  instruction i;

  /* Compute "copy from" sets for the instruction : recursion */
  switch(instruction_tag( i = statement_instruction( st ))) {
    case is_instruction_block:
      copy_to_block(st, instruction_block( i ));
      break;
    case is_instruction_test:
      copy_to_test(st, instruction_test( i ));
      break;
    case is_instruction_loop:
      copy_to_loop(st, instruction_loop( i ));
      break;
    case is_instruction_whileloop:
      copy_to_whileloop(st, instruction_whileloop( i ));
      break;
    case is_instruction_forloop:
      copy_to_forloop(st, instruction_forloop( i ));
      break;
    case is_instruction_call:
      copy_to_call(st, instruction_call( i ));
      break;
    case is_instruction_expression: {
      expression e = instruction_expression( i );
      if(expression_call_p(e)) {
        copy_to_call(st, expression_call(e));
      } else {
        pips_user_warning("Unsupported stmt expression : ");
        print_expression(e);
        fprintf(stderr,"\n");
        // Just propagate then...
        set copy_to_in = COPY_TO_IN( st );
        set copy_to_out = COPY_TO_OUT( st );
        /* The ins are initially the out minus CopyFrom*/
        set_assign(copy_to_in, copy_to_out);
      }
      break;
    }
    case is_instruction_goto:
      pips_error("inout_statement", "Unexpected tag %d\n", i);
      break;
    case is_instruction_unstructured:
      copy_to_unstructured(st, instruction_unstructured( i ));
      break;
    default:
      pips_internal_error("Unknown tag %d\n", instruction_tag(i) );
  }


  // We remove from "copy to out" what is not written by this statement
  set allowed_to_copy_to= MAKE_SET();
  gen_context_recurse(st,allowed_to_copy_to,statement_domain,gen_true,get_written_entities);
  ifdebug(4) {
    pips_debug(4,"Removing from copy_to_out what is not written here : ");
    set removed = MAKE_SET();
    set_difference(removed,COPY_TO_OUT(st),allowed_to_copy_to);
    list t_to = set_to_sorted_list(removed,(gen_cmp_func_t)compare_entities);
    FOREACH(entity, e, t_to) {
      fprintf(stderr, "%s ", entity_name(e));
    }
    fprintf(stderr, "\n");
  }
  set_intersection(COPY_TO_OUT(st),COPY_TO_OUT(st),allowed_to_copy_to);
  set_free(allowed_to_copy_to);


  ifdebug(4) {
    fprintf(stderr,
            "** stmt (%d) associated copy to : **\n",
            (int)statement_ordering(st));
    fprintf(stderr, "*\n* OUT : ");
    {
      string str = concatenate("Copy to out : ", NULL);
      list t_to = set_to_sorted_list(COPY_TO_OUT(st),(gen_cmp_func_t)compare_entities);
      FOREACH(entity, e, t_to) {
        fprintf(stderr, "%s ", entity_name(e));
        str = concatenate(str, " ", entity_local_name(e), NULL);
      }
      str = concatenate(str, "\n", NULL);
      if(!statement_block_p(st) || ! empty_statement_or_continue_without_comment_p(st) )
        insert_comments_to_statement(st, strdup(str));
    }
    fprintf(stderr, "* IN  : ");
    {
      string str = concatenate("Copy to in : ", NULL);
      list t_to = set_to_sorted_list(COPY_TO_IN(st),(gen_cmp_func_t)compare_entities);
      FOREACH(entity, e, t_to) {
        fprintf(stderr, "%s ", entity_name(e));
        str = concatenate(str, " ", entity_local_name(e), NULL);
      }
      str = concatenate(str, "\n", NULL);
      if(!statement_block_p(st) || !empty_statement_or_continue_without_comment_p(st) )
        insert_comments_to_statement(st, strdup(str));
    }
    fprintf(stderr, "*\n**********************************\n");
  }


}

/****************************************************************
 * GENERATE TRANSFERT
 */

static void transfert_statement(statement st,
                                set already_transfered_to,
                                set already_transfered_from,
                                set array_to_transfer_to_after,
                                set array_to_transfer_from_before);

static void transfert_block(statement st,
                            list sts,
                            set already_transfered,
                            set already_transfered_from) {

  pips_debug(4,"Entering transfert_block\n");

  /* loop over statements inside the block */
  for (list current = (sts); !ENDP(current); POP(current)) {
    statement one = STATEMENT(CAR(current));

    // These are here to record what a statement require to transfer
    // We generate transfers only in sequence
    set transfert_to = MAKE_SET();
    set transfert_from = MAKE_SET();

    /* Compute the transfer for this statement */
    transfert_statement(one,
                        already_transfered,
                        already_transfered_from,
                        transfert_to,
                        transfert_from);
    // Really insert the transfers statement
    list t_from = set_to_sorted_list(transfert_from,(gen_cmp_func_t)compare_entities);
    FOREACH(entity, e_from, t_from) {
      statement new_st_from = make_dma_transfert(e_from, dma_store);
      CDR(current) = CONS(statement,one,CDR(current));
      CAR(current).e = new_st_from;
      POP(current);
    }
    gen_free_list(t_from);


    list t_to = set_to_sorted_list(transfert_to,(gen_cmp_func_t)compare_entities);
    FOREACH(entity, e_to, t_to) {
      statement new_st_to = make_dma_transfert(e_to, dma_load);
      CDR(current) = CONS(statement,new_st_to,CDR(current));
      POP(current);
    }
    gen_free_list(t_to);
  }


}
static void transfert_test(statement st,
                           test t,
                           set already_transfered,
                           set already_transfered_from,
                           set transfert_to,
                           set transfert_from) {
  set already_transfered_from_tmp = set_make(set_pointer);
  set already_transfered_tmp = set_make(set_pointer);
  set transfert_to_tmp = set_make(set_pointer);
  set transfert_from_tmp = set_make(set_pointer);

  set_assign(already_transfered_from_tmp,already_transfered_from);
  set_assign(already_transfered_tmp,already_transfered);
  set_assign(transfert_to_tmp,transfert_to);
  set_assign(transfert_from_tmp,transfert_from);


  transfert_statement(test_true(t),
                      already_transfered_tmp,
                      already_transfered_from_tmp,
                      transfert_to_tmp,
                      transfert_from_tmp);


  set_assign(already_transfered_from_tmp,already_transfered_from);
  set_assign(already_transfered_tmp,already_transfered);
  set_assign(transfert_to_tmp,transfert_to);
  set_assign(transfert_from_tmp,transfert_from);

  transfert_statement(test_false(t),
                      already_transfered_tmp,
                      already_transfered_from_tmp,
                      transfert_to_tmp,
                      transfert_from_tmp);


}



static void transfert_loop(statement st,
                           loop l,
                           set already_transfered,
                           set already_transfered_from,
                           set transfert_to,
                           set transfert_from) {
  statement body = loop_body(l);

  /* Compute loop body */
  transfert_statement(body,
                      already_transfered,
                      already_transfered_from,
                      transfert_to,
                      transfert_from);


  /*
   *  At the end of a loop, we check that any transfer that need to be done
   */

  // Retrieve the last statement of the loop body
  set last_copy_from;
  if(!statement_sequence_p(body)) {
    last_copy_from=COPY_FROM_OUT(body);
  } else {
    FOREACH(statement,s,sequence_statements(statement_sequence(body))) {
      set copy_from = COPY_FROM_OUT(s);
      if(copy_from && !set_undefined_p(copy_from))
        last_copy_from=copy_from;
    }
  }

  /*
   *  Compute the set of entity that are not on the GPU at the begin of the loop
   *  but need a copy-out at the end, after the last statement
   */
  set fixed_transfert_from = MAKE_SET();
  set_difference(fixed_transfert_from,last_copy_from,COPY_FROM_OUT(body));
  list t_from = set_to_sorted_list(fixed_transfert_from,(gen_cmp_func_t)compare_entities);
  set_free(fixed_transfert_from);
  // Insert transfers at the end of the loop !
  FOREACH(entity, e_from, t_from) {
    insert_statement(body,make_dma_transfert(e_from, dma_store),false);
  }


}

static void transfert_whileloop(statement st,
                                whileloop l,
                                set already_transfered,
                                set already_transfered_from,
                                set transfert_to,
                                set transfert_from) {
  statement body = whileloop_body(l);

  /* Compute loop body */
  transfert_statement(body,
                      already_transfered,
                      already_transfered_from,
                      transfert_to,
                      transfert_from);
}
static void transfert_forloop(statement st,
                              forloop l,
                              set already_transfered,
                              set already_transfered_from,
                              set transfert_to,
                              set transfert_from) {
  statement body = forloop_body(l);

  /* Compute loop body */
  transfert_statement(body,
                      already_transfered,
                      already_transfered_from,
                      transfert_to,
                      transfert_from);
}
static void transfert_call(statement st,
                           call c,
                           set already_transfered,
                           set already_transfered_from) {
}

static void transfert_unstructured(statement st,
                                   unstructured u,
                                   set already_transfered,
                                   set already_transfered_from) {
  pips_user_warning("Unimplemented !! Results may be wrong...\n");
}

static void transfert_statement(statement st,
                                set already_transfered_to,
                                set already_transfered_from,
                                set array_to_transfer_to_after,
                                set array_to_transfer_from_before) {

  instruction i = statement_instruction( st );

  set transferts_to = MAKE_SET();
  set_difference(transferts_to, COPY_TO_OUT(st), COPY_TO_IN(st));
  set_difference(transferts_to, transferts_to, COPY_FROM_OUT(st));
  set_difference(transferts_to, transferts_to, already_transfered_to);

  // We remove from "transfers_to" what is not written by this statement
  set allowed_to_transfer_to= MAKE_SET();
  gen_context_recurse(st,allowed_to_transfer_to,statement_domain,gen_true,get_written_entities);
  ifdebug(4) {
    pips_debug(4,"Removing from transfer_to what is not written here : ");
    set removed = MAKE_SET();
    set_difference(removed,transferts_to,allowed_to_transfer_to);
    list t_to = set_to_sorted_list(removed,(gen_cmp_func_t)compare_entities);
    FOREACH(entity, e, t_to) {
      fprintf(stderr, "%s ", entity_name(e));
    }
    fprintf(stderr, "\n");
  }
  set_intersection(transferts_to,transferts_to,allowed_to_transfer_to);
  set_free(allowed_to_transfer_to);


  SET_FOREACH(entity, e, transferts_to) {
    set_add_element(already_transfered_to, already_transfered_to, e);
    set_add_element(array_to_transfer_to_after, array_to_transfer_to_after, e);

    ifdebug(2) {
      string str = strdup(concatenate("Transfert to accel : ",
                                      entity_local_name(e),
                                      "\n",
                                      NULL));
      pips_debug(4,"(%zd) %s",statement_number(st),concatenate("Inserting :",str,"\n",NULL));
      insert_comments_to_statement(st, str);
      free(str);
    }
  }

  set transferts_from = MAKE_SET();
  set_difference(transferts_from, COPY_FROM_IN(st), COPY_FROM_OUT(st));
  set_difference(transferts_from, transferts_from, COPY_TO_IN(st));
  set_difference(transferts_from, transferts_from, already_transfered_from);

  {
    SET_FOREACH(entity, e, transferts_from)
    {
      set_add_element(already_transfered_from, already_transfered_from, e);
      set_add_element(array_to_transfer_from_before,
                      array_to_transfer_from_before,
                      e);
      ifdebug(2) {
        string str = strdup(concatenate("Transfert from accel : ",
                                        entity_local_name(e),
                                        "\n",
                                        NULL));
        pips_debug(4,"Inserting : %s\n",str);
        insert_comments_to_statement(st, str);
        free(str);
      }
    }
  }

  /* Compute "copy from" sets for the instruction : recursion */
  switch(instruction_tag(i)) {
    case is_instruction_block:
      transfert_block(st,
                      instruction_block( i ),
                      already_transfered_to,
                      already_transfered_from);
      break;
    case is_instruction_test:
      transfert_test(st,
                     instruction_test( i ),
                     already_transfered_to,
                     already_transfered_from,
                     array_to_transfer_to_after,
                     array_to_transfer_from_before);
      break;
    case is_instruction_loop:
      transfert_loop(st,
                     instruction_loop( i ),
                     already_transfered_to,
                     already_transfered_from,
                     array_to_transfer_to_after,
                     array_to_transfer_from_before);
      break;
    case is_instruction_whileloop:
      transfert_whileloop(st,
                          instruction_whileloop( i ),
                          already_transfered_to,
                          already_transfered_from,
                          array_to_transfer_to_after,
                          array_to_transfer_from_before);
      break;
    case is_instruction_forloop:
      transfert_forloop(st,
                        instruction_forloop( i ),
                        already_transfered_to,
                        already_transfered_from,
                        array_to_transfer_to_after,
                        array_to_transfer_from_before);
      break;
    case is_instruction_call:
      transfert_call(st,
                     instruction_call( i ),
                     already_transfered_to,
                     already_transfered_from);
      break;
    case is_instruction_expression:
      /* The second argument is not used */
      transfert_call(st,
                     (call)instruction_expression( i ),
                     already_transfered_to,
                     already_transfered_from);
      break;
    case is_instruction_goto:
      pips_error("inout_statement", "Unexpected tag %d\n", i);
      break;
    case is_instruction_unstructured:
      transfert_unstructured(st,
                             instruction_unstructured( i ),
                             already_transfered_to,
                             already_transfered_from);
      break;
    default:
      pips_internal_error("Unknown tag %d\n", instruction_tag(i) );
  }



  set killed_transfer = MAKE_SET();
  set_difference(killed_transfer, COPY_TO_OUT(st), COPY_TO_IN(st));
  set_difference(already_transfered_to, already_transfered_to, killed_transfer);
  set_difference(killed_transfer, COPY_FROM_IN(st), COPY_FROM_OUT(st));
  set_difference(already_transfered_from,
                 already_transfered_from,
                 killed_transfer);
  set_free(killed_transfer);

}

bool ikernel_load_store(__attribute__((unused)) char * module_name) {
  statement module_stat;

  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE,
                                                                 module_name,
                                                                 true));
  module_stat = get_current_module_statement();
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("KERNEL_DATA_MAPPING_DEBUG_LEVEL");

  string pe = db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true);
  set_proper_rw_effects((statement_effects)pe);

  /* out regions */
  string or = db_get_memory_resource(DBR_OUT_REGIONS, module_name, true);
  set_out_effects((statement_effects)or);

  /* in regions */
  string ir = db_get_memory_resource(DBR_IN_REGIONS, module_name, true);
  set_in_effects((statement_effects)ir);

  /* preconditions */
  string precond = db_get_memory_resource(DBR_PRECONDITIONS, module_name, true);
  set_precondition_map((statement_mapping)precond);

  transfert_statement(module_stat, MAKE_SET(), MAKE_SET(), NULL, NULL);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
      module_name,
      module_stat);

  reset_proper_rw_effects();
  reset_out_effects();
  reset_in_effects();
  reset_precondition_map();
  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}

bool kernel_data_mapping(char * module_name) {
  statement module_stat;

  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE,
                                                                 module_name,
                                                                 true));
  module_stat = get_current_module_statement();
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("KERNEL_DATA_MAPPING_DEBUG_LEVEL");


  /* regions */
  string region = db_get_memory_resource(DBR_REGIONS, module_name, true);
  set_proper_rw_effects((statement_effects)region);


  /* out regions */
  string or = db_get_memory_resource(DBR_OUT_REGIONS, module_name, true);
  set_out_effects((statement_effects)or);

  /* in regions */
  string ir = db_get_memory_resource(DBR_IN_REGIONS, module_name, true);
  set_in_effects((statement_effects)ir);

  /* preconditions */
  string precond = db_get_memory_resource(DBR_PRECONDITIONS, module_name, true);
  set_precondition_map((statement_mapping)precond);

  /* preconditions */
  string cumu = db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true);
  set_cumulated_rw_effects((statement_effects)cumu);

  // Stuff for ... ?
  module_to_value_mappings(get_current_module_entity());


  /* Initialize global hashtables */
  copies_from_in = hash_table_make(hash_pointer, INIT_STATEMENT_SIZE);
  copies_from_out = hash_table_make(hash_pointer, INIT_STATEMENT_SIZE);
  copies_to_in = hash_table_make(hash_pointer, INIT_STATEMENT_SIZE);
  copies_to_out = hash_table_make(hash_pointer, INIT_STATEMENT_SIZE);

  /* Initialize set for each statement */
  gen_recurse(module_stat, statement_domain,
      init_one_statement,
      gen_identity);

  /* Run the pass except if current module is a kernel */
  if(!is_a_kernel(module_name)) {
    copy_from_statement(module_stat);
    copy_to_statement(module_stat);
    // Is it a bad hack ? Should we need a fix point here ?
    copy_from_statement(module_stat);

    transfert_statement(module_stat, MAKE_SET(), MAKE_SET(), MAKE_SET(), MAKE_SET());
  }

  ifdebug(1) {
    pips_debug(1,"Interprocedural summary for %s :\n To :",module_name);
    print_entities(set_to_list(COPY_TO_IN(module_stat)));
    fprintf(stderr,"\n From :");
    print_entities(set_to_list(COPY_FROM_OUT(module_stat)));
    fprintf(stderr,"\n");
  }

  DB_PUT_MEMORY_RESOURCE(DBR_CODE,
      module_name,
      module_stat);

  DB_PUT_MEMORY_RESOURCE(DBR_KERNEL_COPY_IN,
      module_name,
      make_memory_mapping(COPY_TO_IN(module_stat)));
  DB_PUT_MEMORY_RESOURCE(DBR_KERNEL_COPY_OUT,
      module_name,
      make_memory_mapping(COPY_FROM_OUT(module_stat)));

#define TABLE_FREE(t)             \
  {HASH_MAP( k, v, {set_free( (set)v ) ;}, t ) ; hash_table_free(t);}

  //  TABLE_FREE(copies_from_in);
  //  TABLE_FREE(copies_from_out);

  reset_proper_rw_effects();
  reset_out_effects();
  reset_in_effects();
  reset_precondition_map();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_cumulated_rw_effects();
  free_value_mappings();

  return true;
}





static entity wrapper_function = NULL;


static void wrap_argument(syntax s) {
  if(syntax_reference_p(s)
      && entity_array_p(reference_variable(syntax_reference(s)))) {
    entity e = reference_variable(syntax_reference(s));


/*
    sizeofexpression size_of = make_sizeofexpression_expression(transfer_size);
    expression size_of_e;
    size_of_e = make_expression(make_syntax_sizeofexpression(size_of),
                                normalized_undefined);
*/
    syntax new_ref = copy_syntax(s);
    syntax_tag(s) = is_syntax_call;
    list args = make_expression_list(make_expression(new_ref,
                                                     normalized_undefined),
                                                     makeTransfertSizeExpression(e)/*size_of_e*/);
    syntax_call(s) = make_call(wrapper_function,args);

    AddEntityToModuleCompilationUnit(wrapper_function, get_current_module_entity());
  }

}

static void wrap_call_argument(call c, const char* module_name) {
  if(!call_intrinsic_p(c)
      && same_string_p(module_local_name(call_function(c)),module_name)) {

    list args = call_arguments(c);
    for (expression e; !ENDP(args); POP(args)) {
      e = EXPRESSION(CAR(args));
      gen_recurse(e,syntax_domain,gen_true,wrap_argument);
    }

  }
}


/**
 * This pass will wrap kernel arguments in a call to a wrapper function.
 */
bool wrap_kernel_argument(char * module_name) {

  entity kernel_entity = local_name_to_top_level_entity(module_name);

  debug_on("WRAP_KERNEL_ARGUMENT_DEBUG_LEVEL");

  callees callers = (callees)db_get_memory_resource(DBR_CALLERS,
                                                    module_name,
                                                    true);

  // Get the wrapper function
  const char* function_name = get_string_property("WRAP_KERNEL_ARGUMENT_FUNCTION_NAME");
  wrapper_function = module_name_to_entity(function_name);
  if(entity_undefined_p(wrapper_function)) {
    wrapper_function
        = make_empty_subroutine(function_name,
                                copy_language(module_language(kernel_entity)));
    pips_user_warning("Cannot find \"%s\" method. Are you sure you have set\n"
        "WRAP_KERNEL_ARGUMENT_FUNCTION_NAME"
        "to a defined entity and added the correct .c file?\n",function_name);
  }




  FOREACH(STRING,caller_name,callees_callees(callers)) {
    /* prelude */
    set_current_module_entity(module_name_to_entity(caller_name));
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE,
                                                                   caller_name,
                                                                   true));

    /*do the job */
    gen_context_recurse(get_current_module_statement(),module_name,call_domain,gen_true,wrap_call_argument);
    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, caller_name,get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, caller_name, compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
  }

  return true;
}

