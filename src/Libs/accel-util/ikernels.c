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
#include "transformations.h"
#include "parser_private.h"
#include "preprocessor.h"
#include "accel-util.h"


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
static string get_dma_name(enum region_to_dma_switch m, int d) {
  char *seeds[] = { "KERNEL_LOAD_STORE_LOAD_FUNCTION",
                    "KERNEL_LOAD_STORE_STORE_FUNCTION",
                    "KERNEL_LOAD_STORE_ALLOCATE_FUNCTION",
                    "KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION" };
  char * propname = seeds[(int)m];
  /* If the DMA is not scalar, the DMA function name is in the property
   of the form KERNEL_LOAD_STORE_LOAD/STORE_FUNCTION_dD: */
  if(d > 0 && (int)m < 2)
    asprintf(&propname, "%s_%dD", seeds[(int)m], (int)d);
  string dmaname = get_string_property(propname);
  asprintf(&dmaname, "%s", dmaname);
  if(d > 0 && (int)m < 2)
    free(propname);
  pips_debug(6,"Get dma name : %s\n",dmaname);
  return dmaname;
}

static statement make_dma_transfert(entity e, enum region_to_dma_switch dma) {

  string function_name = get_dma_name(dma, 0);
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


bool is_a_kernel(string func_name) {
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
  set_intersection(COPY_FROM_OUT( st ), COPY_FROM_OUT( test_true(t)), COPY_FROM_OUT( test_false(t)));
}

static void copy_from_all_kind_of_loop(statement st, statement body) {
  /* Compute "in" sets for the loop body */
  set_assign(COPY_FROM_IN( body ), COPY_FROM_IN( st ));

  /* Compute loop body */
  copy_from_statement(body);

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
  string func_name = entity_local_name(callee);

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
    // We remove from "copy from" what is used by this statement
    FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        pips_debug(6,"Removing %s from copy_out\n",entity_name(e_used));
        copy_from_out = set_del_element(copy_from_out, copy_from_out, e_used);
      }
    }

    // We add the inter-procedural results for copy out, but not for intrinsics
    if(!call_intrinsic_p(c)) { // Ignoring intrinsics
      set
          func_copy_from_formal =
              memory_mapping_map((memory_mapping)db_get_memory_resource(DBR_KERNEL_COPY_OUT,
                      func_name,
                      TRUE));

      list /* of entity */l_formals = module_formal_parameters(callee);
      int arg_num, n_formals = gen_length(l_formals);
      gen_free_list(l_formals);

      /* if there are more actuals than formals, they are skipped.
       */
      set func_copy_from = MAKE_SET();
      list real_args = call_arguments(c);
      for (arg_num = 1; !ENDP(real_args) && arg_num <= n_formals; real_args
          = CDR(real_args), arg_num++) {
        entity formal_ent = find_ith_formal_parameter(callee, arg_num);
        bool found = false;
        SET_FOREACH(entity, e, func_copy_from_formal) {
          if(same_entity_p(e, formal_ent)) {
            found = true;
            break;
          }
        }
        if(found) {
          expression real_exp = EXPRESSION(CAR(real_args));

          if(expression_reference_p(real_exp)) {
            reference r = syntax_reference(expression_syntax(real_exp));
            set_add_element(func_copy_from,
                            func_copy_from,
                            reference_variable(r));
          } else {
            pips_user_warning(
                "Copy from out on %s map on a non ref expression ",
                entity_name(formal_ent));
            print_syntax(expression_syntax(real_exp));
            fprintf(stderr, "\n");
          }
        }
      }

      copy_from_out = set_union(copy_from_out, func_copy_from, copy_from_out);
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
    case is_instruction_expression:
      /* The second argument is not used */
      copy_from_call(st, expression_call(instruction_expression( i )));
      break;
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
      SET_FOREACH(entity, e, COPY_FROM_OUT(st)) {
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
      SET_FOREACH(entity, e, COPY_FROM_IN(st)) {
        fprintf(stderr, "%s ", entity_name(e));
        str = concatenate(str, " ", entity_local_name(e), NULL);
      }
      str = concatenate(str, "\n", NULL);
      if(!statement_block_p(st) || ! empty_statement_or_continue_without_comment_p(st) )
        insert_comments_to_statement(st, strdup(str));
    }
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

    /* Compute the "copy from" for this statement */
    copy_to_statement(one);

    /* Get the outs for next statement */
    set_assign(copy_to_out, COPY_TO_IN( one ));
    set_difference(copy_to_out, copy_to_out, COPY_FROM_IN(one));
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


  /* Compute "out" sets for the test */
  set_intersection(COPY_TO_IN( st ), COPY_TO_IN( test_true(t)), COPY_TO_IN( test_false(t)));
}
static void copy_to_all_kind_of_loop(statement st, statement body) {
  /* Compute "out" sets for the loop body */
  set_assign(COPY_TO_OUT( body ), COPY_TO_OUT( st ));

  /* Compute loop body */
  copy_to_statement(body);

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
  string func_name = entity_local_name(callee);

  /* The initial */
  set copy_to_in = COPY_TO_IN( st );
  set copy_to_out = COPY_TO_OUT( st );

  /* The ins are initially the out minus CopyFrom*/
  set_assign(copy_to_in, copy_to_out);
  if(is_a_kernel(func_name)) {
    pips_debug(3,"%s is a Kernel !\n",
        func_name);
    // it's a kernel call !!
    // Let's add the "region in" to copy_to set !
    list l_in = load_statement_in_regions(st);
    FOREACH(EFFECT, eff, l_in )
    {
      print_effect(eff);
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        pips_debug(6,"Adding %s into copy_in\n",entity_name(e_used));
        copy_to_in = set_add_element(copy_to_in, copy_to_in, e_used);
      }
    }
  } else {
    // Standard call (not a kernel)
    pips_debug(3,"%s is a simple call\n",
        func_name);
    // We remove from "copy to" what is used by this statement
    FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
      entity e_used = reference_variable(effect_any_reference(eff));
      if(entity_array_p(e_used)) {
        pips_debug(6,"Removing %s from copy_in\n",entity_name(e_used));
        copy_to_in = set_del_element(copy_to_in, copy_to_in, e_used);
      }
    }

    // We add the inter-procedural results for copy in, but not for intrinsics
    if(!call_intrinsic_p(c)) { // Ignoring intrinsics
      set
          func_copy_to_formal =
              memory_mapping_map((memory_mapping)db_get_memory_resource(DBR_KERNEL_COPY_IN,
                      func_name,
                      TRUE));

      list /* of entity */l_formals = module_formal_parameters(callee);
      int arg_num, n_formals = gen_length(l_formals);
      gen_free_list(l_formals);

      /* if there are more actuals than formals, they are skipped.
       */
      set func_copy_to = MAKE_SET();
      list real_args = call_arguments(c);
      for (arg_num = 1; !ENDP(real_args) && arg_num <= n_formals; real_args
          = CDR(real_args), arg_num++) {
        entity formal_ent = find_ith_formal_parameter(callee, arg_num);
        bool found = false;
        SET_FOREACH(entity, e, func_copy_to_formal) {
          if(same_entity_p(e, formal_ent)) {
            found = true;
            break;
          }
        }
        if(found) {
          expression real_exp = EXPRESSION(CAR(real_args));

          if(expression_reference_p(real_exp)) {
            reference r = syntax_reference(expression_syntax(real_exp));
            entity e_used = reference_variable(r);
            pips_debug(6,"Adding %s into copy_in\n",entity_name(e_used));
            set_add_element(func_copy_to, func_copy_to, e_used);
          } else {
            pips_user_warning(
                "Copy to from out on %s map on a non ref expression ",
                entity_name(formal_ent));
            print_syntax(expression_syntax(real_exp));
            fprintf(stderr, "\n");
          }
        }
      }

      copy_to_in = set_union(copy_to_in, func_copy_to, copy_to_in);
    }
  }
  //  copy_to_in = set_difference(copy_to_in, copy_to_in, COPY_FROM_IN(st));
}

static void copy_to_unstructured(statement st, unstructured u) {
  pips_user_warning("Unimplemented !! Results may be wrong...\n");
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
    case is_instruction_expression:
      /* The second argument is not used */
      copy_to_call(st, expression_call(instruction_expression( i )));
      break;
    case is_instruction_goto:
      pips_error("inout_statement", "Unexpected tag %d\n", i);
      break;
    case is_instruction_unstructured:
      copy_to_unstructured(st, instruction_unstructured( i ));
      break;
    default:
      pips_internal_error("Unknown tag %d\n", instruction_tag(i) );
  }

  ifdebug(4) {
    fprintf(stderr,
            "** stmt (%d) associated copy to : **\n",
            (int)statement_ordering(st));
    fprintf(stderr, "*\n* OUT : ");
    {
      string str = concatenate("Copy to out : ", NULL);
      SET_FOREACH(entity, e, COPY_TO_OUT(st)) {
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
      SET_FOREACH(entity, e, COPY_TO_IN(st)) {
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
                                set stmt_to_insert_after,
                                set stmt_to_insert_before);

static void transfert_block(statement st,
                            list sts,
                            set already_transfered,
                            set already_transfered_from) {

  pips_debug(4,"Entering transfert_block\n");

  /* Store mapping between statement and copy from */
//  hash_table statement_transfert_from = hash_table_make(hash_pointer, 10);
//  hash_table statement_transfert_to = hash_table_make(hash_pointer, 10);

  /* loop over statements inside the block */
  for (list current = (sts); !ENDP(current); POP(current)) {
    statement one = STATEMENT(CAR(current));

    set transfert_to = MAKE_SET();
    set transfert_from = MAKE_SET();

    /* Compute the transfert for this statement */
    transfert_statement(one,
                        already_transfered,
                        already_transfered_from,
                        transfert_to,
                        transfert_from);
    pips_debug(0,"t_to len : %d\n",set_size(transfert_to));
    SET_FOREACH(statement, new_st_from, transfert_from) {
      printf("INSERT STATEMENT !!\n");
      CDR(current) = CONS(statement,one,CDR(current));
      CAR(current).e = new_st_from;
      POP(current);
    }
    SET_FOREACH(statement, new_st_to, transfert_to)
    {
      pips_debug(0,"INSERT STATEMENT !!\n");
      CDR(current) = CONS(statement,new_st_to,CDR(current));
      POP(current);
    }
  }
  /*
   HASH_MAP(st, st_tr_to,
   {
   SET_FOREACH(statement, new_st, st_tr_to)
   {
   // loop over statements inside the block
   for( list current = (sts);!ENDP(current);POP(current))
   {
   statement one = (statement)CAR(current);
   if(one==st) {
   list l = CONS(STATEMENT, st, l);
   }
   }
   }
   }, statement_transfert_to);
   */
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
                                set stmt_to_insert_after,
                                set stmt_to_insert_before) {

  instruction i = statement_instruction( st );

  set transferts_to = MAKE_SET();
  set_difference(transferts_to, COPY_TO_OUT(st), COPY_TO_IN(st));

  set_difference(transferts_to, transferts_to, COPY_FROM_IN(st));
  set_difference(transferts_to, transferts_to, already_transfered_to);
  SET_FOREACH(entity, e, transferts_to) {
    if(!set_belong_p(already_transfered_to, e)) {
      set_add_element(already_transfered_to, already_transfered_to, e);
      statement new_one = make_dma_transfert(e, dma_load);
      set_add_element(stmt_to_insert_after, stmt_to_insert_after, new_one);

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
  }

  set transferts_from = MAKE_SET();
  set_difference(transferts_from, COPY_FROM_IN(st), COPY_FROM_OUT(st));
  set_difference(transferts_from, transferts_from, COPY_TO_IN(st));
  set_difference(transferts_from, transferts_from, already_transfered_from);

  // We remove from "transfer from" what is not used by this statement
  set touched_entity = MAKE_SET();
  FOREACH(EFFECT, eff, load_proper_rw_effects_list(st) ) {
    entity e_used = reference_variable(effect_any_reference(eff));
    if(entity_array_p(e_used)) {
      pips_debug(6,"Removing %s from transferts_from\n",entity_name(e_used));
      set_add_element(touched_entity, touched_entity, e_used);
    }
  }
  /*
   transferts_from = set_intersection(transferts_from,
   transferts_from,
   touched_entity);*/
  set_free(touched_entity);

  {
    SET_FOREACH(entity, e, transferts_from)
    {
      if(!set_belong_p(already_transfered_to, e)) {
        set_add_element(already_transfered_from, already_transfered_from, e);
        set_add_element(stmt_to_insert_before,
                        stmt_to_insert_before,
                        make_dma_transfert(e, dma_store));
        ifdebug(2) {
          string str = strdup(concatenate("Transfert from accel : ",
                                          entity_local_name(e),
                                          "\n",
                                          NULL));
          pips_debug(4,"%s",concatenate("Inserting :",str,"\n",NULL));
          insert_comments_to_statement(st, str);
          free(str);
        }
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
                     stmt_to_insert_after,
                     stmt_to_insert_before);
      break;
    case is_instruction_loop:
      transfert_loop(st,
                     instruction_loop( i ),
                     already_transfered_to,
                     already_transfered_from,
                     stmt_to_insert_after,
                     stmt_to_insert_before);
      break;
    case is_instruction_whileloop:
      transfert_whileloop(st,
                          instruction_whileloop( i ),
                          already_transfered_to,
                          already_transfered_from,
                          stmt_to_insert_after,
                          stmt_to_insert_before);
      break;
    case is_instruction_forloop:
      transfert_forloop(st,
                        instruction_forloop( i ),
                        already_transfered_to,
                        already_transfered_from,
                        stmt_to_insert_after,
                        stmt_to_insert_before);
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
                                                                 TRUE));
  module_stat = get_current_module_statement();
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("KERNEL_DATA_MAPPING_DEBUG_LEVEL");

  string pe = db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE);
  set_proper_rw_effects((statement_effects)pe);

  /* out regions */
  string or = db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE);
  set_out_effects((statement_effects)or);

  /* in regions */
  string ir = db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE);
  set_in_effects((statement_effects)ir);

  /* preconditions */
  string precond = db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE);
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
                                                                 TRUE));
  module_stat = get_current_module_statement();
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("KERNEL_DATA_MAPPING_DEBUG_LEVEL");

  string pe = db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE);
  set_proper_rw_effects((statement_effects)pe);

  /* out regions */
  string or = db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE);
  set_out_effects((statement_effects)or);

  /* in regions */
  string ir = db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE);
  set_in_effects((statement_effects)ir);

  /* preconditions */
  string precond = db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE);
  set_precondition_map((statement_mapping)precond);

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

    transfert_statement(module_stat, MAKE_SET(), MAKE_SET(), NULL, NULL);
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

static void wrap_call_argument(call c, string module_name) {
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
  string function_name = get_string_property("WRAP_KERNEL_ARGUMENT_FUNCTION_NAME");
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

