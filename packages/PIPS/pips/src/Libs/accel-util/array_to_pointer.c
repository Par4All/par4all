/*
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

/**
 * @file linearize_array.c
 * transform arrays to low-level pointers or 1D arrays
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-07-01
 */
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include <ctype.h>

#include "genC.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "callgraph.h"
#include "misc.h"
#include "control.h"
#include "expressions.h"
#include "preprocessor.h"
#include "accel-util.h"

typedef struct {
  /**
   * This bool is controlled by the "LINEARIZE_ARRAY_CAST_AT_CALL_SITE"
   * property Turning it on break further effects analysis, but without
   * the cast it might
   * break compilation or at least generate warnings for type mismatch
   */
  bool cast_at_call_site_p;
  /**
   * This bool is controlled by the "LINEARIZE_ARRAY_USE_POINTERS" property
   */
  bool use_pointers_p;
  /**
   * This bool is controlled by the "LINEARIZE_ARRAY_MODIFY_CALL_SITE"
   * property
   */
  bool modify_call_site_p;
} param_t;


bool do_convert_this_array_to_pointer_p(entity e) {
  if(get_bool_property("LINEARIZE_ARRAY_USE_POINTERS")) {
    if(get_bool_property("LINEARIZE_ARRAY_SKIP_STATIC_LENGTH_ARRAYS") && !entity_variable_length_array_p(e))
      return false;
    value v =entity_initial(e);
    if ( !value_undefined_p(v) && value_expression_p(v) && !expression_brace_p(value_expression(v) ) ) 
        return true;
    if( get_bool_property("LINEARIZE_ARRAY_SKIP_LOCAL_ARRAYS") && !entity_formal_p(e) ) 
      return false;
    return true;
  }
  return false;
}


/* @return the number of dimensions in @param t, 
 * counting pointers as a dimension
 *
 * BEWARE: does not take structs and unions into account
 *
 * */
size_t type_dereferencement_depth(type t) {
  t = ultimate_type(t);
  if(type_variable_p(t)) {
    ifdebug(8) {
      pips_debug(0,"Type is : ");
      print_type(t);
    }
    variable v = type_variable(t);
    basic b = variable_basic(v);
    return gen_length(variable_dimensions(v)) + (basic_pointer_p(b) ? 1
        + type_dereferencement_depth(basic_pointer(b)) : 0);
  }
  return 0;
}




static void do_linearize_array_reference(reference r) {
  entity e =reference_variable(r);
  if(entity_variable_p(e)) {
    list indices = reference_indices(r);
    list to_be_free = indices;
    bool fortran_p = fortran_module_p(get_current_module_entity());
    if (fortran_p) {
      indices = gen_nreverse (indices);
    }
    if(!ENDP(indices)) {
      type et = ultimate_type(entity_type(e));
      list new_indices = NIL;
      while(!ENDP(indices)) {
        expression new_index = expression_undefined;
        variable v = type_variable(et);
        /* must first check the dimensions , then the pointer type */
        list vdims = variable_dimensions(v);
        /* one dimension variable, nothing to do */
        if(ENDP(vdims)||ENDP(CDR(vdims))) {
        }
        else {
          /* merge all */
          new_index=int_to_expression(0);/* better start with this than nothing */
          while(!ENDP(vdims) && !ENDP(indices) ) {
            expression curr_exp = copy_expression(EXPRESSION(CAR(indices)));
            if (fortran_p) {
              // in fortran we have to take care of the lower bound that can
              // be set to any value
              // First compute the lower bound
              expression lower = copy_expression(dimension_lower(DIMENSION(CAR(vdims))));
              if (ENDP(CDR(indices))) {
                // for the last dimension (the most contiguous in the memory)
                // substract the lower bound minus 1 since the first index is
                // one
                lower = add_integer_to_expression (lower, -1);
                curr_exp = make_op_exp (MINUS_OPERATOR_NAME, curr_exp, lower);
              }
              else {
                // substract the lower bound to the index to compute the
                // dimension stride in the linearized array
                curr_exp = make_op_exp (MINUS_OPERATOR_NAME, curr_exp, lower);
              }
            }
            new_index=make_op_exp(PLUS_OPERATOR_NAME,
                make_op_exp(MULTIPLY_OPERATOR_NAME,
                  curr_exp,
                  SizeOfDimensions(CDR(vdims))
                  ),
                new_index
                );
            POP(vdims);
            POP(indices);
          }
        }
        /* it's a pointer: pop type */
        if(basic_pointer_p(variable_basic(v))) {
          et = basic_pointer(variable_basic(v));
        }
        if(expression_undefined_p(new_index)) {
          new_index =copy_expression(EXPRESSION(CAR(indices)));
          POP(indices);
        }
        new_indices=CONS(EXPRESSION,new_index,new_indices);
      }
      reference_indices(r)=gen_nreverse(new_indices);
      gen_full_free_list (to_be_free);
    }
  }
}

static void do_linearize_array_subscript(subscript s) {
  pips_user_warning("subscript linearization not handled yet\n");
}

static bool type_void_or_void_pointer_p(type t) {
  if(type_void_p(t)) return true;
  else if(type_variable_p(t)) {
    basic b = variable_basic(type_variable(t));
    return basic_pointer_p(b) &&
      type_void_or_void_pointer_p(ultimate_type(basic_pointer(b)));
  }
  return false;
}

static bool do_linearize_type(type *t, bool *rr) {
  bool linearized =false;
  if(type_void_or_void_pointer_p(*t)) {
    pips_user_warning("cannot linearize void type\n");
  }
  else {
    pips_assert ("variable expected", type_variable_p (*t));
    pips_debug (5, "try to linearize type: %s\n", type_to_string (*t));
    if(rr)*rr=false;
    variable v = type_variable(*t);
    type ut = ultimate_type(*t);
    variable uv = type_variable(ut);
    size_t uvl = gen_length(variable_dimensions(uv));
    size_t vl = gen_length(variable_dimensions(v));
    if(uvl > 1 ) {
      dimension nd = dimension_undefined;
      if (fortran_module_p(get_current_module_entity()) ) {
        nd = make_dimension(int_to_expression(1),
            SizeOfDimensions(variable_dimensions(uv)));
      }
      else {
        nd = make_dimension(int_to_expression(0),
            make_op_exp(MINUS_OPERATOR_NAME,
              SizeOfDimensions(variable_dimensions(uv)),
              int_to_expression(1)));
      }

      type nt = type_undefined;
      bool free_it = false;
      // copy only if needed, otherwise modify in place
      if (ut == *t) {
        nt = *t;
        free_it = false;
      }
      else {
        nt = copy_type(uvl>vl?ut:*t);
        free_it = true;
      }
      variable nv = type_variable(nt);
      gen_full_free_list(variable_dimensions(nv));
      variable_dimensions(nv)=CONS(DIMENSION,nd,NIL);
      if (free_it) {
        // it might be dangerous to free the type that can be reused somewhere
        // else in the RI
        free_type(*t);
        *t=nt;
      }
      linearized=true;
      if(rr)*rr=true;
      pips_debug (5, "type has been linearized\n");
    }

    if(basic_pointer_p(variable_basic(type_variable(*t))))
      return do_linearize_type(&basic_pointer(variable_basic(type_variable(*t))),rr) || linearized;
  }
  return linearized;
}

static void do_array_to_pointer_type_aux(type *t) {
  variable v = type_variable(*t);
  if(basic_pointer_p(variable_basic(v)))
    do_array_to_pointer_type_aux(&basic_pointer(variable_basic(v)));
  list dimensions = variable_dimensions(v);
  variable_dimensions(v)=NIL;
  FOREACH(DIMENSION,d,dimensions) {
    *t=make_type_variable(
        make_variable(
          make_basic_pointer(*t),
          NIL,NIL
          )
        );
  }
}

  /* returns true if a dereferencment has been supressed */
static bool do_array_to_pointer_type(type *t) {
  bool remove = false;
  if(!type_void_or_void_pointer_p(*t)) {
    if(pointer_type_p(*t)){
      variable vt = type_variable(*t);
      basic bt = variable_basic(vt);
      type t2 = basic_pointer(bt);
      if(array_type_p(t2)) {
        basic_pointer(bt) = type_undefined;
        free_type(*t);
        *t=t2;
        remove=true;
      }
    }
    do_array_to_pointer_type_aux(t);
  }
  return remove;
}


static void do_linearize_array_manage_callers(entity m,set linearized_param, param_t *param) {
  list callers = callees_callees((callees)db_get_memory_resource(DBR_CALLERS,module_local_name(m), true));
  list callers_statement = callers_to_statements(callers);
  list call_sites = callers_to_call_sites(callers_statement,m);

  /* we may have to change the call sites, prepare iterators over call sites arguments here */
  FOREACH(CALL,c,call_sites) {
    list args = call_arguments(c);
    FOREACH(PARAMETER,p,module_functional_parameters(m)) {
      if(set_belong_p(linearized_param,p)) {
        expression * arg = (expression*)REFCAR(args);
        type type_at_call_site = expression_to_type(*arg);
        type type_in_func_prototype = parameter_type(p);
        if(!pointer_type_p(type_at_call_site)) {
          /*
             type t = make_type_variable(
             make_variable(
             make_basic_pointer(
             copy_type(parameter_type(p))
             ),
             NIL,NIL)
             );
             */
          if(array_type_p(type_at_call_site)) {
            if(param->cast_at_call_site_p && !fixed_length_array_type_p(type_in_func_prototype)) {
              *arg = 
                make_expression(
                    make_syntax_cast(
                      make_cast(
                        copy_type(type_in_func_prototype),
                        *arg
                        )
                      ),
                    normalized_undefined
                    );
              if(!param->use_pointers_p) {
                *arg=MakeUnaryCall(
                    entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                    *arg);
              }
            }
          }
          else {
              if(!fixed_length_array_type_p(type_in_func_prototype))
            *arg = 
              make_expression(
                  make_syntax_cast(
                    make_cast(
                      copy_type(type_in_func_prototype),
                      MakeUnaryCall(
                        entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                        *arg
                        )
                      )
                    ),
                  normalized_undefined
                  );
            if(!param->use_pointers_p) {
              *arg=MakeUnaryCall(
                  entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                  *arg);
            }
          }
        }
        else if(!param->use_pointers_p && !type_equal_p(type_at_call_site,type_in_func_prototype)) {
          *arg =
            MakeUnaryCall(
                entity_intrinsic(DEREFERENCING_OPERATOR_NAME),*arg);
        }
        free_type(type_at_call_site);
      }
      POP(args);
    }
  }
  for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter))
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, STRING(CAR(citer)),STATEMENT(CAR(siter)));

}
static void do_linearize_array_cast(cast c) {
  do_linearize_type(&cast_type(c),NULL);
}
static void do_linearize_array_walker(void* obj) {
  gen_multi_recurse(obj,
      reference_domain,gen_true,do_linearize_array_reference,
      subscript_domain,gen_true,do_linearize_array_subscript,
      cast_domain,gen_true,do_linearize_array_cast,
      NULL);
}

static void do_linearize_expression_is_pointer(expression exp, hash_table ht) {
  basic b = basic_of_expression(exp);
  hash_put(ht,exp,(void*)(intptr_t)basic_pointer_p(b));
  free_basic(b);
}

static void do_linearize_pointer_is_expression(expression exp, hash_table ht) {
  intptr_t t = (intptr_t)hash_get(ht,exp);
  if(t != (intptr_t)HASH_UNDEFINED_VALUE ) {
    basic b = basic_of_expression(exp);
    /*SG: let us hope that by fixing only references, it will be enough */
    if(t && !basic_pointer_p(b) && expression_reference_p(exp)){
      syntax syn = expression_syntax(exp);
      expression_syntax(exp) = syntax_undefined;
      update_expression_syntax(exp,
          make_syntax_call(
            make_call(
              entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
              CONS(EXPRESSION,
                make_expression(syn,normalized_undefined),
                NIL)
              )
            )
          );
    }
    free_basic(b);
  }
}

static hash_table init_expression_is_pointer(void* obj) {
  hash_table ht = hash_table_make(hash_int,HASH_DEFAULT_SIZE);
  gen_context_recurse(obj,ht,expression_domain,gen_true,do_linearize_expression_is_pointer);
  FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
    if(entity_variable_p(e))
      gen_context_recurse(entity_initial(e),ht,expression_domain,gen_true,do_linearize_expression_is_pointer);
  }
  return ht;
}

static void do_linearize_patch_expressions(void* obj, hash_table ht) {
  gen_context_recurse(obj,ht,expression_domain,gen_true,do_linearize_pointer_is_expression);
  FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
    if(entity_variable_p(e))
      gen_context_recurse(entity_initial(e),ht,expression_domain,gen_true,do_linearize_pointer_is_expression);
  }
}

static void do_linearize_array_init(value v) {
  if(value_expression_p(v)) {
    expression exp = value_expression(v);
    if(expression_call_p(exp)) {
      call c = expression_call(exp);
      entity op = call_function(c);
      if(ENTITY_BRACE_INTRINSIC_P(op)) {
        list inits = NIL;
        for(list iter = call_arguments(c); !ENDP(iter) ; POP(iter)) {
          expression *eiter = (expression*)REFCAR(iter);
          if(expression_call_p(*eiter)) {
            call c2 = expression_call(*eiter);
            if(ENTITY_BRACE_INTRINSIC_P(call_function(c2))) {
              iter=gen_append(iter,call_arguments(c2));
              call_arguments(c2)=NIL;
              continue;
            }
          }
          inits=CONS(EXPRESSION,copy_expression(*eiter),inits);
        }
        inits=gen_nreverse(inits);
        gen_full_free_list(call_arguments(c));
        call_arguments(c)=inits;
      }
    }
  }
}

static void do_linearize_remove_dereferencment_walker(expression exp, entity e) {
  if(expression_call_p(exp)) {
    call c = expression_call(exp);
    if(ENTITY_DEREFERENCING_P(call_function(c))) {
      expression arg = EXPRESSION(CAR(call_arguments(c)));
      if(expression_reference_p(arg)) {
        reference r = expression_reference(arg);
        if(same_entity_p(reference_variable(r),e)) {
          syntax syn = expression_syntax(arg);
          expression_syntax(arg)=syntax_undefined;
          update_expression_syntax(exp,syn);
        }
      }
      else if(expression_call_p(arg)) {
        call c2 = expression_call(arg);
        if(ENTITY_PLUS_C_P(call_function(c2))) {
          bool remove =false;
          FOREACH(EXPRESSION,exp2,call_arguments(c2)) {
            if(expression_reference_p(exp2))
              remove|=same_entity_p(reference_variable(expression_reference(exp2)),e);
          }
          if(remove) {
            syntax syn = expression_syntax(arg);
            expression_syntax(arg)=syntax_undefined;
            update_expression_syntax(exp,syn);
          }
        }

      }
    }
  }
}
static void do_linearize_remove_dereferencment(statement s, entity e) {
  gen_context_recurse(s,e,expression_domain,gen_true,do_linearize_remove_dereferencment_walker);
  FOREACH(ENTITY,e,entity_declarations(get_current_module_entity()))
    if(entity_variable_p(e))
      gen_context_recurse(entity_initial(e),e,expression_domain,gen_true,do_linearize_remove_dereferencment_walker);
}

static void do_linearize_prepatch_type(type t) {
  if(pointer_type_p(t)) {
    type t2 = basic_pointer(variable_basic(type_variable(t)));
    type t3 = ultimate_type(t2);
    if(array_type_p(t2)) {
      variable v = type_variable(t2);
      variable_dimensions(v)=CONS(DIMENSION,
          make_dimension(int_to_expression(0),int_to_expression(0)),
          variable_dimensions(v));
      basic_pointer(variable_basic(type_variable(t)))=type_undefined;
      free_variable(type_variable(t));
      type_variable(t)=v;
    }
    else if(array_type_p(t3)) {
      variable v = copy_variable(type_variable(t2));
      variable_dimensions(v)=CONS(DIMENSION,
          make_dimension(int_to_expression(0),int_to_expression(0)),
          variable_dimensions(v));
      free_variable(type_variable(t));
      type_variable(t)=v;
    }
  }
}

/* subscripts of the form (*a)[n] are transformed into a[n] 
 * it is coherent with other transformations scattered here and there in this file :p
 */
static void do_linearize_prepatch_subscript(subscript s) {
  expression exp = subscript_array(s);
  if(expression_call_p(exp)) {
    call c =expression_call(exp);
    entity op = call_function(c);
    if(ENTITY_DEREFERENCING_P(op)) {
      expression arg = EXPRESSION(CAR(call_arguments(c)));
      if(expression_reference_p(arg)) {
        reference r = expression_reference(arg);
        entity var = reference_variable(r);
        if(entity_pointer_p(var)) {
          type pointed_type = basic_pointer(variable_basic(type_variable(ultimate_type(entity_type(var)))));
          if(array_type_p(ultimate_type(pointed_type))) {
              reference_indices(r)=CONS(EXPRESSION,int_to_expression(0),reference_indices(r));
              update_expression_syntax(exp,copy_syntax(expression_syntax(arg)));
          }
        }
      }
    }
  }
}

/* transform some subscripts for generic handling later */
static void do_linearize_prepatch_subscripts(statement s) {
  gen_recurse(s,subscript_domain,gen_true,do_linearize_prepatch_subscript);
  cleanup_subscripts(s);
}

static void do_linearize_prepatch(entity m,statement s) {
  FOREACH(ENTITY,e,entity_declarations(m))
    if(entity_variable_p(e)) {
        if(local_entity_of_module_p(e,m) && 
                entity_pointer_p(e) &&
                value_unknown_p(entity_initial(e))) {
            free_value(entity_initial(e));
            entity_initial(e) = make_value_expression(int_to_expression(0));
        }
      do_linearize_prepatch_type(entity_type(e));
    }
  FOREACH(PARAMETER,p,module_functional_parameters(m)) {
    dummy d = parameter_dummy(p);
    if(dummy_identifier_p(d))
    {
      entity di = dummy_identifier(d);
      do_linearize_prepatch_type(entity_type(di));
    }
    do_linearize_prepatch_type(parameter_type(p));
    pips_assert("everything went well",parameter_consistent_p(p));
  }
}

static void do_linearize_array(entity m, statement s, param_t *param) {
  /* step 0: remind all expressions types */
  hash_table e2t = init_expression_is_pointer(s);

  /* step 0.25: hack some subscripts typically found in pips inputs */
  do_linearize_prepatch_subscripts(s);

  /* step 0.5: transform int (*a) [3] into int a[*][3] */
  do_linearize_prepatch(m,s);

  /* step1: the statements */
  do_linearize_array_walker(s);
  FOREACH(ENTITY,e,entity_declarations(m))
    if(entity_variable_p(e))
      do_linearize_array_walker(entity_initial(e));

  /* step2: the declarations */
  FOREACH(ENTITY,e,entity_declarations(m)) {
    if(entity_variable_p(e)) {
      bool rr;
      pips_debug (5, "linearizing entity %s\n", entity_name (e));
      do_linearize_type(&entity_type(e),&rr);
      if(rr) do_linearize_remove_dereferencment(s,e);
      do_linearize_array_init(entity_initial(e));
    }
  }

  /* pips bonus step: the consistency */
  set linearized_param = set_make(set_pointer);
  FOREACH(PARAMETER,p,module_functional_parameters(m)) {
    dummy d = parameter_dummy(p);
    pips_debug (5, "linearizing parameters\n");
    if(dummy_identifier_p(d))
    {
      entity di = dummy_identifier(d);
      do_linearize_type(&entity_type(di),NULL);
      pips_debug (5, "linearizing dummy parameter %s\n", entity_name (di));
    }

    if(do_linearize_type(&parameter_type(p),NULL))
      set_add_element(linearized_param,linearized_param,p);

    // Convert to pointer if requested
    if(param->use_pointers_p) {
      if(dummy_identifier_p(d)) {
        entity di = dummy_identifier(d);
        if(do_convert_this_array_to_pointer_p(di)) {
          do_array_to_pointer_type(&entity_type(di));
          do_array_to_pointer_type(&parameter_type(p));
        }
      }
    }
    pips_assert("everything went well",parameter_consistent_p(p));
  }

  /* step3: change the caller to reflect the new types accordingly */
  if (param->modify_call_site_p) {
    do_linearize_array_manage_callers(m,linearized_param,param);
  }
  set_free(linearized_param);

  /* final step: fix expressions if we have disturbed typing in the process */
  do_linearize_patch_expressions(s,e2t);
  hash_table_free(e2t);
  FOREACH(PARAMETER,p,module_functional_parameters(get_current_module_entity())) {
    pips_assert("everything went well",parameter_consistent_p(p));
  }
}

static void do_array_to_pointer_walk_expression(expression exp) {
  if(expression_reference_p(exp)) {
    reference r = expression_reference(exp);
    entity e =reference_variable(r);
    if(do_convert_this_array_to_pointer_p(e)) {
      list indices = reference_indices(r);
      if(!ENDP(indices)) {
        expression new_expression = expression_undefined;
        reference_indices(r)=NIL;
        indices=gen_nreverse(indices);
        FOREACH(EXPRESSION,index,indices) {
          new_expression=MakeUnaryCall(
              entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
              MakeBinaryCall(
                entity_intrinsic(PLUS_C_OPERATOR_NAME),
                expression_undefined_p(new_expression)?
                entity_to_expression(e):
                new_expression,
                index
                )
              );
        }
        syntax syn = expression_syntax(new_expression);
        expression_syntax(new_expression)=syntax_undefined;
        update_expression_syntax(exp,syn);
      }
    }
  }
  else if(syntax_subscript_p(expression_syntax(exp))) {
    pips_user_warning("subscript are not well handled (yet)!\n");
  }
}

/* fix some strange constructs introduced by previous processing */
static bool do_array_to_pointer_patch_call_expression(expression exp) {
  if(expression_call_p(exp)) {
    call c = expression_call(exp);
    entity op = call_function(c);
    if(ENTITY_ADDRESS_OF_P(op)) {
      expression arg = EXPRESSION(CAR(call_arguments(c)));
      if(expression_call_p(arg)) {
        call c2 = expression_call(arg);
        if(ENTITY_DEREFERENCING_P(call_function(c2))) {
          syntax syn = expression_syntax( EXPRESSION(CAR(call_arguments(c2))) );
          expression_syntax( EXPRESSION(CAR(call_arguments(c2))) )=syntax_undefined;
          update_expression_syntax(exp,syn);
        }
        else if( ENTITY_ADDRESS_OF_P(call_function(c2))) {
          update_expression_syntax(arg,copy_syntax(expression_syntax(EXPRESSION(CAR(call_arguments(c2))))));
        }
      }
    }
  }
  return true;
}

/* special ad-hoc handler for pointer to arrays */
static void do_array_to_pointer_walk_call_and_patch(call c) {
  entity op = call_function(c);
  if(ENTITY_DEREFERENCING_P(op)) {
    expression exp = EXPRESSION(CAR(call_arguments(c)));
    if(expression_reference_p(exp)) {
      reference r = expression_reference(exp);
      entity e = reference_variable(r);
      /* pointer to an array ... */
      if(entity_pointer_p(e)) {
        type pointed_type = basic_pointer(variable_basic(type_variable(ultimate_type(entity_type(e)))));
        if(array_type_p(pointed_type)) {
          update_expression_syntax(exp,
              make_syntax_call(
                make_call(
                  entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                  CONS(EXPRESSION,copy_expression(exp),NIL)
                  )
                )
              );
        }
      }
    }
  }
}

static void do_array_to_pointer_walk_cast(cast ct){
  do_array_to_pointer_type(&cast_type(ct));
}

/* converts arrays to pointer */
static void do_array_to_pointer_walker(void *obj) {
  gen_multi_recurse(obj,
      expression_domain,gen_true,do_array_to_pointer_walk_expression,
      call_domain,gen_true,do_array_to_pointer_walk_call_and_patch,
      cast_domain,gen_true,do_array_to_pointer_walk_cast,
      NULL);
  gen_recurse(obj,expression_domain,do_array_to_pointer_patch_call_expression,do_array_to_pointer_patch_call_expression);

}

/* create a list of statements from entity declarations */
static
list initialization_list_to_statements(entity e) {
  list stats = NIL;
  if(entity_array_p(e)) {
    value v = entity_initial(e);
    if(value_expression_p(v)) {
      expression exp = value_expression(v);
      if(expression_call_p(exp)) {
        call c = expression_call(exp);
        entity op = call_function(c);
        /* we assume that we only have one level of braces, linearize_array should have done the previous job 
         * incomplete type are not handled ...
         * */
        if(ENTITY_BRACE_INTRINSIC_P(op)) {
          expression i = copy_expression(dimension_lower(DIMENSION(CAR(variable_dimensions(type_variable(ultimate_type(entity_type(e))))))));
          FOREACH(EXPRESSION,exp,call_arguments(c)) {
            stats=CONS(STATEMENT,
                make_assign_statement(
                  reference_to_expression(
                    make_reference(
                      e,
                      CONS(EXPRESSION,i,NIL)
                      )
                    ),
                  make_expression(
                    expression_syntax(exp),
                    normalized_undefined
                    )
                  ),
                stats);
            expression_syntax(exp)=syntax_undefined;
            i=make_op_exp(PLUS_OPERATOR_NAME,
                copy_expression(i),
                int_to_expression(1)
                );
          }
        }
      }
    }
    if(!formal_parameter_p(e)) {
        value v = entity_initial(e);
        expression ev;
        if(value_expression_p(v) && (!expression_call_p(ev=value_expression(v)) ||
                !ENTITY_BRACE_INTRINSIC_P(call_function(expression_call(ev)) ))) {
        }
        else {
            /* use alloca when converting array to pointers, to make sure everything is initialized correctly */
            free_value(entity_initial(e));
            type ct = copy_type(ultimate_type(entity_type(e)));
            if(array_type_p(ct)) {
                POP(variable_dimensions(type_variable(ct))); //leak spotted !
                ct = make_type_variable(
                        make_variable(
                            make_basic_pointer(ct),NIL,NIL
                            )
                        );
            }

            entity_initial(e) = make_value_expression(
                    syntax_to_expression(
                        make_syntax_cast(
                            make_cast(ct,
                                MakeUnaryCall(
                                    entity_intrinsic(ALLOCA_FUNCTION_NAME),
                                    make_expression(
                                        make_syntax_sizeofexpression(
                                            make_sizeofexpression_type(
                                                copy_type(entity_type(e))
                                                )
                                            ),
                                        normalized_undefined
                                        )
                                    )
                                )
                            )
                        )
                    );
            AddEntityToModuleCompilationUnit(entity_intrinsic(ALLOCA_FUNCTION_NAME),
                    get_current_module_entity());
        }
    }
  }
  return gen_nreverse(stats);
}

/* initialization statements are added right after declarations */
static void insert_statements_after_declarations(statement st, list stats) {
  if(!ENDP(stats)) {
    if(ENDP(statement_declarations(st))) {
      insert_statement(st,make_block_statement(stats),true);
    }
    else {
      for(list iter=statement_block(st),prev=NIL;!ENDP(iter);POP(iter)) {
        if(declaration_statement_p(STATEMENT(CAR(iter)))) {
          prev=iter;
        } else if(prev == NIL) {
          /* No declarations */
          insert_statement(st, make_block_statement(stats), true);
        } else {
          CDR(prev) = stats;
          while(!ENDP(CDR(stats)))
            POP(stats);
          CDR(stats) = iter;
          break;
        }
      }
    }

  }
}

/* transform each array type in module @p m with statement @p s */
static void do_array_to_pointer(entity m, statement s, param_t *p) {
  /* step1: the statements */
  do_array_to_pointer_walker(s);
  FOREACH(ENTITY,e,entity_declarations(m)) {
    if(entity_variable_p(e) && do_convert_this_array_to_pointer_p(e) ) {
      do_array_to_pointer_walker(entity_initial(e));
    }
  }

  /* step2: the declarations */
  list inits = NIL;
  FOREACH(ENTITY,e,entity_declarations(m))
    if(entity_variable_p(e) && do_convert_this_array_to_pointer_p(e)) {
      // must do this before the type conversion
      inits=gen_append(inits,initialization_list_to_statements(e));
      if(do_array_to_pointer_type(&entity_type(e)))
        do_linearize_remove_dereferencment(s,e);
    }
  /* step3: insert the initialization statement just after declarations */
  insert_statements_after_declarations(get_current_module_statement(),inits);

  /* pips bonus step: the consistency */
  FOREACH(PARAMETER,p,module_functional_parameters(m)) {
    dummy d = parameter_dummy(p);
    if(dummy_identifier_p(d))
    {
      entity di = dummy_identifier(d);
      if( do_convert_this_array_to_pointer_p(di) )  {
        do_array_to_pointer_type(&entity_type(di));
        do_array_to_pointer_type(&parameter_type(p));
      }
    }
    pips_assert("everything went well",parameter_consistent_p(p));
  }

}

/* linearize accesses to an array, and use pointers if asked to */
bool linearize_array_generic (const char* module_name)
{

  debug_on("LINEARIZE_ARRAY_DEBUG_LEVEL");
  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));

  param_t param = { .use_pointers_p = false , .modify_call_site_p = false, .cast_at_call_site_p = false };
  /* Do we have to cast the array at call site ? */
  if (c_module_p(get_current_module_entity())) {
    param.use_pointers_p      = get_bool_property("LINEARIZE_ARRAY_USE_POINTERS");
    param.cast_at_call_site_p = get_bool_property("LINEARIZE_ARRAY_CAST_AT_CALL_SITE");
  }
  param.modify_call_site_p  = get_bool_property("LINEARIZE_ARRAY_MODIFY_CALL_SITE");

  /* it is too dangerous to perform this task on compilation unit, system variables may be changed */
  if(!compilation_unit_entity_p(get_current_module_entity())) {

    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

    /* just linearize accesses and change signature from n-D arrays to 1-D arrays */
    do_linearize_array(get_current_module_entity(),get_current_module_statement(),&param);

    /* additionally perform array-to-pointer conversion for c modules only */
    if(param.use_pointers_p) {
      if(c_module_p(get_current_module_entity())) {
        do_array_to_pointer(get_current_module_entity(),get_current_module_statement(),&param);
        cleanup_subscripts(get_current_module_statement());
      }
      else pips_user_warning("no pointers in fortran !,LINEARIZE_ARRAY_USE_POINTERS ignored\n");
    }

    /* validate */
    pips_assert("everything went well",statement_consistent_p(get_current_module_statement()));
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    if (fortran_module_p(get_current_module_entity()) ) {
      // remove decls_text or the prettyprinter will use that field
      discard_module_declaration_text (get_current_module_entity ());
    } else {
      //compilation unti doesn't exit in fortran
      db_touch_resource(DBR_CODE,compilation_unit_of_module(module_name));
    }
    /*postlude*/
    reset_current_module_statement();
  }
  reset_current_module_entity();
  debug_off();
  return true;
}

/* linearize accesses to an array, and use pointers if asked to */
bool linearize_array(const char* module_name)
{
  return linearize_array_generic (module_name);
}

bool linearize_array_fortran(const char* module_name)
{
  return linearize_array_generic (module_name);
}
