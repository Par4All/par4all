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
 * @file kernels.c
 * kernels manipulation
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2010-01-03
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <ctype.h>


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
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
#include "transformer.h"
#include "expressions.h"
#include "semantics.h"
#include "parser_private.h"
#include "preprocessor.h"
#include "accel-util.h"
#include "c_syntax.h"

struct dma_pair {
  entity new_ent;
  enum region_to_dma_switch s;
};


/* Some constant intended to have a more readable code */
static const int dmaScalar = 0;
static const int dma1D = 1;

static size_t get_dma_dimension(entity to) {
    size_t n = type_dereferencement_depth(entity_type(to)) - 1; /* -1 because we always have pointer to area ... in our case*/
    return n;
}

/**
 * converts a region_to_dma_switch to corresponding dma name
 * according to properties
 */
static string get_dma_name(enum region_to_dma_switch m, size_t d) {
    char *seeds[] = {
        "KERNEL_LOAD_STORE_LOAD_FUNCTION",
        "KERNEL_LOAD_STORE_STORE_FUNCTION",
        "KERNEL_LOAD_STORE_ALLOCATE_FUNCTION",
        "KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION"
    };
    char * propname = seeds[(int)m];
    /* If the DMA is not scalar, the DMA function name is in the property
       of the form KERNEL_LOAD_STORE_LOAD/STORE_FUNCTION_dD: */
    if(d > 0 /* not scalar*/ && (int)m < 2)
        asprintf(&propname,"%s_%dD", seeds[(int)m], (int)d);
    string dmaname = get_string_property(propname);
    if(d > 0 /* not scalar*/ && (int)m <2) free(propname);
    return dmaname;
}

static bool region_to_dimensions(region reg, transformer tr, list *dimensions, list * offsets, expression* condition) {
    if(region_to_minimal_dimensions(reg,tr,dimensions,offsets,true,condition)) {
        return true;
    }
    else
    {
        pips_user_warning("failed to convert regions to minimal array dimensions, using whole array instead\n");
        return false;
    }
}

static void effect_to_dimensions(effect eff, transformer tr, list *dimensions, list * offsets, expression *condition) {
    pips_assert("effects are regions\n",effect_region_p(eff));
    if( ! region_to_dimensions(eff,tr,dimensions,offsets,condition) ) {
        /* let's try with the definition region instead */
        descriptor d = effect_descriptor(eff);
        if(descriptor_convex_p(d)) {
            sc_free(descriptor_convex(d));
            descriptor_convex(d)=entity_declaration_sc(reference_variable(region_any_reference(eff)));
            if( ! region_to_dimensions(eff,tr,dimensions,offsets,condition) )  {
                /* there is still a possibility: expand the sizeof */
                pips_user_warning("failed to compute definition region\n"
                        "This is certainly due to the presence of sizeof or complex size expression in the dimension, trying to take care of this\n");
                /* create a fake no-write effect ... unsure this is safe */
                effects effs = make_effects(NIL);
                partial_eval_declaration(
                        reference_variable(region_any_reference(eff)),
                        predicate_system(transformer_relation(tr)),
                        effs);
                free_effects(effs);
                sc_free(descriptor_convex(d));
                descriptor_convex(d)=entity_declaration_sc(reference_variable(region_any_reference(eff)));
                if( ! region_to_dimensions(eff,tr,dimensions,offsets,condition) )  {
                    pips_internal_error("failed to compute dma from regions appropriately\n");
                }
            }
        }
    }
}

static expression entity_to_address(entity e) {
    size_t n = gen_length(variable_dimensions(type_variable(ultimate_type(entity_type(e)))));
    list indices = NIL;
    while(n--) indices = CONS(EXPRESSION,int_to_expression(0),indices);
    reference r = make_reference(e,indices);
    return MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), reference_to_expression(r));
}


static expression get_sizeofexpression_for_reference(entity variable, list indices) {
  expression sizeof_exp;

  // Never free'd but unclear when we can/should
  reference r = make_reference(variable,indices);

  type element_type = make_type_variable(make_variable(basic_of_reference(r),
                                                       NIL,
                                                       NIL));

  if(type_struct_variable_p(element_type)) {
    expression r_exp = reference_to_expression(r);
    sizeof_exp = MakeSizeofExpression(r_exp);
    free_type(element_type);
  } else {
    sizeof_exp = MakeSizeofType(element_type);
  }
  return sizeof_exp;
}


/**
 * converts dimensions to a dma call from a memory @a from to another memory @a to
 *
 * @param from expression giving the adress of the input memory
 * @param to expression giving the adress of the output memory
 * @param ld list of dimensions to analyze
 * @param m kind of call to generate
 *
 * @return
 */
static
call dimensions_to_dma(entity from,
		  entity to,
		  list/*of dimensions*/ ld,
		  list/*of offsets*/    lo,
		  enum region_to_dma_switch m)
{
  expression dest;
  list args = NIL;
  string function_name = get_dma_name(m,get_dma_dimension(to));

  entity mcpy = module_name_to_entity(function_name);
  if (entity_undefined_p(mcpy)) {
      mcpy=make_empty_subroutine(function_name,copy_language(module_language(get_current_module_entity())));
    pips_user_warning("Cannot find \"%s\" method. Are you sure you have set\n"
		    "KERNEL_LOAD_STORE_..._FUNCTION "
		    "to a defined entity and added the correct .c file?\n",function_name);
  }
  else {
      AddEntityToModuleCompilationUnit(mcpy,get_current_module_entity());
  }

  /* Scalar detection: */
  bool scalar_entity = entity_scalar_p(from);

  if (dma_allocate_p(m)) {
      /* Need the address for the allocator to modify the pointer itself: */
      dest = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),entity_to_expression(to));
      /* Generate a "void **" type: */
      type voidpp = make_type_variable(
              make_variable(
                  make_basic_pointer(
                      make_type_variable(
                          make_variable(
                              make_basic_pointer(
                                  make_type_void(NIL)
                                  ),
                              NIL,NIL
                              )
                          )
                      ),
                  NIL,NIL
                  )
              );
      /* dest = "(void **) &to" */
      dest = make_expression(
              make_syntax_cast(
                  make_cast(voidpp,dest)
                  ),
              normalized_undefined);
  }
  else if (!dma_deallocate_p(m) && !scalar_entity)
    /* Except for the deallocation or if we have a scalar and then we have
       already created a pointer to it, the original array is referenced
       through pointer dereferencing: */
    dest = MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
			 entity_to_expression(to));
  else
    dest=entity_to_expression(to);



  switch(m) {
      case dma_deallocate:
          args = make_expression_list(dest);
          break;
      case dma_allocate:
          {
            expression sizeof_exp = get_sizeofexpression_for_reference(from,lo);

            /* sizeof(element)*number elements of the array: */
              expression transfer_size = SizeOfDimensions(ld);
              transfer_size=MakeBinaryCall(
                      entity_intrinsic(MULTIPLY_OPERATOR_NAME),
                      sizeof_exp,
                      transfer_size);

              args = make_expression_list(dest, transfer_size);
          } break;
      case dma_load:
      case dma_store:
	/* Generate communication functions: */
	{
	  //if(!scalar_entity) {
	    /* Build the sizes of the array block to transfer: */
	    list /*of expressions*/ transfer_sizes = NIL;
	    FOREACH(DIMENSION,d,ld) {
	      expression transfer_size=
		SizeOfDimension(d);
	      transfer_sizes=CONS(EXPRESSION,transfer_size,transfer_sizes);
	    }
	    transfer_sizes=gen_nreverse(transfer_sizes);

	    /* Build the sizes of the array with element to transfer: */
	    list/* of expressions*/ from_dims = NIL;
	    /* We may skip the size of the first dimension since it is not
	       used in adress calculation. But since it depends of Fortran
	       or C in the runtime, postpone this micro-optimization... */
	    FOREACH(DIMENSION,d,variable_dimensions(type_variable(ultimate_type(entity_type(from))))) {
	      from_dims=CONS(EXPRESSION,SizeOfDimension(d),from_dims);
	    }
	    from_dims=gen_nreverse(from_dims);

	    /* Build the offsets of the array block to transfer: */
	    list/* of expressions*/ offsets = NIL;
	    FOREACH(EXPRESSION,e,lo)
	      offsets=CONS(EXPRESSION,e,offsets);
	    offsets=gen_nreverse(offsets);
	/* Use a special transfert function for scalars instead of reusing
	   the 1D function. It may useful for example if it is implemented
	   as a FIFO at the hardware level: */
	//} else {
	//    /* If we have a scalar variable to transfert, generate
	//       synthetic transfer parameters: */
	//    /* 1 element to transfert */
	//    transfer_sizes = make_expression_list(int_to_expression(1));
	//    /* 1 dimension */
	//    from_dims = make_expression_list(int_to_expression(1));
	//    /* At the begining of the « array »: */
	//    offsets = make_expression_list(int_to_expression(0));
	//  }

	  expression source = entity_to_address(from);
	  /* Generate host and accel adresses: */
	  args = CONS(EXPRESSION,source,CONS(EXPRESSION,dest,NIL));
	  //if(dma_load_p(m))
	  //    args=gen_nreverse(args);
	  /* Output parameters in an order compatible with some C99
	     implementation of the runtime: size and block size first, so
	     that some arguments can be defined with them: */
	  /* Insert offset: */
	  args = gen_append(offsets, args);
	  /* Insert the block size to transfert: */
	  args = gen_append(transfer_sizes, args);
	  /* Insert the array sizes: */
	  args = gen_append(from_dims, args);
	  /* Insert the element size expression: */
	  expression sizeof_exp = get_sizeofexpression_for_reference(from,lo);
	  args = CONS(EXPRESSION,
                sizeof_exp,
                args);
	} break;
      default:
          pips_internal_error("should not happen");
  }
  return make_call(mcpy, args);
}

static bool effect_on_non_local_variable_p(effect eff) {
    return !same_string_p(
            entity_module_name(reference_variable(effect_any_reference(eff))),
            get_current_module_name()
            );
}

static bool effects_on_non_local_variable_p(list effects) {
    FOREACH(EFFECT,eff,effects)
        if( effect_on_non_local_variable_p(eff)) {
            char * seffect = effect_to_string(eff);
            pips_user_warning("effect on non local variable: %s\n",seffect);
            free(seffect);
            return true;
        }
    return false;
}


/* Compute a call to a DMA function from the effects of a statement

   @return a statement of the DMA transfers or statement_undefined if
   nothing needed or if the dma function has been set to "" in the relevant property
 */
static
statement effects_to_dma(statement stat,
			 enum region_to_dma_switch s,
			 hash_table e2e, expression * condition,
             bool fine_grain_analysis)
{
    /* if no dma is provided, skip the computation
     * it is used for scalope at least */
  if(empty_string_p(get_dma_name(s,dma1D)))
    return statement_undefined;

  list rw_effects= load_cumulated_rw_effects_list(stat);
  transformer tr = transformer_range(load_statement_precondition(stat));

  list effects = NIL;

  /* filter out relevant effects depending on operation mode */
  FOREACH(EFFECT,e,rw_effects) {
    if ((dma_load_p(s) || dma_allocate_p(s) || dma_deallocate_p(s))
	&& action_read_p(effect_action(e)))
      effects=CONS(EFFECT,e,effects);
    else if ((dma_store_p(s)  || dma_allocate_p(s) || dma_deallocate_p(s))
	     && action_write_p(effect_action(e)))
        effects=CONS(EFFECT,e,effects);
  }

  /* handle the may approximations here: if the approximation is may,
   * we have to load the data, otherwise the store may store
   * irrelevant data
   */
  if (dma_load_p(s) || dma_allocate_p(s) || dma_deallocate_p(s)) {
      /* first step is to check for may-write effects */
      list may_write_effects = NIL;
      FOREACH(EFFECT,e,rw_effects) {
          if(approximation_may_p(effect_approximation(e)) &&
                      action_write_p(effect_action(e)) ) {
              effect fake = copy_effect(e);
              action_tag(effect_action(fake))=is_action_read;
              may_write_effects=CONS(EFFECT,fake,may_write_effects);
          }
      }
      /* then we will merge these effects with those 
       * that were already gathered
       * because we are manipulating sets, it is not very efficient
       * but there should not be that many effects anyway
       */
      FOREACH(EFFECT,e_new,may_write_effects) {
        bool merged = false; // if we failed to merge e_new in effects, we just add it to the list */
        for(list iter=effects;!ENDP(iter);POP(iter)){
          effect * e_origin = (effect*)REFCAR(iter); // get a reference to change it in place if needed
          if(same_entity_p(
                effect_any_entity(*e_origin),
                effect_any_entity(e_new))) {
            merged=true;
            region tmp = regions_must_convex_hull(*e_origin,e_new);
            // there should be a free there, but it fails
            *e_origin=tmp;
          }
        }
        /* no data was copy-in, add this effect */
        if(!merged) {
          effects=CONS(EFFECT,copy_effect(e_new),effects);
        }
      }
      gen_full_free_list(may_write_effects);
  }

  /* if we failed to provide a fine_grain_analysis, we can still rely on the definition region to over approximate the result
   */
  if(!fine_grain_analysis) {
    FOREACH(EFFECT,eff,effects) {
      descriptor d = effect_descriptor(eff);
      if(descriptor_convex_p(d)) {
        sc_free(descriptor_convex(d));
        descriptor_convex(d)=entity_declaration_sc(reference_variable(region_any_reference(eff)));
      }
    }
  }

  if(effects_on_non_local_variable_p(effects)){
      pips_user_warning("Cannot handle non local variables in isolated statement\n");
      return statement_undefined;
  }

  /* builds out transfer from gathered effects */
  list statements = NIL;
  FOREACH(EFFECT,eff,effects) {
    statement the_dma = statement_undefined;
    reference r = effect_any_reference(eff);
    entity re = reference_variable(r);
    struct dma_pair * val = (struct dma_pair *) hash_get(e2e, re);

    if( val == HASH_UNDEFINED_VALUE || (val->s != s) ) {
        if(!entity_scalar_p(re) || get_bool_property("KERNEL_LOAD_STORE_SCALAR")) {
            list /*of dimensions*/ the_dims = NIL,
                 /*of expressions*/the_offsets = NIL;
            effect_to_dimensions(eff,tr,&the_dims,&the_offsets,condition);

            entity eto;
            if(val == HASH_UNDEFINED_VALUE) {

                /* initialized with NULL value */
                expression init = int_to_expression(0);

                /* Replace the reference to the array re to *eto: */
                entity renew = make_new_array_variable(get_current_module_entity(),copy_basic(entity_basic(re)),the_dims);
                eto = make_temporary_pointer_to_array_entity_with_prefix(entity_local_name(re),renew,init);
                AddLocalEntityToDeclarations(eto,get_current_module_entity(),stat);
                isolate_patch_entities(stat,re,eto,the_offsets);

                val=malloc(sizeof(*val));
                val->new_ent=eto;
                val->s=s;
                hash_put(e2e,re,val);
            }
            else {
                eto = val->new_ent;
                val->s=s;/*to avoid duplicate*/
            }
            the_dma = instruction_to_statement(make_instruction_call(dimensions_to_dma(re,eto,the_dims,the_offsets,s)));
            statements=CONS(STATEMENT,the_dma,statements);
        }
    }
  }
  gen_free_list(effects);
  if (statements == NIL)
    return statement_undefined;
  else
    return make_block_statement(statements);
}

static bool do_isolate_statement_preconditions(statement s)
{
    callees c = compute_callees(s);
    bool nocallees = ENDP(callees_callees(c));
    free_callees(c);
    if(! nocallees) {
        pips_user_warning("cannot isolate statement with callees\n");
        return false;
    }
    return true;
}

/* perform statement isolation on statement @p s
 * that is make sure that all access to variables in @p s 
 * are made either on private variables or on new entities declared on a new memory space
 */
void do_isolate_statement(statement s) {
    bool fine_grain_analysis = true;
    statement allocates, loads, stores, deallocates;
    /* this hash table holds an entity to (entity + tag ) binding */
    hash_table e2e ;
    if(!do_isolate_statement_preconditions(s)) {
        pips_user_warning("isolated statement has callees, transfers will be approximated\n");
        fine_grain_analysis = false;
    }
    e2e = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    expression condition = expression_undefined;
    allocates = effects_to_dma(s,dma_allocate,e2e,&condition,fine_grain_analysis);
    loads = effects_to_dma(s,dma_load,e2e,NULL,fine_grain_analysis);
    stores = effects_to_dma(s,dma_store,e2e,NULL,fine_grain_analysis);
    deallocates = effects_to_dma(s,dma_deallocate,e2e,NULL,fine_grain_analysis);
    HASH_MAP(k,v,free(v),e2e);
    hash_table_free(e2e);

    /* Add the calls now if needed, in the correct order: */
    if (loads != statement_undefined)
        insert_statement(s, loads,true);
    if (stores != statement_undefined)
        insert_statement(s, stores,false);
    if (deallocates != statement_undefined)
        insert_statement(s, deallocates,false);
    if (allocates != statement_undefined)
        insert_statement(s, allocates,true);
    /* guard the whole block by according conditions */
    if(!expression_undefined_p(condition)) {

        /* prends ton couteau suisse et viens jouer avec moi dans pips */
        pips_assert("statement is a block",statement_block_p(s));
        for(list prev=NIL,iter=statement_block(s);!ENDP(iter);POP(iter)) {
            if(declaration_statement_p(STATEMENT(CAR(iter)))) prev=iter;
            else {
                pips_assert("there must be at least one declaration",!ENDP(prev));
                statement cond = 
                    instruction_to_statement(
                            make_instruction_test(
                                make_test(
                                    condition,
                                    make_block_statement(iter),
                                    make_empty_statement()
                                    )
                                )
                            );
                CDR(prev)=CONS(STATEMENT,cond,NIL);
                break;
            }
        }
    }
}

static void
kernel_load_store_generator(statement s, string module_name)
{
    if(statement_call_p(s))
    {
        call c = statement_call(s);
        if(!call_intrinsic_p(c) &&
                same_string_p(module_local_name(call_function(c)),module_name))
        {
            do_isolate_statement(s);
        }
    }
}

/* run kernel load store using either region or effect engine,
 */
static bool kernel_load_store_engine(char *module_name,const char * enginerc) {
    /* generate a load stores on each caller */

    debug_on("KERNEL_LOAD_STORE_DEBUG_LEVEL");

    callees callers = (callees)db_get_memory_resource(DBR_CALLERS,module_name,true);
    FOREACH(STRING,caller_name,callees_callees(callers)) {
        /* prelude */
        set_current_module_entity(module_name_to_entity( caller_name ));
        set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, caller_name, true) );
        set_cumulated_rw_effects((statement_effects)db_get_memory_resource(enginerc, caller_name, true));
        module_to_value_mappings(get_current_module_entity());
        set_precondition_map( (statement_mapping) db_get_memory_resource(DBR_PRECONDITIONS, caller_name, true) );
        /*do the job */
        gen_context_recurse(get_current_module_statement(),module_name,statement_domain,gen_true,kernel_load_store_generator);
        /* validate */
        module_reorder(get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CODE, caller_name,get_current_module_statement());
        DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, caller_name, compute_callees(get_current_module_statement()));

        /*postlude*/
        reset_precondition_map();
        free_value_mappings();
        reset_cumulated_rw_effects();
        reset_current_module_entity();
        reset_current_module_statement();
    }

    /*flag the module as kernel if not done */
    callees kernels = (callees)db_get_memory_resource(DBR_KERNELS,"",true);
    bool found = false;
    FOREACH(STRING,kernel_name,callees_callees(kernels))
        if( (found=(same_string_p(kernel_name,module_name))) ) break;
    if(!found)
        callees_callees(kernels)=CONS(STRING,strdup(module_name),callees_callees(kernels));
    db_put_or_update_memory_resource(DBR_KERNELS,"",kernels,true);

    debug_off();

    return true;
}


/** Generate malloc/copy-in/copy-out on the call sites of this module.
  * based on convex array regions
  */
bool kernel_load_store(char *module_name) {
    return kernel_load_store_engine(module_name,DBR_REGIONS);
}



/**
 * create a statement eligible for outlining into a kernel
 * #1 find the loop flagged with loop_label
 * #2 make sure the loop is // with local index
 * #3 perform strip mining on this loop to make the kernel appear
 * #4 perform two outlining to separate kernel from host
 *
 * @param s statement where the kernel can be found
 * @param loop_label label of the loop to be turned into a kernel
 *
 * @return true as long as the kernel is not found
 */
static
bool do_kernelize(statement s, entity loop_label)
{
  if( same_entity_p(statement_label(s),loop_label) ||
      (statement_loop_p(s) && same_entity_p(loop_label(statement_loop(s)),loop_label)))
    {
      if( !instruction_loop_p(statement_instruction(s)) )
	pips_user_error("you choosed a label of a non-doloop statement\n");



      loop l = instruction_loop(statement_instruction(s));

      /* gather and check parameters */
      int nb_nodes = get_int_property("KERNELIZE_NBNODES");
      while(!nb_nodes)
        {
	  string ur = user_request("number of nodes for your kernel?\n");
	  nb_nodes=atoi(ur);
        }

      /* verify the loop is parallel */
      if( execution_sequential_p(loop_execution(l)) )
	pips_user_error("you tried to kernelize a sequential loop\n");
      if( !entity_is_argument_p(loop_index(statement_loop(s)),loop_locals(statement_loop(s))) )
	pips_user_error("you tried to kernelize a loop whose index is not private\n");

      if(nb_nodes >1 )
      {
      /* we can strip mine the loop */
      loop_strip_mine(s,nb_nodes,-1);
      /* unfortunetly, the strip mining does not exactly does what we
	 want, fix it here

	 it is legal because we know the loop index is private,
	 otherwise the end value of the loop index may be used
	 incorrectly...
      */
      {
	statement s2 = loop_body(statement_loop(s));
	entity outer_index = loop_index(statement_loop(s));
	entity inner_index = loop_index(statement_loop(s2));
	replace_entity(s2,inner_index,outer_index);
	loop_index(statement_loop(s2))=outer_index;
	replace_entity(loop_range(statement_loop(s2)),outer_index,inner_index);
	if(!ENDP(loop_locals(statement_loop(s2)))) replace_entity(loop_locals(statement_loop(s2)),outer_index,inner_index);
	loop_index(statement_loop(s))=inner_index;
	replace_entity(loop_range(statement_loop(s)),outer_index,inner_index);
    RemoveLocalEntityFromDeclarations(outer_index,get_current_module_entity(),get_current_module_statement());
	loop_body(statement_loop(s))=make_block_statement(make_statement_list(s2));
	AddLocalEntityToDeclarations(outer_index,get_current_module_entity(),loop_body(statement_loop(s)));
	l = statement_loop(s);
      }
      }

      string kernel_name=get_string_property_or_ask("KERNELIZE_KERNEL_NAME","name of the kernel ?");
      string host_call_name=get_string_property_or_ask("KERNELIZE_HOST_CALL_NAME","name of the fucntion to call the kernel ?");

      /* validate changes */
      callees kernels=(callees)db_get_memory_resource(DBR_KERNELS,"",true);
      callees_callees(kernels)= CONS(STRING,strdup(host_call_name),callees_callees(kernels));
      DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);

      entity cme = get_current_module_entity();
      statement cms = get_current_module_statement();
      module_reorder(get_current_module_statement());
      DB_PUT_MEMORY_RESOURCE(DBR_CODE, get_current_module_name(),get_current_module_statement());
      DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, get_current_module_name(), compute_callees(get_current_module_statement()));
      reset_current_module_entity();
      reset_current_module_statement();

      /* recompute effects */
      proper_effects(module_local_name(cme));
      cumulated_effects(module_local_name(cme));
      set_current_module_entity(cme);
      set_current_module_statement(cms);
      set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, get_current_module_name(), TRUE));
      /* outline the work and kernel parts*/
      outliner(kernel_name,make_statement_list(loop_body(l)));
      (void)outliner(host_call_name,make_statement_list(s));
      reset_cumulated_rw_effects();




      /* job done */
      gen_recurse_stop(NULL);

    }
  return true;
}


/**
 * turn a loop flagged with LOOP_LABEL into a kernel (GPU, terapix ...)
 *
 * @param module_name name of the module
 *
 * @return true
 */
bool kernelize(char * module_name)
{
  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

  /* retreive loop label */
  string loop_label_name = get_string_property_or_ask("LOOP_LABEL","label of the loop to turn into a kernel ?\n");
  entity loop_label_entity = find_label_entity(module_name,loop_label_name);
  if( entity_undefined_p(loop_label_entity) )
    pips_user_error("label '%s' not found in module '%s' \n",loop_label_name,module_name);


  /* run kernelize */
  gen_context_recurse(get_current_module_statement(),loop_label_entity,statement_domain,do_kernelize,gen_null);

  /* validate */
  module_reorder(get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

  /*postlude*/
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}

bool flag_kernel(char * module_name)
{
  if (!db_resource_p(DBR_KERNELS, ""))
    pips_internal_error("kernels not initialized");
  callees kernels=(callees)db_get_memory_resource(DBR_KERNELS,"",true);
  callees_callees(kernels)= CONS(STRING,strdup(module_name),callees_callees(kernels));
  DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);
  return true;
}

bool bootstrap_kernels(__attribute__((unused)) char * module_name)
{
  if (db_resource_p(DBR_KERNELS, ""))
    pips_internal_error("kernels already initialized");
  callees kernels=make_callees(NIL);
  DB_PUT_MEMORY_RESOURCE(DBR_KERNELS,"",kernels);
  return true;
}


