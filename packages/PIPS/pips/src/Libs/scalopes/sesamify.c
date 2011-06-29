/* A phase that transform simple tasks in SCMP code.
   clement.marguet@hpc-project.com
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
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "text-util.h"
#include "parser_private.h"
#include "scalopes.h"

/*Use to store action on entities*/
struct dma_action {
  int read;
  int write;
};

hash_table shared_mem;
hash_table entity_action;
/*Convert pointer to fixed size type to normal pointer*/
static type convert_local_to_pointer_array(type local_type){
  list ls        = variable_dimensions(type_variable(local_type));
  size_t size    = gen_length(ls);
  type pointer_type = make_type_variable(make_variable(copy_basic(variable_basic(type_variable(local_type))),
						       NIL,
						       NIL));
  basic b;
  for(unsigned int i = 0; i<size-1; i++){
    b = make_basic_pointer(pointer_type);
    pointer_type = make_type_variable(make_variable(b,NIL,NIL));
  }
  return pointer_type;
}

bool sesamify (char* module_name) {
  debug_on("SESAMIFY_DEBUG_LEVEL");
  
  list args,args2,args3, args4, args5, malloc_statements, map_statements, unmap_statements, wait_statements, send_statements,  rw_effects, entity_declaration;
  callees callees_list = (callees) db_get_memory_resource(DBR_CALLEES, module_name, true);
  intptr_t id;
  entity reserve_data     = local_name_to_top_level_entity("sesam_reserve_data");
  entity get_page_size    = local_name_to_top_level_entity("sesam_get_page_size");
  entity data_assignation = local_name_to_top_level_entity("sesam_data_assignation");
  entity map_data         = local_name_to_top_level_entity("sesam_map_data");
  entity unmap_data       = local_name_to_top_level_entity("sesam_unmap_data");
  entity wait_dispo       = local_name_to_top_level_entity("sesam_wait_dispo");
  entity send_dispo       = local_name_to_top_level_entity("sesam_send_dispo");
  entity chown_data       = local_name_to_top_level_entity("sesam_chown_data");
  entity new, re;
  struct dma_action * val;

  shared_mem      = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
  entity_action   = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);

  intptr_t counter = 1;
  int nb_task_total = gen_length(callees_callees(callees_list));
  int num_task=1;

  FOREACH(STRING,callee_name,gen_nreverse(callees_callees(callees_list))) {

    pips_debug(1,"%s\n", callee_name);
    //Change context
    set_current_module_entity(module_name_to_entity( callee_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE,
								    callee_name,
								    true));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
								       callee_name,
								       true));

    //reset tables
    hash_table_clear(entity_action);

    rw_effects = load_cumulated_rw_effects_list(get_current_module_statement());
    map_statements     = NIL;
    malloc_statements  = NIL;
    unmap_statements   = NIL;
    wait_statements    = NIL;
    send_statements    = NIL;
    entity_declaration = NIL;

    /*list effects in the task*/
    FOREACH(EFFECT,e,rw_effects) {

      re = reference_variable(effect_any_reference(e));

       val = (struct dma_action *) hash_get(entity_action, re);
       /*check if the entities has already been processed with the same action to avoid doublons*/
      if(val==HASH_UNDEFINED_VALUE|| (action_read_p(effect_action(e)) && val->read==0) || (action_write_p(effect_action(e)) && val->write==0)){

	/*if the entity is not stored yet*/
	if(val==HASH_UNDEFINED_VALUE)
	  val = (struct dma_action *) malloc(sizeof(struct dma_action));
	//flags entity effects
	if(action_read_p(effect_action(e)))
	  val->read=1;
	else
	  val->write=1;

	//MEMORY ALLOCATION
	if(hash_get(shared_mem,re)==HASH_UNDEFINED_VALUE){
	  //shared memory ID
	  id=counter-1;
	  hash_put(shared_mem, re, (void*)counter);
	  //compute table memory size
	  range the_range = make_range(int_to_expression(0),
				     make_op_exp(MINUS_OPERATOR_NAME,
						 make_expression(make_syntax_sizeofexpression(make_sizeofexpression_type(entity_type(re))),
								 normalized_undefined),
						 int_to_expression(1)),
				     int_to_expression(1));

	  expression memory_size  = range_to_expression(the_range,range_to_distance);
	  expression size = MakeBinaryCall(entity_intrinsic(DIVIDE_OPERATOR_NAME),
					   memory_size,
					   call_to_expression(make_call(get_page_size,NIL)));

	  size = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				size,
				int_to_expression(1));
	  args  = make_expression_list(size);

	  args2 = CONS(EXPRESSION, int_to_expression(1), NIL);
	  args2 = CONS(EXPRESSION, size,args2);
	  args2 = CONS(EXPRESSION, int_to_expression(counter-1),args2);

	  malloc_statements=CONS(STATEMENT,
				 instruction_to_statement(make_instruction_call(make_call(reserve_data,args))),
				 malloc_statements);
	  malloc_statements=CONS(STATEMENT,
				 instruction_to_statement(make_instruction_call(make_call(data_assignation,args2))),
				 malloc_statements);
	  counter++;
	}
	else{
	  id = (intptr_t) hash_get(shared_mem, re)-1;
	}

	//MAP_DATA + pointer creation +UNMAP + CHMOD
	type t = convert_local_to_pointer_array(entity_type(re));
	new = make_new_scalar_variable(get_current_module_entity(),
					      make_basic_pointer(t));
	entity_declaration = CONS(ENTITY,new, entity_declaration);
	args3 =  make_expression_list(int_to_expression(id));
	expression map_data_exp = call_to_expression(make_call(map_data,args3));
	expression new_exp = entity_to_expression(new);
	replace_entity_by_expression(get_current_module_statement(), re, new_exp);
	args4 = CONS(EXPRESSION,map_data_exp, NIL);
	args4 = CONS(EXPRESSION,new_exp,args4);

	map_statements=CONS(STATEMENT,
			    instruction_to_statement(make_instruction_call(make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME),args4))),
			    map_statements);
	unmap_statements=CONS(STATEMENT,
			      instruction_to_statement(make_instruction_call(make_call(unmap_data,make_expression_list(new_exp)))),
			      unmap_statements);

	//change data owner
	if(num_task < nb_task_total){
	  args5 =  make_expression_list(int_to_expression(id),int_to_expression(num_task));
	  unmap_statements=CONS(STATEMENT,
				instruction_to_statement(make_instruction_call(make_call(chown_data,args5))),
				unmap_statements);
	}

	//SEND + WAIT dispo
	print_effect(e);
	if(action_read_p(effect_action(e))){
	  print_entities(CONS(ENTITY,re,NIL));
	  pips_debug(1,"READ\n");
	  wait_statements=CONS(STATEMENT,
			       instruction_to_statement(make_instruction_call(make_call(wait_dispo,
											make_expression_list(int_to_expression(id),
													     int_to_expression(0),
													     int_to_expression(0),
													     int_to_expression(0))))),
			       wait_statements);
	  send_statements=CONS(STATEMENT,
			       instruction_to_statement(make_instruction_call(make_call(send_dispo,
											make_expression_list(int_to_expression(id),
													     int_to_expression(0),
													     int_to_expression(1))))),
			       send_statements);
	}
	else{
	  print_entities(CONS(ENTITY,re,NIL));
	  pips_debug(1,"WRITE\n");
	  wait_statements=CONS(STATEMENT,
			       instruction_to_statement(make_instruction_call(make_call(wait_dispo,
											make_expression_list(int_to_expression(id),
													     int_to_expression(0),
													     int_to_expression(1),
													     int_to_expression(0))))),
			       wait_statements);
	   send_statements=CONS(STATEMENT,
				instruction_to_statement(make_instruction_call(make_call(send_dispo,
											 make_expression_list(int_to_expression(id),
													      int_to_expression(0),
													      int_to_expression(0))))),
				send_statements);
	}
      }
    }

    //insert all statements
    insert_statement(get_current_module_statement(),
		       make_block_statement(gen_nreverse(wait_statements)),true);
    insert_statement(get_current_module_statement(),
		       make_block_statement(gen_nreverse(map_statements)),true);
    insert_statement(get_current_module_statement(),
		       make_block_statement(gen_nreverse(malloc_statements)),true);
    insert_statement(get_current_module_statement(),
		       make_block_statement(gen_nreverse(send_statements)),false);
    insert_statement(get_current_module_statement(),
		       make_block_statement(gen_nreverse(unmap_statements)),false);

    //add declaration
    FOREACH(ENTITY,ent,entity_declaration){
      AddEntityToCurrentModule(ent);
    }

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, callee_name,
			   get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, callee_name,
			   compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_cumulated_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();
    num_task++;
  }

  debug_off();
  return true;
}
