/* A simple phase that change wrap references to flow outside kernel for dataflow runtime manager

   clement.marguet@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"
#include "callgraph.h"
#include "pipsdbm.h"
#include "properties.h"
#include "accel-util.h"
#include "resources.h"



static bool call_load_store_p(call c){
  
  if(strstr(entity_name(call_function(c)),get_string_property("KERNEL_LOAD_STORE_LOAD_FUNCTION"))!=NULL
     || strstr(entity_name(call_function(c)),get_string_property("KERNEL_LOAD_STORE_STORE_FUNCTION"))!=NULL){
    return TRUE;
  }

  return FALSE;

}

static bool find_call_to_wrap(call c,list *call_to_wrap){
  *call_to_wrap = CONS(CALL,c,*call_to_wrap); 
  return FALSE;
}


static bool find_entities_to_wrap(call c,set entities_to_wrap){

  if(call_load_store_p(c)){
    
    expression exp = EXPRESSION(CAR(call_arguments(c)));
    
    if(expression_call_p(exp)){
      call expcall = expression_call(exp);
      if(call_intrinsic_p(expcall)){
	c = expcall;
      }
    }
    
    print_expression(exp);
    pips_debug(1,"=========================\n");

    entity e = expression_variable(EXPRESSION(CAR(call_arguments(c))));
    

    if(!set_belong_p(entities_to_wrap,e)) {
        set_add_element(entities_to_wrap,entities_to_wrap,e);
    }
  }  
  return TRUE;
}


static bool pointer_to_array_p(entity e){

  list ls = variable_dimensions(type_variable(entity_type(e)));

  if(ENDP(ls)) return true;
  
  return false;
}

static type convert_local_to_pointer_array(type local_type){

  list ls        = variable_dimensions(type_variable(local_type));
  size_t size    = gen_length(ls);
  type pointer_type = make_type_variable(make_variable(copy_basic(variable_basic(type_variable(local_type))),NIL,NIL));
  basic b;
	  
  for(unsigned int i = 0; i<size; i++){
    b = make_basic_pointer(pointer_type);
    pointer_type = make_type_variable(make_variable(b,NIL,NIL));
  }
  
  return pointer_type;
    
}

bool scalopify (char* module_name) {
    
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "SCALOPIFY_DEBUG_LEVEL");
  set entities_to_wrap=set_make(set_pointer);
  list call_to_wrap=NIL;

  expression exp;
  type t;

  entity get_call_entity = local_name_to_top_level_entity("P4A_scmp_flow");

  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

  gen_recurse(module_statement,call_domain,find_entities_to_wrap,gen_identity);

  gen_context_recurse(module_statement,&call_to_wrap,call_domain,find_call_to_wrap,gen_identity);


  FOREACH(CALL, c , call_to_wrap){
 
    if(!call_load_store_p(c)){
  
      SET_FOREACH(entity, e, entities_to_wrap){
 
	/*Two cases: vector or scalar*/
	if(entity_array_p(e)) {	

	 exp =  MakeUnaryCall(get_call_entity, entity_to_expression(e));

	  if(!pointer_to_array_p(e)){
	    t = convert_local_to_pointer_array(entity_type(e));
	  }
	  else{
	    t= entity_type(e);
	  }
	  
	  exp = make_expression( make_syntax_cast(make_cast(t,exp)) ,normalized_undefined);
	}
	else{
	 
	  basic b   = make_basic_pointer(copy_type(entity_type(e)));
	  t   = make_type_variable(make_variable(b,NIL,NIL));

	  exp = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), entity_to_expression(e));
	  exp = MakeUnaryCall(get_call_entity, exp);
	  exp = make_expression(make_syntax_cast(make_cast(t, exp)),normalized_undefined);
	  exp = MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME), exp);
	}
	replace_entity_by_expression(c, e, exp);
      }    
    }
  }


 
  gen_free_list(call_to_wrap);
  set_free(entities_to_wrap);

  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));

  reset_cumulated_rw_effects();
  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);

  return TRUE;
}
