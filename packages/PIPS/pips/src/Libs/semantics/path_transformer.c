#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"
#include <stdbool.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "accel-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

#include "effects-convex.h"
#include "genC.h"

#include "complexity_ri.h"

#include "semantics.h"

//modes
#define PATH_MODE int 
#define MODE_SEQUENCE 0
#define MODE_PROLOGUE 1
#define MODE_PERMANENT 2
#define MODE_EPILOGUE 3



//static bool sbegin_present;
//static bool send_present;


/*initialization for the use of path transformer */
//static int myindices[6];
static path pbegin;
static  path pend;
//static int count;


static bool gen_indices(statement sl)
{
  if(statement_loop_p(sl)){
    loop l = statement_loop(sl);
    expression index_l = range_lower(loop_range(l));//int_to_expression(myindices[count++]);
    gen_array_addto(pbegin.indices, statement_ordering(sl), (void *)index_l);
    expression index_u = range_upper(loop_range(l));//int_to_expression(myindices[count++]);
    gen_array_addto(pend.indices, statement_ordering(sl),(void *)index_u);
  }
  return true;
} 

void path_initialize(statement s, statement sbegin, statement send, path *pb, path *pe)
{
  pb->indices = gen_array_make(0);
  pe->indices = gen_array_make(0);
  pbegin = *pb;
  pend = *pe;
  /*myindices[0]= 2; myindices[1]= 5;  myindices[2]= 2; myindices[3]= 5;
				       myindices[4]= 1;  myindices[5]= 2;*/
  //myindices[0]= 1; myindices[1]= 19;  myindices[2]= 1; myindices[3]= 19;
  //count =0;
  gen_recurse(s,statement_domain,gen_indices,gen_null);
  /*expression index_l = int_to_expression(myindices[0]);
  gen_array_addto(pbegin.indices, statement_ordering(sbegin),(intptr_t)index_l);
  expression index_u = int_to_expression(myindices[1]);
  gen_array_addto(pend.indices, statement_ordering(send),(intptr_t)index_u);*/
  //gen_array_append(pbegin.indices, statement_ordering(sbegin));
  //gen_array_append(pend.indices, statement_ordering(send));
  *pb = pbegin;
  *pe = pend;
  pb->statement = sbegin;
  pe->statement = send;
  return; 
}

static transformer precondition_in_off(statement s)
{
  if(get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT"))
    return  load_statement_precondition(s); 
  else
    return transformer_identity();
}

static transformer do_loop_to_init_transformer(loop l, transformer pre __attribute__ ((unused)))  
{
  entity i = loop_index(l);
  expression init = range_lower(loop_range(l));
  list eff = expression_to_proper_constant_path_effects(init);
  transformer t_init = any_scalar_assign_to_transformer(i, init, eff, transformer_undefined);
  gen_full_free_list(eff);
  return t_init;
}

static transformer do_loop_to_enter_condition_transformer(loop l, transformer pre)  
{
  entity i = loop_index(l);
  expression e1 = entity_to_expression(i);
  expression e2 = copy_expression(range_upper(loop_range(l)));
  expression e3 = binary_intrinsic_expression(GREATER_THAN_OPERATOR_NAME, e1, e2);
  //transformer t_init = do_loop_to_init_transformer(l, transformer_undefined);
  transformer t_enter_up = condition_to_transformer(e3, pre, false);
  free_expression(e3);
  e1 = entity_to_expression(i);
  e2 = copy_expression(range_lower(loop_range(l)));
  e3 = binary_intrinsic_expression(GREATER_OR_EQUAL_OPERATOR_NAME, e1, e2);
  transformer t_enter_low = condition_to_transformer(e3,  transformer_undefined, true);
  free_expression(e3);
  transformer t_enter = transformer_combine(t_enter_up, t_enter_low);
  free_transformer(t_enter_low);
  return t_enter;
}

static transformer do_loop_to_exit_condition_transformer(loop l, transformer pre __attribute__ ((unused)))  
{
  entity i = loop_index(l);
  expression e1 = entity_to_expression(i);
  expression e2 = copy_expression(range_upper(loop_range(l)));
  expression e3 = binary_intrinsic_expression(GREATER_THAN_OPERATOR_NAME, e1, e2);
  transformer t_exit = condition_to_transformer(e3,  /*pre*/ transformer_undefined, true);
  free_expression(e3);
  return t_exit;
}





static transformer approximative(statement sb, statement s, path pbegin, path pend)
{
  statement sbegin = pbegin.statement;
  loop l = statement_loop(s);
  transformer tb, t_star, tb_complete;
  transformer pre = precondition_in_off(sb); 
  tb = path_transformer_on(sb, pbegin, pend, MODE_PERMANENT);
  transformer t_enter = do_loop_to_enter_condition_transformer(l, pre);
  tb_complete = transformer_combine(copy_transformer(t_enter), tb);
  transformer t_inc = transformer_add_loop_index_incrementation(transformer_identity(), l, pre);
  // safe_expression_to_transformer(inc_e, transformer_undefined);
  tb_complete = transformer_combine(tb_complete, t_inc);
  //tb_complete = transformer_combine(tb_complete, t_enter);
  t_star = transformer_derivative_fix_point(tb_complete);
  //t_star = transformer_combine(t_star, t_enter);
  //t_star = transformer_convex_hull(t_enter, t_star_plus
  if(belong_to_statement(s,sbegin, false)){
    t_star = transformer_combine(t_star, tb_complete);
  }
  free_transformer(t_enter);
  //free_transformer(tb); 
  //free_transformer(t_star);
  free_transformer(tb_complete); 
  //print_transformer(t);
  return t_star;
}



static transformer iterate(statement sb, statement s, path pbegin, path pend, expression low, expression high, PATH_MODE m)
{
  loop l = statement_loop(s);
  transformer t = transformer_identity();
  if(expression_constant_p( range_upper(loop_range(l))) && expression_constant_p( range_lower(loop_range(l)))) {
    transformer tb, tb_complete;
    transformer pre = precondition_in_off(sb);
    transformer t_enter = transformer_identity();
    t_enter = do_loop_to_enter_condition_transformer(l, precondition_in_off(s));
    expression iteration_set = MakeBinaryCall(entity_intrinsic("-"),high,low);
    iteration_set = MakeBinaryCall(entity_intrinsic("+"),iteration_set,int_to_expression(1));
    tb = path_transformer_on(sb, pbegin, pend, m);
    for(int i=1;i<=expression_to_int(iteration_set);i++)//low<=i<=high
      {
	tb_complete = transformer_combine(t_enter, tb);
	tb_complete = transformer_add_loop_index_incrementation(tb_complete, l, pre);
	t = transformer_combine(t, tb_complete);
      }
    free_transformer(t_enter);
    //free_transformer(tb);
  }
  else
    t = approximative(sb, s, pbegin, pend);
  return t;
}

static transformer path_transformer_on_loop(statement sb, statement s, path pbegin, path pend, PATH_MODE m, expression ibegin, expression iend)
{
  loop l = statement_loop(s);
  statement sbegin = pbegin.statement;
  statement send = pend.statement;
  transformer pre = precondition_in_off(sb);
  transformer t_exit = do_loop_to_exit_condition_transformer(l, pre);
  //transformer t_init = do_loop_to_init_transformer(l, pre);
  transformer t;
  transformer tp = transformer_identity(), te = transformer_identity(), tp2;
  if(m == MODE_SEQUENCE || m == MODE_PROLOGUE)
    {
      expression low =  MakeBinaryCall(entity_intrinsic("+"),ibegin,int_to_expression(1));
      expression high = (m == MODE_SEQUENCE) ? MakeBinaryCall(entity_intrinsic("-"),iend,int_to_expression(1))
	:range_upper(loop_range(l));
      if(belong_to_statement(s,sbegin, false)){
	 tp = path_transformer_on(sb, pbegin, pend, MODE_PROLOGUE);
	 transformer t_inc = transformer_add_loop_index_incrementation(transformer_identity(), l, precondition_in_off(s));
	 tp = transformer_combine(tp, t_inc); 
	 tp2 = copy_transformer(tp);
      }
      t = iterate(sb, s, pbegin, pend, low, high,  MODE_PERMANENT);
      tp = transformer_combine (tp,t);
      /*optimization: T^* = T^+ union id
       only if prologue exists*/
      if(!belong_to_statement(s,send, false)){
	tp = transformer_combine(tp, t_exit);	
      }
      if(expression_constant_p( range_upper(loop_range(l))) && expression_constant_p( range_lower(loop_range(l)))) {
	if(belong_to_statement(s,sbegin, false)){
	  if(!belong_to_statement(s,send, false)){
	    tp2 = transformer_combine(tp2, t_exit);	
	  }
	  tp = transformer_convex_hull( tp2, tp);
	}
      }
    }
  if(m == MODE_SEQUENCE || m == MODE_EPILOGUE)
    {
      t = transformer_identity();
      expression low = range_lower(loop_range(l));
      expression high = (m == MODE_SEQUENCE)? int_to_expression(-1): MakeBinaryCall(entity_intrinsic("-"),iend,int_to_expression(1)); 
      if(m == MODE_EPILOGUE)
	te = iterate(sb, s, pbegin, pend, low, high, MODE_PERMANENT);
      else{ //avoid T* for the case of sequence: T^(-k)
	te = transformer_identity();
      }
      if(belong_to_statement(s,send, false)){
	t = path_transformer_on(sb, pbegin, pend, MODE_EPILOGUE);
	transformer t_enter = do_loop_to_enter_condition_transformer(l, pre);
	t = transformer_combine(t_enter, t);
      }
      te = transformer_combine (t, te);
    }
  tp = transformer_combine(tp,te);
  return tp;
}
/* We assume that foreach loop, |upper-lower|>0*/

transformer path_transformer_on(statement s, path pbegin, path pend, PATH_MODE m)
{
  instruction inst = statement_instruction(s);
  statement sbegin = pbegin.statement;
  statement send = pend.statement;
  if(statement_ordering(sbegin) == statement_ordering(s) && statement_ordering(send) == statement_ordering(s) &&  (m == MODE_SEQUENCE || m == MODE_EPILOGUE)) {
    if(get_bool_property("IDENTITY_EMPTY_PATH_TRANSFORMER"))
      return transformer_identity();
    else
      return transformer_empty();
  }
  if(statement_ordering(sbegin) == statement_ordering(s)) 
    {
      return complete_statement_transformer(load_statement_transformer(s), precondition_in_off(s),s);
    }
  else {
    if(statement_ordering(send) == statement_ordering(s))
       {
	 return transformer_identity();
       }
  }
  switch (instruction_tag(inst))
    {
    case is_instruction_call:
      {
	statement sbegin = pbegin.statement;
	statement send = pend.statement;
	transformer ts;
	ts = ((m == MODE_PERMANENT)
	      ||(m == MODE_SEQUENCE && statement_ordering(sbegin) <= statement_ordering(s) 
		 && statement_ordering(s) <= statement_ordering(send))
	      ||(m == MODE_PROLOGUE &&  statement_ordering(sbegin) <= statement_ordering(s))
	      ||(m == MODE_EPILOGUE && statement_ordering(s) <= statement_ordering(send)))
	  ?complete_statement_transformer(load_statement_transformer(s), precondition_in_off(s),s): transformer_identity();
	return ts;
      }
    case is_instruction_block :
      {
	transformer ts = transformer_identity(); 
	MAPL( stmt_ptr,
	      {
		statement s1 = STATEMENT(CAR( stmt_ptr ));
		transformer ts1 = path_transformer_on(s1, pbegin, pend, m);
		ts = transformer_combine(ts, ts1);
	      },
	      instruction_block( inst ) );
	return ts;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	transformer tt = path_transformer_on(test_true(t), pbegin, pend, m);
	transformer tf = path_transformer_on(test_false(t), pbegin, pend, m);
	//transformer tc = transformer_identity();
	if(!belong_to_statement(s,sbegin, false) && !belong_to_statement(s,send, false))
	  complete_statement_transformer(load_statement_transformer(s), load_statement_precondition(s)/*precondition_in_off(s)*/,s);
	  //return transformer_convex_hull(tt, tf);
	else {
	  if((belong_to_statement(test_true(t),sbegin, false) && belong_to_statement(test_false(t),send, false)) || (belong_to_statement(test_true(t),send, false) && belong_to_statement(test_false(t),sbegin, false))){
	    if(m == MODE_PERMANENT)
	      return transformer_convex_hull( tt, tf);
	    else
	      return transformer_empty();
	  }
	  else {
	    if(belong_to_statement(test_true(t),sbegin, false) || belong_to_statement(test_true(t),send, false)){
	      transformer t_cond = condition_to_transformer(test_condition(t), /*pre*/ transformer_undefined, true);
	      return transformer_combine(t_cond, tt);
	    }
	    else { 
	      if(belong_to_statement(test_false(t),sbegin, false) || belong_to_statement(test_false(t),send, false)){
		transformer t_cond = condition_to_transformer(test_condition(t), /*pre*/ transformer_undefined, false);
		return transformer_combine(t_cond, tf);
	      }
	    }
	  
	    /*if(!transformer_identity_p(tt) && !transformer_identity_p(tf))
	      return transformer_empty();//false
	      else
	      return transformer_combine( tt, tf);*/
	  }
	}
	return transformer_convex_hull( tt, tf);
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(s);
	statement sb = loop_body(l);
	transformer pre = precondition_in_off(sb);
	expression ibegin =(gen_array_item(pbegin.indices,statement_ordering(s)))?gen_array_item(pbegin.indices,statement_ordering(s)):range_lower(loop_range(l));
	expression iend = (gen_array_item(pend.indices, statement_ordering(s)))?gen_array_item(pend.indices,statement_ordering(s)):range_upper(loop_range(l));
	transformer t_exit = do_loop_to_exit_condition_transformer(l, pre);
	transformer t_init = do_loop_to_init_transformer(l, precondition_in_off(s));
	transformer t;
	if(!belong_to_statement(s,sbegin, false) && !belong_to_statement(s,send, false))
	  return complete_statement_transformer(load_statement_transformer(s), load_statement_precondition(s)/*precondition_in_off(s)*/,s);
	else if(m == MODE_PERMANENT){
	  t = iterate(sb, s, pbegin, pend, range_lower(loop_range(l)), range_upper(loop_range(l)), m);
	  t = transformer_combine(copy_transformer(t_init), t);
	  t = transformer_combine(t, t_exit);	
	  return t;
	}
	else{
	  transformer t_one_iteration = path_transformer_on(sb, pbegin, pend, m);
	  t = path_transformer_on_loop(sb, s, pbegin, pend, m, ibegin, iend);
	  if(!belong_to_statement(s,sbegin, false)){
	    t = transformer_combine(copy_transformer(t_init), t);
	  }
	  if(belong_to_statement(s,sbegin, false) && !belong_to_statement(s,send, false)){
	    transformer t_inc = transformer_add_loop_index_incrementation(transformer_identity(), l, precondition_in_off(s));
	    t_one_iteration = transformer_combine(t_one_iteration, t_inc); 
	    t_one_iteration = transformer_combine(t_one_iteration, t_exit);	
	  }
	  if(belong_to_statement(s,send, false) && !belong_to_statement(s,sbegin, false)){
	    transformer t_enter = do_loop_to_enter_condition_transformer(l, pre);
	    t_one_iteration = transformer_combine(t_enter, t_one_iteration);
	    t_one_iteration = transformer_combine(t_init, t_one_iteration);
	  }
	  if(expression_constant_p(range_upper(loop_range(l))) && expression_constant_p( range_lower(loop_range(l)))) {
	    if(m == MODE_SEQUENCE && expression_equal_p(ibegin, iend))
	      return t_one_iteration;
	    else 
	      return t;
	  }
	  else{
	    t = transformer_convex_hull(t_one_iteration, t);
	    return t;
	  }
	}
	break;
	return t;
      }
    case is_instruction_forloop :
      {
	pips_user_warning("Not implemented yet");
	return transformer_identity();
      }
    default:
      return load_statement_transformer(s);
    }
}

transformer compute_path_transformer (statement s, path pbegin, path pend)
{
  transformer patht = path_transformer_on(s, pbegin, pend, 0);
  return patht;
}


bool path_transformer(char * module_name)
{ 
  entity	module;
  statement	module_stat;
 
  module = local_name_to_top_level_entity(module_name);
  module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
  set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS, module_name, true));
  set_transformer_map((statement_mapping)
		      db_get_memory_resource(DBR_TRANSFORMERS, module_name, true));
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(get_current_module_entity());
  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);

  /*Properties to set the parameters of the path transformer*/
  string sbegin_label_name = (string) get_string_property("PATH_TRANSFORMER_BEGIN");
  string send_label_name = (string) get_string_property("PATH_TRANSFORMER_END");
  if(!label_defined_in_current_module_p(find_label_entity(get_current_module_name(), sbegin_label_name)))
    pips_error("path transformer", "The sbegin label does not exist");
  if(!label_defined_in_current_module_p(find_label_entity(get_current_module_name(), send_label_name)))
    pips_error("path transformer", "The send label does not exist");
  statement sbegin = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),sbegin_label_name);
  statement send =  find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),send_label_name);
  path_initialize(module_stat,  sbegin, send, &pbegin, &pend);
  transformer cumulated_transformer = compute_path_transformer(module_stat, pbegin, pend);
  cumulated_transformer = transformer_normalize(cumulated_transformer,2);
  
  string local = db_build_file_resource_name(DBR_PATH_TRANSFORMER_FILE, module_name, ".pt");
  string dir = db_get_current_workspace_directory();
  string full = strdup(concatenate(dir,"/",local, NULL));
  free(dir);
  FILE * fp = safe_fopen(full,"w");
  text txt = make_text(NIL);
  MERGE_TEXTS(txt, text_module(module, module_stat));
  ADD_SENTENCE_TO_TEXT(txt,MAKE_ONE_WORD_SENTENCE(0,"\nThe path transformer between Sbegin and Send is:"));
  //init_prettyprint(semantic_to_text);
  set_prettyprint_transformer();
  MERGE_TEXTS(txt, text_transformer(cumulated_transformer));
  print_text(fp,txt);
  //fprint_transformer(fp, transformer_normalize(cumulated_transformer,2), (get_variable_name_t)  external_value_name);
  //fprint_transformer(fp, cumulated_transformer, (get_variable_name_t)  external_value_name);
  free_text(txt);
  safe_fclose(fp,full);
  free(full);
  DB_PUT_FILE_RESOURCE(DBR_PATH_TRANSFORMER_FILE, module_name, local);

  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_precondition_map();
  reset_transformer_map();
  reset_current_module_entity();
  generic_effects_reset_all_methods();
  free_value_mappings();
  return true;
}
