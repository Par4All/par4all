/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"
#include "preprocessor.h"

#define OUTLINE_SUFFIX ".outlined"
#define TARGET_LABEL_NAME_LEN 64

GENERIC_GLOBAL_FUNCTION(outline,outline_map);


static outline_data outline_outlining=outline_data_undefined;

static int outline_from=0;
static int outline_to=0;
static int outline_step=0;
static entity outlined_module=entity_undefined;
static list outline_list=list_undefined;
static list outline_local=list_undefined;
static char target_label[TARGET_LABEL_NAME_LEN];

//######################## OUTLINE INIT ########################

bool outlining_init(string __attribute__ ((unused)) module_name)
{
  init_outline();
  outlining_save();
  return true;
}

void outlining_load()
{
  set_outline((outline_map)db_get_memory_resource(DBR_OUTLINED, "", true));
}

void outlining_save()
{
  DB_PUT_MEMORY_RESOURCE(DBR_OUTLINED, "", get_outline());
  reset_outline();
}

//####################### OUTLINE STEP 1 #######################

/* block isn't duplicated in the outline_data genereted by outlining_close */
entity outlining_start(const char* new_module_name)
{
  debug_on("STEP_TUTORIAL_LEVEL");
  pips_debug(1, "new_module_name = %s\n", new_module_name);

  pips_assert("step", outline_step==0);
  pips_assert("list", list_undefined_p(outline_list));
  pips_assert("local", list_undefined_p(outline_local));
  pips_assert("outlined_module", entity_undefined_p(outlined_module));
  pips_assert("outline_outlining", outline_data_undefined_p(outline_outlining));

  // ensure "NEW_MODULE_NAME" from "nEW_MOdULe_name"
  char * mname=strupper(strdup(new_module_name),new_module_name);

  outlined_module = make_empty_subroutine(mname,copy_language(module_language(get_current_module_entity())));

  if (entity_undefined_p(outlined_module)) return false;

  outline_list = NIL;
  outline_local = NIL;
  outline_outlining = make_outline_data(get_current_module_entity(),NIL,NIL,NIL);
  outline_step = 1;
  pips_debug(1, "outlined_module = %p\n", outlined_module);
	debug_off();

  return outlined_module;
}

entity outlining_add_declaration(entity e)
{
  pips_debug(1, "e = %p\n", e);

  pips_assert("step",outline_step==1);
  if (!gen_in_list_p(e,outline_list))
    outline_list=CONS(ENTITY,e,outline_list);

  pips_debug(1, "e = %p\n", e);
  return e;
}

entity outlining_local_declaration(entity e)
{
  outline_local=CONS(ENTITY,e,outline_local);
  return e;
}

//####################### OUTLINE STEP 2 #######################

expression entity_to_expr(e)
     entity e;
{
  switch (type_tag(entity_type(e))){
  case is_type_variable:
    return reference_to_expression(make_reference(e,NIL));
    break;
  case is_type_functional:
    return call_to_expression(make_call(e,NIL));
    break;
  default:
    pips_internal_error("unexpected entity tag: %d", type_tag(entity_type(e)));
    return expression_undefined; 
  }
}

static bool outlining_entity_filter(entity e);

static bool outlining_reference_filter(reference r)
{
  return outlining_entity_filter(reference_variable(r));
}

static bool outlining_call_filter(call c)
{
  return outlining_entity_filter(call_function(c));
}

static bool outlining_loop_filter(loop l)
{
  return outlining_entity_filter(loop_index(l));
}

static bool outlining_entity_filter(entity e)
{
  pips_debug(5, "e = %p\n", e);

  if (gen_in_list_p(e,outline_list))
    {
      pips_debug(5,"entity %s already treated\n", entity_global_name(e));
      return true;
    }
  if (!gen_in_list_p(e,code_declarations(value_code(entity_initial(get_current_module_entity())))))
    {
      pips_debug(5,"entity %s not in %s declaration\n", entity_global_name(e), entity_global_name(get_current_module_entity()));
      return true;
    }
  if (gen_in_list_p(e,outline_local))
    {
      pips_debug(5,"entity %s local in %s declaration\n", entity_global_name(e), entity_global_name(outlined_module));
      return true;
    }
  if(entity_variable_p(e)) // scan entity in type
      gen_multi_recurse(entity_type(e),
			reference_domain, outlining_reference_filter, gen_null,
			call_domain,outlining_call_filter,gen_null,
			NULL);

  pips_debug(5,"new scanned entity %s\n", entity_global_name(e));
  outline_list=CONS(ENTITY,e,outline_list);      

  pips_debug(5, "end\n");
  return true;
}

list outlining_scan_block(list block_list)
{
  statement block;
  pips_debug(1, "block_list = %p\n", block_list);

  pips_assert("step",outline_step==1);
  pips_assert("list",!list_undefined_p(block_list));

  block = make_block_statement(gen_full_copy_list(block_list));
  gen_multi_recurse(block,
		    reference_domain, outlining_reference_filter, gen_null,
		    call_domain,outlining_call_filter,gen_null,
		    loop_domain,outlining_loop_filter,gen_null,
		    NULL);
  free_statement(block);
  outline_data_block(outline_outlining) = gen_full_copy_list(block_list);
  outline_step = 2;

  pips_debug(1, "outline_list = %p\n", outline_list);
  return outline_list;
}

static void patch_outlined_reference(expression x, entity e)
{
  if(expression_reference_p(x))
    {
        reference r =expression_reference(x);
        entity e1 = reference_variable(r);
	if(same_entity_p(e, e1))
	  {
	    variable v = type_variable(entity_type(e));
	    if( (!basic_pointer_p(variable_basic(v))) && (ENDP(variable_dimensions(v))) ) /* not an array / pointer */
	      {
		expression X = make_expression(expression_syntax(x),normalized_undefined);
		expression_syntax(x)=make_syntax_call(make_call(
								CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
								CONS(EXPRESSION,X,NIL)));
	      }
	    
	  }
    }
}

static
void bug_in_patch_outlined_reference(loop l , entity e)
{
    if( same_entity_p(loop_index(l), e))
    {
        statement parent = (statement)gen_get_ancestor(statement_domain,l);
        pips_assert("child's parent child is me",statement_loop_p(parent) && statement_loop(parent)==l);
        statement body =loop_body(l);
        range r = loop_range(l);
        instruction new_instruction = make_instruction_forloop(
                    make_forloop(
                        make_assign_expression(
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_lower(r)),
                        MakeBinaryCall(entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_upper(r)),
                        MakeBinaryCall(entity_intrinsic(PLUS_UPDATE_OPERATOR_NAME),
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_increment(r)),
                        body
                        )
                    );

        statement_instruction(parent)=instruction_undefined;/*SG this is a small leak */
        update_statement_instruction(parent,new_instruction);
        pips_assert("statement consistent",statement_consistent_p(parent));
    }
}
static void outlining_make_argument(entity e, expression expr)
{
  pips_debug(3, "e = %p, expr = %p\n", e, expr);

  pips_assert("outlined_module",!entity_undefined_p(outlined_module));
  pips_assert("outline_list",!list_undefined_p(outline_list));

  bool outline_symbolic_p=get_bool_property("OUTLINING_SYMBOLIC");
  functional ft = type_functional(entity_type(outlined_module));
  entity new_entity=FindOrCreateEntity(entity_local_name(outlined_module),entity_user_name(e));
  expression x = expr;
  bool is_fortran = fortran_module_p(get_current_module_entity());

  entity_initial(new_entity) = copy_value(entity_initial(e));

  if (outline_symbolic_p && value_symbolic_p(entity_initial(e)))
    entity_type(new_entity) = copy_type(functional_result(type_functional(entity_type(e))));
  else
    entity_type(new_entity) = copy_type(entity_type(e));
  
  if (expression_undefined_p(x))
    {
      variable v = type_variable(entity_type(e));
      x = entity_to_expr(e);
      if (!is_fortran && 
	  !basic_pointer_p(variable_basic(v)) && 
	  ENDP(variable_dimensions(v)))
	{
	  x = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, x, NIL));
	  entity_type(new_entity) = make_type_variable(make_variable(
								     make_basic_pointer(copy_type(entity_type(new_entity))),
								     NIL,
								     NIL
								     )
						       );
	  /* remplacement dans le body des references a l'entity e par un pointeur vers l'entity e
	   */
	  gen_context_multi_recurse(make_sequence(outline_data_block(outline_outlining)), e,
				    //statement_domain,gen_true,patch_outlined_reference_in_declarations,
				    loop_domain,gen_true,bug_in_patch_outlined_reference,
				    expression_domain,gen_true,patch_outlined_reference,
				    0);
	}
    }

  if (entity_variable_p(e)||(outline_symbolic_p && value_symbolic_p(entity_initial(e))))
    {
      entity_storage(new_entity)=make_storage(is_storage_formal, make_formal(outlined_module,gen_length(outline_data_formal(outline_outlining))+1));
      functional_parameters(ft) =
	CONS(PARAMETER,make_parameter(entity_type(new_entity),MakeModeReference(),make_dummy_identifier(new_entity)),
		 functional_parameters(ft));
      outline_data_formal(outline_outlining)=CONS(ENTITY,new_entity,outline_data_formal(outline_outlining));
      outline_data_arguments(outline_outlining)=CONS(EXPRESSION, x, outline_data_arguments(outline_outlining));
      pips_debug(3,"add argument %s\n", entity_global_name(e));
    }
  else
    entity_storage(new_entity) = copy_storage(entity_storage(e));
  
  pips_debug(3,"add entity %s\n", entity_global_name(new_entity));

  code_declarations(value_code(entity_initial(outlined_module)))=CONS(ENTITY,new_entity,code_declarations(value_code(entity_initial(outlined_module))));
  
  pips_debug(3, "fin\n");
}

void outlining_add_argument(entity e,expression expr)
{
  pips_debug(3, "e = %p, expr = %p\n", e, expr);

  pips_assert("step", outline_step==2);
  gen_remove(&outline_list,e);
  outlining_make_argument(e,expr);

  pips_debug(3, "fin\n");
}

//####################### OUTLINE STEP 3 #######################

statement outlining_close(string new_user_file)
{
  bool saved;
  entity label = entity_undefined;
  statement call = statement_undefined;
  statement source_body = statement_undefined;
  bool is_fortran=fortran_module_p(get_current_module_entity());

  pips_debug(1, "begin\n");

  pips_assert("step", outline_step==2);

  // argument building
  outline_list = gen_nreverse(outline_list);
  FOREACH(ENTITY, e, outline_list)
    {
      outlining_make_argument(e, expression_undefined);
    }
  functional_parameters(type_functional(entity_type(outlined_module)))=gen_nreverse(functional_parameters(type_functional(entity_type(outlined_module))));
  code_declarations(value_code(entity_initial(outlined_module))) = gen_nreverse(code_declarations(value_code(entity_initial(outlined_module))));
  outline_data_formal(outline_outlining) = gen_nreverse(outline_data_formal(outline_outlining));
  outline_data_arguments(outline_outlining) = gen_nreverse(outline_data_arguments(outline_outlining));

  /*
    call building
  */
  if (!ENDP(outline_data_block(outline_outlining)))
    label = statement_label(STATEMENT(CAR(outline_data_block(outline_outlining))));
  else
    label = entity_empty_label();

  /*
    ensure a label to designate the statement for reverse inlining transformation
  */
  if (entity_empty_label_p(label)) 
    label = make_new_label(get_current_module_entity());

  call = make_statement(label,STATEMENT_NUMBER_UNDEFINED,STATEMENT_ORDERING_UNDEFINED,
			strdup(concatenate("* Outlined code -> ",entity_user_name(outlined_module),"\n",NULL)),
			make_instruction_call(make_call(outlined_module,gen_full_copy_list(outline_data_arguments(outline_outlining)))),NIL,NULL,empty_extensions ());

  /*
    outline source generating
  */
  source_body = make_block_statement(gen_nconc(gen_full_copy_list(outline_data_block(outline_outlining)), CONS(STATEMENT, make_return_statement(outlined_module), NIL)));

  FOREACH(ENTITY, e, outline_local)
    {
      AddLocalEntityToDeclarations(e,outlined_module,source_body);
    }
 
  saved = get_bool_property("PRETTYPRINT_STATEMENT_NUMBER");
  set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", false);
   
  if (is_fortran)
    { 
      set_prettyprint_language_tag(is_language_fortran);
    }
  else 
    {
      set_prettyprint_language_tag(is_language_c);
    }
  add_new_module(entity_local_name(outlined_module), outlined_module, source_body, is_fortran);


  set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", saved);
  free_statement(source_body);

  if(new_user_file && !string_undefined_p(new_user_file) && is_fortran)
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, module_local_name(outlined_module), new_user_file);

  store_outline(outlined_module,outline_outlining);
  pips_debug(1,"New outlined module : %s\n", entity_user_name(outlined_module));
  
  if (!is_fortran)
    { 
      /* pour eviter une double declaration des entites lorsque le code produit est parse. */
      code_declarations(value_code(entity_initial(outlined_module)))=NIL;
    }

  outline_outlining = outline_data_undefined;
  outlined_module = entity_undefined;
  gen_free_list(outline_list);
  outline_list = list_undefined;
  gen_free_list(outline_local);
  outline_local = list_undefined;
  outline_step = 0;

  pips_debug(1,"call = %p\n", call);
  return call;
}

//####################### OUTLINE  #######################

static bool outline_statement(statement stmt,const char* name,int from, int to)
{
  pips_debug(1,"stmt = %p, name = %s\n", stmt, name);

  if (from!=statement_number(stmt) || to!=statement_number(stmt))
    return false;

  if(outlining_start(name))
    {
      statement call=statement_undefined;
      list block=CONS(STATEMENT,stmt,NIL);
      outlining_scan_block(block);
      gen_free_list(block);
      call=outlining_close(NULL);
      free_instruction(statement_instruction(stmt));
      statement_instruction(stmt)=statement_instruction(call);
      statement_comments(stmt)=statement_comments(call);
      statement_number(stmt)=statement_number(call);
      return true;
    }
  return false;
}

static bool outline_block(statement stmt, const char* name, int from, int to)
{
  list treated = NIL;
  list block_list=NIL;
  bool record_p = false;
  bool end_p = false;
  statement call = statement_undefined;

  pips_debug(1,"stmt = %p, name = %s, from = %d, to = %d\n", stmt, name, from, to);

  if (!instruction_sequence_p(statement_instruction(stmt)))
    return false;

  FOREACH(STATEMENT,s,sequence_statements(instruction_sequence(statement_instruction(stmt))))
    {
      if (!end_p && !record_p && from==statement_number(s))
	record_p=true;
      if (!end_p && record_p)
	block_list=CONS(STATEMENT,s,block_list);
      else
	treated=CONS(STATEMENT,s,treated);
      if (!end_p && to==statement_number(s))
	{
	  end_p=true;
	  // block outlining
	  if(outlining_start(name))
	    {
	      outlining_scan_block(block_list);
	      call=outlining_close(NULL);
	      treated=CONS(STATEMENT,call,treated);
	    }
	}
    }
  if (end_p)
      sequence_statements(instruction_sequence(statement_instruction(stmt)))=gen_nreverse(treated);
  else
    gen_free_list(treated);

  gen_free_list(block_list);

  pips_debug(1,"end_p = %d\n", end_p);
  return end_p;
}

static bool outline_block_filter(statement stmt)
{
  const char* name;

  pips_debug(1,"stmt = %p\n", stmt);

  name = get_string_property("OUTLINING_NAME");

  if (outline_from==0 || outline_to==0)
    return false;

  if (outline_block(stmt,name,outline_from,outline_to) ||
      outline_statement(stmt,name,outline_from,outline_to))
    {
      outline_from=0;
      outline_to=0;
      return false;
    }
  else
    return true;  
}


/* Phase non utilisee pour l'instant */
bool step_outlining(const char* module_name)
{
  statement body;

  body = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
  pips_assert("",outline_data_undefined_p(outline_outlining));

  outline_from = get_int_property("OUTLINING_FROM");
  outline_to = get_int_property("OUTLINING_TO");

  if (outline_from == 0 || outline_to == 0)
    {
      pips_debug(1,"Nothing to do.\n");
      return true;
    }
  debug_on("OUTLINING_DEBUG_LEVEL");  

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  outlining_load();

  gen_recurse(body, statement_domain, outline_block_filter, gen_null);

  if (outline_from!=0 || outline_to!=0)
    pips_debug(1,"Block [%d-%d] not found\n",outline_from,outline_to);

  module_reorder(body);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(body));
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, body);	

  outlining_save();
  reset_current_module_entity();
 
  debug_off();
  return true;
}

//####################### INLINE  #######################

static bool inline_outlined_statement(statement stmt)
{
  entity function;
  statement block;

  pips_debug(1,"stmt = %p\n", stmt);

  STEP_DEBUG_STATEMENT(3,"current",stmt);
  if (!statement_call_p(stmt) || strcmp(target_label,entity_name(statement_label(stmt)))!=0)
    return false;

  function = call_function(instruction_call(statement_instruction(stmt)));
  if(!bound_outline_p(function))
    return false;

  outline_outlining = load_outline(function);
  block = STATEMENT(CAR(outline_data_block(outline_outlining)));
  STEP_DEBUG_STATEMENT(2,"body",block);

  outline_outlining = outline_data_undefined;

  statement_instruction(stmt)=copy_instruction(statement_instruction(block));
  statement_comments(stmt)=statement_comments(block);
  statement_number(stmt)=statement_number(block);

  delete_outline(function);

  pips_debug(1,"Fin\n");
  return true;
}

static bool inline_outlined_block(statement stmt)
{
  list treated = NIL;
  bool find_p = false;

  pips_debug(1,"stmt = %p\n", stmt);

  STEP_DEBUG_STATEMENT(3,"current",stmt);
  
  if (!statement_block_p(stmt))
    return false;

  FOREACH(STATEMENT,s,sequence_statements(instruction_sequence(statement_instruction(stmt))))
    {
      if(statement_call_p(s) && strcmp(target_label,entity_name(statement_label(s)))==0)
	{
	  entity function = call_function(instruction_call(statement_instruction(s)));
	  if (bound_outline_p(function))
	    {
	      find_p = true;
	      outline_outlining = load_outline(function);
	      list block = outline_data_block(outline_outlining);
	      outline_outlining = outline_data_undefined;
	      treated = gen_nconc(gen_nreverse(gen_full_copy_list(block)),treated);
	      delete_outline(function);
	    }
	}
      else
	treated=CONS(STATEMENT,s,treated); 
    }

  if(find_p)
      sequence_statements(instruction_sequence(statement_instruction(stmt)))=gen_nreverse(treated);
  else
    gen_free_list(treated);

  pips_debug(1,"find_p = %d\n", find_p);
  return find_p;
}

static bool inline_not_outlined(statement stmt)
{
  STEP_DEBUG_STATEMENT(3,"current",stmt);
  if (strcmp(target_label,entity_name(statement_label(stmt)))!=0)
    return false;
  if (statement_call_p(stmt) &&
      intrinsic_entity_p(call_function(instruction_call(statement_instruction(stmt)))))
    {
      pips_debug(1,"Nothing to do at label %s.\n",entity_user_name(statement_label(stmt)));
      return true;
    }
  pips_debug(1,"Only inlining as revert outlining is implemented.\n");
  return true;
}

static bool inline_call_filter(statement stmt)
{
  if(get_int_property("INLINING_LABEL")==0)
    return false;

  if(inline_outlined_block(stmt) ||
     inline_outlined_statement(stmt) ||
     inline_not_outlined(stmt))
    {
      set_int_property("INLINING_LABEL",0);
      return false;
    }

  return true;  
}

/* Phase non utilisee pour l'instant */
bool step_inlining(const char* module_name)
{
  int label_id;
  statement body;

  label_id = get_int_property("INLINING_LABEL");
  body = (statement)db_get_memory_resource(DBR_CODE, module_name, true);

  if(label_id == 0)
    {
      pips_debug(1,"Nothing to do.\n");
      return true;
    }
  snprintf(target_label,TARGET_LABEL_NAME_LEN,"%s%s%s%d",
	   module_name, MODULE_SEP_STRING, LABEL_PREFIX, label_id);

  debug_on("INLINING_DEBUG_LEVEL");

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  outlining_load();

  gen_recurse(body, statement_domain, inline_call_filter, gen_null);

  if((label_id=get_int_property("INLINING_LABEL"))!=0)
      pips_debug(1,"Label %d not found.\n",label_id);
  
  module_reorder(body);
  if(ordering_to_statement_initialized_p())
    reset_ordering_to_statement();

  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(body));
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, body);	

  outlining_save();
  reset_current_module_entity();
  debug_off();
  return true;
}
