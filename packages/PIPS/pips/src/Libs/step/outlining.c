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
static char target_label[TARGET_LABEL_NAME_LEN];

//######################## OUTLINE INIT ########################

bool outlining_init(string __attribute__ ((unused)) module_name)
{
  init_outline();
  outlining_save();
  return TRUE;
}

void outlining_load()
{
  set_outline((outline_map)db_get_memory_resource(DBR_OUTLINED, "", TRUE));
}

void outlining_save()
{
  DB_PUT_MEMORY_RESOURCE(DBR_OUTLINED, "", get_outline());
  reset_outline();
}

//####################### OUTLINE STEP 1 #######################

/* block isn't duplicated in the outline_data genereted by outlining_close */
entity outlining_start(string new_module_name)
{
  int i=0;

  pips_debug(1, "new_module_name = %s\n", new_module_name);

  pips_assert("step", outline_step==0);
  pips_assert("list", list_undefined_p(outline_list));
  pips_assert("outlined_module", entity_undefined_p(outlined_module));
  pips_assert("outline_outlining", outline_data_undefined_p(outline_outlining));

  // ensure "NEW_MODULE_NAME" from "nEW_MOdULe_name"
  for( i = 0; new_module_name[ i ]; i++)
    new_module_name[ i ] = toupper( new_module_name[ i ] );

  outlined_module = make_empty_subroutine(new_module_name);

  if (entity_undefined_p(outlined_module)) return FALSE;

  outline_list = NIL;
  outline_outlining = make_outline_data(get_current_module_entity(),NIL,NIL,NIL);
  outline_step = 1;

  pips_debug(1, "outlined_module = %p\n", outlined_module);
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

//####################### OUTLINE STEP 2 #######################
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
  pips_debug(1, "e = %p\n", e);

  if (gen_in_list_p(e,outline_list))
    {
      pips_debug(3,"entity %s already treated\n", entity_global_name(e));
      return TRUE;
    }
  if (!gen_in_list_p(e,code_declarations(value_code(entity_initial(get_current_module_entity())))))
    {
      pips_debug(4,"entity %s not in %s declaration\n", entity_global_name(e), entity_global_name(get_current_module_entity()));
      return TRUE;
    }
  if(entity_variable_p(e)) // scan entity in type
      gen_multi_recurse(entity_type(e),
			reference_domain, outlining_reference_filter, gen_null,
			call_domain,outlining_call_filter,gen_null,
			NULL);

  pips_debug(3,"new scanned entity %s\n", entity_global_name(e));
  outline_list=CONS(ENTITY,e,outline_list);      

  pips_debug(1, "end\n");
  return TRUE;
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

static void outlining_make_argument(entity e, expression expr)
{
  pips_debug(1, "e = %p, expr = %p\n", e, expr);

  pips_assert("outlined_module",!entity_undefined_p(outlined_module));
  pips_assert("outline_list",!list_undefined_p(outline_list));

  bool outline_symbolic_p=get_bool_property("OUTLINING_SYMBOLIC");
  functional ft = type_functional(entity_type(outlined_module));
  entity new_entity=FindOrCreateEntity(entity_local_name(outlined_module),entity_local_name(e));
  entity_initial(new_entity) = copy_value(entity_initial(e));
  if (outline_symbolic_p && value_symbolic_p(entity_initial(e)))
    entity_type(new_entity) = copy_type(functional_result(type_functional(entity_type(e))));
  else
    entity_type(new_entity) = copy_type(entity_type(e));
  
  if (entity_variable_p(e)||(outline_symbolic_p && value_symbolic_p(entity_initial(e))))
    {
      entity_storage(new_entity)=make_storage(is_storage_formal, make_formal(outlined_module,gen_length(outline_data_formal(outline_outlining))+1));
      functional_parameters(ft) =
	CONS(PARAMETER,make_parameter(entity_type(new_entity),MakeModeReference(),make_dummy_unknown()),
		 functional_parameters(ft));
      outline_data_formal(outline_outlining)=CONS(ENTITY,new_entity,outline_data_formal(outline_outlining));
      outline_data_arguments(outline_outlining)=CONS(EXPRESSION,expr,outline_data_arguments(outline_outlining));
      pips_debug(2,"add argument %s\n", entity_global_name(e));
    }
      else
	entity_storage(new_entity) = copy_storage(entity_storage(e));
  
  pips_debug(2,"add entity %s\n", entity_global_name(new_entity));
  code_declarations(value_code(entity_initial(outlined_module)))=CONS(ENTITY,new_entity,code_declarations(value_code(entity_initial(outlined_module))));

  pips_debug(1, "fin\n");
}

void outlining_add_argument(entity e,expression expr)
{
  pips_debug(1, "e = %p, expr = %p\n", e, expr);

  pips_assert("step", outline_step==2);
  gen_remove(&outline_list,e);
  outlining_make_argument(e,expr);

  pips_debug(1, "fin\n");
}

//####################### OUTLINE STEP 3 #######################

statement outlining_close(void)
{
  entity label = entity_undefined;
  statement call = statement_undefined;
  statement source_body = statement_undefined;

  pips_debug(1, "begin\n");

  pips_assert("step", outline_step==2);

  // argument building
  outline_list = gen_nreverse(outline_list);
  MAP(ENTITY, e,{
      outlining_make_argument(e,entity_to_expr(e));
    }, outline_list);

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
    label = make_new_label(entity_user_name(get_current_module_entity()));

  call = make_statement(label,STATEMENT_NUMBER_UNDEFINED,STATEMENT_ORDERING_UNDEFINED,
			strdup(concatenate("* Outlined code -> ",entity_user_name(outlined_module),"\n",NULL)),
			make_instruction_call(make_call(outlined_module,gen_full_copy_list(outline_data_arguments(outline_outlining)))),NIL,NULL,empty_extensions ());

  /*
    outline source generating
  */
  source_body = make_block_statement(gen_nconc(gen_full_copy_list(outline_data_block(outline_outlining)), CONS(STATEMENT, make_return_statement(outlined_module), NIL)));
  add_new_module(entity_local_name(outlined_module), outlined_module, source_body, TRUE);
  free_statement(source_body);

  store_outline(outlined_module,outline_outlining);
  pips_debug(1,"New outlined module : %s\n", entity_user_name(outlined_module));
  
  outline_outlining = outline_data_undefined;
  outlined_module = entity_undefined;
  gen_free_list(outline_list);
  outline_list = list_undefined;
  outline_step = 0;

  pips_debug(1,"call = %p\n", call);
  return call;
}

//####################### OUTLINE  #######################

static bool outline_statement(statement stmt,string name,int from, int to)
{
  pips_debug(1,"stmt = %p, name = %s\n", stmt, name);

  if (from!=statement_number(stmt) || to!=statement_number(stmt))
    return FALSE;

  if(outlining_start(name))
    {
      statement call=statement_undefined;
      list block=CONS(STATEMENT,stmt,NIL);
      outlining_scan_block(block);
      gen_free_list(block);
      call=outlining_close();
      free_instruction(statement_instruction(stmt));
      statement_instruction(stmt)=statement_instruction(call);
      statement_comments(stmt)=statement_comments(call);
      statement_number(stmt)=statement_number(call);
      return TRUE;
    }
  return FALSE;
}

static bool outline_block(statement stmt, string name, int from, int to)
{
  list treated = NIL;
  list block_list=NIL;
  bool record_p = FALSE;
  bool end_p = FALSE;
  statement call = statement_undefined;

  pips_debug(1,"stmt = %p, name = %s, from = %d, to = %d\n", stmt, name, from, to);

  if (!instruction_sequence_p(statement_instruction(stmt)))
    return FALSE;

  MAP(STATEMENT,s,
      {
	if (!end_p && !record_p && from==statement_number(s))
	  record_p=TRUE;
	if (!end_p && record_p)
	  block_list=CONS(STATEMENT,s,block_list);
	else
	  treated=CONS(STATEMENT,s,treated);
	if (!end_p && to==statement_number(s))
	  {
	    end_p=TRUE;
	    // block outlining
	    if(outlining_start(name))
	      {
		outlining_scan_block(block_list);
		call=outlining_close();
		treated=CONS(STATEMENT,call,treated);
	      }
	  }
      },sequence_statements(instruction_sequence(statement_instruction(stmt))));
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
  string name;

  pips_debug(1,"stmt = %p\n", stmt);

  name = get_string_property("OUTLINING_NAME");

  if (outline_from==0 || outline_to==0)
    return FALSE;

  if (outline_block(stmt,name,outline_from,outline_to) ||
      outline_statement(stmt,name,outline_from,outline_to))
    {
      outline_from=0;
      outline_to=0;
      return FALSE;
    }
  else
    return TRUE;  
}


/* Phase non utilisee pour l'instant */
bool step_outlining(string module_name)
{
  statement body;

  body = (statement)db_get_memory_resource(DBR_CODE, module_name, TRUE);
  pips_assert("",outline_data_undefined_p(outline_outlining));

  outline_from = get_int_property("OUTLINING_FROM");
  outline_to = get_int_property("OUTLINING_TO");

  if (outline_from == 0 || outline_to == 0)
    {
      pips_debug(1,"Nothing to do.\n");
      return TRUE;
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
  return TRUE;
}

//####################### INLINE  #######################

static bool inline_outlined_statement(statement stmt)
{
  entity function;
  statement block;

  pips_debug(1,"stmt = %p\n", stmt);

  STEP_DEBUG_STATEMENT(3,"current",stmt);
  if (!statement_call_p(stmt) || strcmp(target_label,entity_name(statement_label(stmt)))!=0)
    return FALSE;

  function = call_function(instruction_call(statement_instruction(stmt)));
  if(!bound_outline_p(function))
    return FALSE;

  outline_outlining = load_outline(function);
  block = STATEMENT(CAR(outline_data_block(outline_outlining)));
  STEP_DEBUG_STATEMENT(2,"body",block);

  outline_outlining = outline_data_undefined;

  statement_instruction(stmt)=copy_instruction(statement_instruction(block));
  statement_comments(stmt)=statement_comments(block);
  statement_number(stmt)=statement_number(block);

  delete_outline(function);

  pips_debug(1,"Fin\n");
  return TRUE;
}

static bool inline_outlined_block(statement stmt)
{
  list treated = NIL;
  bool find_p = FALSE;

  pips_debug(1,"stmt = %p\n", stmt);

  STEP_DEBUG_STATEMENT(3,"current",stmt);
  
  if (!block_statement_p(stmt))
    return FALSE;
  
  MAP(STATEMENT,s,{
      if(statement_call_p(s) && strcmp(target_label,entity_name(statement_label(s)))==0)
	{
	  entity function = call_function(instruction_call(statement_instruction(s)));
	  if (bound_outline_p(function))
	    {
	      find_p = TRUE;
	      outline_outlining = load_outline(function);
	      list block = outline_data_block(outline_outlining);
	      outline_outlining = outline_data_undefined;
	      treated = gen_append(gen_nreverse(gen_full_copy_list(block)),treated);
	      delete_outline(function);
	    }
	}
      else
	treated=CONS(STATEMENT,s,treated); 
    },sequence_statements(instruction_sequence(statement_instruction(stmt))));

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
    return FALSE;
  if (statement_call_p(stmt) &&
      intrinsic_entity_p(call_function(instruction_call(statement_instruction(stmt)))))
    {
      pips_debug(1,"Nothing to do at label %s.\n",entity_user_name(statement_label(stmt)));
      return TRUE;
    }
  pips_debug(1,"Only inlining as revert outlining is implemented.\n");
  return TRUE;
}

static bool inline_call_filter(statement stmt)
{
  if(get_int_property("INLINING_LABEL")==0)
    return FALSE;

  if(inline_outlined_block(stmt) ||
     inline_outlined_statement(stmt) ||
     inline_not_outlined(stmt))
    {
      set_int_property("INLINING_LABEL",0);
      return FALSE;
    }

  return TRUE;  
}

/* Phase non utilisee pour l'instant */
bool step_inlining(string module_name)
{
  int label_id;
  statement body;

  label_id = get_int_property("INLINING_LABEL");
  body = (statement)db_get_memory_resource(DBR_CODE, module_name, TRUE);

  if(label_id == 0)
    {
      pips_debug(1,"Nothing to do.\n");
      return TRUE;
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
  return TRUE;
}
