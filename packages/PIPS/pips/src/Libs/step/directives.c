/* Copyright 2007, 2008, 2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/


#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "defines-local.h"

GENERIC_GLOBAL_FUNCTION(global_directives,directives);


bool directives_init(string __attribute__ ((unused)) module_name)
{
  init_global_directives();
  global_directives_save();
  return TRUE;
}

void global_directives_load()
{
  set_global_directives((directives)db_get_memory_resource(DBR_DIRECTIVES, "", TRUE));
}

void global_directives_save()
{
  DB_PUT_MEMORY_RESOURCE(DBR_DIRECTIVES, "", get_global_directives());
  reset_global_directives();
}



/* directive stack used to outline directive body */
DEFINE_GLOBAL_STACK(current_directives, directive);

/*
  Phase PIPS de pres traitement :  les directives sont remplacees par des appels a la fonction DIR_CALL
  dont l'argument est le texte de la directive et le nouveau fichier a pour suffix STEP_FILTERED_SUFFIX.
*/
#define DIR_CALL "STEP_DIRECTIVES_"
#define STEP_FILTERED_SUFFIX ".step_filtered"
static int sequence_level = 0;

bool directive_filter(string name)
{
    string file_name, dir_name, new_name, src_name;

    debug_on("STEP_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", name);
    
    dir_name = db_get_current_workspace_directory();
    file_name = db_get_file_resource(DBR_SOURCE_FILE, name, TRUE);
    new_name = strdup(concatenate(name,"/",name, STEP_FILTERED_SUFFIX, NULL));
    src_name = strdup(concatenate(dir_name, "/", file_name, NULL));

    pips_debug(2,"cmd=%s",concatenate("step_directives"," < ", src_name, " > ", dir_name, "/", new_name,"\n", NULL));
    safe_system(concatenate("step_directives"," < ", src_name, " > ", dir_name, "/", new_name, NULL));

    DB_PUT_FILE_RESOURCE(DBR_DIRECTIVE_FILTERED_FILE, name, new_name);

    free(src_name);
    free(dir_name);

    pips_debug(1, "fin step_filter\n");
    debug_off();
    return TRUE;
}

/*######################################################################################################*/
string step_make_new_directive_module_name(string suffix, string id)
{
  int i;
  string new_name, final_name, i_s;
  entity current_module;

  pips_debug(1,"suffix = %s, id = %s\n", suffix, id);

  current_module = get_current_module_entity();
  pips_assert("current module defined", !entity_undefined_p(current_module));
  pips_assert("stack defined", !stack_undefined_p(current_directives_stack));

  new_name = string_undefined;
  final_name = string_undefined;

  if (current_directives_empty_p())
    {
      /* stack is empty */
      new_name = module_local_name(current_module);
    }
  else
    {
      string previous_module_name;
      
      previous_module_name = directive_module_name(current_directives_head());

      if(suffix)
	  new_name = strdup(concatenate(previous_module_name,"_",suffix, id, NULL));
      else
	  new_name = strdup(previous_module_name);
    }

  pips_debug(2,"temporary new_name = %s\n", new_name);

  if (suffix)      //  ensure name unicity
    {
      i = 1;
      if (string_undefined_p(id) || strlen(id) > 0)
	i_s = strdup("");
      else
	i_s=i2a(i++);
      
      final_name = strdup(concatenate(new_name, i_s, NULL));
      while (!entity_undefined_p(gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, final_name,NULL), entity_domain)) ||
	     !entity_undefined_p(find_label_entity(get_current_module_name(), i_s)))
	{
	  free(i_s);
	  i_s = i2a(i++);
	  free(final_name);
	  final_name = strdup(concatenate(new_name, i_s, NULL));
	} 

      free(i_s);
      free(new_name);
      make_empty_subroutine(final_name); // for matching the next try with the same "new_name" but for a new directive
      new_name = final_name;
    }

  pips_debug(2,"final new_name = %s\n", new_name);
  return new_name;
}

void step_print_directive(directive drt)
{
  pips_assert("directive defined!", !directive_undefined_p(drt));
  
  pips_debug(1, "txt = %s, module_name = %s\n", directive_txt(drt), directive_module_name(drt));

  if (directive_body(drt) == NIL)
    {
      pips_debug(1, "empty body\n");
    }
  else
    {
      statement stmt;
      stmt = make_block_statement(gen_full_copy_list(directive_body(drt)));
      pips_debug(1, "body:\n");
      print_statement(stmt);
      free_statement(stmt);
    }

  if (directive_clauses(drt) == NIL)
    {
      pips_debug(1, "no clause\n");
    }
  else
    {
      pips_debug(1, "clause(s) exist\n");
    }
}

static directive make_directive_none(statement stmt)
{
  string id;
  string new_name;
  list stmt_list;
  directive drt;
  pips_debug(1,"stmt = %p\n", stmt);

  if (statement_undefined_p(stmt))
    {
      id = NULL;
      stmt_list = NIL;
    }
  else
    {
      id = "";
      stmt_list = CONS(STATEMENT, copy_statement(stmt), NIL);
    }

  new_name = step_make_new_directive_module_name(NULL, id);
  drt = make_directive(strdup("none"), new_name, make_type_directive_none(), stmt_list, NIL);

  pips_debug(1,"drt = %p\n", drt);
  return drt;
}

static instruction handle_directive_none(directive begin, directive end)
{
  instruction instr;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  instr = instruction_undefined;

  pips_assert("directive none", type_directive_none_p(directive_type(end)));
  switch (gen_length(directive_body(end)))
    {
    case 0:
      instr = make_continue_instruction();
      break;
    case 1:
      instr = copy_instruction(statement_instruction(STATEMENT(CAR(directive_body(end)))));
      break;
    default:
      instr = make_instruction_block(gen_full_copy_list(directive_body(end)));
      break;
    }
  update_label_do_directive_module_name(begin, end);
  //  free_directive(end);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

static bool optional_end_directive_p(directive current)
{
  bool b;

  pips_debug(1,"current = %p\n", current);

  b = FALSE;

  if(type_directive_omp_do_p(directive_type(current)) &&
     !ENDP(directive_body(current)))
    {
      /* if body not empty then omp_do is considered as ended */
      b = TRUE;
    }

  if(type_directive_omp_parallel_do_p(directive_type(current)) &&
     !ENDP(directive_body(current)))
    {
      /* if body not empty then omp_parallel_do is considered as ended */
      b = TRUE;
    }

  pips_debug(1,"b = %d\n", b);
  return b;
}

static bool hook_block_FALSE(directive __attribute__ ((unused)) current,directive __attribute__ ((unused)) next)
{
  return FALSE;
}

/*######################################################################################################*/
typedef directive (*make_directive_hook) (statement);
typedef bool (*block_directive_hook) (directive,directive);
typedef instruction (*handle_directive_hook) (directive,directive); 

typedef struct {
  enum type_directive_utype directive_type_tag;
  string directive_txt;
  make_directive_hook make; /* generation de la representation interne de la directive (type newgen) */
  block_directive_hook begin;
  block_directive_hook end;
  handle_directive_hook handle; /* traitement de la directive: outlining */ 
} directive_hooks;

static directive_hooks directives_hooks[]= {
  {is_type_directive_omp_parallel_sections,
   PARALLEL_SECTIONS_TEXT,
   make_directive_omp_parallel_sections,
   is_begin_directive_omp_parallel_sections,
   hook_block_FALSE,
   handle_omp_section
  },
  {is_type_directive_omp_end_parallel_sections,
   END_PARALLEL_SECTIONS_TEXT,
   make_directive_omp_end_parallel_sections,
   hook_block_FALSE,
   is_end_directive_omp_end_parallel_sections,
   handle_omp_section
  },
  {is_type_directive_omp_sections,
   SECTIONS_TEXT,
   make_directive_omp_sections,
   is_begin_directive_omp_sections,
   hook_block_FALSE,
   handle_omp_section
  },
  {is_type_directive_omp_section,
   SECTION_TEXT,
   make_directive_omp_section,
   hook_block_FALSE,
   is_end_directive_omp_section,
   handle_omp_section
  },
  {is_type_directive_omp_end_sections,
   END_SECTIONS_TEXT,
   make_directive_omp_end_sections,
   hook_block_FALSE,
   is_end_directive_omp_end_sections,
   handle_omp_section
  },
  {is_type_directive_omp_parallel_do,
   PARALLEL_DO_TEXT,
   make_directive_omp_parallel_do,
   is_begin_directive_omp_parallel_do,
   hook_block_FALSE,
   handle_omp_parallel_do
  },
  {is_type_directive_omp_end_parallel_do,
   END_PARALLEL_DO_TEXT,
   make_directive_omp_end_parallel_do,
   hook_block_FALSE,
   is_end_directive_omp_end_parallel_do,
   handle_omp_parallel_do
  },
  {is_type_directive_omp_parallel,
   PARALLEL_TEXT,
   make_directive_omp_parallel,
   is_begin_directive_omp_parallel,
   hook_block_FALSE,
   handle_omp_parallel
  },
  {is_type_directive_omp_end_parallel,
   END_PARALLEL_TEXT,
   make_directive_omp_end_parallel,
   hook_block_FALSE,
   is_end_directive_omp_end_parallel,
   handle_omp_parallel
  },
  {is_type_directive_omp_do,
   DO_TEXT,
   make_directive_omp_do,
   is_begin_directive_omp_do,
   hook_block_FALSE,
   handle_omp_do
  },
  {is_type_directive_omp_end_do,
   END_DO_TEXT,
   make_directive_omp_end_do,
   hook_block_FALSE,
   is_end_directive_omp_end_do,
   handle_omp_do
   },
  {is_type_directive_omp_master,
   MASTER_TEXT,
   make_directive_omp_master,
   begin_directive_omp_master,
   hook_block_FALSE,
   handle_omp_master
  },
  {is_type_directive_omp_end_master,
   END_MASTER_TEXT,
   make_directive_omp_end_master,
   hook_block_FALSE,
   end_directive_omp_end_master,
   handle_omp_master
  },
  {is_type_directive_omp_barrier,
   BARRIER_TEXT,
   make_directive_omp_barrier,
   begin_directive_omp_barrier,
   end_directive_omp_barrier,
   handle_omp_barrier
  },
  /*must be the last*/
  {is_type_directive_none,/* fake directive for sequence instruction handling*/
   string_undefined,
   (make_directive_hook)make_directive_none,
   (block_directive_hook)hook_block_FALSE,
   (block_directive_hook)hook_block_FALSE,
   (handle_directive_hook)handle_directive_none
  }
};

bool step_directive_p(string directive, string entity_name) 
{
  int retcode;
  char *debut;

  pips_debug(3, "%zd directive = %s, entity_name = %s\n",strlen(directive), directive, entity_name);

  if (string_undefined_p(directive))
    return FALSE;

  debut=strcasestr(entity_name,directive);
  retcode=((debut-(char*)entity_name)==1);

  pips_debug(3, "retcode = %d\n", retcode);
  return retcode;
}

static directive_hooks hooks_by_txt(string txt)
{
  int id_hook=0;
  while (id_hook < is_type_directive_none &&
	 !step_directive_p(directives_hooks[id_hook].directive_txt, txt))
    id_hook++;
  return directives_hooks[id_hook];
}

static directive_hooks hooks_by_directive(directive d)
{
  int id_hook=0;
  while (id_hook < is_type_directive_none &&
	 type_directive_tag(directive_type(d)) != directives_hooks[id_hook].directive_type_tag)
    id_hook++;
  return directives_hooks[id_hook];
}

/*######################################################################################################*/
static bool directive_statement_p(statement stmt)
{
  entity f;
  if (statement_undefined_p(stmt) || !statement_call_p(stmt))
    return FALSE;
  
  f = call_function(statement_call(stmt));
  return (top_level_entity_p(f) && strncmp(DIR_CALL, entity_local_name(f), strlen(DIR_CALL))==0);
}

string statement_to_directive_txt(statement stmt)
{
  pips_assert("call",!statement_undefined_p(stmt) && statement_call_p(stmt));
  entity f = call_function(statement_call(stmt));
  pips_assert("directive",(top_level_entity_p(f) && strncmp(DIR_CALL, entity_local_name(f), strlen(DIR_CALL))==0));
  return entity_local_name(expr_to_entity(EXPRESSION(CAR(call_arguments(statement_call(stmt))))));
}

// un parser serait pas mal pour reconnaitre la type de directive avec ses clauses :)
static directive make_directive_from_statement(statement stmt)
{
  pips_debug(1, "stmt = %p\n", stmt);

  if (!directive_statement_p(stmt))
    return make_directive_none(stmt);
  else
    /* Where the directive is created */
    return ((hooks_by_txt(statement_to_directive_txt(stmt))).make)(stmt);
}

static bool directive_none_p(directive d)
{
  return type_directive_none_p(directive_type(d));
}

static bool begin_directive_p(directive d)
{
  return ((hooks_by_directive(d)).begin)(current_directives_head(),d);
}

static bool end_directive_p(directive d)
{ 
  return ((hooks_by_directive(d)).end)(current_directives_head(),d);
}

static instruction handle_directive(directive begin,directive end)
{
  if (directive_none_p(begin)&&directive_none_p(end))
    {
       /*
	 Case of the fake directive
       */ 
      return ((hooks_by_directive(end)).handle)(current_directives_head(),end);
    }
  else
    {
      /*
	Case of a directive
      */
      return ((hooks_by_directive(begin)).handle)(begin,end);
    }
}

/*######################################################################################################*/
static bool dir_filter(statement);

static void add_instruction_to_list(instruction i, list l)
{
  pips_debug(1, "i = %p, l = %p\n", i, l);

  pips_assert("list",!ENDP(l));
  
  /* add at the beginning of the list */
  if(instruction_sequence_p(i))
    {
      CDR(l)=gen_nconc(gen_full_copy_list(instruction_block(i)),CDR(l));
      free_instruction(i);
    }
  else
    CDR(l)=CONS(STATEMENT,make_statement(entity_empty_label(), 
					 STATEMENT_NUMBER_UNDEFINED,
					 STATEMENT_ORDERING_UNDEFINED, 
					 empty_comments,
					 i,
					 NIL,NULL,empty_extensions ()),CDR(l));

  pips_debug(1, "end\n");
  return;
}

statement step_keep_directive_txt(directive begin,statement stmt,directive end)
{
  
  if(!(directive_undefined_p(begin) || directive_none_p(begin)))
    insert_statement(stmt,make_statement(entity_empty_label(), 
					 STATEMENT_NUMBER_UNDEFINED,
					 STATEMENT_ORDERING_UNDEFINED, 
					 strdup(concatenate("C     !$omp ",step_remove_quote(directive_txt(begin)),"\n",NULL)),
					 make_continue_instruction(),
					 NIL,NULL,empty_extensions ()),
		     TRUE);
  if(!(directive_undefined_p(end) || directive_none_p(end)))
    insert_statement(stmt,make_statement(entity_empty_label(), 
					 STATEMENT_NUMBER_UNDEFINED,
					 STATEMENT_ORDERING_UNDEFINED, 
					 strdup(concatenate("C     !$omp ",step_remove_quote(directive_txt(end)),"\n",NULL)),
					 make_continue_instruction(),
					 NIL,NULL,empty_extensions ()),
		     FALSE);
  return stmt;
}

static void directive_statement_filter(statement stmt)
{
  directive new_directive;
  pips_debug(1, "stmt = %p\n", stmt);

  pips_assert("non block",!statement_block_p(stmt) && directive_statement_p(stmt));
  new_directive = make_directive_from_statement(stmt);
  pips_debug(2,"directive %s\n",directive_txt(new_directive));

  /*
  a directive that is both begin and end is a single statement directive
  for instance: barrier
  */
  if(!directive_none_p(new_directive) && 
     begin_directive_p(new_directive) && end_directive_p(new_directive))
	{
	  free_instruction(statement_instruction(stmt));
	  statement_instruction(stmt)=handle_directive(new_directive,new_directive);
	}
  /*
    a block begin/end  or an unknown directive is found in a non-block statement
  */
  else 
    {
      if (directive_none_p(new_directive) &&  // an unknown directive
	  directive_statement_p(STATEMENT(CAR(directive_body(new_directive)))))
	pips_error("directive_statement_filter","directive unknown.\n");
      else
	pips_error("directive_statement_filter","begin/end directive out of block.\n");
      free_directive(new_directive); 
    }

  STEP_DEBUG_STATEMENT(1,"end filter statement ",stmt);
}

static void _directives_outline(list remaining, statement stmt)
{
  pips_debug(1,"remaining = %p, stmt = %p\n", remaining, stmt);

  STEP_DEBUG_DIRECTIVE(2, "current_directives head", current_directives_head());

  if (optional_end_directive_p(current_directives_head())) 
    {
      /*
	Case where the end directive is optional (and present or not)
	Where the current directive is handled (outlined)
      */
      
      directive begin;

      pips_debug(2, "optional end directive\n");

      begin = current_directives_pop();
      STEP_DEBUG_DIRECTIVE(2,"pop current_directives", begin);

      /* handle (outline) the current directive */
      add_instruction_to_list(handle_directive(begin, make_directive_none(stmt)), remaining);
    }
  else
    {
      directive new_directive;
      /* 
	 Converts any instruction in a directive.
	 For a statement which is not an OpenMP directive,
	 it creates a directive of type directive_none
      */
      
      new_directive = make_directive_from_statement(stmt);
      pips_debug(2,"new_directive = %s\n", directive_txt(new_directive));
      
      if(begin_directive_p(new_directive)) 
	{
	  /* add a new directive into the stack */
	  pips_debug(2, "begin directive\n");
	  STEP_DEBUG_DIRECTIVE(2,"push current_directives", new_directive);
	  current_directives_push(new_directive);
	}
      
      /* a master directive is both begin and end */
      if(end_directive_p(new_directive))
	{
	  /*
	    Case where the end directive is present
	    Where the current directive is handled (outlined)
	  */
	  directive begin;
	  
	  pips_debug(2, "end directive\n");
	  begin = current_directives_pop();
	  STEP_DEBUG_DIRECTIVE(2,"pop current_directives", begin);

	  /* handle (outline) the current directive */
	  add_instruction_to_list(handle_directive(begin, new_directive), remaining);
	}
      else if(directive_none_p(new_directive))
	{
	  /*
	    Case where the statement is not a directive
	    (directive_none)
	    
	    Add the none directive to the current directive body
	  */
	  
	  directive current;
	  pips_debug(2, "none directive\n");

	  current = current_directives_head();
	  STEP_DEBUG_DIRECTIVE(2,"head current_directives", current);

	  /* free instr corresponding to new_directive */
	  free_instruction(statement_instruction(stmt));
	  statement_instruction(stmt) = handle_directive(new_directive, new_directive);
	  pips_debug(2, "Add the none directive to the body of the head of current_directives\n");
	  directive_body(current) = gen_nconc(directive_body(current), CONS(STATEMENT, copy_statement(stmt), NIL));

	}
    }

  
  pips_debug(1,"end\n");
}

static void directives_outline(list remaining)
{
  statement stmt;
  
  pips_debug(1, "remaining = %p\n", remaining);

  stmt = STATEMENT(CAR(remaining));
  
  STEP_DEBUG_STATEMENT(2, "stmt", stmt);

  if(!directive_statement_p(stmt))
    {
      pips_debug(2, "not a directive statement\n");

      gen_recurse(stmt, statement_domain, dir_filter, gen_null); 
    }
  else 
    {
      pips_debug(2, "directive statement\n");
    }

  pips_debug(2, "goes on...\n");

  if (statement_block_p(stmt)) 
    {
      /* sequence flattening */
      pips_debug(2, "block statement: sequence flattening\n");

      CDR(remaining) = gen_nconc(gen_full_copy_list(instruction_block(statement_instruction(stmt))), CDR(remaining));
    }
  else 
    {
      /* look for the directive */
      pips_debug(2, "not a block statement\n");
      
      _directives_outline(remaining, stmt);
    }

  pips_debug(2, "sequence_level = %d, nb remaining = %zd\n", sequence_level, gen_length(remaining) - 1);
  pips_debug(2, "Size of the current_directives stack = %d\n", current_directives_size());
  STEP_DEBUG_DIRECTIVE(2, "current_directives head", current_directives_head());
  pips_debug(1, "end\n");
}

static void directive_block_filter(statement stmt)
{
  list sequence;
  directive new_directive;

  pips_debug(1, "stmt = %p\n", stmt);
  pips_assert("begin filter block", statement_block_p(stmt));

  STEP_DEBUG_STATEMENT(2, "stmt", stmt);

  sequence = gen_full_copy_list(sequence_statements(instruction_sequence(statement_instruction(stmt))));

  /* 
     make a fake directive to record sequence statement in the directive body
  */

  pips_debug(2, "push a fake none directive to record sequence statement in the directive body\n");
  current_directives_push(make_directive_none(statement_undefined));

  /*
    for each statement of the sequence (not yet traited)
  */
  sequence_level ++;
  pips_debug(2, "directives_outline on the block level %d\n", sequence_level);
  MAPL(remaining,
       {
	 pips_debug(2, "next stmt of sequence level %d\n", sequence_level);
	 directives_outline(remaining);
       }, sequence);
  gen_full_free_list(sequence);
  sequence_level --;

  /*
    handle the fake directive
  */
  pips_debug(2, "pop the fake none directive\n");
  new_directive = current_directives_pop(); 
  if (!type_directive_none_p(directive_type(new_directive)))
    {
      free_directive(new_directive);
      pips_error("directive_block_filter", "directive not well formed.\n");
    }
  else // convert the fake directive into instruction
    {
      free_instruction(statement_instruction(stmt));
      statement_instruction(stmt) = handle_directive_none(new_directive, new_directive);
    }

  STEP_DEBUG_STATEMENT(2,"end filter block stmt", stmt);
  pips_debug(1, "end\n");
}

static bool dir_filter(statement stmt)
{
  pips_debug(1, "stmt = %p\n", stmt);

  pips_assert("statement_defined",!statement_undefined_p(stmt));
  STEP_DEBUG_STATEMENT(2, "start filter", stmt);
  
  /*
    directive_statement corresponds to a one-line directive: barrier
    and flush or directive_block after treatment

  */

  if (!statement_block_p(stmt))
    {
      pips_debug(2, "not a block statement\n");
      if(directive_statement_p(stmt))
	{
	  pips_debug(2, "directive_statement (CALL_DIR)\n");
	  
	  directive_statement_filter(stmt);
	}
      else
	{
	  pips_debug(2, "not a directive statement\n");
	}
    }
  /*
    directive_block corresponds to directives with associated blocks
    of statements as do, master...
    
    a directive may produce a block statement...
  */
  if (statement_block_p(stmt)) 
    {
      pips_debug(2, "block statement\n");

      directive_block_filter(stmt); 
      return FALSE;
    }

  STEP_DEBUG_STATEMENT(2, "end filter", stmt);

  pips_debug(1, "end\n");
  return TRUE;
}

bool step_directives(string module_name)
{
  debug_on("STEP_DEBUG_LEVEL");
  pips_debug(1, "%d module_name = %s\n", __LINE__, module_name);
   
  debug_on("STEP_DIRECTIVES_DEBUG_LEVEL");

  statement stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  global_directives_load();
  outlining_load();

  pips_debug(2, "Initialization of the current_directives stack\n");
  make_current_directives_stack();

  /* where directives are outlined */
  gen_recurse(stmt, statement_domain, dir_filter, gen_null); 

  pips_debug(2, "Finalization of the current_directives stack\n");
  free_current_directives_stack();
  outlining_save();
  global_directives_save();
  reset_current_module_entity();


  module_reorder(stmt);
  if(ordering_to_statement_initialized_p())
    reset_ordering_to_statement();

  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(stmt));
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stmt);

  pips_debug(1, "Fin step_directives\n");
  debug_off(); 
  debug_off();
  
  return TRUE;
}

static string clauses_to_string(list clauses,bool close)
{
  string s;
  string_buffer sb = string_buffer_make(FALSE);

  FOREACH(CLAUSE,c,clauses)
    {
      s = string_undefined;
      switch (clause_tag(c))
	{
	case is_clause_step_reduction:
	  if (close==false)
	    s=clause_reduction_to_string(clause_step_reduction(c));
	  break;
	case is_clause_step_private:
	  if (close==false)
	    s=clause_private_to_string(clause_step_private(c));
	  break;
	default:
	  pips_user_warning("unexpected clause\n");
	}
      if(s != string_undefined)
	string_buffer_append(sb, s);
    }
  s = string_buffer_to_string(sb);
  string_buffer_free_all(&sb);
  return s;
}


static string directive_formate(string unformated,string directive_guard,string directive_continue)
{
  char tmp[MAX_LINE_LENGTH+1];
  string_buffer formated_bs = string_buffer_make(FALSE);
  string formated;
  size_t size;

  tmp[0]='\0';
  strcat(tmp,directive_guard);
  size=MAX_LINE_LENGTH-strlen(directive_guard);
  string_buffer_append(formated_bs, strdup(strncat(tmp,unformated,size)));

  while (strlen(unformated)>size)
    {
      unformated+=size;

      tmp[0]='\0';
      strcat(tmp,"\n");
      strcat(tmp,directive_continue);
      size=MAX_LINE_LENGTH-strlen(directive_continue);      
      string_buffer_append(formated_bs,strdup(strncat((char*)tmp,unformated,size)));
    }

  formated = string_buffer_to_string(formated_bs);
  string_buffer_free_all(&formated_bs);

  return formated;
}


string directive_to_string(directive d,bool close)
{
  pips_debug(1, "d=%p, close=%u\n",d,close);
  string d_txt=string_undefined;
  string c_txt=string_undefined;
  string directive_formated;
  switch (type_directive_tag(directive_type(d)))
    {
    case is_type_directive_omp_parallel:
      d_txt = directive_omp_parallel_to_string(d,close);
      break;
    case is_type_directive_omp_do:
      d_txt = directive_omp_do_to_string(d,close);
      break;
    case is_type_directive_omp_parallel_do:
      d_txt = directive_omp_parallel_do_to_string(d,close);
      break;
    case is_type_directive_omp_master:
      d_txt = directive_omp_master_to_string(d,close);
      break;
    case is_type_directive_omp_barrier:
      d_txt = directive_omp_barrier_to_string(d,close);
      break;
    default:
      d_txt = string_undefined;
    }

  if (d_txt ==string_undefined )
    {
      pips_user_warning("directive handling not yet implemented\n");
      if(close==false)
	d_txt=strdup(concatenate(step_remove_quote(directive_txt(d)),NULL));
    }
  else
    {
      c_txt=clauses_to_string(directive_clauses(d),close);
      d_txt=strdup(concatenate(d_txt,c_txt,NULL));
    }

  if (!string_undefined_p(d_txt))
    {
      directive_formated=directive_formate(d_txt, OMP_DIRECTIVE, OMP_DIR_CONT);
      if (!string_undefined_p(c_txt)) free(c_txt);
      free(d_txt);
    }
  else
    directive_formated=strdup("");
    
  return directive_formated;

}

