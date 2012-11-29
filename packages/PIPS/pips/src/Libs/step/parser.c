#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "defines-local.h"
#include <ctype.h> // for toupper
#include "control.h" // for module_reorder
#include "pipsmake.h"  // for compilation_unit_of_module
extern void step_bison_parse(pragma pgm, statement stmt);
extern void step_comment2pragma_handle(statement stmt);

static statement current_statement = statement_undefined;
static pragma current_pragma = pragma_undefined;
static int current_transform = -1;

static void print_step_blocks();
static step_directive new_step_directive(statement directive_stmt, int type,  string s);
static void reset_step_transform();
static step_directive get_current_step_directive(bool open);


/*
  Stack of blocks/sequences used during parsing to construct sequences
  of statements associated to a directive.
  Fundamental for Fortran programs.
*/

DEFINE_LOCAL_STACK(step_blocks, sequence);

/* 
 *
 *
 *
 *
 * Functions called from the BISON parser 
 *
 *
 *
 *
 */

void set_current_transform(int transform)
{
  current_transform = transform;
}

void insert_optional_pragma(int type)
{
  statement new_stmt = make_plain_continue_statement();

  switch (type)
    {
    case STEP_DO:
      add_pragma_str_to_statement(new_stmt, "ompenddo\n", true);
      break;
    case STEP_PARALLEL_DO:
      add_pragma_str_to_statement(new_stmt, "ompendparalleldo\n", true);
      break;
    default:
      pips_user_error("Unexpected pragma type %d\n", type);
    }
  statement parent_stmt = (statement)gen_get_ancestor(statement_domain, current_statement);
  pips_assert("parent", parent_stmt != NULL);
  pips_assert("block", statement_block_p(parent_stmt));
  gen_insert_after(new_stmt, current_statement, statement_block(parent_stmt));
}

entity entity_from_user_name(string name)
{
  entity e = entity_undefined;
  
  pips_debug(1, "begin\n");

  if(fortran_module_p(get_current_module_entity()))
    {
      /* passage de name a NAME */
      size_t i;
      for (i=0; i<strlen(name); i++)
	name[i]=toupper(name[i]);

      string full_name = strdup(concatenate(entity_user_name(get_current_module_entity()), MODULE_SEP_STRING, name, NULL));
      e = gen_find_tabulated(full_name, entity_domain);
      if (entity_undefined_p(e))
	pips_user_error("\nFortran entity \"%s\" not found\n", full_name);
      free(full_name);
    }
  else if (c_module_p(get_current_module_entity()))
    {
      /* determiner le nom avec le bon scope... */
      statement stmt_declaration = current_statement;
      pips_debug(4,"##### ENTITY DECL #####\n");
      while (entity_undefined_p(e) && stmt_declaration)
	{
	  FOREACH(entity, ee, statement_declarations(stmt_declaration))
	    {
	      pips_debug(4, "entity decl : \"%s\"\n", entity_name(ee));
	      if (strcmp(name, entity_user_name(ee)) == 0)
		{
		  e = ee;
		  break;
		}
	    }
	  stmt_declaration = (statement)gen_get_ancestor(statement_domain, stmt_declaration);
	}

      if (entity_undefined_p(e))
	{
	  FOREACH(entity, ee, code_declarations(entity_code(get_current_module_entity())))
	    {
	      pips_debug(4, "entity decl : \"%s\"\n", entity_name(ee));
	      if (strcmp(name, entity_user_name(ee)) == 0)
		{
		  e = ee;
		  break;
		}
	    }
	}
      if (entity_undefined_p(e))
	{
	  string cu = compilation_unit_of_module(get_current_module_name());
	  string full_name = strdup(concatenate(cu, MODULE_SEP_STRING, name, NULL));
	  e = gen_find_tabulated(full_name, entity_domain);
	  free(full_name);
	}
     if (entity_undefined_p(e))
	{
	  string full_name = strdup(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, name, NULL));
	  e = gen_find_tabulated(full_name, entity_domain);
	  free(full_name);
	}

      if (entity_undefined_p(e))
	pips_user_error("\nC Entity \"%s\" not found\n", name);
    }

  pips_debug(1, "end\n");;
  return e;
}

void remove_old_pragma(void)
{
  pips_assert("statement defined", !statement_undefined_p(current_statement));
  pips_assert("pragma defined", !pragma_undefined_p(current_pragma));

  extensions es = statement_extensions(current_statement);

  list el = extensions_extension(es);

  extensions_extension(es) = NIL;
  FOREACH(EXTENSION, e, el)
    {
      pragma p=extension_pragma(e);
      if (p != current_pragma)
	extensions_extension(es) = gen_extension_cons(e, extensions_extension(es));
    }

  extensions_extension(es) = gen_nreverse(extensions_extension(es));
}

step_directive begin_omp_construct(int type, string s)
{
  statement directive_stmt;

  pips_debug(1,"begin type = %d, str = %s\n", type, s);

  directive_stmt = current_statement;

  if(fortran_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  pips_assert("loop statement", statement_loop_p(directive_stmt)||statement_forloop_p(directive_stmt));
	  insert_optional_pragma(type);
	case STEP_PARALLEL:
	case STEP_MASTER:
	case STEP_SINGLE:
	case STEP_BARRIER:
	  directive_stmt = make_empty_block_statement();
	  break;
	case STEP_THREADPRIVATE:
	  directive_stmt = make_empty_block_statement();
	  break;
	default:
	  pips_user_error("unknown directive type %d\n", type);
	  break;
	}
    }
  else if (c_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  {
	    statement new_stmt;
	    pips_assert("loop statement", statement_loop_p(directive_stmt)||statement_forloop_p(directive_stmt));
	    pips_debug(2,"Block conversion (C)\n");

	    new_stmt = instruction_to_statement(statement_instruction(directive_stmt));

	    move_statement_attributes(directive_stmt, new_stmt);
	    statement_label(directive_stmt) = entity_empty_label();
	    statement_comments(directive_stmt) = empty_comments;
	    statement_declarations(directive_stmt) = NIL;
	    statement_decls_text(directive_stmt) = NULL;
	    statement_extensions(directive_stmt) = empty_extensions();
	    statement_instruction(directive_stmt) = make_instruction_block(CONS(STATEMENT, new_stmt, NIL));
	  }
	case STEP_PARALLEL:
	case STEP_MASTER:
	case STEP_SINGLE:
	  /* In C, these OpenMP pragmas are already supported by blocks: no block conversion needed */
	  break;
	case STEP_BARRIER:
	  directive_stmt = make_empty_block_statement();
	  break;
	case STEP_THREADPRIVATE:
	  directive_stmt = make_empty_block_statement();
	  break;
	default:
	  pips_user_error("unknown directive type %d\n", type);
	  break;
	}
    }
  else
    pips_user_error("language not supported");

  step_directive drt = new_step_directive(directive_stmt, type, s);

  if(fortran_module_p(get_current_module_entity()))
    {
      step_blocks_push(statement_sequence(directive_stmt));
    }

  pips_debug(1,"end drt = %p\n", drt);
  return drt;
}

/*
 * Function used for Fortran programs
 *
 *
  -> substitution dans une sequence : I1,...,Ii+OMPpragma,... , Ij-1, Ij+OMPpragma, ..., In en :
  I1,..., Block(Ii, ..., Ij-1 )+STEPpragma, Ij..., In
*/
step_directive end_omp_construct(int type)
{
  step_directive drt;

  pips_debug(1,"begin type = %d\n", type);

  drt = step_directive_undefined;
  if(fortran_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  {
	    sequence current_block;
	    statement last_stmt;


	    current_block = step_blocks_head();
	    last_stmt = STATEMENT(CAR(sequence_statements(current_block)));
	    /* suppression du CONTINUE portant le pragma optionnel dans le futur block */
	    while (empty_statement_or_labelless_continue_p(last_stmt))
	      {
		pips_debug(2,"POP current_block");
		POP(sequence_statements(current_block));
		last_stmt = STATEMENT(CAR(sequence_statements(current_block)));
	      }
	    if(step_directives_bound_p(last_stmt))
	      {
		drt = step_directives_load(last_stmt);
		if (step_directive_type(drt) != type)
		  pips_user_error("\nDirective end-loop not well formed\n");
		else
		  pips_debug(2,"loop directive already closed\n");
	      }
	    else if (!statement_loop_p(last_stmt))
	      pips_user_error("\nDirective end-loop after a no-loop statement\n");
	  }
	default:
	  break;
	}

      if(step_directive_undefined_p(drt))
	{
	  drt = get_current_step_directive(true);
	  
	  if (step_directive_undefined_p(drt) || step_directive_type(drt) != type)
	    pips_user_error("\nDirective not well formed\n");
	  
	  int size = step_blocks_size();
	  sequence new_seq = step_blocks_pop();
	  pips_debug(2,"pop block directive\n");
	  pips_debug(2, "block_stack size :%d\t new_seq_length=%d\n", size, (int)gen_length(sequence_statements(new_seq)));
	  
	  statement directive_stmt = step_directive_block(drt);
	  sequence_statements(statement_sequence(directive_stmt)) = gen_nreverse(statement_block(directive_stmt));
	}
    }

  pips_debug(1, "end drt = %p\n", drt);
  return drt;
}

/* 
 *
 *
 *
 *
 * Functions used for gen_multi_recurse 
 *
 *
 *
 *
 */
static step_directive new_step_directive(statement directive_stmt, int type,  string s)
{
  step_directive drt;
  sequence current_block;

  pips_debug(1,"begin directive_stmt = %p, type = %d, str = %s\n", directive_stmt, type, s);

  if(!statement_block_p(directive_stmt))
    {
      pips_debug(0,"Directive type : %s\n", s);
      STEP_DEBUG_STATEMENT(0, "on statement", directive_stmt);
      pips_assert("block statement", false);
    }


  pips_debug(2, "make_step_directive\n");
  drt = make_step_directive(type, directive_stmt,  CONS(STEP_CLAUSE, make_step_clause_transformation(current_transform), NIL));
  step_directives_store(directive_stmt, drt);
  add_pragma_str_to_statement(directive_stmt, strdup(concatenate(STEP_SENTINELLE, s, NULL)), false);

  current_block = step_blocks_head();
  sequence_statements(current_block) = CONS(STATEMENT, directive_stmt, sequence_statements(current_block));

  reset_step_transform();

  pips_debug(1,"end drt = %p\n", drt);
  return drt;
}


static void print_step_blocks()
{
  int size;
  int head_len;

  size = step_blocks_size();
  head_len = size?gen_length(sequence_statements(step_blocks_head())):0;
  pips_debug(4, "size = %d, head_length = %d\n", size, head_len);
}

static void reset_step_transform()
{
  const char* transformation = get_string_property("STEP_DEFAULT_TRANSFORMATION");

  if (strncasecmp(transformation, STEP_DEFAULT_TRANSFORMATION_MPI_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_MPI_TXT))==0)
    current_transform = STEP_TRANSFORMATION_MPI;
  else if (strncasecmp(transformation, STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT))==0)
    current_transform = STEP_TRANSFORMATION_HYBRID;
  else if (strncasecmp(transformation, STEP_DEFAULT_TRANSFORMATION_OMP_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_OMP_TXT))==0)
    current_transform = STEP_TRANSFORMATION_OMP;
  else
    current_transform = STEP_TRANSFORMATION_SEQ;

  pips_debug(2, "step transform type : %d\n", current_transform);
}

/*
  Le statement correspondant a la directive courante est en debut de sequence
  (l'ordre des statements dans les séquences est inversé dans la pile step_blocks)
*/
static step_directive get_current_step_directive(bool open)
{
  if(open && fortran_module_p(get_current_module_entity()))
    {
      pips_assert("stack size", step_blocks_size() > 1);

      sequence parent_block = step_blocks_nth(2);

      statement last_stmt = STATEMENT(CAR(sequence_statements(parent_block)));
      STEP_DEBUG_STATEMENT(1, "last_stmt 1", last_stmt);

      pips_assert("directive", step_directives_bound_p(last_stmt));
      pips_assert("sequence", statement_sequence_p(last_stmt));
      pips_assert("same sequence", statement_sequence(last_stmt)==step_blocks_head());

      return step_directives_load(last_stmt);
    }
  else if(!step_blocks_empty_p())
    {
      sequence current_block = step_blocks_head();

      if (!ENDP(sequence_statements(current_block)))
	{
	  statement last_stmt = STATEMENT(CAR(sequence_statements(current_block)));
	  STEP_DEBUG_STATEMENT(1,"last_stmt 2",last_stmt);

	  if(step_directives_bound_p(last_stmt))
	    return step_directives_load(last_stmt);
	}
    }

  return step_directive_undefined;
}

/*
 * pragma analysis 
 */

static void step_pragma_handle(statement stmt)
{
  pips_debug(1,"begin\n");

  pips_assert("statement undefined", statement_undefined_p(current_statement));
  pips_assert("pragma undefined", pragma_undefined_p(current_pragma));

  current_statement = stmt;

  reset_step_transform();

  STEP_DEBUG_STATEMENT(1, "begin stmt", current_statement);
  pips_debug(1,"end stmt\n");

  FOREACH(EXTENSION, e, extensions_extension(statement_extensions(stmt)))
    {
      pips_debug(2,"for each extension\n");
      current_pragma = extension_pragma(e);
      if (pragma_string_p(current_pragma))
	{
	  step_bison_parse(current_pragma, current_statement);
	}
    }
  current_pragma = pragma_undefined;
  current_statement = statement_undefined;
  pips_debug(1,"end\n");
}

static bool statement_filter(statement stmt)
{
  pips_debug(1,"begin\n");
  STEP_DEBUG_STATEMENT(2,"begin in stmt",stmt);
  pips_debug(2,"end in stmt\n");

  print_step_blocks();

  /* For fortran programs, converting comments into pragmas */
  if(fortran_module_p(get_current_module_entity()))
    step_comment2pragma_handle(stmt);

  if (step_blocks_empty_p())
    {
      pips_debug(2,"empty step_blocks: nothing to do in the module body\n");
    }
  else
    {
      statement parent_stmt;

      pips_debug(2,"Analysing and handling pragmas\n");

      parent_stmt = (statement)gen_get_ancestor(statement_domain, stmt);
      pips_assert("parent", parent_stmt != NULL);

      if (!ENDP(extensions_extension(statement_extensions(stmt))) &&
	  !statement_block_p(parent_stmt))
	{
	  /*
	    On transforme un statement portant un pragma en block s'il ne l'etait pas deja.
	    L'analyse aura lieu lors du parcours du block nouvellement cree
	  */
	  pips_debug(2,"Block conversion (Fortran)\n");

	  statement new_stmt = instruction_to_statement(statement_instruction(stmt));
	  move_statement_attributes(stmt, new_stmt);

	  statement_label(stmt) = entity_empty_label();
	  statement_comments(stmt) = empty_comments;
	  statement_declarations(stmt) = NIL;
	  statement_decls_text(stmt) = NULL;
	  statement_extensions(stmt) = empty_extensions();

	  statement_instruction(stmt) = make_instruction_block(CONS(STATEMENT, new_stmt, NIL));
	}
      else
	{
	  step_pragma_handle(stmt);

	  /*
	    Si dans une sequence (== block), on met a jour le block courant
	    Si c'est une directive, elle a ete ajoute a son identification
	  */
	  if(!step_directives_bound_p(stmt) && statement_block_p(parent_stmt))
	    {
	      sequence current_block = step_blocks_head();
	      pips_debug(2,"ADD stmt to the current block\n");
	      sequence_statements(current_block)=CONS(STATEMENT, stmt,
						      sequence_statements(current_block));
	    }
	  else
	    {
	      pips_debug(2,"NOT added to current step_blocks\n");
	    }
	}
    }

  STEP_DEBUG_STATEMENT(2,"begin out stmt",stmt);
  pips_debug(2,"end out stmt\n");
  pips_debug(1,"end\n");
  return true;
}

static void statement_rewrite(statement stmt)
{
  pips_debug(1,"begin\n");
  STEP_DEBUG_STATEMENT(5, "begin in stmt", stmt);
  pips_debug(5, "end in stmt");

  print_step_blocks();

  STEP_DEBUG_STATEMENT(5, "begin out stmt", stmt);
  pips_debug(5, "end out stmt");

  pips_debug(1,"end\n");
  return;
}

static bool sequence_filter(sequence __attribute__ ((unused)) seq)
{
  pips_debug(1,"begin\n");
  pips_debug(2,"PUSH empty sequence on step_blocks\n");
  step_blocks_push(make_sequence(NIL));
  
  print_step_blocks();
  pips_debug(1,"end\n");
  return true;
}

static void sequence_rewrite(sequence seq)
{
  int size;
  sequence previous_block;

  pips_debug(1,"begin\n");

  size = step_blocks_size();
  previous_block = step_blocks_nth(2);

  if(previous_block && !sequence_undefined_p(previous_block))
    {
      statement last_stmt = STATEMENT(CAR(sequence_statements(previous_block)));
      if (fortran_module_p(get_current_module_entity()) && step_directives_bound_p(last_stmt))
	pips_user_error("\nDirective not well formed\n");
    }

  sequence new_seq = step_blocks_pop();
  pips_debug(2, "POP step_blocks\n");
  pips_debug(2, "block_stack size :%d\t new_seq_length=%d\n", size, (int)gen_length(sequence_statements(new_seq)));

  gen_free_list(sequence_statements(seq));
  sequence_statements(seq) = gen_nreverse(sequence_statements(new_seq));
  sequence_statements(new_seq) = NIL;
  free_sequence(new_seq);

  pips_debug(1,"end\n");
  return;
}

static void step_directive_parser(statement body)
{
  /* For fortran programs, converting comments into pragmas into declaration txt */
  if(fortran_module_p(get_current_module_entity()))
    step_comment2pragma_handle(statement_undefined);

  make_step_blocks_stack();

  // where the OpenMP constructs are identified
  gen_multi_recurse(body,
		    statement_domain, statement_filter, statement_rewrite,
		    sequence_domain, sequence_filter, sequence_rewrite,
		    NULL);

  free_step_blocks_stack();
}

bool step_parser(const char* module_name)
{
  debug_on("STEP_PARSER_DEBUG_LEVEL");
  pips_debug(1, "%d module_name = %s\n", __LINE__, module_name);

  statement stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  stmt = make_block_statement(CONS(STATEMENT, stmt, NIL));

  set_current_module_entity(local_name_to_top_level_entity(module_name));

  step_directives_init(1);

  step_directive_parser(stmt);

  ifdebug(1)
    {
      step_directives_print();
    }

  step_directives_save();
  reset_current_module_entity();


  module_reorder(stmt);
  if(ordering_to_statement_initialized_p())
    reset_ordering_to_statement();

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stmt);

  pips_debug(1, "End step_parser\n");
  debug_off();

  return true;
}

