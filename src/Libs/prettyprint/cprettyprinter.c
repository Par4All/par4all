/* 
   $Id$

   Try to prettyprint the RI in C.
   Very basic at the time.
   Functionnal. 
   All arguments are assumed newly allocated.
   It might be really slow, but it should be safe.
   I should use some kind of string accumulator (array/list...)

   print_c_rough > MODULE.crough
                 < PROGRAM.entities
                 < MODULE.code

   print_c_code  > MODULE.c_printed_file
                 < MODULE.crough
*/

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#define EMPTY         ""
#define NL            "\n"
#define SEMICOLON     ";" NL
#define SPACE         " "

#define OPENBRACKET   "["
#define CLOSEBRACKET  "]"

#define OPENPAREN     "("
#define CLOSEPAREN    ")"

#define OPENBRACE     "{"
#define CLOSEBRACE    "}"

#define SHARPDEF      "#define"
#define COMMENT	      "//" SPACE

/* forward declaration. */
static string c_expression(expression);

/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
  (entity_function_p(get_current_module_entity()))

/************************************************************** DECLARATIONS */

/* 
   integer a(n,m) -> int a[m][n];
   parameter (n=4) -> #define n 4
 */

static string
c_basic_string(basic b)
{
  string result = "UNKNOWN_BASIC_TYPE" SPACE;

  switch (basic_tag(b))
    {
    case is_basic_int:
      switch (basic_int(b))
	{
	case 2: result = "short" SPACE; 
	  break;
	case 4: result = "int" SPACE; 
	  break;
	case 6: result = "long" SPACE; 
	  break;
	case 8: result = "long long" SPACE; 
	  break;
	}
      break;
    case is_basic_float:
      switch (basic_float(b))
	{
	case 4: result = "float" SPACE;
	  break;
	case 8: result = "double" SPACE;
	  break;
	}
      break;
    case is_basic_logical:
      result = "int" SPACE;
      break;
    case is_basic_string:
      result = "char" SPACE;
      break;

      /* add (short, long, long long) signed, unsigned, bit, pointer, derived, typedef*/
    }

  return strdup(result);
}

static string 
int_to_string(int i)
{
  char buffer[50];
  sprintf(buffer, "%d", i);
  return strdup(buffer);
}

/* returns a string for dimension list dim.
   lower is assumed to be 0 in references.
   Caution: the dimension order is reversed! I want it!
 */
static string c_dim_string(list ldim)
{
  string result = strdup("");
  if (ldim) 
    {
      MAP(DIMENSION, dim,
      {
	expression elow = dimension_lower(dim);
	expression eup = dimension_upper(dim);
	int low;
	int up;
	string slow;
	string sup;

	/* In fact, the lower bound of array in C is always equal to 0, 
	   we only need to print the upper dimension 
	   but in order to handle Fortran code, we check all other possibilities.


	NHAM ROI, X[N] TUONG DUONG VOI X[0] DEN X[N-1], nen khong phai tru 1*/

	if (expression_integer_value(elow, &low))
	  {
	    if (low == 0)
	      result = strdup(concatenate(OPENBRACKET,words_to_string(words_expression(eup)),
					  CLOSEBRACKET,result,NULL));
	    else 
	      {
		if (expression_integer_value(eup, &up))
		  result = strdup(concatenate(OPENBRACKET,int_to_string(up-low),CLOSEBRACKET,result,NULL));
		else
		  {
		    sup = words_to_string(words_expression(eup));
		    result = strdup(concatenate(OPENBRACKET,sup,"-",int_to_string(low),CLOSEBRACKET,result,NULL));
		    free(sup);
		  }
	      }
	  }
	else 
	  {
	    slow = words_to_string(words_expression(elow));
	    sup = words_to_string(words_expression(eup));
	    result = strdup(concatenate(OPENBRACKET,sup,"-",slow,CLOSEBRACKET,result,NULL));
	    free(slow);
	    free(sup);
	  }
      },
	  ldim);
    }
  /* otherwise the list is empty, no dimension to declare */
  return result;
}

#define RESULT_NAME	"result"

/* the local name in C: lower case, special handling possible...
 */
static string 
c_entity_local_name(entity var)
{
  string name;
  char * car;

  if (current_module_is_a_function() &&
      var != get_current_module_entity() &&
      same_string_p(entity_local_name(var), 
		    entity_local_name(get_current_module_entity())))
    {
      name = strdup(RESULT_NAME);
    }
  else
    {
      name = strdup(entity_local_name(var));
    }

  /* switch to lower cases... */
  for (car = name; *car; car++)
    *car = (char) tolower(*car);

  return name;
}

static string 
this_entity_cdeclaration(entity var)
{
  string result = NULL;
  storage s = entity_storage(var);
  
  if (storage_rom_p(s))
    {
      value va = entity_initial(var);
      constant c = NULL;

      if (value_constant_p(va))
	c = value_constant(va);
      else if (value_symbolic_p(va))
	c = symbolic_constant(value_symbolic(va));
      
      if (c)
	{
	  if (constant_int_p(c))
	    {
	      string sval = int_to_string(constant_int(c));
	      string svar = c_entity_local_name(var);

	      result = strdup(concatenate(SHARPDEF, SPACE, svar,
					  SPACE, sval, NL, NULL));

	      free(svar);
	      free(sval);
	    }
	  /* What about real, double, string, ... ?*/
	}
    }
  else
    {
      type t;
      variable v;
      string st, sd, svar;
      
      t = entity_type(var);

      /* add other types here : functional, struct, union, enum */

      pips_assert("it is a variable", type_variable_p(t));
      v = type_variable(t);
  
      st = c_basic_string(variable_basic(v));
      sd = c_dim_string(variable_dimensions(v));

      /* add variable qualifiers here */

      svar = c_entity_local_name(var);
      
      result= strdup(concatenate(st, SPACE, svar, sd, NULL));

      free(st); free(sd); free(svar);
    }

  return result? result: strdup("");
}

static bool parameter_p(entity e)
{
  /* Constant variables */
  return storage_rom_p(entity_storage(e)) && 
    value_symbolic_p(entity_initial(e)) &&
    type_functional_p(entity_type(e));
}

static bool variable_p(entity e)
{
  storage s = entity_storage(e);
  return type_variable_p(entity_type(e)) &&
    (storage_ram_p(s) || storage_return_p(s));
}

static bool argument_p(entity e)
{
  /* Formal variables */
  return type_variable_p(entity_type(e)) && 
    storage_formal_p(entity_storage(e));
}

static string 
c_declarations(entity module,
	       bool (*consider_this_entity)(entity),
	       string separator,
	       bool lastsep)
{
  string result = strdup("");
  code c;
  bool first = TRUE;

  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  MAP(ENTITY, var,
  {
    debug(2, "\n Prettyprinter declaration for variable :",entity_local_name(var));   
    if (consider_this_entity(var))
      {
	string old = result;
	string svar = this_entity_cdeclaration(var);
	result = strdup(concatenate(old, !first && !lastsep? separator: "",
				    svar, lastsep? separator: "", NULL));
	free(old);
	free(svar);
	first = FALSE;
      }
  },code_declarations(c));
  return result;
}

/********************************************************************** HEAD */

/* returns the head of the function/subroutine/program.
   declarations look ANSI C. 
 */
#define MAIN_DECLARATION	"int main(int argc, char *argv[])" NL

static string c_head(entity module)
{
  string result;

  pips_assert("it is a function", type_functional_p(entity_type(module)));

  if (entity_main_module_p(module))
    {
      /* another kind : "int main(void)" ?*/
      result = strdup(MAIN_DECLARATION);
    }
  else
    {
      string head, args, svar;
      functional f = type_functional(entity_type(module));

      /* define type head. */
      if (entity_subroutine_p(module))
	{
	  head = strdup("void");
	}
      else
	{
	  variable v;
	  pips_assert("type of result is a variable", 
		      type_variable_p(functional_result(f)));
	  v = type_variable(functional_result(f));
	  head = c_basic_string(variable_basic(v));
	}
      
      /* define args. */
      if (functional_parameters(f))
	{
	  args = c_declarations(module, argument_p, ", ", FALSE);
	}
      else
	{
	  args = strdup("void");
	}

      svar = c_entity_local_name(module);

      result = strdup(concatenate(head, SPACE, svar,
				  OPENPAREN, args, CLOSEPAREN, NL, NULL));

      free(head); 
      free(args);
      free(svar);
    }

  return result;
}

/*************************************************************** EXPRESSIONS */

/* generate a basic c expression.
   no operator priority is assumed...
 */
typedef string (*prettyprinter)(string, list);

struct s_ppt
{
  char * fortran;
  char * c;
  prettyprinter ppt;
};

static bool expression_needs_parenthesis_p(expression);

static string ppt_binary(string in_c, list le)
{
  string result;
  expression e1, e2;
  string s1, s2;
  bool p1, p2;

  pips_assert("2 arguments to binary call", gen_length(le)==2);

  e1 = EXPRESSION(CAR(le));
  p1 = expression_needs_parenthesis_p(e1);
  s1 = c_expression(e1);

  e2 = EXPRESSION(CAR(CDR(le)));
  p2 = expression_needs_parenthesis_p(e2);
  s2 = c_expression(e2);

  result = strdup(concatenate(p1? OPENPAREN: EMPTY, s1, p1? CLOSEPAREN: EMPTY,
			      SPACE, in_c, SPACE, 
			      p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
			      NULL));

  free(s1);
  free(s2);

  return result;
}

static string ppt_unary(string in_c, list le)
{
  string e, result;
  pips_assert("one arg to unary call", gen_length(le)==1);
  e = c_expression(EXPRESSION(CAR(le)));
  result = strdup(concatenate(in_c, SPACE, e, NULL));
  free(e);
  return result;
}

static string ppt_call(string in_c, list le)
{
  string scall, old;
  if (le == NIL)
    { 
      scall = strdup(concatenate(in_c, NULL));
    }
  else 
    {
      bool first = TRUE;
      scall = strdup(concatenate(in_c, OPENPAREN, NULL));
     
      /* Attention: not like this for io statements*/
      MAP(EXPRESSION, e,
      {
	string arg = c_expression(e);
	old = scall;
	scall = strdup(concatenate(old, first? "": ", ", arg, NULL));
	free(arg);
	free(old);
	first = FALSE;
      },le);

      old = scall;
      scall = strdup(concatenate(old, CLOSEPAREN, NULL));
      free(old);
    }
  return scall;
}

static struct s_ppt fortran_to_c[] = 
{
  { "=", "=", ppt_binary },
  { "*", "*", ppt_binary },
  { "+", "+", ppt_binary  },
  { "-", "-", ppt_binary },
  { "/", "/", ppt_binary },
  { ".EQ.", "==", ppt_binary },
  { ".NE.", "!=", ppt_binary },
  { ".LE.", "<=", ppt_binary },
  { ".LT.", "<", ppt_binary },
  { ".GE.", ">=", ppt_binary },
  { ".GT.", ">", ppt_binary },
  { ".AND.", "&&", ppt_binary },
  { ".OR.", "||", ppt_binary },
  { ".NOT.", "!", ppt_unary },
  { NULL, NULL, ppt_call }
  /* add things here ? */
};

/* return the prettyprinter structure for c.
 
Not only from Fortran ?*/
static struct s_ppt * get_ppt(entity f)
{
  string called = entity_local_name(f);
  struct s_ppt * table = fortran_to_c;
  while (table->fortran && !same_string_p(called, table->fortran))
    table++;
  return table;
}


/*
 */
static bool expression_needs_parenthesis_p(expression e)
{
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      {
	struct s_ppt * p = get_ppt(call_function(syntax_call(s)));
	return p->ppt==ppt_binary;
      }
    case is_syntax_reference:
    case is_syntax_range:
    default:
      return FALSE;
    }
}

#define RET	"return"

static string c_call(call c)
{
  entity called = call_function(c);
  struct s_ppt * ppt = get_ppt(called);
  string result;

  /* special case... */
  if (same_string_p(entity_local_name(called), "RETURN"))
    {
      if (entity_main_module_p(get_current_module_entity()))
	result = strdup(RET " 0");
      else if (current_module_is_a_function())
	result = strdup(RET SPACE RESULT_NAME);
      else
	result = strdup(RET);
    }
  else if (call_constant_p(c))
    {
      result = c_entity_local_name(called);
    }
  else
    {
      string s = c_entity_local_name(called);
      result = ppt->ppt(ppt->c? ppt->c: s, call_arguments(c));
      free(s);
    }

  return result;
}

/* the indexes are reversed.
 */
static string c_reference(reference r)
{
  string result = strdup(EMPTY), old, svar;
  MAP(EXPRESSION, e, 
  {
    string s = c_expression(e);
    old = result;
    result = strdup(concatenate(OPENBRACKET, s, CLOSEBRACKET, old, NULL));
    free(old);
    free(s);
  }, reference_indices(r));

  old = result;
  svar = c_entity_local_name(reference_variable(r));
  result = strdup(concatenate(svar, old, NULL));
  free(old);
  free(svar);
  return result;
}

static string c_expression(expression e)
{
  string result = NULL;
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      result = c_call(syntax_call(s));
      break;
    case is_syntax_range:
      result = strdup("range not implemented");
      break;
    case is_syntax_reference:
      result = c_reference(syntax_reference(s));
      break;
      /* add cast, sizeof here */
    default:
      pips_internal_error("unexpected syntax tag");
    }
  return result;
}

static string c_statement(statement s);

static string c_unstructured(unstructured u)
{
  string result =  strdup(EMPTY);
  /* build an arbitrary reverse trail of control nodes */
  list trail = unstructured_to_trail(u);
  list cc = NIL;
  trail = gen_nreverse(trail);
  dump_trail(trail);
  
  for(cc=trail; !ENDP(cc); POP(cc)) 
    {
      control c = CONTROL(CAR(cc));
      int nsucc = gen_length(control_successors(c));
      statement st = control_statement(c);
      string oldresult = result;   
      switch(nsucc) 
	{
	case 0:
	  {	       
	    result = strdup(concatenate(oldresult,c_statement(st),NULL));
	    fprintf(stderr, "Where 0 \n");
	    break;
	  }
	case 1: 
	  {
	    control succ = CONTROL(CAR(control_successors(c)));
	    if(check_io_statement_p(control_statement(succ)) &&
	       !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) 
	       {
		 succ = CONTROL(CAR(CDR(control_successors(succ))));
		 if(check_io_statement_p(control_statement(succ)) &&
		   !get_bool_property("PRETTYPRINT_CHECK_IO_STATEMENTS")) 
		  {
		    
		    succ = CONTROL(CAR(CDR(control_successors(succ))));
		  }
		 pips_assert("The successor is not a check io statement",
			     !check_io_statement_p(control_statement(succ)));
	       }
	   
	    result = strdup(concatenate(oldresult,c_statement(st),NULL));
	    if(statement_does_return(st)) {
	      if(!ENDP(CDR(cc))) {
		control tsucc = CONTROL(CAR(CDR(cc)));
		if(tsucc==succ) 
		  {
		    break;
		  }
	      }
	    }
	    fprintf(stderr, "Where 1 \n");
	    break;
	  }
	case 2: 
	  {
	    control succ1 = CONTROL(CAR(control_successors(c)));
	    control succ2 = CONTROL(CAR(CDR(control_successors(c))));
	    instruction i = statement_instruction(st);
	    test t = instruction_test(i);
	    pips_assert("must be a test", instruction_test_p(i));
	    fprintf(stderr, "Where 2 \n");
	    result = strdup(concatenate(oldresult,c_statement(st),NULL));
	    break;
	  }
	default:
	  pips_internal_error("Too many successors for a control node\n");
	}
      free(oldresult);
    }   
  
  gen_free_list(trail);
  return result;
}

static string c_test(test t)
{
  string result;
  bool no_false;
  string cond, strue, sfalse;
  cond = c_expression(test_condition(t));
  strue = c_statement(test_true(t));
  no_false = empty_statement_p(test_false(t));
  
  sfalse = no_false? NULL: c_statement(test_false(t));
  
  result = strdup(concatenate("if (", cond, ") {" NL, 
			      strue, 
			      no_false? "}" NL: "} else {" NL,
			      sfalse, "}" NL, NULL));
  free(cond);
  free(strue);
  if (sfalse) free(sfalse);
  return result;
}

static string c_sequence(sequence seq)
{
  string result = strdup(EMPTY);
  MAP(STATEMENT, s,
  {
    string oldresult = result;
    string current = c_statement(s);
    result = strdup(concatenate(oldresult, current, NULL));
    free(current);
    free(oldresult);
  }, sequence_statements(seq));
  return result;
}

static string c_loop(loop l)
{
  /* partial implementation...
     However, there is not this kind of loop in C */
  string result;
  string body = c_statement(loop_body(l));
  string index = c_entity_local_name(loop_index(l));
  range r = loop_range(l);
  string low = c_expression(range_lower(r));
  string up = c_expression(range_upper(r));
  /* what about step*/
  result = strdup(concatenate("for (", index, "=", low, "; ",
			      index, "<=", up, "; ",
			      index, "++)", SPACE, OPENBRACE, NL, 
			      body, CLOSEBRACE, NL, NULL));
  free(body);
  free(low);
  free(up);
  free(index);
  return result;
}

static string c_whileloop(whileloop w)
{
  /* partial implementation... */
  string result;
  string body = c_statement(whileloop_body(w));
  string cond = c_expression(whileloop_condition(w));
  /*do while and while do loops */
  result = strdup(concatenate("while (", cond, ") {" NL, 
			      body, "}" NL, NULL));
  
  free(cond);
  free(body);
  return result;
}
/**************************************************************** STATEMENTS */

static string c_statement(statement s)
{
  string result;
  instruction i = statement_instruction(s);
  switch (instruction_tag(i))
    {
    case is_instruction_test:
      {
	test t = instruction_test(i);
	result = c_test(t);
	break;
      }
    case is_instruction_sequence:
      {
	sequence seq = instruction_sequence(i);
	result = c_sequence(seq);
	break;
      }
    case is_instruction_loop:
      {
	loop l = instruction_loop(i);
	result = c_loop(l);
	break;
      }
    case is_instruction_whileloop:
      {
	whileloop w = instruction_whileloop(i);
	result = c_whileloop(w);
	break;
      }
    case is_instruction_call:
      {
	string scall = c_call(instruction_call(i));
	result = strdup(concatenate(scall, SEMICOLON, NULL));
	break;
      }
    case is_instruction_unstructured:
      {
	unstructured u = instruction_unstructured(i);
	result = c_unstructured(u);
	break;
      }
    case is_instruction_goto:
      {
	statement g = instruction_goto(i);
	entity el = statement_label(g);
	string l = entity_local_name(el) + strlen(LABEL_PREFIX);
	result = strdup(concatenate("goto ",l, SEMICOLON, NULL));
	break;
      }
      /* add switch, forloop break, continue, return instructions here*/
    default:
      result = strdup(concatenate(COMMENT, " Instruction not implemented" NL, NULL));
      break;
    }

  return result;
}

static string c_code_string(entity module, statement stat)
{
  string before_head, head, decls, body, result;

  /* What about declarations that are external a module scope ?
     Consider a source file as a module entity, put all declarations in it 
     (external static + TOP-LEVEL) */

  /* before_head only generates the constant declarations, such as #define*/

  before_head = c_declarations(module, parameter_p, NL, TRUE);
  head        = c_head(module);
  /* What about declarations associated to statements */
  decls       = c_declarations(module, variable_p, SEMICOLON, TRUE);
  body        = c_statement(stat);
  
  result = strdup(concatenate(before_head, head, OPENBRACE, NL, 
			      decls, NL,
			      body, CLOSEBRACE, NL, NULL));

  free(before_head);
  free(head);
  free(decls);
  free(body);

  return result;
}

/******************************************************** PIPSMAKE INTERFACE */

#define INDENT		"indent"
#define CROUGH		".crough"
#define CPRETTY		".c"

bool print_c_rough(string module_name)
{
  FILE * out;
  string ppt, crough, dir, filename;
  entity module;
  statement stat;

  crough = db_build_file_resource_name(DBR_CROUGH, module_name, CROUGH);
  module = local_name_to_top_level_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", crough, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("CPRETTYPRINTER_DEBUG_LEVEL");
  pips_debug(1, "Begin C prettyprrinter for %s\n", entity_name(module));
  ppt = c_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();  

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "/* C pretty print for module %s. */\n%s", module_name, ppt);
  safe_fclose(out, filename);

  free(ppt);
  free(dir);
  free(filename);

  DB_PUT_FILE_RESOURCE(DBR_CROUGH, module_name, crough);

  reset_current_module_statement();
  reset_current_module_entity();

  return TRUE;
}

/* C indentation thru indent.
 */
bool print_c_code(string module_name)
{
  string crough, cpretty, dir, cmd;

  crough = db_get_memory_resource(DBR_CROUGH, module_name, TRUE);
  cpretty = db_build_file_resource_name(DBR_C_PRINTED_FILE, module_name, CPRETTY);
  dir = db_get_current_workspace_directory();

  cmd = strdup(concatenate(INDENT, " ", 
			   dir, "/", crough, " -o ", 
			   dir, "/", cpretty, NULL));

  safe_system(cmd);

  DB_PUT_FILE_RESOURCE(DBR_C_PRINTED_FILE, module_name, cpretty);
  free(cmd);
  free(dir);

  return TRUE;
}
