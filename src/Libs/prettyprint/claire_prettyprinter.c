/* 
   $Id$

   Try to prettyprint the RI in CLAIRE.
   Very basic at the time.

   print_claire_code        > MODULE.claire_printed_file
                            < PROGRAM.entities
                            < MODULE.code

   $Log: claire_prettyprinter.c,v $
   Revision 1.3  2004/03/24 16:02:03  hurbain
   First version for claire pretty printer. Currently only manages (maybe :p) arrays declarations.

   Revision 1.2  2004/03/11 15:09:43  irigoin
   function print_claire_code() declared for the link but not programmed yet.

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
#include "text-util.h"

#define EMPTY         ""
#define NL            "\n"
#define TAB           "\t"
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


#define RESULT_NAME	"result"

static string c_entity_local_name(entity var)
{
  string name;
  char * car;

  if (current_module_is_a_function() &&
      var != get_current_module_entity() &&
      same_string_p(entity_local_name(var), 
		    entity_local_name(get_current_module_entity())))
    name = RESULT_NAME;
  else
    {
      name = entity_local_name(var);
  
      /* Delete all the prefixes */

      if (strstr(name,STRUCT_PREFIX) != NULL)
	name = strstr(name,STRUCT_PREFIX) + 1;
      if (strstr(name,UNION_PREFIX) != NULL)
	name = strstr(name,UNION_PREFIX) + 1;
      if (strstr(name,ENUM_PREFIX) != NULL)
	name = strstr(name,ENUM_PREFIX) + 1;      
      if (strstr(name,TYPEDEF_PREFIX) != NULL)
	name = strstr(name,TYPEDEF_PREFIX) + 1;
      if (strstr(name,MEMBER_SEP_STRING) != NULL)
	name = strstr(name,MEMBER_SEP_STRING) + 1;
    }

  /* switch to lower cases... */
  for (car = name; *car; car++)
    *car = (char) tolower(*car);
  
  return name;
}

/************************************************************** DECLARATIONS */

/* 
   integer a(n,m) -> int a[m][n];
   parameter (n=4) -> #define n 4
 */

static string c_basic_string(basic b);

static string c_type_string(type t)
{
  string result = "UNKNOWN_TYPE" SPACE;
  switch (type_tag(t))
    {
    case is_type_variable:
      {
	basic b = variable_basic(type_variable(t));
	result = c_basic_string(b);
	break;
      }
    case is_type_void:
      {
	result = "void" SPACE;
	break;
      }
    case is_type_struct:
      {
	result = "struct" SPACE;
	break;
      }
    case is_type_union:
      {
	result = "union" SPACE;
	break;
      }
    case is_type_enum:
      {
	result = "enum" SPACE;
	break;
      }
    }
  return strdup(result);
}

static string c_basic_string(basic b)
{
  string result = "UNKNOWN_BASIC" SPACE;
  switch (basic_tag(b))
    {
    case is_basic_int:
      {
	pips_debug(2,"Basic int\n");
	switch (basic_int(b))
	  {
	  case 1: result = "char" SPACE; 
	    break;
	  case 2: result = "short" SPACE; 
	    break;
	  case 4: result = "int" SPACE; 
	    break;
	  case 6: result = "long" SPACE; 
	    break;
	  case 8: result = "long long" SPACE; 
	    break;
	  case 11: result = "unsigned char" SPACE;
	    break;
	  case 12: result = "unsigned short" SPACE;
	    break;
	  case 14: result = "unsigned int" SPACE;
	    break;
	  case 16: result = "unsigned long" SPACE;
	    break;
	  case 18: result = "unsigned long long" SPACE;
	    break;
	  case 21: result = "signed char" SPACE;
	    break;
	  case 22: result = "signed short" SPACE;
	    break;
	  case 24: result = "signed int" SPACE;
	    break;
	  case 26: result = "signed long" SPACE;
	    break;
	  case 28: result = "signed long long" SPACE;
	    break;
	  }
	break;
      }
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
    case is_basic_bit:
      {
	int i = basic_bit(b);
	pips_debug(2,"Bit field basic: %d\n",i);
	result = "int" SPACE; /* ignore if it is signed or unsigned */
	break;
      }
    case is_basic_pointer:
      {
	type t = basic_pointer(b);
	pips_debug(2,"Basic pointer\n");
	result = concatenate(c_type_string(t),"* ",NULL);
	break;
      }
    case is_basic_derived:
      {
	entity ent = basic_derived(b);
	type t = entity_type(ent);
	string name = c_entity_local_name(ent);
	result = concatenate(c_type_string(t),name,NULL);
	break;
      }
    case is_basic_typedef:
      {
	entity ent = basic_typedef(b);
	result = c_entity_local_name(ent);
	break;
      }  
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

static string claire_dim_string(list ldim)
{
  string result = "";
  int nbdim = 0;
  string origins = "origins = list<integer>(";
  string dimensions = "dimSizes = list<integer>(";

  if (ldim)
    {
      result = strdup(contatenate("A_A", " :: ", "DATA_ARRAY(",
				  "name = symbol!(" "\"A_A\"", ")", ",", NL));
      result = strdup(concatenate(result, TAB, "dim = ");
      MAP(DIMENSION, dim, {
	nbdim++;
	expression elow = dimension_lower(dim);
	expression eup = dimension_upper(dim);
	int low;
	int up;
	if (expression_integer_value(elow, &low)){
	  if(nbdim != 1)
	    origins = strdup(concatenate(origins, ",",int_to_string(low)));
	  else
	    origins = strdup(concatenate(origins, int_to_string(low)));
	}
	else pips_user_error("Array origins must be integer");

	if (expression_integer_value(esup, &sup)){
	  if(nbdim != 1)
	    dimensions = strdup(concatenate(dimensions, ", ",int_to_string(sup-low+1)));
	  else
	    dimensions = strdup(concatenate(dimensions, int_to_string(sup-low+1)));
	}
	else pips_user_error("Array dimensions must be integer");
      }, ldim);
      result = strdup(concatenate(result, int_to_string(nbdim), ",", NL));
      result = strdup(concatenate(result, TAB, origins, "),", NL));
      result = strdup(concatenate(result, TAB, dimensions, "),", NL));
      result = strdup(concatenate(result, TAB, "dataType = INTEGER)", NL, NL);
    }
  return result;
}

static string c_dim_string(list ldim)
{
  string result = "";
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
	   we only need to print (upper dimension + 1) 
	   but in order to handle Fortran code, we check all other possibilities
	   and print (upper - lower + 1). Problem : the order of dimensions is reversed !!!! */

	if (expression_integer_value(elow, &low))
	  {
	    if (low == 0)
	      {
		if (expression_integer_value(eup, &up))
		  result = strdup(concatenate(result,OPENBRACKET,int_to_string(up+1),CLOSEBRACKET,NULL));
		else
		  /* to be refined here to make more beautiful expression */
		  result = strdup(concatenate(result,OPENBRACKET,
				       words_to_string(words_expression(MakeBinaryCall(CreateIntrinsic("+"),
										       eup,int_to_expression(1)))),
				       CLOSEBRACKET,NULL));
	      }
	    else 
	      {
		if (expression_integer_value(eup, &up))
		  result = strdup(concatenate(result,OPENBRACKET,int_to_string(up-low+1),CLOSEBRACKET,NULL));
		else
		  {
		    sup = words_to_string(words_expression(eup));
		    result = strdup(concatenate(result,OPENBRACKET,sup,"-",int_to_string(low-1),CLOSEBRACKET,NULL)); 
		    free(sup);
		  }
	      }
	  }
	else 
	  {
	    slow = words_to_string(words_expression(elow));
	    sup = words_to_string(words_expression(eup));
	    result = strdup(concatenate(result,OPENBRACKET,sup,"-",slow,"+ 1",CLOSEBRACKET,NULL));
	    free(slow);
	    free(sup);
	  }
      }, ldim);
    }
  /* otherwise the list is empty, no dimension to declare */
  return result;
}

static string c_qualifier_string(list l)
{
  string result="";
  MAP(QUALIFIER,q,
  {
    switch (qualifier_tag(q)) {
    case is_qualifier_register:
      result = concatenate(result,"register ",NULL);
      break; 
    case is_qualifier_const:
      result = concatenate(result,"const ",NULL);
      break;
    case is_qualifier_restrict:
      result = concatenate(result,"restrict ",NULL);
      break;  
    case is_qualifier_volatile:
      result = concatenate(result,"volatile ",NULL);
      break; 
    }
  },l);
  return strdup(result); 
}

static bool brace_expression_p(expression e)
{
  if (expression_call_p(e))
    {
      entity f = call_function(syntax_call(expression_syntax(e)));
      if (ENTITY_BRACE_INTRINSIC_P(f))
	return TRUE;
    }
  return FALSE;
}

static string c_brace_expression_string(expression exp)
{
  string result = "{";
  list args = call_arguments(syntax_call(expression_syntax(exp)));
  
  bool first = TRUE;
  MAP(EXPRESSION,e,
  {
    if (brace_expression_p(e))
      result = strdup(concatenate(result,first?"":",",c_brace_expression_string(e),NULL));
    else
      result = strdup(concatenate(result,first?"":",",words_to_string(words_expression(e)),NULL));
    first = FALSE;
  },args);
  result = strdup(concatenate(result,"}",NULL));
  return result;
}

static string this_entity_cdeclaration(entity var)
{
  string result = NULL;
  string name = entity_local_name(var);
  type t = entity_type(var);
  storage s = entity_storage(var);
  pips_debug(2,"Entity name : %s\n",entity_name(var));
  /*  Many possible combinations */

  if (strstr(name,TYPEDEF_PREFIX) != NULL)
    /* This is a typedef name, what about typedef int myint[5] ???  */
    return strdup(concatenate("typedef ", c_type_string(t),SPACE,c_entity_local_name(var),NULL));
  
  switch (storage_tag(s)) {
  case is_storage_rom: 
    {
      value va = entity_initial(var);
      if (!value_undefined_p(va))
	{
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
		  return result;
		}
	      /*What about real, double, string, ... ?*/
	    }
	}
      break;
    }
  case is_storage_ram: 
    {
      /*     ram r = storage_ram(s);
      entity sec = ram_section(r);
      if ((sec == CurrentSourceFileStaticArea) || (sec == CurrentStaticArea))
      result = "static ";*/
      break;
    }
  default: 
  }

  switch (type_tag(t)) {
  case is_type_variable:
    {
      variable v = type_variable(t);  
      string st, sd, svar, sq;
      value val = entity_initial(var);
      st = c_basic_string(variable_basic(v));
      sd = c_dim_string(variable_dimensions(v));
      sq = c_qualifier_string(variable_qualifiers(v));
      svar = c_entity_local_name(var);
     
      /* problems with order !*/
      result = strdup(concatenate(sq, st, SPACE, svar, sd, NULL));
      if (!value_undefined_p(val))
	{
	  if (value_expression_p(val))
	    {
	      expression exp = value_expression(val);
	      if (brace_expression_p(exp))
		result = strdup(concatenate(result,"=",c_brace_expression_string(exp),NULL));
	      else 
		result = strdup(concatenate(result,"=",words_to_string(words_expression(exp)),NULL));
	    }
	}
      if (basic_bit_p(variable_basic(v)))
	{
	  int i = basic_bit(variable_basic(v));
	  pips_debug(2,"Basic bit %d",i);
	  result = strdup(concatenate(result,":",int_to_string(i),NULL));
	}
      free(st); free(sd); free(svar);
      break;
    }
  case is_type_struct:
    {
      list l = type_struct(t);
      result = strdup(concatenate("struct ",c_entity_local_name(var), "{", NL,NULL));
      MAP(ENTITY,ent,
      {
	string s = this_entity_cdeclaration(ent);	    
	result = strdup(concatenate(result, s, SEMICOLON, NULL));
	free(s);
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  case is_type_union:
    {
      list l = type_union(t);
      result = strdup(concatenate("union ",c_entity_local_name(var), "{", NL,NULL));
      MAP(ENTITY,ent,
      {
	string s = this_entity_cdeclaration(ent);	    
	result = strdup(concatenate(result, s, SEMICOLON, NULL));
	free(s);
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  case is_type_enum:
    {
      list l = type_enum(t);
      bool first = TRUE;
      result = strdup(concatenate("enum ",c_entity_local_name(var), " {",NULL));
      MAP(ENTITY,ent,
      { 
	result = strdup(concatenate(result,first?"":",",c_entity_local_name(ent),NULL));
	first = FALSE;
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  default:
  }
 
  return result? result: strdup("");
}

static string this_entity_clairedeclaration(entity var)
{
  string result = NULL;
  string name = entity_local_name(var);
  type t = entity_type(var);
  storage s = entity_storage(var);
  pips_debug(2,"Entity name : %s\n",entity_name(var));
  /*  Many possible combinations */

  if (strstr(name,TYPEDEF_PREFIX) != NULL)
    /* This is a typedef name, what about typedef int myint[5] ???  */
    return strdup(concatenate("typedef ", c_type_string(t),SPACE,c_entity_local_name(var),NULL));
  
  switch (storage_tag(s)) {
  case is_storage_rom: 
    {
      value va = entity_initial(var);
      if (!value_undefined_p(va))
	{
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
		  return result;
		}
	      /*What about real, double, string, ... ?*/
	    }
	}
      break;
    }
  case is_storage_ram: 
    {
      /*     ram r = storage_ram(s);
      entity sec = ram_section(r);
      if ((sec == CurrentSourceFileStaticArea) || (sec == CurrentStaticArea))
      result = "static ";*/
      break;
    }
  default: 
  }

  switch (type_tag(t)) {
  case is_type_variable:
    {
      variable v = type_variable(t);  
      string st, sd, svar, sq;
      value val = entity_initial(var);
      st = c_basic_string(variable_basic(v));
      sd = claire_dim_string(variable_dimensions(v));
      sq = c_qualifier_string(variable_qualifiers(v));
      svar = c_entity_local_name(var);
     
      /* problems with order !*/
      result = strdup(concatenate(sq, st, SPACE, svar, sd, NULL));
      if (!value_undefined_p(val))
	{
	  if (value_expression_p(val))
	    {
	      expression exp = value_expression(val);
	      if (brace_expression_p(exp))
		result = strdup(concatenate(result,"=",c_brace_expression_string(exp),NULL));
	      else 
		result = strdup(concatenate(result,"=",words_to_string(words_expression(exp)),NULL));
	    }
	}
      if (basic_bit_p(variable_basic(v)))
	{
	  int i = basic_bit(variable_basic(v));
	  pips_debug(2,"Basic bit %d",i);
	  result = strdup(concatenate(result,":",int_to_string(i),NULL));
	}
      free(st); free(sd); free(svar);
      break;
    }
  case is_type_struct:
    {
      list l = type_struct(t);
      result = strdup(concatenate("struct ",c_entity_local_name(var), "{", NL,NULL));
      MAP(ENTITY,ent,
      {
	string s = this_entity_clairedeclaration(ent);	    
	result = strdup(concatenate(result, s, SEMICOLON, NULL));
	free(s);
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  case is_type_union:
    {
      list l = type_union(t);
      result = strdup(concatenate("union ",c_entity_local_name(var), "{", NL,NULL));
      MAP(ENTITY,ent,
      {
	string s = this_entity_clairedeclaration(ent);	    
	result = strdup(concatenate(result, s, SEMICOLON, NULL));
	free(s);
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  case is_type_enum:
    {
      list l = type_enum(t);
      bool first = TRUE;
      result = strdup(concatenate("enum ",c_entity_local_name(var), " {",NULL));
      MAP(ENTITY,ent,
      { 
	result = strdup(concatenate(result,first?"":",",c_entity_local_name(ent),NULL));
	first = FALSE;
      },l);
      result = strdup(concatenate(result,"}", NULL));
      break;
    }
  default:
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
    debug(2, "\n Prettyprinter declaration for variable :",c_entity_local_name(var));   
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
  char * intrinsic;
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

static string ppt_unary_post(string in_c, list le)
{
  string e, result;
  pips_assert("one arg to unary call", gen_length(le)==1);
  e = c_expression(EXPRESSION(CAR(le)));
  result = strdup(concatenate(e, SPACE, in_c, NULL));
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

static struct s_ppt intrinsic_to_c[] = 
{
  { "+", "+", ppt_binary  },
  { "-", "-", ppt_binary },
  { "/", "/", ppt_binary },
  { "*", "*", ppt_binary },
  { "--", "-", ppt_unary },
  { "**", "**", ppt_binary },
  { "=", "=", ppt_binary },
  { ".OR.", "||", ppt_binary },
  { ".AND.", "&&", ppt_binary },
  { ".NOT.", "!", ppt_unary },
  { ".LT.", "<", ppt_binary },
  { ".GT.", ">", ppt_binary },
  { ".LE.", "<=", ppt_binary },
  { ".GE.", ">=", ppt_binary },
  { ".EQ.", "==", ppt_binary },
  { ".NE.", "!=", ppt_binary },
  { ".", ".", ppt_binary },
  { "->", "->", ppt_binary},
  { "post++", "++", ppt_unary_post },
  {"post--", "--" , ppt_unary_post },
  {"++pre", "++" , ppt_unary },
  {"--pre", "--" , ppt_unary },
  {"&", "&" , ppt_unary },
  {"*indirection", "*" , ppt_unary },
  {"+unary", "+", ppt_unary },
  {"-unary", "-", ppt_unary },
  {"~", "~", ppt_unary },
  {"!", "!", ppt_unary },  
  {"%", "%" , ppt_binary },  
  {"+C", "+" , ppt_binary },
  {"-C", "-", ppt_binary }, 
  {"<<", "<<", ppt_binary },
  {">>", ">>", ppt_binary }, 
  {"<", "<" , ppt_binary },
  {">", ">" , ppt_binary },
  {"<=", "<=", ppt_binary },
  {">=", ">=", ppt_binary }, 
  {"==", "==", ppt_binary },
  {"!=", "!=", ppt_binary },  
  {"&bitand", "&", ppt_binary}, 
  {"^", "^", ppt_binary },
  {"|", "|", ppt_binary },
  {"&&", "&&", ppt_binary }, 
  {"||", "||", ppt_binary },  
  {"*=", "*=", ppt_binary },
  {"/=", "/=", ppt_binary },
  {"%=", "%=", ppt_binary },
  {"+=", "+=", ppt_binary },
  {"-=", "-=", ppt_binary },
  {"<<=", "<<=" , ppt_binary },
  {">>=", ">>=", ppt_binary },
  {"&=", "&=", ppt_binary },
  {"^=", "^=", ppt_binary },
  {"|=","|=" , ppt_binary },
  { NULL, NULL, ppt_call }
};

/* return the prettyprinter structure for c.*/

static struct s_ppt * get_ppt(entity f)
{
  string called = entity_local_name(f);
  struct s_ppt * table = intrinsic_to_c;
  while (table->intrinsic && !same_string_p(called, table->intrinsic))
    table++;
  return table;
}

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

/* Attention with Fortran: the indexes are reversed. */
static string c_reference(reference r)
{
  string result = strdup(EMPTY), old, svar;
  MAP(EXPRESSION, e, 
  {
    string s = c_expression(e);
    
    old = result;
    result = strdup(concatenate(old, OPENBRACKET, s, CLOSEBRACKET, NULL));
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
  string result = "";
  /* build an arbitrary reverse trail of control nodes */
  list trail = unstructured_to_trail(u);
  list cc = NIL;
  trail = gen_nreverse(trail);
  ifdebug(3)
    {
      printf("Print trail: \n");
      dump_trail(trail);
    }
  /* Copy from text_trail ...*/
  for(cc=trail; !ENDP(cc); POP(cc)) 
    {
      control c = CONTROL(CAR(cc));
      string l = string_undefined;
      int nsucc = gen_length(control_successors(c));
      statement st = control_statement(c);
      ifdebug(3)
	{
	  printf("Processing statement:\n");
	  print_statement(st);
	}
      switch(nsucc) 
	{
	case 0:
	  {	  
	    printf("nsucc = 0 \n");
	    result = strdup(concatenate(result,c_statement(st),NULL));
	    break;
	  }
	case 1: 
	  {
	    control succ = CONTROL(CAR(control_successors(c)));
	    printf("nsucc = 1 \n");
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
	   
	    result = strdup(concatenate(result,c_statement(st),NULL));
	    if(statement_does_return(st))
	      {
		if(!ENDP(CDR(cc)))
		  {
		    control tsucc = CONTROL(CAR(CDR(cc)));
		    if(tsucc==succ) 
		      {
			break;
		      }
		  }
		/* A GOTO must be generated to reach the control successor */
		
		l = label_local_name(statement_label(control_statement(succ)));
		pips_assert("Must be labelled", l!= string_undefined);
		result = strdup(concatenate(result,"goto ",l,SEMICOLON,NULL));
	      }
	    break;
	  }
	case 2: 
	  {
	    control succ1 = CONTROL(CAR(control_successors(c)));
	    control succ2 = CONTROL(CAR(CDR(control_successors(c))));
	    instruction i = statement_instruction(st);
	    test t = instruction_test(i);
	    bool no_endif = FALSE;
	    string str = NULL;
	    printf("nsucc = 2 \n");
	    pips_assert("must be a test", instruction_test_p(i));

	    result = strdup(concatenate(result,"if (",c_expression(test_condition(t)), ") {", NL, NULL));
	    printf("Result = %s\n",result);
	 
	    /* Is there a textual successor? */
	    if(!ENDP(CDR(cc)))
	      {
		control tsucc = CONTROL(CAR(CDR(cc)));
		if(tsucc==succ1)
		  {
		    if(tsucc==succ2)
		      {
			/* This may happen after restructuring */
			printf("This may happen after restructuring\n");
			;
		      }
		    else 
		      {
			/* succ2 must be reached by GOTO */
			printf("succ2 must be reached by GOTO\n");
			l = label_local_name(statement_label(control_statement(succ2)));
			pips_assert("Must be labelled", l!= string_undefined);
			str = strdup(concatenate("}",NL, "else {",NL,"goto ", l, SEMICOLON,"}",NL,NULL));
			printf("str = %s\n",str);
		      }
		  }
		else 
		  {
		    if(tsucc==succ2)
		      {
			/* succ1 must be reached by GOTO */
			printf("succ1 must be reached by GOTO\n");
			l = label_local_name(statement_label(control_statement(succ1)));
			pips_assert("Must be labelled", l!= string_undefined);
			no_endif = TRUE;
		      }
		    else
		      {
			/* Both successors must be labelled */
			printf("Both successors must be labelled\n");
			l = label_local_name(statement_label(control_statement(succ1)));
			pips_assert("Must be labelled", l!= string_undefined);
			str = strdup(concatenate("goto ", l, SEMICOLON, "}", NL,"else {",NL,NULL));
			l = label_local_name(statement_label(control_statement(succ2)));
			pips_assert("Must be labelled", l!= string_undefined);	      
			str = strdup(concatenate(str,"goto ", l, SEMICOLON, NULL));
			printf("str = %s\n",str);
		      }
		  }
	      }
	    else
	      {
		/* Both successors must be textual predecessors */
		printf("Both successors must be textual predecessors \n");
		l = label_local_name(statement_label(control_statement(succ1)));
		pips_assert("Must be labelled", l!= string_undefined);
		str = strdup(concatenate("goto ", l, SEMICOLON, "}",NL,"else {",NL,NULL));
		l = label_local_name(statement_label(control_statement(succ2)));
		pips_assert("Must be labelled", l!= string_undefined);
		str = strdup(concatenate(str,"goto ", l, SEMICOLON, "}",NL, NULL));
		printf("str = %s\n",str);
	      }
	    
	    if(no_endif)
	      {
		printf("No endif\n");
		result = strdup(concatenate(result," goto ", l, SEMICOLON, "}",NL,NULL));
		printf("Result = %s\n",result);
	      }
	    printf("Result before = %s\n",result);
	    if (str != NULL)
	      {
		printf("str before = %s\n",str);
		result = strdup(concatenate(result,str,NULL));
	      }
	    printf("Result after = %s\n",result);
	    break;
	  }
	default:
	  pips_internal_error("Too many successors for a control node\n");
	}
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
  evaluation eval = whileloop_evaluation(w);
  /*do while and while do loops */
  if (evaluation_before_p(eval))
    result = strdup(concatenate("while (", cond, ") {" NL, 
				body, "}" NL, NULL));
  else   
    result = strdup(concatenate("do " NL, "{" NL, 
				body, "}" NL,"while (", cond, ");" NL, NULL));
  free(cond);
  free(body);
  return result;
}

static string c_forloop(forloop f)
{
  /* partial implementation... */
  string result;
  string body = c_statement(forloop_body(f));
  string init = c_expression(forloop_initialization(f));
  string cond = c_expression(forloop_condition(f));
  string inc = c_expression(forloop_increment(f));
  result = strdup(concatenate("for (", init, ";",cond,";",inc,") {" NL, 
			      body, "}" NL, NULL));
  
  free(inc);
  free(cond);
  free(init);
  free(body);
  return result;
}
/**************************************************************** STATEMENTS */

static string c_statement(statement s)
{
  string result;
  instruction i = statement_instruction(s);
  list l = statement_declarations(s);
  printf("\nCurrent statement : \n");
  print_statement(s);
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
    case is_instruction_forloop:
      {
	forloop f = instruction_forloop(i);
	result = c_forloop(f);
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

  if (!ENDP(l))
    {
      string decl = ""; 
      MAP(ENTITY, var,
      {
	string svar;
	debug(2, "\n In block declaration for variable :",c_entity_local_name(var));   
	svar = this_entity_cdeclaration(var);
	decl = strdup(concatenate(decl, svar, SEMICOLON, NULL));
	free(svar);
      },l);
      result = strdup(concatenate(decl,result,NULL));
    }

  return result;
}

static string claire_statement(statement s)
{
  string result;
  instruction i = statement_instruction(s);
  list l = statement_declarations(s);
  printf("\nCurrent statement : \n");
  print_statement(s);
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
    case is_instruction_forloop:
      {
	forloop f = instruction_forloop(i);
	result = c_forloop(f);
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

  if (!ENDP(l))
    {
      string decl = ""; 
      MAP(ENTITY, var,
      {
	string svar;
	debug(2, "\n In block declaration for variable :",c_entity_local_name(var));   
	svar = this_entity_clairedeclaration(var);
	decl = strdup(concatenate(decl, svar, SEMICOLON, NULL));
	free(svar);
      },l);
      result = strdup(concatenate(decl,result,NULL));
    }

  return result;
}


static string claire_code_string(entity module, statement stat)
{
  string before_head, head, decls, body, result;

  /* What about declarations that are external a module scope ?
     Consider a source file as a module entity, put all declarations in it 
     (external static + TOP-LEVEL) */

  /* before_head only generates the constant declarations, such as #define*/
  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

  before_head = c_declarations(module, parameter_p, NL, TRUE);
  head        = c_head(module);
  /* What about declarations associated to statements */
  decls       = c_declarations(module, variable_p, SEMICOLON, TRUE);
  body        = claire_statement(stat);
  
  result = strdup(concatenate(before_head, head, OPENBRACE, NL, 
			      decls, NL,
			      body, CLOSEBRACE, NL, NULL));

  free(before_head);
  free(head);
  free(decls);
  free(body);

  return result;
}

static string c_code_string(entity module, statement stat)
{
  string before_head, head, decls, body, result;

  /* What about declarations that are external a module scope ?
     Consider a source file as a module entity, put all declarations in it 
     (external static + TOP-LEVEL) */

  /* before_head only generates the constant declarations, such as #define*/
  ifdebug(2)
    {
      printf("Module statement: \n");
      print_statement(stat);
      printf("and declarations: \n");
      print_entities(statement_declarations(stat));
    }

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

bool print_claire_rough(string module_name)
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
  pips_debug(1, "Begin Claire prettyprrinter for %s\n", entity_name(module));
  ppt = claire_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();  

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "/* Claire pretty print for module %s. */\n%s", module_name, ppt);
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
bool print_claire_code(string module_name)
{
  string crough, cpretty, dir, cmd;

  pips_internal_error("Not implemented yet\n");

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
