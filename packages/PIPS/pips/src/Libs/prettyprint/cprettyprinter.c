/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*

   Try to prettyprint the RI in C.
   Very basic at the time.
   Functionnal.
   All arguments are assumed newly allocated.
   It might be really slow, but it should be safe.
   I should use some kind of string accumulator (array/list...)

   print_crough > MODULE.crough
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
#include "effects.h"

#include "resources.h"
#include "properties.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "effects-generic.h"
#include "pipsdbm.h"
#include "text-util.h"

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

// extension relative to pipsmake
#define INDENT		"indent"
#define CROUGH		".crough"
#define CPRETTY		".c"
#define INTERFACE	"_interface.f08"

// define an extension to append to scalar name in function signature
#define SCALAR_IN_SIG_EXT "_p4a_copy"

// define some C macro to support some fortran intrinsics
#define MAX_FCT    "crough_max"
#define MIN_FCT    "crough_min"
#define MAX_DEF    "#define " MAX_FCT "(a,b) (((a)>(b))?(a):(b))\n"
#define MIN_DEF    "#define " MIN_FCT "(a,b) (((a)<(b))?(a):(b))\n"
#define POW_PRE    "crough_"
#define POW_DEF    "#define " POW_PRE "powi(a,b) ((a)^(b))\n"
#define CMPLX_FCT  "init_complex"
#define CMPLX_DEF  "#define " CMPLX_FCT "(a,b) (a + b*I)\n"

/* forward declaration. */
static string c_expression(expression,bool);

// Create some list to keep track of scalar variable that are arguments of the
// function. They need to be renamed in the function signature, copy at
// function entrance and updated at fonction output if they have been written.
static list l_type    = NIL;
static list l_name    = NIL;
static list l_rename  = NIL;
static list l_entity  = NIL;
static list l_written = NIL;

/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
    (entity_function_p(get_current_module_entity()))


#define RESULT_NAME	"result"

/**
 * @brief test if the string looks like a REAL*8 (double in C) declaration
 * i.e something like 987987D54654 : a bunch of digit with a letter in the
 * middle. if yes convert it to C (i.e replace D by E) and return true
 */
static bool convert_double_value(char**str) {
  bool result = true;
  int match = 0;
  int i = 0;
  pips_debug (5, "test if str : %s is a double value. %c 0 = \n", *str, '0');
  for (i = 0; ((*str)[i] != '\0') && (result == true); i++) {
	bool cond = ((*str)[i] == 'D') && (match == 0);
	if (cond == true) {
	  match = i;
	  continue;
	}
	result &= (((*str)[i] >= '0') && ((*str)[i] <= '9')) || ((*str)[i] == '.') || (cond);
  }
  pips_debug (5, "end with i = %d, match = %d result = %s\n",
			  i, match, (result)?"true":"false");
  result &= ((*str)[i] == '\0') && (match + 1 != i) && (match != 0);
  if (result == true) {
	*str = strdup (*str);
	(*str)[match] = 'E';
  }
  return result;
}

/*
 * convert some fortran constant to their equivalent in C
 */
static void const_wrapper(char** s)
{
  static char * const_to_c[][2] = {	{ ".true." , "1" } , { ".false." , "0" }};
  static const int const_to_c_sz = sizeof(const_to_c)/sizeof(*const_to_c);
  int i;
  pips_debug (5, "constant to convert : %s\n", *s);
  if (convert_double_value (s) == false) {
	/* search fortran constant */
	char *name = strlower(strdup(*s),*s);
	for(i=0;i<const_to_c_sz;i++)
	  {
		if(strcmp(name,const_to_c[i][0]) == 0 )
        {
            free(*s);
            *s = strdup(const_to_c[i][1]);
            break;
        }
	  }
	free(name);
  }
  pips_debug (5, "constant converted : %s\n", *s);
}

/*
 * warning : return allocated string, otherwise it leads to modification (through strlower)
 * of critical entities
 */
static char* c_entity_local_name(entity var)
{
    const char* name;

    if (current_module_is_a_function() &&
            var != get_current_module_entity() &&
            same_string_p(entity_local_name(var), entity_local_name(get_current_module_entity()))
       )
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

        /* switch to lower cases... */

    }
    char *rname=strlower(strdup(name),name);
	pips_debug (5, "local name %s found\n", rname);
    return rname;
}

// build the list of written entity
static void build_written_list (list l) {
  list l_effects = l;
  FOREACH (EFFECT, eff,  l_effects) {
    if (effect_write_p (eff)) {
      entity e = effect_any_entity (eff);
      l_written = gen_entity_cons(e, l_written);
      pips_debug (5, "entity %s (%p) is written\n", entity_local_name (e), e);
    } else {
      entity e = effect_any_entity (eff);
      pips_debug (5, "entity %s (%p) is not written\n", entity_local_name (e), e);
    }
  }
}

//@return true if the entity is in the writen list
static bool written_p (entity e) {
  return gen_in_list_p (e, l_written);
}
/**************************************************** Function Pre and Postlude */

static string scalar_prelude () {
  string result = NULL;
  string previous = NULL;
  list t = l_type;
  list n = l_name;
  list r = l_rename;
  for (; n != NIL && r!= NIL && t!= NIL; n = n->cdr, r = r->cdr, t = t->cdr) {
    result = strdup (concatenate ((char*) gen_car (t), SPACE,
				  (string) gen_car (n), " = ", "*",
				  (string) gen_car (r), ";\n", previous,
				  NULL));
    if (previous != NULL) free (previous);
    previous = result;
  }
  return (result == NULL) ? strdup("") : result;
}

static string scalar_postlude () {
  string result = NULL;
  string previous = NULL;
  list n = l_name;
  list r = l_rename;
  list e = l_entity;

  for (; n != NIL && r != NIL && e != NIL; n = n->cdr, r = r->cdr, e = e->cdr) {
    if (written_p (gen_car (e))) {
      result = strdup (concatenate ("*", (string) gen_car (r), " = ",
				    (string) gen_car (n), ";\n", previous, NULL));
      if (previous != NULL) free (previous);
      previous = result;
      pips_debug (5, "entity %s (%p) restored\n",
		  entity_local_name(gen_car (e)),
		  gen_car (e));
    } else {
      pips_debug (5, "entity %s (%p) not restored\n",
		  entity_local_name(gen_car (e)),
		  gen_car (e));
    }
  }
  return (result == NULL) ? strdup("") : result;
}

/// @brief we want to decide if a scalar variable need to be
/// passed by pointer or by value to a C function.
/// Fortran77 assumes that all scalars are
/// passed by pointer. Starting With Fotran95,
/// the arguments can be passed by value by using interfaces.
/// @return true if the variable has to be passed by pointer
/// @param var, the variable to be test as an entity
static bool scalar_by_pointer (entity var) {
  // init result to false
  bool result = false;
  if ((get_bool_property ("CROUGH_FORTRAN_USES_INTERFACE") == false)) {
    // no interface, var is a scalar
    result = true;
  }
  else if ((get_bool_property ("CROUGH_FORTRAN_USES_INTERFACE") == true) &&
	   (get_bool_property ("CROUGH_SCALAR_BY_VALUE_IN_FCT_DECL") == false) &&
	   (written_p (var) == true)) {
    // interface exists but var is written (and user doesn't use the property
    // to force the passing of scalar by  value)
    result = true;
  }

  return result;
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
    default:
      pips_user_warning("case not handled yet \n");
    }
    return strdup(result);
}

// Convert the fortran type to its c string value
static string c_basic_string(basic b)
{
  const char* result = "UNKNOWN_BASIC" SPACE;
  char * aresult=NULL;
  bool user_type = get_bool_property ("CROUGH_USER_DEFINED_TYPE");
  switch (basic_tag(b)) {
  case is_basic_int: {
    pips_debug(2,"Basic int\n");
    if (user_type == false) {
      switch (basic_int(b)) {
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
    } else {
      result = get_string_property ("CROUGH_INTEGER_TYPE");
    }
    break;
  }
  case is_basic_float: {
    if (user_type == false) {
      switch (basic_float(b)){
      case 4: result = "float" SPACE;
	break;
      case 8: result = "double" SPACE;
	break;
      }
    } else {
      result = get_string_property ("CROUGH_REAL_TYPE");
    }
    break;
  }
  case is_basic_logical:
    result = "int" SPACE;
    break;
  case is_basic_string:
    result = "char" SPACE;
    break;
  case is_basic_bit:
    {
      /* An expression indeed... To be fixed... */
      _int i = (_int) basic_bit(b);
      pips_debug(2,"Bit field basic: %td\n", i);
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
      char* name = c_entity_local_name(ent);
      result = concatenate(c_type_string(t),name,NULL);
      free(name);
      break;
    }
  case is_basic_typedef:
    {
      entity ent = basic_typedef(b);
      aresult = c_entity_local_name(ent);
      break;
    }
  case is_basic_complex:
    result = "complex" SPACE; /* c99 style with include of complex.h*/
    break;
  default:
    pips_internal_error("unhandled case");
  }
  return aresult ? aresult : strdup(result);
}

/// @return a newly allocated string of the dimensions in C
/// @param ldim the variable dimension
/// @param fct_sig, set to true if the variable is part of a function signature
static string c_dim_string(list ldim, bool fct_sig)
{
    string result = "";
    if (ldim != NIL )
    {
        FOREACH(DIMENSION, dim,ldim)
        {
            expression elow = dimension_lower(dim);
            expression eup = dimension_upper(dim);
            intptr_t low;
            intptr_t up;
            string slow;
            string sup;

            /* In fact, the lower bound of array in C is always equal to 0,
               we only need to print (upper dimension + 1)
               but in order to handle Fortran code, we check all other possibilities
               and print (upper - lower + 1). Problem : the order of dimensions is reversed !!!! */
#if 1
            if (expression_integer_value(elow, &low))
            {
                if (low == 0)
                {
                    if (expression_integer_value(eup, &up))
                        result = strdup(concatenate(OPENBRACKET,i2a(up+1),CLOSEBRACKET,result,NULL));
                    else
                        /* to be refined here to make more beautiful expression */
                        result = strdup(concatenate(OPENBRACKET,
                                    words_to_string(words_expression(MakeBinaryCall(CreateIntrinsic("+"),
										    eup,int_to_expression(1)), NIL)),
                                    CLOSEBRACKET,result,NULL));
                }
                else
                {
		  if (expression_integer_value(eup, &up)) {
		    result = strdup(concatenate(OPENBRACKET,i2a(up-low+1),CLOSEBRACKET,result,NULL));
		  } else {
		    sup = words_to_string(words_expression(eup, NIL));
		    if (fct_sig == true) {
		      string tmp = NULL;
		      if (get_bool_property ("CROUGH_FORTRAN_USES_INTERFACE") == false) {
			tmp = strdup (concatenate ("(*",sup, SCALAR_IN_SIG_EXT, ")", NULL));
			free (sup);
			sup = tmp;
		      }
		    }
		    result = strdup(concatenate(OPENBRACKET,sup,"-",i2a(low-1),CLOSEBRACKET,result,NULL));
		    free(sup);
		  }
                }
            }
            else
#endif
            {
                slow = c_expression(elow,false);
                sup = c_expression(eup,false);
                result = strdup(concatenate(OPENBRACKET,sup,"-",slow,"+ 1",CLOSEBRACKET,result,NULL));
                free(slow);
                free(sup);
            }
        }
    }
    /* otherwise the list is empty, no dimension to declare */
    return strlower (strdup (result), result);
}

static string c_qualifier_string(list l)
{
    string result="";
    FOREACH(QUALIFIER,q,l)
    {
        switch (qualifier_tag(q)) {
            case is_qualifier_register:
                result = concatenate(result,"register ",NULL);
                break;
            case is_qualifier_thread:
                result = concatenate(result,"__thread ",NULL);
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
            case is_qualifier_auto:
                result = concatenate(result,"auto ",NULL);
            case is_qualifier_asm:
                result = concatenate(result,"__asm(",qualifier_asm(q),") ", NULL);
                break;
        }
    }
    return strdup(result);
}

static string c_brace_expression_string(expression exp)
{
    string result = "{";
    list args = call_arguments(syntax_call(expression_syntax(exp)));

    bool first = true;
    FOREACH (EXPRESSION,e,args)
    {
        if (brace_expression_p(e))
            result = strdup(concatenate(result,first?"":",",c_brace_expression_string(e),NULL));
        else
	  result = strdup(concatenate(result,first?"":",",words_to_string(words_expression(e, NIL)),NULL));
        first = false;
    }
    result = strdup(concatenate(result,"}",NULL));
    return result;
}

/// @param var, the variable to get the c declaration
/// @param fct_sig, set to true if the variable is part of a function signature
static string this_entity_cdeclaration(entity var, bool fct_sig)
{
    string result = NULL;
    //string name = entity_local_name(var);
    type t = entity_type(var);
    storage s = entity_storage(var);
    pips_debug(2,"Entity name : %s\n",entity_name(var));
    /*  Many possible combinations */

    /* This is a typedef name, what about typedef int myint[5] ???  */
    if (typedef_entity_p(var))
    {
        string tmp = NULL;
        tmp=c_entity_local_name(var);
        result = strdup(concatenate("typedef ", c_type_string(t),SPACE,tmp,NULL));
        free(tmp);
        return result;
    }

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
                            string sval = i2a(constant_int(c));
                            string svar = c_entity_local_name(var);
                            enum language_utype old_lang = get_prettyprint_language_tag();
                            set_prettyprint_language_tag (is_language_c);
                            string sbasic = basic_to_string(entity_basic(var));
                            set_prettyprint_language_tag(old_lang);
                            asprintf(&result,"static const %s %s = %s\n",sbasic,svar,sval);
                            free(sval);
                            free(svar);
                            free(sbasic);
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
            break;
    }

    switch (type_tag(t)) {
        case is_type_variable:
            {
                variable v = type_variable(t);
                string sptr, st, sd, svar, sq, ext;
                value val = entity_initial(var);
                st = c_basic_string(variable_basic(v));
                sd = c_dim_string(variable_dimensions(v), fct_sig);
                sq = c_qualifier_string(variable_qualifiers(v));
                svar = c_entity_local_name(var);

				// In the case of a signature check if the scalars need to
				// be passed by pointers. If the check return true
				// a "*" must be added
				if ((fct_sig == true) && (variable_dimensions(v) == NIL) &&
					(scalar_by_pointer (var) == true)) {
				  ext = SCALAR_IN_SIG_EXT;
				  sptr = "*";
				  l_type   = gen_string_cons(strdup(concatenate(sq, st, NULL)),
											 l_type);
				  l_name   = gen_string_cons(strdup(concatenate(svar, NULL)),
											 l_name);
				  l_rename = gen_string_cons(strdup(concatenate(svar,ext,NULL)),
											 l_rename);
				  l_entity = gen_entity_cons(var, l_entity);
				}
				// In case of a signature check if the arrays need to
				// be passed by pointers. If the check return true
				// a "*" must be added and the dim must be remove
				else if ((fct_sig == true) && (variable_dimensions(v) != NIL) &&
						 (get_bool_property("CROUGH_ARRAY_PARAMETER_AS_POINTER") == true)) {
				  ext = "";
				  sptr = "*";
				  free (sd);
				  sd = strdup ("");
				}
				else {
				  ext = "";
				  sptr = "";
				}


                /* problems with order !*/
                result = strdup(concatenate(sq, st, sptr, SPACE, svar, ext,
					    sd, NULL));
                free(svar);
                if (!value_undefined_p(val))
                {
                    if (value_expression_p(val))
                    {
                        expression exp = value_expression(val);
                        if (brace_expression_p(exp))
                            result = strdup(concatenate(result,"=",c_brace_expression_string(exp),NULL));
                        else
			  result = strdup(concatenate(result,"=",words_to_string(words_expression(exp, NIL)),NULL));
                    }
                }
                if (basic_bit_p(variable_basic(v)))
                {
                    /* It is an expression... */
                    _int i = (_int) basic_bit(variable_basic(v));
                    pips_debug(2,"Basic bit %td",i);
                    result = strdup(concatenate(result,":",i2a(i),NULL));
                    user_error("this_entity_cdeclaration",
                            "Bitfield to be finished...");
                }
                free(st);
                free(sd);
                break;
            }
        case is_type_struct:
            {
                list l = type_struct(t);
                string tmp =NULL;
                tmp = c_entity_local_name(var);
                result = strdup(concatenate("struct ",tmp, "{", NL,NULL));
                free(tmp);
                MAP(ENTITY,ent,
                        {
			string s = this_entity_cdeclaration(ent, fct_sig);
                        result = strdup(concatenate(result, s, SEMICOLON, NULL));
                        free(s);
                        },l);
                result = strdup(concatenate(result,"}", NULL));
                break;
            }
        case is_type_union:
            {
                list l = type_union(t);
                string tmp =NULL;
                tmp = c_entity_local_name(var);
                result = strdup(concatenate("union ",tmp, "{", NL,NULL));
                free(tmp);
                MAP(ENTITY,ent,
                        {
			  string s = this_entity_cdeclaration(ent, fct_sig);
                        result = strdup(concatenate(result, s, SEMICOLON, NULL));
                        free(s);
                        },l);
                result = strdup(concatenate(result,"}", NULL));
                break;
            }
        case is_type_enum:
            {
                list l = type_enum(t);
                bool first = true;
                string tmp = NULL;
                tmp = c_entity_local_name(var);
                result = strdup(concatenate("enum ", tmp, " {",NULL));
                free(tmp);
                MAP(ENTITY,ent,
                        {
                        tmp = c_entity_local_name(ent);
                        result = strdup(concatenate(result,first?"":",",tmp,NULL));
                        free(tmp);
                        first = false;
                        },l);
                result = strdup(concatenate(result,"}", NULL));
                break;
            }
        default:
            break;
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

static bool parameter_or_variable_p(entity e)
{
    return parameter_p(e) || variable_p(e);
}

/// @return true if the entity is an argument
static bool argument_p(entity e)
{
    /* Formal variables */
    return type_variable_p(entity_type(e)) &&
        storage_formal_p(entity_storage(e));
}

/// @return the string representation of the given declarations.
/// @param module, the module to get the declaration.
/// @param consider_this_entity, the function test pointer.
/// @param separator, the separatot to be used between vars.
/// @param lastsep, set to true if a final separator is needed.
/// @param fct_sig, set to true if in a function signature.
static string c_declarations(
        entity module,
        bool (*consider_this_entity)(entity),
        string separator,
        bool lastsep,
	bool fct_sig
        )
{
    string result = strdup("");
    code c;
    bool first = true;

    pips_assert("it is a code", value_code_p(entity_initial(module)));

    c = value_code(entity_initial(module));
    FOREACH(ENTITY, var,code_declarations(c))
    {
        string tmp = NULL;
        tmp = c_entity_local_name(var);
        pips_debug(2, "Prettyprinter declaration for variable :%s\n",tmp);
        free(tmp);
        if (consider_this_entity(var))
        {
            string old = result;
            string svar = this_entity_cdeclaration(var, fct_sig);
            pips_debug(6, "svar = %s\n", svar);
            result = strdup(concatenate(old, !first ? separator: "",
                        svar, NULL));
	    pips_debug(6, "result = %s\n", result);
            free(svar);
            free(old);
            first = false;
        }
    }
    // insert the last separtor if required and if at least one declaration
    // has been inserted.
    if (lastsep && !first)
      result = strdup(concatenate(result, separator, NULL));
    return result;
}
/******************************************************************* INCLUDE */
static string c_include (void) {
  string result = NULL;

  // add some c include files in order to support fortran intrinsic
  result = strdup (concatenate ("//needed include to compile the C output\n"
								"#include \"math.h\"\n",   // fabs
								"#include \"stdlib.h\"\n", // abs
								"#include \"complex.h\"\n", // abs
								"\n",
								NULL));

  // take care of include file required by the user
  const char* user_req = get_string_property ("CROUGH_INCLUDE_FILE_LIST");
  pips_debug (5, "including the user file list %s\n", user_req);
  string match = NULL;
  string tmp = strdup(user_req);
  match = strtok (tmp, " ,");
  while (match != NULL) {
	string old = result;
	pips_debug (7, "including the file %s\n", match);
	result = strdup (concatenate (result, "#include \"", match, "\"\n", NULL));
	match = strtok (NULL, " ,");
	free (old);
  }
  free (match);free(tmp);

  // user might use its own type that are define in a specific file
  bool user_type = get_bool_property ("CROUGH_USER_DEFINED_TYPE");
  pips_debug (5, "includind the user define type file %s\n", user_req);
  if (user_type == true) {
	string old = result;
	const char* f_name = get_string_property ("CROUGH_INCLUDE_FILE");
	pips_debug (7, "including the file %s\n", f_name);
	result = strdup (concatenate (result, "#include \"", f_name, "\"\n",
								  NULL));
	free (old);
  }
  pips_debug (5, "include string : %s\n", result);
  return result;
}

/********************************************************************* MACRO */
static string c_macro (void) {
  string result = NULL;
  // add some macro to support fortran intrinsics
  result = strdup (concatenate ("// The macros to support some fortran intrinsics\n",
								"// and complex declaration\n"
								MAX_DEF, MIN_DEF, POW_DEF, CMPLX_DEF, "\n",
								NULL));
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

    if (entity_main_module_p(module)) {
	    /* another kind : "int main(void)" ?*/
	    result = strdup(MAIN_DECLARATION);
    }
    else {
       string head, args, svar;
	   functional f = type_functional(entity_type(module));

	   /* define type head. */
	   if (get_bool_property ("DO_RETURN_TYPE_AS_TYPEDEF") == true) {
           head = strdup (get_string_property ("SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE"));
	   }
	   else if (entity_subroutine_p(module)) {
           head = strdup("void");
	   }
       else {
		   variable v;
           pips_assert("type of result is a variable",
					   type_variable_p(functional_result(f)));
           v = type_variable(functional_result(f));
           head = c_basic_string(variable_basic(v));
        }

        /* define args. */
        if (functional_parameters(f))
        {
            args = c_declarations(module, argument_p, ", ", false, true);
        }
        else
        {
            args = strdup("void");
        }

        svar = c_entity_local_name(module);
	if (get_bool_property("PRETTYPRINT_C_FUNCTION_NAME_WITH_UNDERSCORE"))

	  result = strdup(concatenate(head, SPACE, svar, "_",
				      OPENPAREN, args, CLOSEPAREN, NL, NULL));

	else
	  result = strdup(concatenate(head, SPACE, svar,
				      OPENPAREN, args, CLOSEPAREN, NL, NULL));

        free(svar);
        free(head);
        free(args);
    }

    return result;
}

/*************************************************************** EXPRESSIONS */

/* generate a basic c expression.
   no operator priority is assumed...
   */
typedef string (*prettyprinter)(const char*, list);

struct s_ppt
{
    char * intrinsic;
    char * c;
    prettyprinter ppt;
};

// Define a struct to easily find the function full name according to its
// base_name. Basically some letters are prepend or append according to the
// size ant type of the opperand. For example abs can become absl or fabsf.
typedef struct
{
  char * c_base_name;
  enum basic_utype type;
  intptr_t size;
  char* prefix;
  char* suffix;
} c_full_name;

static bool expression_needs_parenthesis_p(expression);

static string ppt_binary(const char* in_c, list le)
{
    string result;
    expression e1, e2;
    string s1, s2;
    bool p1, p2;

    pips_assert("2 arguments to binary call", gen_length(le)==2);

    e1 = EXPRESSION(CAR(le));
    p1 = expression_needs_parenthesis_p(e1);
    s1 = c_expression(e1,false);

    e2 = EXPRESSION(CAR(CDR(le)));
    p2 = expression_needs_parenthesis_p(e2);
    s2 = c_expression(e2,false);

    result = strdup(concatenate(p1? OPENPAREN: EMPTY, s1, p1? CLOSEPAREN: EMPTY,
                SPACE, in_c, SPACE,
                p2? OPENPAREN: EMPTY, s2, p2? CLOSEPAREN: EMPTY,
                NULL));

    free(s1);
    //free(s2);

    return result;
}

static string ppt_unary(const char* in_c, list le)
{
    string e, result;
    pips_assert("one arg to unary call", gen_length(le)==1);
    e = c_expression(EXPRESSION(CAR(le)),false);
    result = strdup(concatenate(in_c, SPACE, e, NULL));
    free(e);
    return result;
}

static string ppt_unary_post(const char* in_c, list le)
{
    string e, result;
    pips_assert("one arg to unary call", gen_length(le)==1);
    e = c_expression(EXPRESSION(CAR(le)),false);
    result = strdup(concatenate(e, SPACE, in_c, NULL));
    free(e);
    return result;
}

/* SG: PBM spotted HERE */
static string ppt_call(const char* in_c, list le)
{
    string scall, old;
	bool pointer = !get_bool_property ("CROUGH_SCALAR_BY_VALUE_IN_FCT_CALL");
    if (le == NIL)
    {
        scall = strdup(concatenate(in_c, "()", NULL));
    }
    else
    {
        bool first = true;
        scall = strdup(concatenate(in_c, OPENPAREN, NULL));

        /* Attention: not like this for io statements*/
        FOREACH (EXPRESSION, e, le)
        {
            string arg = c_expression(e,false);
            old = scall;
            scall = strdup(concatenate(old, first? "" : ", ",
				       expression_scalar_p(e) && pointer ? "&" : "",
				       arg, NULL));
            free(arg);
            free(old);
            first = false;
        }

        old = scall;
        scall = strdup(concatenate(old, CLOSEPAREN, NULL));
        free(old);
    }
    return scall;
}

static c_full_name c_base_name_to_c_full_name [] = {
  {"abs"   , is_basic_int     , 1  , ""     , "" }, //char
  {"abs"   , is_basic_int     , 2  , ""     , "" }, //short
  {"abs"   , is_basic_int     , 4  , ""     , "" }, //int
  {"abs"   , is_basic_int     , 6  , "l"    , "" }, //long
  {"abs"   , is_basic_int     , 8  , "ll"   , "" }, //long long
  {"abs"   , is_basic_float   , 4  , "f"    , "f"}, //float
  {"abs"   , is_basic_float   , 8  , "f"    , "" }, //double
  {"abs"   , is_basic_complex , 8  , "c"    , "f"}, //float complex
  {"abs"   , is_basic_complex , 16 , "c"    , "" }, //double complex
  {"pow"   , is_basic_int     , 1  , POW_PRE, "i"}, //char
  {"pow"   , is_basic_int     , 2  , POW_PRE, "i"}, //short
  {"pow"   , is_basic_int     , 4  , POW_PRE, "i"}, //int
  {"pow"   , is_basic_int     , 6  , POW_PRE, "i"}, //long
  {"pow"   , is_basic_int     , 8  , POW_PRE, "i"}, //long long
  {"pow"   , is_basic_float   , 4  , ""     , "f"}, //float
  {"pow"   , is_basic_float   , 8  , ""     , "" }, //double
  {"pow"   , is_basic_complex , 8  , "c"    , "f"}, //float complex
  {"pow"   , is_basic_complex , 16 , "c"    , "" }, //double complex
  {NULL    , is_basic_int     , 0  , ""     , "" }
};

/// @brief fill the c_base_name to get the c full name accorgind to its basic
static void get_c_full_name (string* base_in_c, basic b) {
  pips_debug (7, "find the C function for \"%s\" according to the basic\n",
			  *base_in_c);
  pips_assert ("cant deal with basic undefined", b != basic_undefined);
  // initialize some varaibles
  c_full_name * table = c_base_name_to_c_full_name;
  enum basic_utype type = basic_tag (b);
  intptr_t size = basic_type_size (b);

  // find the correct row
  while ((table->c_base_name != NULL) &&
		 !(same_string_p(*base_in_c, table->c_base_name) &&
		   (table->type == type) &&
		   (table->size == size)))
	table++;
  if (table->c_base_name == NULL) {
    pips_internal_error("can not determin the c function to call");
  }
  str_append  (base_in_c, table->suffix);
  str_prepend (base_in_c, table->prefix);
  return;
}

// fortran intrinsic accepts different types but c function only
// accept one type. This type of intrinsic is handle by this ppt_math
// function, it calls the right c function according to its input types.
static string ppt_math(const char* in_c, list le)
{
  basic res_basic = basic_undefined;
  pips_assert ("need at least one argument", 0 != gen_length (le));
  FOREACH (EXPRESSION, exp, le) {
	pips_debug (7, "let's analyse the expression to find the involved types\n");
	type tmp = expression_to_type (exp);
	pips_assert ("type must be a variable", type_variable_p (tmp) == true);
	basic cur_b = variable_basic (type_variable (tmp));
	pips_assert ("expression_to_type returns a basic undefined",
				 cur_b != basic_undefined);
	if (res_basic == basic_undefined) {
	  res_basic = copy_basic (cur_b);
	}
	else {
	  basic old = res_basic;
	  res_basic = basic_maximum (old, cur_b);
	  free_basic (old);
	  pips_assert ("expression_to_type returns a basic undefined",
		       !basic_overloaded_p (res_basic));
	}
	free_type (tmp);
  }
  string str_copy = strdup (in_c);
  get_c_full_name (&str_copy, res_basic);
  string result = ppt_call (str_copy, le);
  if (res_basic != basic_undefined)
	free_basic (res_basic);
  free (str_copy);
  return result;
}

// fortran min and max intrinsic accept from 2 to n elements. This can be done
// in c using an ellipse or using a simple macro. The second possibility is
// chosen
static string ppt_min_max (const char* in_c, list le)
{
  bool flag = false;
  bool pointer = !get_bool_property ("CROUGH_SCALAR_BY_VALUE_IN_FCT_CALL");
  expression exp = EXPRESSION (CAR (le));
  string arg = c_expression (exp, false);
  string result = strdup(concatenate ((expression_scalar_p(exp) &&
									   pointer)? "&" : "", arg, NULL));
  POP (le);
  free (arg);

  FOREACH (EXPRESSION, e, le){
	arg = c_expression(e,false);
	string old = result;
	result = strdup(concatenate(in_c , OPENPAREN, old, ", ",
								expression_scalar_p(e) && pointer ? "&" : "",
								arg, CLOSEPAREN, NULL));
	free(arg);
	free(old);
	flag = true;
  }

  pips_assert ("min and max should have at least 2 arguments", flag == true);
  return result;
}

// @brief Generate a pips_user_error for intrinsic that can not be handle
// right now according to the property defined by the user
///@param in_f, the instrinsic in fortran
static string ppt_unknown(const char* in_f, list le)
{
  if (get_bool_property ("CROUGH_PRINT_UNKNOWN_INTRINSIC") == false)
	pips_user_error ("This intrinsic can not be tranbslated in c: %s\n", in_f);
  string result = ppt_call (in_f, le);
  return result;
}

// @brief Generate a pips_user_error for intrinsic that must not be fined in a
// fortran code
///@param in_f, the instrinsic in fortran
static string ppt_must_error(const char* in_f, list le)
{
  string result = strdup ("");
  pips_user_error("This intrinsic should not be found in a fortran code: %s\n",
				  in_f);
  return result;
}

static struct s_ppt intrinsic_to_c[] = {
  { "+"                        , "+"                         , ppt_binary    },
  { "-"                        , "-"                         , ppt_binary    },
  { "/"                        , "/"                         , ppt_binary    },
  { "*"                        , "*"                         , ppt_binary    },
  { "--"                       , "-"                         , ppt_unary     },
  { "="                        , "="                         , ppt_binary    },
  { ".OR."                     , "||"                        , ppt_binary    },
  { ".AND."                    , "&&"                        , ppt_binary    },
  { ".NOT."                    , "!"                         , ppt_unary     },
  { ".LT."                     , "<"                         , ppt_binary    },
  { ".GT."                     , ">"                         , ppt_binary    },
  { ".LE."                     , "<="                        , ppt_binary    },
  { ".GE."                     , ">="                        , ppt_binary    },
  { ".EQ."                     , "=="                        , ppt_binary    },
  { ".EQV."                    , "=="                        , ppt_binary    },
  { ".NE."                     , "!="                        , ppt_binary    },
  { "."                        , "."                         , ppt_binary    },
  { "->"                       , "->"                        , ppt_binary    },
  { "post++"                   , "++"                        , ppt_unary_post},
  {"post--"                    , "--"                        , ppt_unary_post},
  {"++pre"                     , "++"                        , ppt_unary     },
  {"--pre"                     , "--"                        , ppt_unary     },
  {"&"                         , "&"                         , ppt_unary     },
  {"*indirection"              , "*"                         , ppt_unary     },
  {"+unary"                    , "+"                         , ppt_unary     },
  {"-unary"                    , "-"                         , ppt_unary     },
  {"~"                         , "~"                         , ppt_unary     },
  {"!"                         , "!"                         , ppt_unary     },
  {PLUS_C_OPERATOR_NAME        , PLUS_C_OPERATOR_NAME        , ppt_must_error},
  {MINUS_C_OPERATOR_NAME       , MINUS_C_OPERATOR_NAME       , ppt_must_error},
  {"<<"                        , "<<"                        , ppt_binary    },
  {">>"                        , ">>"                        , ppt_binary    },
  {"<"                         , "<"                         , ppt_binary    },
  {">"                         , ">"                         , ppt_binary    },
  {"<="                        , "<="                        , ppt_binary    },
  {">="                        , ">="                        , ppt_binary    },
  {"=="                        , "=="                        , ppt_binary    },
  {"!="                        , "!="                        , ppt_binary    },
  {"&bitand"                   , "&"                         , ppt_binary    },
  {"^"                         , "^"                         , ppt_binary    },
  {"|"                         , "|"                         , ppt_binary    },
  {"&&"                        , "&&"                        , ppt_binary    },
  {C_OR_OPERATOR_NAME          , C_OR_OPERATOR_NAME          , ppt_must_error},
  {"*="                        , "*="                        , ppt_binary    },
  {"/="                        , "/="                        , ppt_binary    },
  {"%="                        , "%="                        , ppt_binary    },
  {"+="                        , "+="                        , ppt_binary    },
  {"-="                        , "-="                        , ppt_binary    },
  {"<<="                       , "<<="                       , ppt_binary    },
  {">>="                       , ">>="                       , ppt_binary    },
  {"&="                        , "&="                        , ppt_binary    },
  {"^="                        , "^="                        , ppt_binary    },
  {"|="                        , "|="                        , ppt_binary    },
  {POWER_OPERATOR_NAME         , "pow"                       , ppt_math      },
  {MODULO_OPERATOR_NAME        , "%"                         , ppt_binary    },
  {ABS_OPERATOR_NAME           , "abs"                       , ppt_math      },
  {IABS_OPERATOR_NAME          , "abs"                       , ppt_call      },
  {DABS_OPERATOR_NAME          , "fabs"                      , ppt_call      },
  {CABS_OPERATOR_NAME          , "cabsf"                     , ppt_call      },
  {CDABS_OPERATOR_NAME         , "cabs"                      , ppt_call      },
  {WRITE_FUNCTION_NAME         , WRITE_FUNCTION_NAME         , ppt_unknown   },
  {PRINT_FUNCTION_NAME         , PRINT_FUNCTION_NAME         , ppt_unknown   },
  {REWIND_FUNCTION_NAME        , REWIND_FUNCTION_NAME        , ppt_unknown   },
  {OPEN_FUNCTION_NAME          , OPEN_FUNCTION_NAME          , ppt_unknown   },
  {CLOSE_FUNCTION_NAME         , CLOSE_FUNCTION_NAME         , ppt_unknown   },
  {INQUIRE_FUNCTION_NAME       , INQUIRE_FUNCTION_NAME       , ppt_unknown   },
  {BACKSPACE_FUNCTION_NAME     , BACKSPACE_FUNCTION_NAME     , ppt_unknown   },
  {READ_FUNCTION_NAME          , READ_FUNCTION_NAME          , ppt_unknown   },
  {BUFFERIN_FUNCTION_NAME      , BUFFERIN_FUNCTION_NAME      , ppt_unknown   },
  {BUFFEROUT_FUNCTION_NAME     , BUFFEROUT_FUNCTION_NAME     , ppt_unknown   },
  {ENDFILE_FUNCTION_NAME       , ENDFILE_FUNCTION_NAME       , ppt_unknown   },
  {FORMAT_FUNCTION_NAME        , FORMAT_FUNCTION_NAME        , ppt_unknown   },
  {MIN_OPERATOR_NAME           , MIN_FCT                     , ppt_min_max   },
  {MIN0_OPERATOR_NAME          , MIN_FCT                     , ppt_min_max   },
  {MIN1_OPERATOR_NAME          , MIN_FCT                     , ppt_min_max   }, // implicit cast
  {AMIN0_OPERATOR_NAME         , MIN_FCT                     , ppt_min_max   }, // implicit cast
  {AMIN1_OPERATOR_NAME         , MIN_FCT                     , ppt_min_max   },
  {DMIN1_OPERATOR_NAME         , MIN_FCT                     , ppt_min_max   },
  {MAX_OPERATOR_NAME           , MAX_FCT                     , ppt_min_max   },
  {MAX0_OPERATOR_NAME          , MAX_FCT                     , ppt_min_max   }, // implicit cast
  {AMAX0_OPERATOR_NAME         , MAX_FCT                     , ppt_min_max   }, // implicit cast
  {MAX1_OPERATOR_NAME          , MAX_FCT                     , ppt_min_max   },
  {AMAX1_OPERATOR_NAME         , MAX_FCT                     , ppt_min_max   },
  {DMAX1_OPERATOR_NAME         , MAX_FCT                     , ppt_min_max   },
  {IMPLIED_COMPLEX_NAME        , CMPLX_FCT                   , ppt_call      },
  {IMPLIED_DCOMPLEX_NAME       , CMPLX_FCT                   , ppt_call      },
  {SIGN_OPERATOR_NAME          , SIGN_OPERATOR_NAME          , ppt_unknown   },
  {ISIGN_OPERATOR_NAME         , ISIGN_OPERATOR_NAME         , ppt_unknown   },
  {DSIGN_OPERATOR_NAME         , DSIGN_OPERATOR_NAME         , ppt_unknown   },
  {DIM_OPERATOR_NAME           , DIM_OPERATOR_NAME           , ppt_unknown   },
  {IDIM_OPERATOR_NAME          , IDIM_OPERATOR_NAME          , ppt_unknown   },
  {DDIM_OPERATOR_NAME          , DDIM_OPERATOR_NAME          , ppt_unknown   },
  {DPROD_OPERATOR_NAME         , DPROD_OPERATOR_NAME         , ppt_unknown   },
  {CONJG_OPERATOR_NAME         , CONJG_OPERATOR_NAME         , ppt_unknown   },
  {DCONJG_OPERATOR_NAME        , DCONJG_OPERATOR_NAME        , ppt_unknown   },
  {SQRT_OPERATOR_NAME          , SQRT_OPERATOR_NAME          , ppt_unknown   },
  {DSQRT_OPERATOR_NAME         , DSQRT_OPERATOR_NAME         , ppt_unknown   },
  {CSQRT_OPERATOR_NAME         , CSQRT_OPERATOR_NAME         , ppt_unknown   },
  {CDSQRT_OPERATOR_NAME        , CDSQRT_OPERATOR_NAME        , ppt_unknown   },
  {EXP_OPERATOR_NAME           , EXP_OPERATOR_NAME           , ppt_unknown   },
  {DEXP_OPERATOR_NAME          , DEXP_OPERATOR_NAME          , ppt_unknown   },
  {CEXP_OPERATOR_NAME          , CEXP_OPERATOR_NAME          , ppt_unknown   },
  {CDEXP_OPERATOR_NAME         , CDEXP_OPERATOR_NAME         , ppt_unknown   },
  {LOG_OPERATOR_NAME           , LOG_OPERATOR_NAME           , ppt_unknown   },
  {ALOG_OPERATOR_NAME          , ALOG_OPERATOR_NAME          , ppt_unknown   },
  {DLOG_OPERATOR_NAME          , DLOG_OPERATOR_NAME          , ppt_unknown   },
  {CLOG_OPERATOR_NAME          , CLOG_OPERATOR_NAME          , ppt_unknown   },
  {CDLOG_OPERATOR_NAME         , CDLOG_OPERATOR_NAME         , ppt_unknown   },
  {LOG10_OPERATOR_NAME         , LOG10_OPERATOR_NAME         , ppt_unknown   },
  {ALOG10_OPERATOR_NAME        , ALOG10_OPERATOR_NAME        , ppt_unknown   },
  {DLOG10_OPERATOR_NAME        , DLOG10_OPERATOR_NAME        , ppt_unknown   },
  {SIN_OPERATOR_NAME           , SIN_OPERATOR_NAME           , ppt_unknown   },
  {DSIN_OPERATOR_NAME          , DSIN_OPERATOR_NAME          , ppt_unknown   },
  {CSIN_OPERATOR_NAME          , CSIN_OPERATOR_NAME          , ppt_unknown   },
  {CDSIN_OPERATOR_NAME         , CDSIN_OPERATOR_NAME         , ppt_unknown   },
  {COS_OPERATOR_NAME           , COS_OPERATOR_NAME           , ppt_unknown   },
  {DCOS_OPERATOR_NAME          , DCOS_OPERATOR_NAME          , ppt_unknown   },
  {CCOS_OPERATOR_NAME          , CCOS_OPERATOR_NAME          , ppt_unknown   },
  {CDCOS_OPERATOR_NAME         , CDCOS_OPERATOR_NAME         , ppt_unknown   },
  {TAN_OPERATOR_NAME           , TAN_OPERATOR_NAME           , ppt_unknown   },
  {DTAN_OPERATOR_NAME          , DTAN_OPERATOR_NAME          , ppt_unknown   },
  {ASIN_OPERATOR_NAME          , ASIN_OPERATOR_NAME          , ppt_unknown   },
  {DASIN_OPERATOR_NAME         , DASIN_OPERATOR_NAME         , ppt_unknown   },
  {ACOS_OPERATOR_NAME          , ACOS_OPERATOR_NAME          , ppt_unknown   },
  {DACOS_OPERATOR_NAME         , DACOS_OPERATOR_NAME         , ppt_unknown   },
  {ATAN_OPERATOR_NAME          , ATAN_OPERATOR_NAME          , ppt_unknown   },
  {DATAN_OPERATOR_NAME         , DATAN_OPERATOR_NAME         , ppt_unknown   },
  {ATAN2_OPERATOR_NAME         , ATAN2_OPERATOR_NAME         , ppt_unknown   },
  {DATAN2_OPERATOR_NAME        , DATAN2_OPERATOR_NAME        , ppt_unknown   },
  {SINH_OPERATOR_NAME          , SINH_OPERATOR_NAME          , ppt_unknown   },
  {DSINH_OPERATOR_NAME         , DSINH_OPERATOR_NAME         , ppt_unknown   },
  {COSH_OPERATOR_NAME          , COSH_OPERATOR_NAME          , ppt_unknown   },
  {DCOSH_OPERATOR_NAME         , DCOSH_OPERATOR_NAME         , ppt_unknown   },
  {TANH_OPERATOR_NAME          , TANH_OPERATOR_NAME          , ppt_unknown   },
  {DTANH_OPERATOR_NAME         , DTANH_OPERATOR_NAME         , ppt_unknown   },
  {LENGTH_OPERATOR_NAME        , LENGTH_OPERATOR_NAME        , ppt_unknown   },
  {INDEX_OPERATOR_NAME         , INDEX_OPERATOR_NAME         , ppt_unknown   },
  {LGE_OPERATOR_NAME           , LGE_OPERATOR_NAME           , ppt_unknown   },
  {LGT_OPERATOR_NAME           , LGT_OPERATOR_NAME           , ppt_unknown   },
  {LLE_OPERATOR_NAME           , LLE_OPERATOR_NAME           , ppt_unknown   },
  {LLT_OPERATOR_NAME           , LLT_OPERATOR_NAME           , ppt_unknown   },
  {AINT_CONVERSION_NAME        , AINT_CONVERSION_NAME        , ppt_unknown   },
  {DINT_CONVERSION_NAME        , DINT_CONVERSION_NAME        , ppt_unknown   },
  {ANINT_CONVERSION_NAME       , ANINT_CONVERSION_NAME       , ppt_unknown   },
  {DNINT_CONVERSION_NAME       , DNINT_CONVERSION_NAME       , ppt_unknown   },
  {NINT_CONVERSION_NAME        , NINT_CONVERSION_NAME        , ppt_unknown   },
  {IDNINT_CONVERSION_NAME      , IDNINT_CONVERSION_NAME      , ppt_unknown   },
  {AIMAG_CONVERSION_NAME       , AIMAG_CONVERSION_NAME       , ppt_unknown   },
  {DIMAG_CONVERSION_NAME       , DIMAG_CONVERSION_NAME       , ppt_unknown   },
  {INT_GENERIC_CONVERSION_NAME   , INT_GENERIC_CONVERSION_NAME   , ppt_unknown   },
  {IFIX_GENERIC_CONVERSION_NAME  , IFIX_GENERIC_CONVERSION_NAME  , ppt_unknown   },
  {IDINT_GENERIC_CONVERSION_NAME , IDINT_GENERIC_CONVERSION_NAME , ppt_unknown   },
  {REAL_GENERIC_CONVERSION_NAME  , REAL_GENERIC_CONVERSION_NAME  , ppt_unknown   },
  {FLOAT_GENERIC_CONVERSION_NAME , FLOAT_GENERIC_CONVERSION_NAME , ppt_unknown   },
  {DFLOAT_GENERIC_CONVERSION_NAME, DFLOAT_GENERIC_CONVERSION_NAME, ppt_unknown   },
  {SNGL_GENERIC_CONVERSION_NAME  , SNGL_GENERIC_CONVERSION_NAME  , ppt_unknown   },
  {DBLE_GENERIC_CONVERSION_NAME  , DBLE_GENERIC_CONVERSION_NAME  , ppt_unknown   },
  {DREAL_GENERIC_CONVERSION_NAME , DREAL_GENERIC_CONVERSION_NAME , ppt_unknown   },
  {CMPLX_GENERIC_CONVERSION_NAME , CMPLX_GENERIC_CONVERSION_NAME , ppt_unknown   },
  {DCMPLX_GENERIC_CONVERSION_NAME, DCMPLX_GENERIC_CONVERSION_NAME, ppt_unknown   },
  {INT_TO_CHAR_CONVERSION_NAME   , INT_TO_CHAR_CONVERSION_NAME   , ppt_unknown   },
  {CHAR_TO_INT_CONVERSION_NAME   , CHAR_TO_INT_CONVERSION_NAME   , ppt_unknown   },
  {NULL                        , NULL                        , ppt_call      }
};

/* return the prettyprinter structure for c.*/

static struct s_ppt * get_ppt(entity f)
{
    const char* called = entity_local_name(f);
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
            return false;
    }
}

#define RET	"return"
#define CONT "continue"

static string c_call(call c,bool breakable)
{
    entity called = call_function(c);
    struct s_ppt * ppt = get_ppt(called);
    char* local_name = strdup(entity_local_name(called));
    string result = NULL;

    /* special case... */
    if (same_string_p(local_name, "RETURN")) {
      string copy_out = scalar_postlude ();
      if (entity_main_module_p(get_current_module_entity()))
	result = RET " 0";
      else if (current_module_is_a_function())
	result = RET SPACE RESULT_NAME;
      else
	result = RET;
      result = strdup(concatenate (copy_out, result, NULL));
      free (copy_out);
    }
    else if (same_string_p(local_name, "CONTINUE") )
    {
        result = breakable?strdup(CONT):strdup("");
    }
    else if (call_constant_p(c))
    {
        const_wrapper(&local_name);
        result = strlower(strdup(local_name),local_name);
    }
    else
    {
        result = ppt->ppt(ppt->c? ppt->c: local_name, call_arguments(c));
        string tmp = result;
        result=strlower(strdup(result),result);
        free(tmp);
    }
    //free(local_name);

    return result;
}

/* Attention with Fortran: the indexes are reversed.
   And array dimensions in C always rank from 0. BC.
*/
static string c_reference(reference r)
{
    string result = strdup(EMPTY), old, svar;

    list l_dim = variable_dimensions(type_variable(ultimate_type(entity_type(reference_variable(r)))));

    FOREACH (EXPRESSION, e,reference_indices(r)) {
      expression e_tmp;
      expression e_lower = dimension_lower(DIMENSION(CAR(l_dim)));
      string s;
      intptr_t itmp;

      if( !expression_equal_integer_p(e_lower, 0))
	e_tmp =
	  MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
			 copy_expression(e),
			 copy_expression(e_lower));
      else
	e_tmp = copy_expression(e);

      if(expression_integer_value(e_tmp, &itmp))
	s = i2a(itmp);
      else
	s = c_expression( e_tmp,false);

      old = result;
      result = strdup(concatenate(OPENBRACKET, s, CLOSEBRACKET,old, NULL));
      //free(old);
      //free(s);
      free_expression(e_tmp);
      POP(l_dim);
    }


    old = result;
    svar = c_entity_local_name(reference_variable(r));
    result = strdup(concatenate(svar, old, NULL));
    free(old);
    free(svar);
    return result;
}

static string c_expression(expression e,bool breakable)
{
    string result = NULL;
    syntax s = expression_syntax(e);
    switch (syntax_tag(s))
    {
        case is_syntax_call:
            result = c_call(syntax_call(s),breakable);
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

static string c_statement(statement s, bool breakable);

static string c_unstructured(unstructured u,bool breakable)
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
        const char* l;
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
                    result = strdup(concatenate(result,c_statement(st,false),NULL));
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

                    result = strdup(concatenate(result,c_statement(st,false),NULL));
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
                    bool no_endif = false;
                    string str = NULL;
                    printf("nsucc = 2 \n");
                    pips_assert("must be a test", instruction_test_p(i));

                    result = strdup(concatenate(result,"if (",c_expression(test_condition(t),breakable), ") {", NL, NULL));
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
                                no_endif = true;
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
                pips_internal_error("Too many successors for a control node");
        }
    }

    gen_free_list(trail);
    return result;
}

static string c_test(test t,bool breakable)
{
    string result;
    bool no_false;
    string cond, strue, sfalse;
    cond = c_expression(test_condition(t),breakable);
    strue = c_statement(test_true(t),breakable);
    no_false = empty_statement_p(test_false(t));

    sfalse = no_false? NULL: c_statement(test_false(t),false);

    result = strdup(concatenate("if (", cond, ") {" NL,
                strue,
                no_false? "}" NL: "} else {" NL,
                sfalse, "}" NL, NULL));
    free(cond);
    free(strue);
    if (sfalse) free(sfalse);
    return result;
}

static string c_sequence(sequence seq, bool breakable)
{
    string result = strdup(EMPTY);
    FOREACH (STATEMENT, s, sequence_statements(seq))
    {
        string oldresult = result;
        string current = c_statement(s,breakable);
        result = strdup(concatenate(oldresult, current, NULL));
        free(current);
        free(oldresult);
    }
    return result;
}

static string c_loop(loop l)
{
    /* partial implementation...
       However, there is not this kind of loop in C */
    string result;
    string body = c_statement(loop_body(l),true);
    string index = c_entity_local_name(loop_index(l));
    range r = loop_range(l);
    string low = c_expression(range_lower(r),true);
    string up = c_expression(range_upper(r),true);
    string theincr = c_expression(range_increment(r),true);
    string incr = 0;
    if( strcmp(theincr,"1")==0 )
      incr = strdup("++");
    else
      incr = strdup(concatenate( "+=", theincr , NULL ));
    free(theincr);
   /* what about step*/
    result = strdup(concatenate("for (", index, "=", low, "; ",
                index, "<=", up, "; ",
                index,  incr, ")", SPACE, OPENBRACE, NL,
                body, CLOSEBRACE, NL, NULL));
    free(body);
    free(index);
    free(incr);
    // TODO: There are some allocation bugs in c_expression()
    //free(low);
    //free(up);
    return result;
}


static string c_whileloop(whileloop w)
{
    /* partial implementation... */
    string result;
    string body = c_statement(whileloop_body(w),true);
    string cond = c_expression(whileloop_condition(w),true);
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
    string body = c_statement(forloop_body(f),true);
    string init = c_expression(forloop_initialization(f),true);
    string cond = c_expression(forloop_condition(f),true);
    string inc = c_expression(forloop_increment(f),true);
    result = strdup(concatenate("for (", init, ";",cond,";",inc,") {" NL,
                body, "}" NL, NULL));

    free(inc);
    free(cond);
    free(init);
    free(body);
    return result;
}
/**************************************************************** STATEMENTS */

static string c_statement(statement s, bool breakable)
{
    string result;
    instruction i = statement_instruction(s);
    list l = statement_declarations(s);
    /*printf("\nCurrent statement : \n");
      print_statement(s);*/
    switch (instruction_tag(i))
    {
        case is_instruction_test:
            {
                test t = instruction_test(i);
                result = c_test(t,breakable);
                break;
            }
        case is_instruction_sequence:
            {
                sequence seq = instruction_sequence(i);
                result = c_sequence(seq,breakable);
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
                string scall = c_call(instruction_call(i),breakable);
                result = strdup(concatenate(scall, SEMICOLON, NULL));
                break;
            }
        case is_instruction_unstructured:
            {
                unstructured u = instruction_unstructured(i);
                result = c_unstructured(u,breakable);
                break;
            }
        case is_instruction_goto:
            {
                statement g = instruction_goto(i);
                entity el = statement_label(g);
                const char* l = entity_local_name(el) + sizeof(LABEL_PREFIX) -1;
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
            string tmp = c_entity_local_name(var);
            debug(2, "\n In block declaration for variable :", tmp);
            free(tmp);
            svar = this_entity_cdeclaration(var, false);
            decl = strdup(concatenate(decl, svar, SEMICOLON, NULL));
            free(svar);
        },l);
        result = strdup(concatenate(decl,result,NULL));
    }

    return result;
}

/*******************************************************PRINT INTERFACE FCTS */
static string interface_type_string (type t, bool value);

/// @brief Convert the fortran basic to its interface string value
/// @param b, the basic to be converted to string
/// @param value, set to true if the var has to be passed by value
static string interface_basic_string(basic b, bool value)
{
  const char* result = "UNKNOWN_BASIC" SPACE;
  char * aresult=NULL;
  bool user_type = get_bool_property ("CROUGH_USER_DEFINED_TYPE");
  switch (basic_tag(b)) {
  case is_basic_int: {
    pips_debug(2,"Basic int\n");
    if (user_type == false) {
      switch (basic_int(b)) {
      /* case 1: result = "char"; */
      /* 	break; */
      /* case 2: result = "short"; */
      /* 	break; */
      case 4:
	result = "integer (c_int)";
	break;
      /* case 6: result = "long"; */
      /* 	break; */
      case 8:
	result = "integer (c_size_t)";
	break;
      /* case 11: result = "unsigned char"; */
      /* 	break; */
      /* case 12: result = "unsigned short"; */
      /* 	break; */
      /* case 14: result = "unsigned int"; */
      /* 	break; */
      /* case 16: result = "unsigned long"; */
      /* 	break; */
      /* case 18: result = "unsigned long long"; */
      /* 	break; */
      /* case 21: result = "signed char"; */
      /* 	break; */
      /* case 22: result = "signed short"; */
      /* 	break; */
      /* case 24: result = "signed int"; */
      /* 	break; */
      /* case 26: result = "signed long"; */
      /* 	break; */
      /* case 28: result = "signed long long"; */
      /* 	break; */
      default:
	pips_assert ("not handle case", false);
	break;
      }
    } else {
      result = get_string_property ("CROUGH_INTEGER_TYPE");
    }
    break;
  }
  case is_basic_float: {
    if (user_type == false) {
      switch (basic_float(b)){
      case 4: result = "real (c_float)";
	break;
      case 8: result = "real (c_double)";
	break;
      }
    } else {
      result = get_string_property ("CROUGH_REAL_TYPE");
    }
    break;
  }
  case is_basic_logical:
    result = "integer (c_int)";
    break;
  case is_basic_string:
    result = "character (c_char)";
    break;
  case is_basic_bit:
    pips_internal_error("unhandled case");
    break;
  case is_basic_pointer:
    {
      if (value == true) {
	type t = basic_pointer(b);
	pips_debug(2,"Basic pointer\n");
	aresult = interface_type_string (t, false);
	if (!type_void_p (t)) 
	return aresult;
      }
      else {
	result = "type (c_ptr)";
      }
      break;
    }
  case is_basic_derived:
    pips_internal_error("unhandled case");
    break;
  case is_basic_typedef:
    pips_internal_error("unhandled case");
  default:
    pips_internal_error("unhandled case");
  }
  if (value == true) {
      if(!aresult)aresult=strdup(result);
      char * tmp =aresult;
        aresult = strdup(concatenate (aresult, ", value", NULL));
        free(tmp);
    return aresult;
  }
  return strdup(result) ;
}

/// @param t, the type to be converted to its string representation
/// @param value, set to true if the associated argument is passed by value
/// (i.e. not by pointer)
static string interface_type_string (type t, bool value)
{
    string result ;
    switch (type_tag(t)) {
    case is_type_variable: {
      basic b = variable_basic(type_variable(t));
      result = interface_basic_string(b, value);
      break;
    }
    case is_type_void: {
      result = strdup ("type (c_ptr)");
      break;
    }
    default:
      pips_user_error("case not handled yet \n");
    }
    return result;
}

/// @brief return a string representation of the type to be used
/// for a variable decalaration in an interface module in order to ensure
/// that the C function can be called from fotran codes
static string interface_argument_type_string (entity var) {
  pips_assert("this function is deicated to arguments", argument_p(var));
  string result = NULL;
  type t = entity_type(var);
  variable v = type_variable(t);
  if (variable_dimensions (v) != NULL) {
    result = strdup ("type (c_ptr), value");
  } else {
    result = interface_type_string(t, true);
  }
  return result;
}

/// @return the string representation of the arguments of the given modules
/// to be used as a variable declaration in an interface.
/// @param module, the module to get the declaration.
/// @param separator, the separatot to be used between vars.
/// @param lastsep, set to true if a final separator is needed.
static string interface_argument_declaration (entity module, string separator,
					      string indent) {
  code c;
  string tmp = NULL;
  string args = strdup ("");
  string result = NULL;

  pips_assert("it is a code", value_code_p(entity_initial(module)));

  c = value_code(entity_initial(module));
  FOREACH(ENTITY, var,code_declarations(c)) {
    if (argument_p(var) == true) {
      tmp = args;
      args = strdup (concatenate (args, indent,
				  interface_argument_type_string (var),
				  " :: ",
				  c_entity_local_name (var),
				  separator,
				  NULL));
      free(tmp);
    }
  }
  result = strdup (args);
  free (args);
  return result;
}

/// @brief return the interface signature for a module, i.e. the list of the
/// variable names that are comma serparated.
static string interface_signature (entity module)
{
    code c = code_undefined;
    bool first = true;
    string tmp = NULL;
    string args = strdup ("(");
    string result = NULL;

    pips_assert("it is a function", type_functional_p(entity_type(module)));
    pips_assert("it is a code", value_code_p(entity_initial(module)));

    c = value_code(entity_initial(module));

    FOREACH(ENTITY, var,code_declarations(c)) {
      if (argument_p (var) == true) {
	tmp = args;
	args = strdup (concatenate (args, first == true ? "" : ", ",
				    c_entity_local_name (var), NULL));
	free(tmp);
	first = false;
      }
    }

    result = strdup (concatenate (args, ")", NULL));
    free (args);
    return result;
}

static string interface_code_string(entity module, statement stat)
{
  string name      = NULL;
  string decls     = NULL;
  string result    = NULL;
  string signature = NULL;

  pips_assert("only available for subroutines, to be implemented for functions",
	      entity_subroutine_p(module));

  name = c_entity_local_name (module);
  signature = interface_signature (module);
  decls = interface_argument_declaration (module, "\n", "\t\t\t");

  result = strdup(concatenate ("module ", name, "_interface\n",
			       "\tinterface\n",
			       "\t\tsubroutine ", name, signature,
			       " bind(C, name = \"", name, "\")\n",
			       "\t\t\tuse iso_c_binding\n", decls,
			       "\t\tend subroutine ", name,
			       "\n\tend interface\n",
			       "end module ", name, "_interface\n",
			       NULL));
  free (name);
  free (decls);
  free (signature);

  return result;
}

static string c_code_string(entity module, statement stat)
{
  string head, decls, body, result, copy_in, include, macro;

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

  // get the needed includes
  include     = c_include ();
  // get the needed macro
  macro       = c_macro ();
  // function declaration
  head        = c_head(module);
  // What about declarations associated to statements
  decls       = c_declarations(module,parameter_or_variable_p,SEMICOLON,true,
							   false);
  body        = c_statement(stat, false);
  copy_in     = scalar_prelude ();

  // concatenate everything to get the code
  result = concatenate(include, macro, head, OPENBRACE, NL, decls,
					   copy_in, NL, body, CLOSEBRACE, NL, NULL);

  free (include);
  free(head);
  free(decls);
  free(body);
  free(copy_in);
  return strdup(result);
}

/******************************************************** PIPSMAKE INTERFACE */

bool print_interface (const char* module_name)
{
  FILE * out;
  string interface_code, interface, dir, filename;
  entity module;
  statement stat;

  // get what is needed from PIPS DBM
  interface = db_build_file_resource_name(DBR_INTERFACE, module_name, INTERFACE);
  module = module_name_to_entity(module_name);
  dir = db_get_current_workspace_directory();
  filename = strdup(concatenate(dir, "/", interface, NULL));
  stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_entity(module);
  set_current_module_statement(stat);

  debug_on("INTERFACE_DEBUG_LEVEL");
  pips_debug(1, "Begin print_interface for %s\n", entity_name(module));

  // get the inteface code as a string
  interface_code = interface_code_string(module, stat);
  pips_debug(1, "end\n");
  debug_off();

  /* save to file */
  out = safe_fopen(filename, "w");
  fprintf(out, "! Fortran interface module for %s. \n", module_name);
  fprintf(out, "%s", interface_code);
  safe_fclose(out, filename);

  DB_PUT_FILE_RESOURCE(DBR_INTERFACE, module_name, INTERFACE);

  reset_current_module_statement();
  reset_current_module_entity();

  free (interface_code);
  free (dir);
  free (filename);

  return true;
}

bool print_crough(const char* module_name)
{
    FILE * out;
    string ppt, crough, dir, filename;
    entity module;
    statement stat;
    list l_effect = NULL;

    // get what is needed from PIPS DBM
    crough = db_build_file_resource_name(DBR_CROUGH, module_name, CROUGH);
    module = module_name_to_entity(module_name);
    dir = db_get_current_workspace_directory();
    filename = strdup(concatenate(dir, "/", crough, NULL));
    stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
    l_effect = effects_to_list((effects)
			       db_get_memory_resource(DBR_SUMMARY_EFFECTS,
						      module_name, true));
    set_current_module_entity(module);
    set_current_module_statement(stat);

    debug_on("CPRETTYPRINTER_DEBUG_LEVEL");
    pips_debug(1, "Begin C prettyprrinter for %s\n", entity_name(module));

    // init the list needed for the function pre and postlude
    l_type    = NIL;
    l_name    = NIL;
    l_rename  = NIL;
    l_entity  = NIL;
    l_written = NIL;
    // build the list of written entity
    build_written_list (l_effect);

    // get the c code as a string
    ppt = c_code_string(module, stat);
    pips_debug(1, "end\n");
    debug_off();

    /* save to file */
    out = safe_fopen(filename, "w");
    fprintf(out, "/* C pretty print for module %s. */\n", module_name);
    fprintf(out, "%s", ppt);
    safe_fclose(out, filename);

    // free and reset strin lists
    gen_free_list (l_entity);
    gen_free_list (l_written);
    gen_free_string_list (l_type);
    gen_free_string_list (l_name);
    gen_free_string_list (l_rename);
    l_type    = NIL;
    l_name    = NIL;
    l_rename  = NIL;
    l_entity  = NIL;
    l_written = NIL;
    free(ppt);
    free(dir);
    free(filename);

    DB_PUT_FILE_RESOURCE(DBR_CROUGH, module_name, crough);

    reset_current_module_statement();
    reset_current_module_entity();

    return true;
}

/* C indentation thru indent.
*/
bool print_c_code(const char* module_name)
{
    string crough, cpretty, dir, cmd;

    crough = db_get_memory_resource(DBR_CROUGH, module_name, true);
    cpretty = db_build_file_resource_name(DBR_C_PRINTED_FILE, module_name, CPRETTY);
    dir = db_get_current_workspace_directory();

    cmd = strdup(concatenate(INDENT, " ",
                dir, "/", crough, " -st > ",
                dir, "/", cpretty, NULL));

    safe_system(cmd);

    DB_PUT_FILE_RESOURCE(DBR_C_PRINTED_FILE, module_name, cpretty);
    free(cmd);
    free(dir);

    return true;
}
