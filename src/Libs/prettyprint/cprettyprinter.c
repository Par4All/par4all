/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#include "resources.h"
#include "properties.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "text-util.h"
#include "transformations.h"

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
static string c_expression(expression,bool);

/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
    (entity_function_p(get_current_module_entity()))


#define RESULT_NAME	"result"



/*
 * convert string to lower case, string is modified in place
 */
static void string_tolower(string s)
{
    string car=NULL;
    for (car = s; *car; car++)
        *car = (char) tolower(*car);
}

/*
 * convert some fortran constant to their equivalent in C
 */
static void const_wrapper(string* s)
{
    static char * const_to_c[][2] = { { ".true." , "1" } , { ".false." , "0" } };
    static const int const_to_c_sz = sizeof(const_to_c)/sizeof(*const_to_c);
    int i;

    /* search fortran constant */
    char *name = strdup(*s);
    string_tolower(name);
    for(i=0;i<const_to_c_sz;i++)
    {
        if(strcmp(name,const_to_c[i][0]) == 0 )
        {
            *s = const_to_c[i][1];
            break;
        }
    }
    free(name);
}

/*
 * warning : return allocated string, otherwise it leads to modification (through string_tolower)
 * of critical entities
 */
static string c_entity_local_name(entity var)
{
    string name;

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
    name=strdup(name);
    string_tolower(name);


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
    default:
      pips_user_warning("case not handled yet \n");
    }
    return strdup(result);
}

static string c_basic_string(basic b)
{
    string result = "UNKNOWN_BASIC" SPACE;
    bool allocated =false;
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
                string name = c_entity_local_name(ent);
                result = concatenate(c_type_string(t),name,NULL);
                free(name);
                break;
            }
        case is_basic_typedef:
            {
                entity ent = basic_typedef(b);
                result = c_entity_local_name(ent);
                allocated=true;
                break;
            }
        default:
            pips_internal_error("unhandled case\n");
    }
    return allocated ? result : strdup(result);
}

static string c_dim_string(list ldim)
{
    string result = "";
    if (ldim != NIL )
    {
        FOREACH(DIMENSION, dim,ldim)
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
#if 0
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
                    if (expression_integer_value(eup, &up))
                        result = strdup(concatenate(OPENBRACKET,i2a(up-low+1),CLOSEBRACKET,result,NULL));
                    else
                    {
		      sup = words_to_string(words_expression(eup, NIL));
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
    return result;
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
                break;
        }
    }
    return strdup(result);
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
	  result = strdup(concatenate(result,first?"":",",words_to_string(words_expression(e, NIL)),NULL));
        first = FALSE;
    },args);
    result = strdup(concatenate(result,"}",NULL));
    return result;
}

static string this_entity_cdeclaration(entity var)
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
        result = strdup(concatenate("typedef ", c_type_string(t),SPACE,c_entity_local_name(var),NULL));
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
                            bool old = get_prettyprint_is_fortran();
                            reset_prettyprint_is_fortran();
                            string sbasic = basic_to_string(entity_basic(var));
                            if(old) set_prettyprint_is_fortran();
                            asprintf(&result,"static const %s %s = %s;\n",sbasic,svar,sval);
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
                string st, sd, svar, sq;
                value val = entity_initial(var);
                st = c_basic_string(variable_basic(v));
                sd = c_dim_string(variable_dimensions(v));
                sq = c_qualifier_string(variable_qualifiers(v));
                svar = c_entity_local_name(var);

                /* problems with order !*/
                result = strdup(concatenate(sq, st, SPACE, svar, sd, NULL));
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
                //free(sd);
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
                string tmp =NULL;
                tmp = c_entity_local_name(var);
                result = strdup(concatenate("union ",tmp, "{", NL,NULL));
                free(tmp);
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
                string tmp = NULL;
                tmp = c_entity_local_name(var);
                result = strdup(concatenate("enum ", tmp, " {",NULL));
                free(tmp);
                MAP(ENTITY,ent,
                        {
                        tmp = c_entity_local_name(ent);
                        result = strdup(concatenate(result,first?"":",",tmp,NULL));
                        free(tmp);
                        first = FALSE;
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

static bool argument_p(entity e)
{
    /* Formal variables */
    return type_variable_p(entity_type(e)) &&
        storage_formal_p(entity_storage(e));
}

static string c_declarations(
        entity module,
        bool (*consider_this_entity)(entity),
        string separator,
        bool lastsep
        )
{
    string result = strdup("");
    code c;
    bool first = TRUE;

    pips_assert("it is a code", value_code_p(entity_initial(module)));

    c = value_code(entity_initial(module));
    FOREACH(ENTITY, var,code_declarations(c))
    {
        string tmp = NULL;
        tmp = c_entity_local_name(var);
        debug(2, "\n Prettyprinter declaration for variable :",tmp);
        free(tmp);
        if (consider_this_entity(var))
        {
            string old = result;
            string svar = this_entity_cdeclaration(var);
            result = strdup(concatenate(old, !first && !lastsep? separator: "",
                        svar, lastsep? separator: "", NULL));
            free(svar);
            free(old);
            first = FALSE;
        }
    }
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

static string ppt_unary(string in_c, list le)
{
    string e, result;
    pips_assert("one arg to unary call", gen_length(le)==1);
    e = c_expression(EXPRESSION(CAR(le)),false);
    result = strdup(concatenate(in_c, SPACE, e, NULL));
    free(e);
    return result;
}

static string ppt_unary_post(string in_c, list le)
{
    string e, result;
    pips_assert("one arg to unary call", gen_length(le)==1);
    e = c_expression(EXPRESSION(CAR(le)),false);
    result = strdup(concatenate(e, SPACE, in_c, NULL));
    free(e);
    return result;
}

/* SG: PBM spotted HERE */
static string ppt_call(string in_c, list le)
{
    string scall, old;
    if (le == NIL)
    {
        scall = strdup(concatenate(in_c, "()", NULL));
    }
    else
    {
        bool first = TRUE;
        scall = strdup(concatenate(in_c, OPENPAREN, NULL));

        /* Attention: not like this for io statements*/
        MAP(EXPRESSION, e,
        {
            string arg = c_expression(e,false);
            old = scall;
            scall = strdup(concatenate(old, first? "": ", ", arg, NULL));
            //free(arg);
            //free(old);
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
    { "**", "pow", ppt_call },
    { "=", "=", ppt_binary },
    { ".OR.", "||", ppt_binary },
    { ".AND.", "&&", ppt_binary },
    { ".NOT.", "!", ppt_unary },
    { ".LT.", "<", ppt_binary },
    { ".GT.", ">", ppt_binary },
    { ".LE.", "<=", ppt_binary },
    { ".GE.", ">=", ppt_binary },
    { ".EQ.", "==", ppt_binary },
    { ".EQV.", "==", ppt_binary },
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
#define CONT "continue"

static string c_call(call c,bool breakable)
{
    entity called = call_function(c);
    struct s_ppt * ppt = get_ppt(called);
    string local_name = entity_local_name(called);
    string result = NULL;

    /* special case... */
    if (same_string_p(local_name, "RETURN"))
    {
        if (entity_main_module_p(get_current_module_entity()))
            result = RET " 0";
        else if (current_module_is_a_function())
            result = RET SPACE RESULT_NAME;
        else
            result = RET;
        result=strdup(result);
    }
    else if (same_string_p(local_name, "CONTINUE") )
    {
        result = breakable?strdup(CONT):strdup("");
    }
    else if (call_constant_p(c))
    {
        const_wrapper(&local_name);
        result = strdup(local_name);
        string_tolower(result);
    }
    else
    {
        result = ppt->ppt(ppt->c? ppt->c: local_name, call_arguments(c));
        string_tolower(result);
    }

    return result;
}

/* Attention with Fortran: the indexes are reversed. 
   And array dimensions in C always rank from 0. BC.
*/
static string c_reference(reference r)
{
    string result = strdup(EMPTY), old, svar;

    list l_dim = variable_dimensions(type_variable(entity_type(reference_variable(r)))); 

    MAP(EXPRESSION, e,
    {
      expression e_tmp;
      expression e_lower = dimension_lower(DIMENSION(CAR(l_dim)));
      string s;
      int itmp;

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
    }, reference_indices(r));


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
                    bool no_endif = FALSE;
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
    MAP(STATEMENT, s,
    {
        string oldresult = result;
        string current = c_statement(s,breakable);
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
            string tmp = c_entity_local_name(var);
            debug(2, "\n In block declaration for variable :", tmp);
            free(tmp);
            svar = this_entity_cdeclaration(var);
            decl = strdup(concatenate(decl, svar, SEMICOLON, NULL));
            free(svar);
        },l);
        result = strdup(concatenate(decl,result,NULL));
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
    ifdebug(2)
    {
        printf("Module statement: \n");
        print_statement(stat);
        printf("and declarations: \n");
        print_entities(statement_declarations(stat));
    }

    //before_head = c_declarations(module, parameter_p, NL, TRUE);
    head        = c_head(module);
    /* What about declarations associated to statements */
    decls       = c_declarations(module, parameter_or_variable_p, SEMICOLON, TRUE);
    body        = c_statement(stat, false);

    result = concatenate(/*before_head,*/ head, OPENBRACE, NL,
            decls, NL, body, CLOSEBRACE, NL, NULL);

    free(before_head);
    free(head);
    free(decls);
    free(body);

    return strdup(result);
}

/******************************************************** PIPSMAKE INTERFACE */

#define INDENT		"indent"
#define CROUGH		".crough"
#define CPRETTY		".c"

bool print_crough(string module_name)
{
    FILE * out;
    string ppt, crough, dir, filename;
    entity module;
    statement stat;

    crough = db_build_file_resource_name(DBR_CROUGH, module_name, CROUGH);
    module = module_name_to_entity(module_name);
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
