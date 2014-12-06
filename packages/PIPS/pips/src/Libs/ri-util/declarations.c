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
/* Regeneration of declarations from the symbol table
 *
 * Regeneration of declarations...
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "syntax.h"

#include "pipsdbm.h"

#include "misc.h"
#include "properties.h"
#include "prettyprint.h"

/*===================== Variables and Function prototypes for C ===========*/

/* pdl is the parser declaration list. It is used to decide if a
   derived entity should be simply declared, "struct s", or fully
   defined, "struct s {....}" */
text c_text_entity(entity module, entity e, int margin, list pdl);
list c_words_entity(type t, list name, list pdl);
static list words_qualifier(list obj);
list words_dimensions(list dims, list pdl);

/********************************************************************* WORDS */

static list
words_constant(constant obj)
{
    list pc=NIL;

    if (constant_int_p(obj)) {
	pc = CHAIN_IWORD(pc,constant_int(obj));
    }
    else {
	pips_internal_error("unexpected tag");
    }
    /*What about real, double, string constants ... ?*/
    return(pc);
}

static list words_value(value obj)
{
    list pc=NIL;

    if (value_symbolic_p(obj)) {
	pc = words_constant(symbolic_constant(value_symbolic(obj)));
    }
    else if (value_constant_p(obj)) {
	pc = words_constant(value_constant(obj));
    }
    else if (value_unknown_p(obj)) {
      // FI: Only good for Fortran
      pc = CHAIN_SWORD(pc, "(*)");
    }
    else {
	pips_internal_error("unexpected tag");
	pc = NIL;
    }

    return(pc);
}

/** #define LIST_SEPARATOR (is_fortran? ", " : ",") */

static list words_parameters(entity e, list pdl)
{
  list pc = NIL;
  type te = entity_type(e);
  functional fe;
  int nparams, i;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  pips_assert("is functionnal", type_functional_p(te));

  fe = type_functional(te);
  nparams = gen_length(functional_parameters(fe));

  for (i = 1; i <= nparams; i++) {
    entity param = find_ith_parameter(e, i);
    if (pc != NIL) {
      pc = CHAIN_SWORD(pc, space_p? ", " : ",");
    }

    /* If prettyprint alternate returns... Property to be added. */
    if ( get_bool_property( "PRETTYPRINT_REGENERATE_ALTERNATE_RETURNS" )
        && formal_label_replacement_p( param ) ) {
      pc = CHAIN_SWORD(pc, "*");
    } else if ( entity_undefined_p(param) ) {
      parameter p = PARAMETER(gen_nth(i-1,functional_parameters(fe)));
      type t = parameter_type(p);
      switch(get_prettyprint_language_tag()) {
        case is_language_fortran:
        case is_language_fortran95:
          pips_user_warning("%dth parameter out of %d parameters not found for "
              "function %s\n", i, nparams, entity_name(e));
          break;
        case is_language_c:
          /* param can be undefined for C language: void foo(void)
           * We do not have an entity corresponding to the 1st argument
           */
          break;
        default:
          pips_internal_error("Language unknown !");
          break;
      }
      pc = gen_nconc( pc, words_type( t, pdl, true ) );
      /* Should be correct, but seems useless */
      //if(!same_string_p(pn, "")) {
      //  pc = gen_nconc(pc, strdup(" "));
      //  pc = gen_nconc(pc, strdup(pn));
      //}
    } else {
      switch(get_prettyprint_language_tag()) {
        case is_language_fortran:
        case is_language_fortran95:
          /*
           * Variable type and dimensions will be (eventually) specified
           * in declarations
           */
          pc = CHAIN_SWORD(pc, entity_local_name(param));
          break;
        case is_language_c: {
          /*
           * We have to print variable's type, dimensions, ... with C
           * This can be also a formal function
           */
          type t = type_undefined;
          t = entity_type(param);
          list entity_word = CHAIN_SWORD(NIL,entity_local_name(param));
          pc = gen_nconc( pc,
                          c_words_entity( t,
                                          entity_word,
                                          pdl ) );
          break;
        }
        default:
          pips_internal_error("Language unknown !");
          break;
      }
    }
  }
  return ( pc );
}

static list words_dimension(dimension obj, list pdl)
{
  list pc = NIL;
  call c = call_undefined;
  entity f = entity_undefined;
  expression eup = expression_undefined;
  expression e1 = expression_undefined;
  expression e2 = expression_undefined;
  intptr_t up, i;
  switch(get_prettyprint_language_tag()) {
    case is_language_fortran95:
      /* Not asterisk for unbound dimension in F95*/
      if(unbounded_dimension_p(obj)) {
        pc = CHAIN_SWORD(pc,":");
        break;
      }
      /* If dimension are bounded, fallback on plain old F77 style */
    case is_language_fortran:
      pc = words_expression( dimension_lower(obj), pdl );
      pc = CHAIN_SWORD(pc,":");
      pc = gen_nconc( pc, words_expression( dimension_upper(obj), pdl ) );
      break;
    case is_language_c:
      /* The lower bound of array in C is always equal to 0,
       we only need to print (upper dimension + 1) */
      if ( unbounded_dimension_p( obj ) )
        pc = CHAIN_SWORD(pc,"");
      else {
        eup = dimension_upper(obj);
        if ( false && expression_integer_value( eup, &up ) ) {
          /*
           * FI: why do you want to change the source code? Because it may no
           * longer be the user source code after partial evaluation
           */
          pc = CHAIN_IWORD(pc,up+1);
        }
        if ( expression_constant_p( eup )
            && constant_int_p(expression_constant(eup)) ) {
          /* To deal with partial eval generated expressions */
          up = expression_to_int( eup );
          pc = CHAIN_IWORD(pc,up+1);
        } else {
          if ( expression_call_p( eup ) ) {
            c = syntax_call(expression_syntax(eup));
            f = call_function(c);
            if ( ENTITY_MINUS_P(f) || ENTITY_MINUS_C_P(f) ) {
              e1 = binary_call_lhs(c);
              e2 = binary_call_rhs(c);
              if ( expression_integer_value( e2, &i ) && i == 1 )
                pc = words_expression( e1, pdl );
            }
          }
          if ( pc == NIL ) {
            /* to be refined here to make more beautiful expression,
             * use normalize ? FI: why would we modify the user C source code?
             */
            pc = words_expression( MakeBinaryCall( CreateIntrinsic( "+" ),
                                                   eup,
                                                   int_to_expression( 1 ) ),
                                   pdl );
          }
        }
      }
      break;
    default:
      pips_internal_error("Language unknown !");
      break;
  }
  return(pc);
}

/* some compilers don't like dimensions that are declared twice.
 * this is the case of g77 used after hpfc. thus I added a
 * flag not to prettyprint again the dimensions of common variables. FC.
 *
 * It is in the standard that dimensions cannot be declared twice in a
 * single module. BC.
 */
list words_declaration(entity e,
                       bool prettyprint_common_variable_dimensions_p,
                       list pdl) {
  list pl = NIL;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  pl = CHAIN_SWORD(pl, entity_user_name(e));

  if (type_variable_p(entity_type(e))) {
    if (prettyprint_common_variable_dimensions_p || !(variable_in_common_p(e)
        || variable_static_p(e))) {
      if (variable_dimensions(type_variable(entity_type(e))) != NIL) {
        list dims = variable_dimensions(type_variable(entity_type(e)));

        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
            pl = CHAIN_SWORD(pl, "(");
            MAPL(pd,
                {
                  pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd)), pdl));
                  if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, space_p? ", " : ",");
                }, dims)
            ;
            pl = CHAIN_SWORD(pl, ")");
            break;
          case is_language_c:
            MAPL(pd,
                {
                  pl = CHAIN_SWORD(pl, "[");
                  pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd)), pdl));
                  pl = CHAIN_SWORD(pl, "]");
                }, dims)
            ;
            break;
          default:
            pips_internal_error("Language unknown !");
        }
      }
    }
  }
  attach_declaration_to_words(pl, e);
  return (pl);
}

/* what about simple DOUBLE PRECISION, REAL, INTEGER... */
list words_basic(basic obj, list pdl)
{
  list pc = NIL;

  if ( basic_undefined_p(obj) ) {
    /* This may happen in debugging statements */
    pc = CHAIN_SWORD(pc,"undefined");
  } else {
    switch ( basic_tag(obj) ) {
      case is_basic_int: {
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
/*            if(basic_int(obj)==4) {
              pc = CHAIN_SWORD(pc,"INTEGER");
            } else { */
              pc = CHAIN_SWORD(pc,"INTEGER*");
              pc = CHAIN_IWORD(pc,basic_int(obj));
//            }
            break;
          case is_language_c:
            switch ( basic_int(obj) ) {
              case 1:
                pc = CHAIN_SWORD(pc,"char");
                break;
              case 2:
                pc = CHAIN_SWORD(pc,"short");
                break;
              case 4:
                pc = CHAIN_SWORD(pc,"int");
                break;
              case 6:
                pc = CHAIN_SWORD(pc,"long int");
                break;
              case 8:
                pc = CHAIN_SWORD(pc,"long long int");
                break;
              case 11:
                pc = CHAIN_SWORD(pc,"unsigned char");
                break;
              case 12:
                pc = CHAIN_SWORD(pc,"unsigned short");
                break;
              case 14:
                pc = CHAIN_SWORD(pc,"unsigned int");
                break;
              case 16:
                pc = CHAIN_SWORD(pc,"unsigned long int");
                break;
              case 18:
                pc = CHAIN_SWORD(pc,"unsigned long long int");
                break;
              case 21:
                pc = CHAIN_SWORD(pc,"signed char");
                break;
              case 22:
                pc = CHAIN_SWORD(pc,"signed short");
                break;
              case 24:
                pc = CHAIN_SWORD(pc,"signed int");
                break;
              case 26:
                pc = CHAIN_SWORD(pc,"signed long int");
                break;
              case 28:
                pc = CHAIN_SWORD(pc,"signed long long int");
                break;
            }
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_float: {
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
            pc = CHAIN_SWORD(pc,"REAL*");
            pc = CHAIN_IWORD(pc,basic_float(obj));
            break;
          case is_language_c:
            switch ( basic_float(obj) ) {
              case 4:
                pc = CHAIN_SWORD(pc,"float");
                break;
              case 8:
                pc = CHAIN_SWORD(pc,"double");
                break;
              case 16:
                pc = CHAIN_SWORD(pc,"long double");
                break;
            }
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_logical: {
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
            pc = CHAIN_SWORD(pc,"LOGICAL*");
            pc = CHAIN_IWORD(pc,basic_logical(obj));
            break;
          case is_language_c:
            pc = CHAIN_SWORD(pc,"int"); /* FI: Use "bool" of stdbool.h instead
					    of "int" but it leads to
					    include issue for
					    generated code; avoid stdbool.h
					    and use "_Bool" directly
					    but it leads to infinite
					    loop from "_Bool" to
					    "_Bool" because "_Bool"
					    is declared as a typedef
					    in anr999 */
            break;
          case is_language_fortran95:
            pips_internal_error("Need to update F95 case");
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_overloaded: {
        /* should be a user error? Or simply bootstrap.c is not accurate? */
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
            pc = CHAIN_SWORD(pc,"OVERLOADED");
            break;
          case is_language_c:
            pc = CHAIN_SWORD(pc,"overloaded");
            break;
          case is_language_fortran95:
            pips_internal_error("Need to update F95 case");
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_complex: {
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
          case is_language_fortran95:
            pc = CHAIN_SWORD(pc,"COMPLEX*");
            pc = CHAIN_IWORD(pc,basic_complex(obj));
            break;
          case is_language_c:
            switch ( basic_complex(obj) ) {
              case 8:
                pc = CHAIN_SWORD(pc,"_Complex");
                break;
              case 9:
                pc = CHAIN_SWORD(pc,"float _Complex");
                break;
              case 16:
                pc = CHAIN_SWORD(pc,"double _Complex");
                break;
              case 32:
                pc = CHAIN_SWORD(pc,"long double _Complex");
                break;
              default:
                pips_internal_error("Unexpected complex size");
            }
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_string: {
        switch(get_prettyprint_language_tag()) {
          case is_language_fortran:
            pc = CHAIN_SWORD(pc,"CHARACTER*");
            pc = gen_nconc( pc, words_value( basic_string(obj) ) );
            break;
          case is_language_c:
            pc = CHAIN_SWORD(pc,"char *"); // FI: should it be char[]?
            break;
          case is_language_fortran95:
            pips_internal_error("Need to update F95 case");
            break;
          default:
            pips_internal_error("Language unknown !");
            break;
        }
        break;
      }
      case is_basic_bit: {
        symbolic bs = basic_bit(obj);
        int i = constant_int(symbolic_constant(bs));
        pips_debug(7,"Bit field basic: %d\n",i);
        pc = CHAIN_SWORD(pc,"int"); /* ignore if it is signed or unsigned */
        break;
      }
      /* The following code maybe redundant, because of tests in c_words_entity*/
      case is_basic_pointer: {
        type t = basic_pointer(obj);
        pips_debug(7,"Basic pointer\n");
        if ( type_undefined_p(t) ) {
          /* This may occur in the parser when a variable is used
           before it is fully defined (see ptr in decl42.c) */
          pc = CHAIN_SWORD(pc,"type_undefined *");
        } else {
	  pc = gen_nconc( pc, words_type( t, pdl, false ) );
          pc = CHAIN_SWORD(pc," *");
        }
        break;
      }
      case is_basic_derived: {
        entity ent = basic_derived(obj);
        const char* name = entity_user_name( ent );
        const char* lname = entity_local_name( ent );
        type t = entity_type(ent);

        if ( strstr( lname, STRUCT_PREFIX DUMMY_STRUCT_PREFIX ) == NULL
            && strstr( lname, UNION_PREFIX DUMMY_UNION_PREFIX ) == NULL
            && strstr( lname, ENUM_PREFIX DUMMY_ENUM_PREFIX ) == NULL ) {
	  pc = gen_nconc( pc, words_type( t, pdl, false ) );
          pc = CHAIN_SWORD(pc," ");
          pc = CHAIN_SWORD(pc,name);
          pc = CHAIN_SWORD(pc," "); /* FI: This space may not be always useful */
        } else {
          pc = gen_nconc( pc, c_words_entity( t, NIL, pdl ) );
        }
        break;
      }
      case is_basic_typedef: {
        entity ent = basic_typedef(obj);
        pc = CHAIN_SWORD(pc,entity_user_name(ent));
        break;
      }
      default:
        pips_internal_error("unexpected basic tag %d", basic_tag(obj));
    }
  }
  return(pc);
}

/**************************************************************** SENTENCE */

sentence sentence_variable(entity e, list pdl)
{
    list pc = NIL;
    type te = entity_type(e);

    pips_assert("is a variable", type_variable_p(te));

    pc = gen_nconc(pc, words_basic(variable_basic(type_variable(te)), pdl));
    pc = CHAIN_SWORD(pc, " ");

    pc = gen_nconc(pc, words_declaration(e, true, pdl));

    return(make_sentence(is_sentence_unformatted,
			 make_unformatted(NULL, 0, 0, pc)));
}


/* We have no way to distinguish between the SUBROUTINE and PROGRAM
 * They two have almost the same properties.
 * For the time being, especially for the PUMA project, we have a temporary
 * idea to deal with it: When there's no argument(s), it should be a PROGRAM,
 * otherwise, it should be a SUBROUTINE.
 * Lei ZHOU 18/10/91
 *
 * correct PROGRAM and SUBROUTINE distinction added, FC 18/08/94
 * approximate BLOCK DATA / SUBROUTINE distinction also added. FC 09/97
 */
sentence sentence_head(entity e, list pdl)
{
    list pc = NIL;
  type te = entity_type(e);
  functional fe;
  type tr;
  list args = words_parameters(e, pdl);

  pips_assert("is functionnal", type_functional_p(te));

  if (static_module_p(e))
    pc = CHAIN_SWORD(pc,"static ");

  fe = type_functional(te);
  tr = functional_result(fe);

  switch(type_tag(tr)) {
    case is_type_void:
      switch(get_prettyprint_language_tag()) {
        case is_language_fortran:
        case is_language_fortran95:
          if (entity_main_module_p(e))
            pc = CHAIN_SWORD(pc,"PROGRAM ");
          else {
            if (entity_blockdata_p(e))
              pc = CHAIN_SWORD(pc, "BLOCKDATA ");
            else if (entity_f95module_p(e))
              pc = CHAIN_SWORD(pc, "MODULE ");
            else
              pc = CHAIN_SWORD(pc,"SUBROUTINE ");
          }
          break;
        case is_language_c:
          pc = CHAIN_SWORD(pc,"void ");
          break;
        default:
          pips_internal_error("Language unknown !");
          break;
      }
      break;
    case is_type_variable: {
      list pdl = NIL;
      pc = gen_nconc(pc, words_basic(variable_basic(type_variable(tr)), pdl));
      switch(get_prettyprint_language_tag()) {
        case is_language_fortran:
        case is_language_fortran95:
          pc = CHAIN_SWORD(pc," FUNCTION ");
          break;
        case is_language_c:
          pc = CHAIN_SWORD(pc," ");
          break;
        default:
          pips_internal_error("Language unknown !");
          break;
      }
      break;
    }
    case is_type_unknown:
      /*
       * For C functions with no return type.
       * It can be treated as of type int, but we keep it unknown
       * for the moment, to make the differences and to regenerate initial code
       */
      break;
    default:
      pips_internal_error("unexpected type for result");
  }

  pc = CHAIN_SWORD(pc, entity_user_name(e));

  if (!ENDP(args)) {
    pc = CHAIN_SWORD(pc, "(");
    pc = gen_nconc(pc, args);
    pc = CHAIN_SWORD(pc, ")");
  } else if (type_variable_p(tr)
              || (prettyprint_language_is_c_p()
                  && (type_unknown_p(tr) || type_void_p(tr)))) {
    pc = CHAIN_SWORD(pc, "()");
  }

  return (make_sentence(is_sentence_unformatted, make_unformatted(NULL,
                                                                  0,
                                                                  0,
                                                                  pc)));
}

static bool
empty_static_area_p(entity e)
{
    if (!static_area_p(e)) return false;
    return ENDP(area_layout(type_area(entity_type(e))));
}

/*  special management of empty commons added.
 *  this may happen in the hpfc generated code.
 */
static sentence sentence_area(entity e, entity module, bool pp_dimensions, list pdl)
{
    const char* area_name = module_local_name(e);
    type te = entity_type(e);
    list pc = NIL, entities = NIL;
    bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

    /* FI: POINTER declarations should be generated for the heap area */
    if (dynamic_area_p(e) || heap_area_p(e) || stack_area_p(e) || pointer_dummy_targets_area_p(e)) /* shouldn't get in? */
	return sentence_undefined;

    assert(type_area_p(te));

    if (!ENDP(area_layout(type_area(te))))
    {
	bool pp_hpfc = get_bool_property("PRETTYPRINT_HPFC");

	if (pp_hpfc)
	    entities = gen_copy_seq(area_layout(type_area(te)));
	else
	    entities = common_members_of_module(e, module, true);

	/*  the common is not output if it is empty
	 */
	if (!ENDP(entities))
	{
	    bool comma = false, is_save = static_area_p(e);

	    if (is_save)
	    {
		pc = CHAIN_SWORD(pc, "SAVE ");
	    }
	    else
	    {
		pc = CHAIN_SWORD(pc, "COMMON ");
		if (strcmp(area_name, BLANK_COMMON_LOCAL_NAME) != 0)
		{
		    pc = CHAIN_SWORD(pc, "/");
		    pc = CHAIN_SWORD(pc, area_name);
		    pc = CHAIN_SWORD(pc, "/ ");
		}
	    }

	    MAP(ENTITY, ee,
	     {
		 if (comma) pc = CHAIN_SWORD(pc, space_p? ", " : ",");
		 else comma = true;
		 pc = gen_nconc(pc,
				words_declaration(ee, !is_save && pp_dimensions, pdl));
	     },
		 entities);

	    gen_free_list(entities);
	}
	else
	{
	    pips_user_warning("empty common %s for module %s encountered...\n",
			      entity_name(e), entity_name(module));
	    return make_sentence(is_sentence_formatted,
	       strdup(concatenate("!! empty common ", entity_local_name(e),
				  " in module ", entity_local_name(module),
				  "\n", NULL)));
	}
    }

    return make_sentence(is_sentence_unformatted,
			 make_unformatted(NULL, 0, get_prettyprint_indentation(), pc));
}

static sentence sentence_basic_declaration(entity e)
{
  list decl = NIL;
  basic b = entity_basic(e);

  pips_assert("b is defined", !basic_undefined_p(b));

  decl = CHAIN_SWORD(decl, basic_to_string(b));
  decl = CHAIN_SWORD(decl, " ");
  decl = CHAIN_SWORD(decl, entity_local_name(e));

  return(make_sentence(is_sentence_unformatted,
		       make_unformatted(NULL, 0, get_prettyprint_indentation(), decl)));
}


/**
 * @brief Create a sentence for a USE directive
 *
 * @description Use directive is handled by copying the string directly in the
 * name of the entity during the parsing. So we juste get the local name and put
 * it in a sentence.
 *
 * @return a sentence with correct indentation containing the whole use
 * directive in one word.
 */
static sentence sentence_f95use_declaration( entity e ) {
  list decl = NIL;

  decl = CHAIN_SWORD(decl, entity_local_name(e));

  return (make_sentence(is_sentence_unformatted, make_unformatted(NULL,
                                                                  0,
                                                                  INDENTATION,
                                                                  decl)));
}

static sentence sentence_external(entity f)
{
  list pc = NIL;

  pc = CHAIN_SWORD(pc, "EXTERNAL ");
  pc = CHAIN_SWORD(pc, entity_local_name(f));

  return(make_sentence(is_sentence_unformatted,
		       make_unformatted(NULL, 0, get_prettyprint_indentation(), pc)));
}

static sentence sentence_symbolic(entity f, list pdl)
{
  list pc = NIL;
  value vf = entity_initial(f);
  expression e = symbolic_expression(value_symbolic(vf));

  pc = CHAIN_SWORD(pc, "PARAMETER (");
  pc = CHAIN_SWORD(pc, entity_local_name(f));
  pc = CHAIN_SWORD(pc, " = ");
  pc = gen_nconc(pc, words_expression(e, pdl));
  pc = CHAIN_SWORD(pc, ")");

  return(make_sentence(is_sentence_unformatted,
		       make_unformatted(NULL, 0, get_prettyprint_indentation(), pc)));
}

/* why is it assumed that the constant is an int ???
 */
static sentence sentence_data(entity e)
{
  list pc = NIL;
  constant c;

  if (! value_constant_p(entity_initial(e)))
    return(sentence_undefined);

  c = value_constant(entity_initial(e));

  if (! constant_int_p(c))
    return(sentence_undefined);

  pc = CHAIN_SWORD(pc, "DATA ");
  pc = CHAIN_SWORD(pc, entity_local_name(e));
  pc = CHAIN_SWORD(pc, " /");
  pc = CHAIN_IWORD(pc, constant_int(c));
  pc = CHAIN_SWORD(pc, "/");

  return(make_sentence(is_sentence_unformatted,
		       make_unformatted(NULL, 0, get_prettyprint_indentation(), pc)));
}

/********************************************************************* TEXT */

#define ADD_WORD_LIST_TO_TEXT(t, l) ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(t, l, 0)
#define ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(t, l, m)\
  if (!ENDP(l)) ADD_SENTENCE_TO_TEXT(t,\
             make_sentence(is_sentence_unformatted, \
               make_unformatted(NULL, 0, m, l)));

/* if the common is declared similarly in all routines, generate
 * "include 'COMMON.h'", and the file is put in Src. otherwise
 * the full local declarations are generated. That's fun.
 */

static text include(const char* file)
{
  return make_text
    (CONS(SENTENCE,
	  make_sentence(is_sentence_formatted,
			strdup(concatenate("      include '", file, "'\n", NULL))),
	  NIL));
}

static text text_area_included(
    entity common /* the common the declaration of which are of interest */,
    entity module /* hte module dealt with */)
{
  string dir, file, local;
  const char* name;
  text t;

  dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  name = module_local_name(common);
  if (same_string_p(name, BLANK_COMMON_LOCAL_NAME))
    name = "blank";
  local = strdup(concatenate(name, ".h", NULL));
  file = strdup(concatenate(dir, "/", local, NULL));
  free(dir);

  if (file_exists_p(file))
    {
      /* the include was generated once before... */
      t = include(local);
    }
  else
    {
      string nofile =
	strdup(concatenate(file, ".sorry_common_not_homogeneous", NULL));
      t = text_common_declaration(common, module);
      if (!file_exists_p(nofile))
	{
	  if (check_common_inclusion(common))
	    {
	      /* same declaration, generate the file! */
	      FILE * f = safe_fopen(file, "w");
	      fprintf(f, "!!\n!! pips: include file for common %s\n!!\n",
		      name);
	      print_text(f, t);
	      safe_fclose(f, file);
	      t = include(local);
	    }
	  else
	    {
	      /* touch the nofile to avoid the inclusion check latter on. */
	      FILE * f = safe_fopen(nofile, "w");
	      fprintf(f,
		      "!!\n!! pips: sorry,  cannot include common %s\n!!\n",
		      name);
	      safe_fclose(f, nofile);
	    }
	  free(nofile);
	}
    }

  free(local); free(file);
  return t;
}

/* We add this function to cope with the declaration
 * When the user declare sth. there's no need to declare sth. for the user.
 * When nothing is declared ( especially there is no way to know whether it's
 * a SUBROUTINE or PROGRAM). We will go over the entire module to find all the
 * variables and declare them properly.
 * Lei ZHOU 18/10/91
 *
 * the float length is now tested to generate REAL*4 or REAL*8.
 * ??? something better could be done, printing "TYPE*%d".
 * the problem is that you cannot mix REAL*4 and REAL*8 in the same program
 * Fabien Coelho 12/08/93 and 15/09/93
 *
 * pf4 and pf8 distinction added, FC 26/10/93
 *
 * Is it really a good idea to print overloaded type variables~? FC 15/09/93
 * PARAMETERS added. FC 15/09/93
 *
 * typed PARAMETERs FC 13/05/94
 * EXTERNALS are missing: added FC 13/05/94
 *
 * Bug: parameters and their type should be put *before* other declarations
 *      since they may use them. Changed FC 08/06/94
 *
 * COMMONS are also missing:-) added, FC 19/08/94
 *
 * updated to fully control the list to be used.
 */
/* hook for commons, when not generated...
 */
static string default_common_hook(entity __attribute__ ((unused)) module,
				  entity common)
{
  return strdup(concatenate
		("common to include: ", entity_local_name(common), "\n", NULL));
}

static string (*common_hook)(entity, entity) = default_common_hook;

void set_prettyprinter_common_hook(string(*f)(entity,entity))
{
  common_hook=f;
}

void reset_prettyprinter_common_hook(void)
{
  common_hook=default_common_hook;
}

/* debugging for equivalences */
#define EQUIV_DEBUG 8

static void equiv_class_debug(list l_equiv)
{
  if (ENDP(l_equiv))
    fprintf(stderr, "<none>");
  MAP(ENTITY, equiv_ent,
      {
	fprintf(stderr, " %s", entity_local_name(equiv_ent));
      }, l_equiv);
  fprintf(stderr, "\n");
}


/* static int equivalent_entity_compare(entity *ent1, entity *ent2)
 * input    : two pointers on entities.
 * output   : an integer for qsort.
 * modifies : nothing.
 * comment  : this is a comparison function for qsort; the purpose
 *            being to order a list of equivalent variables.
 * algorithm: If two variables have the same offset, the longest 
 * one comes first; if they have the same length, use a lexicographic
 * ordering.
 * author: bc.
 */
static int equivalent_entity_compare(entity *ent1, entity *ent2)
{
  int result;
  int offset1 = ram_offset(storage_ram(entity_storage(*ent1)));
  int offset2 = ram_offset(storage_ram(entity_storage(*ent2)));
  Value size1, size2;

  result = offset1 - offset2;

  /* pips_debug(1, "entities: %s %s\n", entity_local_name(*ent1),
     entity_local_name(*ent2)); */

  if (result == 0)
    {
      /* pips_debug(1, "same offset\n"); */
      size1 = ValueSizeOfArray(*ent1);
      size2 = ValueSizeOfArray(*ent2);
      result = value_compare(size2,size1);

      if (result == 0)
	{
	  /* pips_debug(1, "same size\n"); */
	  result = strcmp(entity_local_name(*ent1), entity_local_name(*ent2));
	}
    }

  return(result);
}

/* static text text_equivalence_class(list  l_equiv)
 * input    : a list of entities representing an equivalence class.
 * output   : a text, which is the prettyprint of this class.
 * modifies : sorts l_equiv according to equivalent_entity_compare.
 * comment  : partially associated entities are not handled.
 * author   : bc.
 */
static text text_equivalence_class(list /* of entities */ l_equiv)
{
  text t_equiv = make_text(NIL);
  list lw = NIL;
  list l1, l2;
  entity ent1, ent2;
  int offset1, offset2;
  Value size1, offset_end1;
  bool first;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  if (gen_length(l_equiv)<=1) return t_equiv;

  /* FIRST, sort the list by increasing offset from the beginning of
     the memory suite. If two variables have the same offset, the longest
     one comes first; if they have the same lenght, use a lexicographic
     ordering */
  ifdebug(EQUIV_DEBUG)
  {
    pips_debug(1, "equivalence class before sorting:\n");
    equiv_class_debug(l_equiv);
  }

  gen_sort_list(l_equiv,
		(int (*)(const void *,const void *)) equivalent_entity_compare);

  ifdebug(EQUIV_DEBUG)
  {
    pips_debug(1, "equivalence class after sorting:\n");
    equiv_class_debug(l_equiv);
  }

  /* THEN, prettyprint the sorted list*/
  pips_debug(EQUIV_DEBUG,"prettyprint of the sorted list\n");

  /* At each step of the next loop, we consider two entities
   * from the equivalence class. l1 points on the first entity list,
   * and l2 on the second one. If l2 is associated with l1, we compute
   * the output string, and l2 becomes the next entity. If l2 is not
   * associated with l1, l1 becomes the next entity, until it is
   * associated with l1. In the l_equiv list, l1 is always before l2.
   */

  /* loop initialization */
  l1 = l_equiv;
  ent1 = ENTITY(CAR(l1));
  offset1 = ram_offset(storage_ram(entity_storage(ent1)));
  size1 = ValueSizeOfArray(ent1);
  l2 = CDR(l_equiv);
  first = true;

  while(!ENDP(l2))
    {
      ent2 = ENTITY(CAR(l2));
      offset2 = ram_offset(storage_ram(entity_storage(ent2)));

      pips_debug(EQUIV_DEBUG, "dealing with: %s %s\n",
		 entity_local_name(ent1),
		 entity_local_name(ent2));

      /* If the two variables have the same offset, their
       * first elements are equivalenced.
       */
      if (offset1 == offset2)
	{
	  pips_debug(EQUIV_DEBUG, "easiest case: offsets are the same\n");

	  if (first) lw = CHAIN_SWORD(lw, "EQUIVALENCE"), first = false;
	  else lw = CHAIN_SWORD(lw, space_p? ", " : ",");

	  lw = CHAIN_SWORD(lw, " (");
	  lw = CHAIN_SWORD(lw, entity_local_name(ent1));
	  lw = CHAIN_SWORD(lw, space_p? ", " : ",");
	  lw = CHAIN_SWORD(lw, entity_local_name(ent2));
	  lw = CHAIN_SWORD(lw, ")");
	  POP(l2);
	}
      /* Else, we first check that there is an overlap */
      else
	{
	  pips_assert("the equivalence class has been sorted\n",
		      offset1 < offset2);

	  offset_end1 = value_plus(offset1, size1);

	  /* If there is no overlap, we change the reference variable */
	  if (value_le(offset_end1,offset2))
	    {
	      pips_debug(1, "second case: there is no overlap\n");
	      POP(l1);
	      ent1 = ENTITY(CAR(l1));
	      offset1 = ram_offset(storage_ram(entity_storage(ent1)));
	      size1 = ValueSizeOfArray(ent1);
	      if (l1 == l2) POP(l2);
	    }

	  /* Else, we must compute the coordinates of the element of ent1
	   * which corresponds to the first element of ent2
	   */
	  else
	    {
	      /* ATTENTION: Je n'ai pas considere le cas
	       * ou il y a association partielle. De ce fait, offset
	       * est divisiable par size_elt_1. */
	      int offset = offset2 - offset1;
	      int rest;
	      int current_dim;
	      int dim_max = NumberOfDimension(ent1);
	      int size_elt_1 = SizeOfElements(
					      variable_basic(type_variable(entity_type(ent1))));
	      list l_tmp = variable_dimensions
		(type_variable(entity_type(ent1)));
	      normalized nlo;
	      Pvecteur pvlo;

	      pips_debug(EQUIV_DEBUG, "third case\n");
	      pips_debug(EQUIV_DEBUG,
			 "offset=%d, dim_max=%d, size_elt_1=%d\n",
			 offset, dim_max,size_elt_1);

	      if (first) lw = CHAIN_SWORD(lw, "EQUIVALENCE"), first = false;
	      else lw = CHAIN_SWORD(lw, space_p? ", " : ",");

	      lw = CHAIN_SWORD(lw, " (");
	      lw = CHAIN_SWORD(lw, entity_local_name(ent1));
	      lw = CHAIN_SWORD(lw, "(");

	      pips_assert("partial association case not implemented:\n"
			  "offset % size_elt_1 == 0",
			  (offset % size_elt_1) == 0);

	      offset = offset/size_elt_1;
	      current_dim = 1;

	      while (current_dim <= dim_max)
		{
		  dimension dim = DIMENSION(CAR(l_tmp));
		  int new_decl;
		  int size;

		  pips_debug(EQUIV_DEBUG, "prettyprinting dimension %d\n",
			     current_dim);
		  size = SizeOfIthDimension(ent1, current_dim);
		  rest = (offset % size);
		  offset = offset / size;
		  nlo = NORMALIZE_EXPRESSION(dimension_lower(dim));
		  pvlo = normalized_linear(nlo);

		  pips_assert("sg", vect_constant_p(pvlo));
		  pips_debug(EQUIV_DEBUG,
			     "size=%d, rest=%d, offset=%d, lower_bound=%d\n",
			     size, rest, offset, (int)VALUE_TO_INT(val_of(pvlo)));

		  new_decl = VALUE_TO_INT(val_of(pvlo)) + rest;
		  lw = CHAIN_SWORD(lw,i2a(new_decl));
		  if (current_dim < dim_max)
		    lw = CHAIN_SWORD(lw, space_p? ", " : ",");

		  POP(l_tmp);
		  current_dim++;

		} /* while */

	      lw = CHAIN_SWORD(lw, ")");
	      lw = CHAIN_SWORD(lw, space_p? ", " : ",");
	      lw = CHAIN_SWORD(lw, entity_local_name(ent2));
	      lw = CHAIN_SWORD(lw, ")");
	      POP(l2);
	    } /* if-else: there is an overlap */
	} /* if-else: not same offset */
    } /* while */
  ADD_WORD_LIST_TO_TEXT(t_equiv, lw);

  pips_debug(EQUIV_DEBUG, "end\n");
  return t_equiv;
}


/* input    : the current module, and the list of declarations.
 * output   : a text for all the equivalences.
 * modifies : nothing
 * comment  :
 */
static text text_equivalences(
    entity __attribute__ ((unused)) module     /* the module dealt with */,
    list ldecl        /* the list of declarations to consider */,
    bool no_commons /* whether to print common equivivalences */)
{
  list equiv_classes = NIL, l_tmp;
  text t_equiv_class;

  pips_debug(1,"begin\n");

  /* FIRST BUILD EQUIVALENCE CLASSES */

  pips_debug(EQUIV_DEBUG, "loop on declarations\n");
  /* consider each entity in the declaration */
  MAP(ENTITY, e,
      {
	storage s = entity_storage(e);
	/* but only variables which have a ram storage must be considered
	 */
	if (type_variable_p(entity_type(e)) && storage_ram_p(s))
	  {
	    ram r = storage_ram(s);
	    entity common = ram_section(r);
	    list l_shared = ram_shared(r);

	    if (no_commons && !entity_special_area_p(common))
	      break;

	    ifdebug(EQUIV_DEBUG)
	    {
	      pips_debug(1, "considering entity: %s\n",entity_local_name(e));
	      pips_debug(1, "shared variables:\n");
	      equiv_class_debug(l_shared);
	    }

	    /* If this variable is statically aliased */
	    if (!ENDP(l_shared))
	      {
		bool found = false;
		list found_equiv_class = NIL;

		/* We first look in already found equivalence classes
		 * if there is already a class in which one of the
		 * aliased variables appears
		 */
		MAP(LIST, equiv_class,
		    {
		      ifdebug(EQUIV_DEBUG)
		      {
			pips_debug(1, "considering equivalence class:\n");
			equiv_class_debug(equiv_class);
		      }

		      MAP(ENTITY, ent,
			  {
			    if (variable_in_list_p(ent, equiv_class))
			      {
				found = true;
				found_equiv_class = equiv_class;
				break;
			      }
			  }, l_shared);

		      if (found) break;
		    },
		    equiv_classes);

		if (found)
		  {
		    pips_debug(EQUIV_DEBUG, "already there\n");
		    /* add the entities of shared which are not already in
		     * the existing equivalence class. Useful ??
		     */
		    MAP(ENTITY, ent,
			{
			  if(!variable_in_list_p(ent, found_equiv_class) &&
			     variable_in_list_p(ent, ldecl)) /* !!! */
			    found_equiv_class =
			      CONS(ENTITY, ent, found_equiv_class);
			}, l_shared)
		      }
		else
		  {
		    list l_tmp = NIL;
		    pips_debug(EQUIV_DEBUG, "not found\n");
		    /* add the list of variables in l_shared; necessary
		     * because variables may appear several times in
		     * l_shared. */
		    MAP(ENTITY, shared_ent,
			{
			  if (!variable_in_list_p(shared_ent, l_tmp) &&
			      variable_in_list_p(shared_ent, ldecl))
			    /* !!! restricted to declared... */
			    l_tmp = CONS(ENTITY, shared_ent, l_tmp);
			},
			l_shared);
		    equiv_classes = CONS(LIST, l_tmp, equiv_classes);
		  }
	      }
	  }
      },
      ldecl);

  ifdebug(EQUIV_DEBUG)
  {
    pips_debug(1, "final equivalence classes:\n");
    MAP(LIST, equiv_class, equiv_class_debug(equiv_class), equiv_classes);
  }

  /* SECOND, PRETTYPRINT THEM */
  t_equiv_class = make_text(NIL);
  MAP(LIST, equiv_class,
      {
	MERGE_TEXTS(t_equiv_class, text_equivalence_class(equiv_class));
      }, equiv_classes);

  /* AND FREE THEM */
  for(l_tmp = equiv_classes; !ENDP(l_tmp); POP(l_tmp))
    {
      list equiv_class = LIST(CAR(l_tmp));
      gen_free_list(equiv_class);
      LIST(CAR(l_tmp)) = NIL;
    }
  gen_free_list(equiv_classes);

  /* THE END */
  pips_debug(EQUIV_DEBUG, "end\n");
  return(t_equiv_class);
}

/* Prettyprint the initializations field of code */
static sentence sentence_data_statement(statement is, list pdl)
{
  unformatted u =
    make_unformatted
    (NULL,
     STATEMENT_NUMBER_UNDEFINED, get_prettyprint_indentation(),
     CONS(STRING, strdup("DATA "), NIL));
  sentence s = make_sentence(is_sentence_unformatted, u);
  list wl = unformatted_words(u);
  instruction ii = statement_instruction(is);
  call ic = instruction_call(ii);
  entity ife = entity_undefined;
  list al = list_undefined; /* Argument List */
  list rl = list_undefined; /* Reference List */
  expression rle = expression_undefined; /* reference list expression, i.e. call to DATA LIST */
  entity rlf = entity_undefined; /* DATA LIST entity function */
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  pips_assert("An initialization instruction is a call", instruction_call_p(ii));
  ife = call_function(ic);
  pips_assert("The static initialization function is called",
	      ENTITY_STATIC_INITIALIZATION_P(ife));
  al = call_arguments(ic);

  /* Find all initialized variables pending from DATA LIST */
  rle = EXPRESSION(CAR(al));
  POP(al); /* Move al to the first value */
  pips_assert("The first argument is a call", expression_call_p(rle));
  rlf = call_function(syntax_call(expression_syntax(rle)));
  pips_assert("This is the DATA LIST function", ENTITY_DATA_LIST_P(rlf));
  rl = call_arguments(syntax_call(expression_syntax(rle)));

  for(; !ENDP(rl); POP(rl)){
    expression ive = expression_undefined;
    list ivwl = list_undefined;

    if(rl!=call_arguments(syntax_call(expression_syntax(rle)))) {
      wl = CHAIN_SWORD(wl, strdup(space_p? ", " : ","));
    }

    ive = EXPRESSION(CAR(rl));
    ivwl = words_expression(ive, pdl);
    wl = gen_nconc(wl, ivwl);
  }

  pips_assert("The value list is not empty", !ENDP(al));

  /* Print all values */

  wl = CHAIN_SWORD(wl, " /");

  for(; !ENDP(al); POP(al)){
    expression ve = EXPRESSION(CAR(al));
    call vc = syntax_call(expression_syntax(ve));
    list iwl = list_undefined;

    pips_assert("Values are encoded as calls", expression_call_p(ve));

    if(strcmp(module_local_name(call_function(vc)), REPEAT_VALUE_NAME)==0) {
      expression rfe = expression_undefined;
      expression rve = expression_undefined;
      list rwl = list_undefined;

      pips_assert("Pseudo-intrinsic REPEAT-VALUE must have two arguments",
		  gen_length(call_arguments(vc))==2);

      rfe = binary_call_lhs(vc);
      rve = binary_call_rhs(vc);

      if(!(integer_constant_expression_p(rfe) && expression_to_int(rfe)==1)) {
	/* print out the repeat factor if it is not one */
	rwl = words_expression(rfe, pdl);
	wl = gen_nconc(wl, rwl);
	wl = gen_nconc(wl, CONS(STRING, strdup("*"), NIL));
      }
      iwl = words_expression(rve, pdl);
      wl = gen_nconc(wl, iwl);
    }
    else {
      iwl = words_expression(ve, pdl);
      wl = gen_nconc(wl, iwl);
    }
    if(!ENDP(CDR(al))) {
      wl = gen_nconc(wl, CONS(STRING, strdup(space_p? ", " : ","), NIL));
    }
  }

  wl = CHAIN_SWORD(wl, "/");

  return s;
}

text text_initializations(entity m)
{
  text t = make_text(NIL);
  list il = list_undefined;

  pips_assert("m is a module", entity_module_p(m));

  il = sequence_statements(code_initializations(value_code(entity_initial(m))));

  FOREACH(STATEMENT, is, il) {
    /* The previous declaration list is useless in Fortran, but the
       signature of functions designed for C or Fortran must be
       respected. */
    list pdl = NIL;
    if(!empty_comments_p(statement_comments(is))) {
      ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted,
					    strdup(statement_comments(is))));
    }
    ADD_SENTENCE_TO_TEXT(t, sentence_data_statement(is, pdl));
  }

  return t;
}

/* returns the DATA initializations.
 * limited to integers, because I do not know where is the value
 * for other types...
 */
static text __attribute__ ((unused))
text_data(entity __attribute__ ((unused)) module, list /* of entity */ ldecl)
{
  list /* of sentence */ ls = NIL;

  MAP(ENTITY, e,
      {
	value v = entity_initial(e);
	if(!value_undefined_p(v) &&
	   value_constant_p(v) && constant_int_p(value_constant(v)))
	  ls = CONS(SENTENCE, sentence_data(e), ls);
      },
      ldecl);

  return make_text(ls);
}

/*************************************************************** PARAMETERS */

static text text_of_parameters(list /* of entity that are parameters */ lp)
{
  list /* of sentence */ ls = NIL;

  /* generate the sentences
   */
  FOREACH(ENTITY, e, lp) {
    list pdl = NIL; // Assumed to be Fortran only
    ls = CONS(SENTENCE, sentence_basic_declaration(e),
	      CONS(SENTENCE, sentence_symbolic(e, pdl), ls));
  }

  return make_text(ls);
}

void check_fortran_declaration_dependencies(list ldecl)
{
  /* Check that each declaration only depends on previous declarations */
  int r = 1;

  FOREACH(ENTITY, v, ldecl) {
    type t = entity_type(v);

    if(type_variable_p(t)) {
      list dep = fortran_type_supporting_entities(NIL, t);
      list cdep = list_undefined;
      storage vs = entity_storage(v);

      /* FOREACH(ENTITY, dv, dep) { */
      for(cdep = dep; !ENDP(cdep); POP(cdep)) {
	entity dv = ENTITY(CAR(cdep));
	int dr = gen_position(dv, ldecl);
	value dvv = entity_initial(dv);

	if(storage_formal_p(vs) && value_symbolic_p(dvv)) {
	  /* Formal parameters are put in ldecl right away when
	     parsing the SUBROUTINE or FUNCTION statement. The
	     placement of their actual declaration is unknown. They
	     may depend on PARAMETERs declared later */
	  ;
	}
	else if(dr>=r) {
	  if(entity_symbolic_p(dv))
	    pips_user_warning("Fortran declaration order may be violated. "
			      "Variable \"%s\" depends on parameter \"%s\""
			      " but is, at least partly, declared first.\n",
			      entity_user_name(v), entity_user_name(dv));
	  else if(entity_scalar_p(dv))
	    pips_user_warning("Fortran declaration order may be violated. "
			      "Variable \"%s\" depends on variable \"%s\" "
			      "but is, at least partly, declared first.\n",
			      entity_user_name(v), entity_user_name(dv));
	  else
	    /* Should be a ParserError() when called from ProcessEntries()... */
	    pips_user_error("Fortran declaration order violated. Variable \"%s\" "
			    "depends on variable \"%s\" but is declared first.\n",
			    entity_user_name(v), entity_user_name(dv));
	}
      }
      gen_free_list(dep);
    }
    r++;
  }
}

/********************************************************** ALL DECLARATIONS */
/**
 * @brief This handle the fact that a Fortran95 declaration use "::" as a
 * separator between type and variable name. It also adds an "allocatable"
 * modifier if requested. Finally it add a "," between each variable if there
 * more than one to declare.
 **/
static list f77_f95_style_management(list prev,
                                     string str,
                                     bool allocatable_pass_p,
                                     bool space_p) {
  list result = prev;
  if (prev == NIL) {
    result = CHAIN_SWORD(result, str);
    if(allocatable_pass_p) {
      result = CHAIN_SWORD(result, ", ALLOCATABLE ");
    }
    if (prettyprint_language_is_fortran95_p ()) {
      result = CHAIN_SWORD(result, ":: ");
    }
  }
  else {
    result = CHAIN_SWORD(result, space_p? ", " : ",");
  }
  return result;
}


/* @brief This function compute the list of declaration at the begining of
 * a module. It's intended to be used with Fortran or Fortran95 only
 *
 * @param ldecl is the list of entity to be prettyprinted
 * @param force_common will force the prettyprint of common in include
 * @param
 * @param pdl is ???
 * @return the text of module declarations
 */
static text text_entity_declaration(entity module,
				    list /* of entity */ ldecl,
				    bool force_common,
				    list pdl)
{
  /*
   * allocatable_pass indicate if we want to prettyprint allocatable or non
   * allocatable entity, this is set to true during a second recursive pass.
   */
  static bool allocatable_pass_p = false;
  list allocatable_list = NULL;

  const char* how_common = get_string_property("PRETTYPRINT_COMMONS");
  bool print_commons = !same_string_p(how_common, "none");
  /* prettyprint common in include if possible... */
  bool pp_cinc = same_string_p(how_common, "include") && !force_common;
  list before = NIL, area_decl = NIL, pi1 = NIL, pi2 = NIL, pi4 = NIL, pi8 =
      NIL, ph1 = NIL, ph2 = NIL, ph4 = NIL, ph8 = NIL, pf4 = NIL, pf8 = NIL,
      pl = NIL, pc8 = NIL, pc16 = NIL, ps = NIL, lparam = NIL, uses = NIL;
  list * ppi = NULL;
  list * pph = NULL;
  text r, t_chars = make_text(NIL), t_area = make_text(NIL);
  const char* pp_var_dim = get_string_property("PRETTYPRINT_VARIABLE_DIMENSIONS");
  bool pp_in_type = false, pp_in_common = false;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");
  /* Declarations cannot be sorted out because Fortran standard impose
   at least an order on parameters. Fortunately here, PARAMETER are
   mostly integers, defined from other integer parameters... I assume
   that PIPS would fail with an ENTRY referencing an integer array
   dimensionned with a real parameter. But real parameters are not
   really well processed by PIPS anyway... Also we are in trouble if
   arrays or functions are used dimension other arrays

   list sorted_ldecl = gen_copy_seq(ldecl);

   gen_sort_list(sorted_ldecl, compare_entities); */

  check_fortran_declaration_dependencies(ldecl);


  /*
   * Deals with indentation
   */
  list indentation_words = NIL;
  if( prettyprint_language_is_fortran95_p() ) {
    for(int i=0; i<INDENTATION; i++) {
      indentation_words = CHAIN_SWORD(indentation_words, " ");
    }
  }

  /* where to put the dimension information.
   */
  if (same_string_p(pp_var_dim, "type")) {
    pp_in_type = true, pp_in_common = false;
  } else if (same_string_p(pp_var_dim, "common")) {
    pp_in_type = false, pp_in_common = true;
  } else {
    pips_internal_error("PRETTYPRINT_VARIABLE_DIMENSIONS=\"%s\""
        " unexpected value\n", pp_var_dim);
  }



  FOREACH(ENTITY, e,ldecl) {
    type te = entity_type(e);
    bool func = type_functional_p(te) && storage_rom_p(entity_storage(e));
    value v = entity_initial(e);
    bool param = func && value_symbolic_p(v);
    bool external = /* subroutines won't be declared */
          (func
        && (value_code_p(v) || value_unknown_p(v) /* not parsed callee */)
        && !(type_void_p(functional_result(type_functional(te)))
            || (type_variable_p(functional_result(type_functional(te)))
                && basic_overloaded_p(variable_basic(type_variable
                        (functional_result(type_functional(te))))))));
    bool area_p = type_area_p(te);
    bool var = type_variable_p(te);
    bool in_ram = storage_ram_p(entity_storage(e));
    bool in_common = in_ram
        && !entity_special_area_p(ram_section(storage_ram(entity_storage(e))));
    bool skip_it = same_string_p(entity_local_name(e),
        entity_local_name(module));

    pips_debug(3, "entity name is %s\n", entity_name(e));

    /* Do not declare variables used to replace formal labels */
    if (storage_formal_p(entity_storage(e))
        && get_bool_property("PRETTYPRINT_REGENERATE_ALTERNATE_RETURNS")
        && formal_label_replacement_p(e))
      continue;

    if (!print_commons && area_p && !entity_special_area_p(e) && !pp_cinc) {
      area_decl = CONS(SENTENCE,
          make_sentence(is_sentence_formatted,
              common_hook(module, e)),
          area_decl);
    }

    if (skip_it) {
      pips_debug(5, "skipping function %s\n", entity_name(e));
    } else if (entity_f95use_p(e)) {
      uses = CONS(SENTENCE, sentence_f95use_declaration(e), uses);
    } else if (!print_commons && (area_p || (var && in_common && pp_cinc))) {
      pips_debug(5, "skipping entity %s\n", entity_name(e));
    } else if (param) {
      /*        PARAMETER
       */
      pips_debug(7, "considered as a parameter\n");
      lparam = CONS(ENTITY, e, lparam);
    } else if (external) {
      /*        EXTERNAL
       */
      pips_debug(7, "considered as an external\n");
      before = CONS(SENTENCE, sentence_basic_declaration(e), before);
      before = CONS(SENTENCE, sentence_external(e), before);
    } else if (area_p && !dynamic_area_p(e) && !heap_area_p(e)
	       && !stack_area_p(e) && !pointer_dummy_targets_area_p(e) && !empty_static_area_p(e) ) {
      /*            AREAS: COMMONS and SAVEs
       */
      pips_debug(7, "considered as a regular common\n");
      if (pp_cinc && !entity_special_area_p(e)) {
        text t = text_area_included(e, module);
        MERGE_TEXTS(t_area, t);
      } else
        area_decl = CONS(SENTENCE,
            sentence_area(e, module, pp_in_common, pdl),
            area_decl);
    } else if (var && !(in_common && pp_cinc)) {
      basic b = variable_basic(type_variable(te));
      bool pp_dim = pp_in_type || variable_static_p(e);

      pips_debug(7, "is a variable...\n");

      switch(basic_tag(b)) {
        case is_basic_int:
          /* simple integers are moved ahead... */

          pips_debug(7, "is an integer\n");
          if (variable_dimensions(type_variable(te))) {
            string s = string_undefined;
            switch(basic_int(b)) {
              case 4:
                ppi = &pi4;
                s = "INTEGER ";
                break;
              case 2:
                ppi = &pi2;
                s = "INTEGER*2 ";
                break;
              case 8:
                ppi = &pi8;
                s = "INTEGER*8 ";
                break;
              case 1:
                ppi = &pi1;
                s = "INTEGER*1 ";
                break;

              default:
                pips_internal_error("Unexpected integer size");
            }

            *ppi = f77_f95_style_management (*ppi, s, allocatable_pass_p, space_p);
            *ppi = gen_nconc(*ppi, words_declaration(e, pp_dim, pdl));
          } else {
            string s = string_undefined;

            switch(basic_int(b)) {
              case 4:
                pph = &ph4;
                s = "INTEGER ";
                break;
              case 2:
                pph = &ph2;
                s = "INTEGER*2 ";
                break;
              case 8:
                pph = &ph8;
                s = "INTEGER*8 ";
                break;
              case 1:
                pph = &ph1;
                s = "INTEGER*1 ";
                break;
              default:
                pips_internal_error("Unexpected integer size");
            }
            *pph = f77_f95_style_management (*pph, s, allocatable_pass_p, space_p);
            *pph = gen_nconc(*pph, words_declaration(e, pp_dim, pdl));
          }
          break;
        case is_basic_float:
          pips_debug(7, "is a float\n");
          switch(basic_float(b)) {
            case 4:
              pf4 = f77_f95_style_management(pf4, "REAL*4 ", allocatable_pass_p, space_p);
              pf4 = gen_nconc(pf4, words_declaration(e, pp_dim, pdl));
              break;
            case 8:
            default:
              pf8 = f77_f95_style_management(pf8, "REAL*8 ", allocatable_pass_p, space_p);
              pf8 = gen_nconc(pf8, words_declaration(e, pp_dim, pdl));
              break;
          }
          break;
        case is_basic_complex:
          pips_debug(7, "is a complex\n");
          switch(basic_complex(b)) {
            case 8:
              pc8 = f77_f95_style_management(pc8, "COMPLEX*8 ", allocatable_pass_p, space_p);
              pc8 = gen_nconc(pc8, words_declaration(e, pp_dim, pdl));
              break;
            case 16:
            default:
              pc16 = f77_f95_style_management(pc16, "COMPLEX*16 ", allocatable_pass_p, space_p);
              pc16 = gen_nconc(pc16, words_declaration(e, pp_dim, pdl));
              break;
          }
          break;
        case is_basic_logical:
          pips_debug(7, "is a logical\n");
          pl = CHAIN_SWORD(pl, pl==NIL ? "LOGICAL " : (space_p? ", " : ","));
          pl = gen_nconc(pl, words_declaration(e, pp_dim, pdl));
          break;
        case is_basic_overloaded:
          /* nothing! some in hpfc I guess...
           */
          break;
        case is_basic_string: {
          value v = basic_string(b);
          pips_debug(7, "is a string\n");

          if (value_constant_p(v) && constant_int_p(value_constant(v))) {
            int i = constant_int(value_constant(v));

            if (i == 1) {
              ps = f77_f95_style_management(ps, "CHARACTER ", allocatable_pass_p, space_p);
              ps = gen_nconc(ps, words_declaration(e, pp_dim, pdl));
            } else {
              list chars = NIL;
              chars = CHAIN_SWORD(chars, "CHARACTER*");
              chars = CHAIN_IWORD(chars, i);
              chars = CHAIN_SWORD(chars, " ");
              chars = gen_nconc(chars, words_declaration(e, pp_dim, pdl));
              attach_declaration_size_type_to_words(chars, "CHARACTER", i);
              ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(t_chars, chars,
                                                get_prettyprint_indentation());
            }
          } else if (value_unknown_p(v)) {
            list chars = NIL;
            chars = CHAIN_SWORD(chars, "CHARACTER*(*) ");
            chars = gen_nconc(chars, words_declaration(e, pp_dim, pdl));
            attach_declaration_type_to_words(chars, "CHARACTER*(*)");
            ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(t_chars, chars,
                                              get_prettyprint_indentation());
          } else if (value_symbolic_p(v)) {
            list chars = NIL;
            symbolic s = value_symbolic(v);
            chars = CHAIN_SWORD(chars, "CHARACTER*(");
            chars = gen_nconc(chars, words_expression(symbolic_expression(s),
                                                      pdl));
            chars = CHAIN_SWORD(chars, ") ");
            chars = gen_nconc(chars, words_declaration(e, pp_dim, pdl));

            attach_declaration_type_to_words(chars, "CHARACTER*(*)");
            ADD_WORD_LIST_TO_TEXT(t_chars, chars);
          } else
            pips_internal_error("unexpected value");
          break;
        }
        case is_basic_derived: {
          if(allocatable_pass_p) {
            pips_internal_error("We got an allocatable but we are inside"
                "allocatable pass !! This should be impossible...\n");
          }
          // Chains the entity to be declared, aka the array inside allocatable
          allocatable_list = CONS(entity,get_allocatable_data_entity(e),allocatable_list);
          break;
        }
        default:
          pips_internal_error("unexpected basic tag (%d)",
              basic_tag(b));
      }
    }
  }


  /* usually they are sorted in order, and appended backwards,
   * hence the reversion.
   */
  r = make_text(uses);
  MERGE_TEXTS(r, make_text(gen_nreverse(before)));

  MERGE_TEXTS(r, text_of_parameters(lparam));
  gen_free_list(lparam), lparam = NIL;

  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, ph1, get_prettyprint_indentation());
  attach_declaration_type_to_words(ph1, "INTEGER*1");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, ph2, get_prettyprint_indentation());
  attach_declaration_type_to_words(ph2, "INTEGER*2");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, ph4, get_prettyprint_indentation());
  attach_declaration_type_to_words(ph4, "INTEGER");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, ph8, get_prettyprint_indentation());
  attach_declaration_type_to_words(ph8, "INTEGER*8");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pi1, get_prettyprint_indentation());
  attach_declaration_type_to_words(pi1, "INTEGER*1");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pi2, get_prettyprint_indentation());
  attach_declaration_type_to_words(pi2, "INTEGER*2");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pi4, get_prettyprint_indentation());
  attach_declaration_type_to_words(pi4, "INTEGER");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pi8, get_prettyprint_indentation());
  attach_declaration_type_to_words(pi8, "INTEGER*8");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pf4, get_prettyprint_indentation());
  attach_declaration_type_to_words(pf4, "REAL*4");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pf8, get_prettyprint_indentation());
  attach_declaration_type_to_words(pf8, "REAL*8");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pl, get_prettyprint_indentation());
  attach_declaration_type_to_words(pl, "LOGICAL");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pc8, get_prettyprint_indentation());
  attach_declaration_type_to_words(pc8, "COMPLEX*8");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, pc16, get_prettyprint_indentation());
  attach_declaration_type_to_words(pc16, "COMPLEX*16");
  ADD_WORD_LIST_TO_TEXT_WITH_MARGIN(r, ps, get_prettyprint_indentation());
  attach_declaration_type_to_words(ps, "CHARACTER");
  MERGE_TEXTS(r, t_chars);

  /* all about COMMON and SAVE declarations
   */
  MERGE_TEXTS(r, make_text(area_decl));
  MERGE_TEXTS(r, t_area);

  /* and EQUIVALENCE statements... - BC
   */
  MERGE_TEXTS(r, text_equivalences(module, /* sorted_ */ldecl,
          pp_cinc || !print_commons));

  /* what about DATA statements! FC
   */
  /* More general way with with call to text_initializations(module) in
   text_named_module() */
  /*
   if(get_bool_property("PRETTYPRINT_DATA_STATEMENTS")) {
   MERGE_TEXTS(r, text_data(module, ldecl));
   }
   */

  /* gen_free_list(sorted_ldecl); */

  if(!allocatable_pass_p && !ENDP(allocatable_list)) {
    /* We have to do a recursive call to get the allocatable declarations */
    allocatable_pass_p = true;
    MERGE_TEXTS(r,text_entity_declaration(module,allocatable_list,force_common,pdl));
    allocatable_pass_p = false;
  }

  return r;
}

/* exported for hpfc.
 */
text text_declaration(entity module)
{
  /* Assume Fortran only! */
  list pdl = NIL;
  text t = text_entity_declaration
      (module, code_declarations(entity_code(module)), false, pdl);
  return t;
}

/* needed for hpfc
 */
text text_common_declaration(
    entity common,
    entity module)
{
    type t = entity_type(common);
    list l;
    text result;
    list pdl = NIL; // Assumed Fortran only
    pips_assert("indeed a common", type_area_p(t));
    l = CONS(ENTITY, common, common_members_of_module(common, module, false));
    result = text_entity_declaration(module, l, true, pdl);
    gen_free_list(l);
    return result;
}

/* ================C prettyprinter functions================= */

static list generic_words_qualifier(list obj, bool initial_p, bool late_p)
{
  list pc = NIL;
  FOREACH(QUALIFIER,q, obj) {
    switch (qualifier_tag(q)) {
    case is_qualifier_register:
      if(initial_p)
	pc = CHAIN_SWORD(pc,"register ");
      break;
    case is_qualifier_thread:
      if(initial_p)
	pc = CHAIN_SWORD(pc,"__thread ");
      break;
    case is_qualifier_const:
      if(late_p)
	pc = CHAIN_SWORD(pc,"const ");
      break;
    case is_qualifier_restrict:
      /* FI: might not be an initial qualifier either */
      if(late_p)
      pc = CHAIN_SWORD(pc,"restrict ");
      break;
    case is_qualifier_volatile:
      if(initial_p)
      pc = CHAIN_SWORD(pc,"volatile ");
      break;
    case is_qualifier_auto:
      /* FI: the auto case was missing; I have no idea why. */
      if(initial_p)
      pc = CHAIN_SWORD(pc,"auto ");
      break;
    case is_qualifier_asm: {
      /* asm qualifiers a reprinted in the end ... */
      } break;
    }
  }

  if(!initial_p && late_p && !ENDP(pc))
    pc = gen_nconc(CHAIN_SWORD(NIL, " "), pc);

  return pc;
}

static list words_qualifier(list obj)
{
  return generic_words_qualifier(obj, true, true);
}

static list words_initial_qualifiers(list obj)
{
  return generic_words_qualifier(obj, true, false);
}

static list words_late_qualifiers(list obj)
{
  return generic_words_qualifier(obj, false, true);
}

/* obj is the type to describe
*
* pdl is a list of already/previously declared data structures (not
* 100 % sure)
*
* argument_p: the type is used as an argument type in a function
* declaration; if an argument appears with the "unknown" type it is
* explicitly declared "int"
*
* returns a list of strings, called "words".
*/
list words_type(type obj, list pdl, bool argument_p)
{
  list pc = NIL;
  switch (type_tag(obj))
    {
    case is_type_variable:
      {
	basic b = variable_basic(type_variable(obj));
	pc = words_basic(b, pdl);
	pc = gen_nconc
	  (pc,
	   words_dimensions(variable_dimensions(type_variable(obj)), pdl));
	break;
      }
    case is_type_void:
      {
	pc = CHAIN_SWORD(pc,"void");
	break;
      }
    case is_type_unknown:
      {
	/* The default type is int. It can be skipped in direct
	   declarations, but not as type of an argument in a function
	   declaration */
	if(argument_p)
	  pc = CHAIN_SWORD(pc,"int");
	break;
      }
    case is_type_struct:
      {
	pc = CHAIN_SWORD(pc,"struct");
	break;
      }
    case is_type_union:
      {
	pc = CHAIN_SWORD(pc,"union");
	break;
      }
    case is_type_enum:
      {
	pc = CHAIN_SWORD(pc,"enum");
	break;
      }
    case is_type_functional:
      {
	string_buffer result = string_buffer_make(true);
	string rs = string_undefined;
	dump_functional(type_functional(obj), result);
	rs = string_buffer_to_string(result);
	pc = gen_nconc(pc, CONS(STRING, rs, NIL));
	string_buffer_free(&result);
	break;
      }
    case is_type_varargs:
      {
	pc = CHAIN_SWORD(pc,"...");
	break;
      }
    default:
      pips_internal_error("unexpected tag %d", type_tag(obj));
    }
  pips_debug(8, "End: \"\%s\"\n", list_to_string(pc));
  return pc;
}

bool c_brace_expression_p(expression e)
{
  if (expression_call_p(e))
    {
      entity f = call_function(syntax_call(expression_syntax(e)));
      if (ENTITY_BRACE_INTRINSIC_P(f))
	return true;
    }
  return false;
}


list words_brace_expression(expression exp, list pdl)
{
  list pc = NIL;
  list args = call_arguments(syntax_call(expression_syntax(exp)));
  bool first = true;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  pc = CHAIN_SWORD(pc,"{");
  MAP(EXPRESSION,e,
  {
    if (!first)
      pc = CHAIN_SWORD(pc, space_p? ", " : ",");
    if (c_brace_expression_p(e))
      pc = gen_nconc(pc,words_brace_expression(e, pdl));
    else
      pc = gen_nconc(pc,words_expression(e, pdl));
    first = false;
  },args);
  pc = CHAIN_SWORD(pc,"}");
  return pc;
}

list words_dimensions(list dims, list pdl)
{
  list pc = NIL;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  switch(get_prettyprint_language_tag()) {
    case is_language_fortran:
    case is_language_fortran95: {
      pc = CHAIN_SWORD(pc, "(");
      string spacer = "";
      FOREACH(dimension,d,dims) {
        pc = CHAIN_SWORD(pc, spacer);
        pc = gen_nconc(pc, words_dimension(d, pdl));
        spacer = space_p ? ", " : ",";
      }
      pc = CHAIN_SWORD(pc, ")");
      break;
    }
    case is_language_c: {
      FOREACH(dimension,d,dims) {
        pc = CHAIN_SWORD(pc, "[");
        pc = gen_nconc(pc, words_dimension(d, pdl));
        pc = CHAIN_SWORD(pc, "]");
      }
      break;
    }
    default:
      pips_internal_error("Language unknown !");
      break;
  }
  return pc;
}

/* This recursive function prints a C variable with its type.
 * It can be a simple variable declaration such as "int a"
 * or complicated one such as "int (* forces[10])()" (an array of
 * 10 pointers, each pointer points to a function
 * with no parameter and the return type is int).
 *
 * Type "t" is recursively traversed and the obtained attribute are
 * accumulated in the list "name" (name of the type, not name of the
 * variable).
 *
 * Purpose of "is_safe"? Seems to be passed down recursively but never
 * to be used...
 *
 * In C, functional type can be decorated by optional dummy parameter names.
 */
list generic_c_words_entity(type t, list name, bool is_safe, bool add_dummy_parameter_name_p, list pdl)
{
// If this function is still used, NIL should be replaced by the
// module declaration list
  return generic_c_words_simplified_entity(t, name, is_safe,
					   add_dummy_parameter_name_p,
					   true, false, false, pdl);
}

/* Same as above, but the bool is_first is used to skip a type
 * specifier which is useful when several variables or types are
 * defined in a unique statement such as "int i, *pi, ai[10],...;"
 *
 * type t: new type to add in front of the word list name
 *
 * list name: later part of the declaration being built
 *
 * bool is_safe: does not seem to be used anymore
 *
 * bool add_dummy_parameter_name_p: for function declarations, add
 * adummy parameter name to the type of each formal parameter
 *
 * bool is_first: prettyprint the qualifiers or not; they should be
 * printed only once when they apply to several declarations as in:
 *
 * "register int i, j;"
 *
 * in_type_declaration is set to true when a variable is declared at
 * the same time as its type
 *
 * argument_p: the type is used as argument type in a function declaration
 *
 * list pdl: declaration list to decide if data structures appearing in
 * another data structure must be declared independently or not. See
 * validation cases struct03.c, struct04.c and struct05.c.
 */
list generic_c_words_simplified_entity(type t, list name, bool is_safe, bool add_dummy_parameter_name_p, bool is_first, bool in_type_declaration, bool argument_p, list pdl)
{
  list pc = NIL;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");

  if(type_undefined_p(t)) {
    pc = CHAIN_SWORD(NIL, "type_undefined");
    pc = gen_nconc(pc,name);
    return pc;
  }

  if (type_functional_p(t))
    {
      functional f = type_functional(t);
      type t2 = functional_result(f);
      list lparams = functional_parameters(f);
      list cparam = list_undefined;
      bool first = true;
      int pnum;

      pips_debug(9,"Function type with name = \"%s\" and length %zd\n",
		 list_to_string(name), gen_length(name));

      if ((gen_length(name) > 1)
	  || ((gen_length(name) == 1) && (strcmp(STRING(CAR(name)),"*")==0)))
	{
	  /* Function name is an expression like *vfs[] in (*vfs[])()
	     (syntax = application), or an abstract function type, so
	     parentheses must be added */
	  pc = CHAIN_SWORD(NIL,"(");
	  pc = gen_nconc(pc,name);
	  pc = CHAIN_SWORD(pc,")(");

	}
      else
	{
	  /* Function name is a simple reference */
	  pc = CHAIN_SWORD(name,"(");
	}

      if(!overloaded_parameters_p(lparams)) {
	for(cparam = lparams, pnum = 1; !ENDP(cparam); POP(cparam), pnum++) {
	  parameter p = PARAMETER(CAR(cparam));
	  type t1 = parameter_type(p);
	  string pn = dummy_unknown_p(parameter_dummy(p))?
	    string_undefined
	    : strdup(entity_local_name(dummy_identifier(parameter_dummy(p))));

	  if(add_dummy_parameter_name_p
	     && string_undefined_p(pn)
	     && !type_varargs_p(t1)
	     && !type_void_p(t1)) {
	    /* RK wants us to use another better function than itoa, but
	       its name is not documented next to itoa() source code and
	       here the string is going to be strduped, which makes
	       itoa() a better choice. */
	    pn = concatenate("f", itoa(pnum), NULL);
	  }

	  /*pips_debug(3,"Parameter type %s\n ",
	    type_undefined_p(t1)? "type_undefined" :
	    words_to_string(words_type(t1, pdl, true))); */
	  if (!first)
	    pc = gen_nconc(pc,CHAIN_SWORD(NIL, space_p? ", " : ","));
	  /* c_words_entity(t1,NIL) should be replaced by c_words_entity(t1,name_of_corresponding_parameter) */
	  pc = gen_nconc(pc,
			 generic_c_words_simplified_entity(t1,
							   string_undefined_p(pn)? NIL : CONS(STRING, strdup(pn), NIL),
							   is_safe, false, true,in_type_declaration, true, pdl));
	  pips_debug(9,"List of parameters \"%s\"\n ",list_to_string(pc));
	  first = false;
	}
      }

      pc = CHAIN_SWORD(pc,")");
      return generic_c_words_simplified_entity(t2, pc, is_safe, false,
					       is_first, in_type_declaration,
					       argument_p, pdl);
    }

  if (pointer_type_p(t))
    {
      type t1 = basic_pointer(variable_basic(type_variable(t)));
      pips_debug(9,"Pointer type with name = %s\n", list_to_string(name));

      /*
       *  No space : it's only "after a comma to separate items in lists such as
       *  declaration lists or parameter lists in order to improve readability"
       */
      // pc = CHAIN_SWORD(NIL, space_p? "*":"*");
      pc = CHAIN_SWORD(NIL, "*");
      if (qualifiers_const_p(variable_qualifiers(type_variable(t)))
	  || qualifiers_restrict_p(variable_qualifiers(type_variable(t)))) {
	pc = gen_nconc(pc, words_late_qualifiers(variable_qualifiers(type_variable(t))) );
	// We may have other qualifiers which must be output somewhere
	// else, for instance "register"
	/*
	pc = gen_nconc(pc,
		       words_qualifier(variable_qualifiers(type_variable(t))));
	*/
      }
      pc = gen_nconc(pc,name);
      pc = generic_c_words_simplified_entity(t1, pc, is_safe, false,
					     is_first, in_type_declaration,
					     argument_p, pdl);
      pc = gen_nconc(words_initial_qualifiers(variable_qualifiers(type_variable(t))), pc);
      return pc;
    }

  /* Add type qualifiers if there are */
  if (( is_first || in_type_declaration )
      && type_variable_p(t)
      && variable_qualifiers(type_variable(t)) != NIL)
    pc = words_qualifier(variable_qualifiers(type_variable(t)));
  else if (( is_first || in_type_declaration )
      && type_void_p(t)
      && type_void(t) != NIL)
    pc = words_qualifier(type_void(t));

  if (basic_type_p(t)) {
      string sname = list_to_string(name);
      pips_debug(9,"Basic type with name = \"%s\"\n", sname);

      if(is_first) {
	pc = gen_nconc(pc,words_type(t, pdl, argument_p));
      }
      if (string_type_p(t)) {
	// pc = CHAIN_SWORD(pc," *");
	;
      }
      /* FI: left out of the previous declaration internal representation */
      if(strlen(sname)!=0 && is_first)
	pc = CHAIN_SWORD(pc," ");
      if(!bit_type_p(t) || (strstr(sname,DUMMY_MEMBER_PREFIX)==NULL)) {
	pc = gen_nconc(pc,name);
	}
      free(sname);
      if (bit_type_p(t)) {
	  symbolic s = basic_bit(variable_basic(type_variable(t)));
	  expression ie = symbolic_expression(s);
	  list iew = words_expression(ie, pdl);
	  pc = CHAIN_SWORD(pc,":");
	  pc = gen_nconc(pc, iew);
      }
      return pc;
    }
  if (array_type_p(t)){
      list dims = variable_dimensions(type_variable(t));
      type t1 = copy_type(t);
      list tmp = NIL;
      pips_debug(9,"Array type with name = %s\n", list_to_string(name));

      if ((gen_length(name) > 1) || ((gen_length(name) == 1) && (strcmp(STRING(CAR(name)),"*")==0)))
	{
	  /* Array name is an expression like __ctype+1 in (__ctype+1)[*np]
	     (syntax = subscript), or abstract type, parentheses must be added */
	  tmp = CHAIN_SWORD(tmp,"(");
	  tmp = gen_nconc(tmp,name);
	  tmp = CHAIN_SWORD(tmp,")");
	}
      else
	{
	  /* Array name is a simple reference  */
	  tmp = name;
	}
      gen_full_free_list(variable_dimensions(type_variable(t1)));
      variable_dimensions(type_variable(t1)) = NIL;
      gen_full_free_list(variable_qualifiers(type_variable(t1)));
      variable_qualifiers(type_variable(t1)) = NIL;
      pips_debug(8, "Before concatenation, pc=\"\%s\"\n", list_to_string(pc));
      if(pc!=NIL)
	pc = CHAIN_SWORD(pc, " ");
      list ret =
	gen_nconc(pc,
		  generic_c_words_simplified_entity(t1,gen_nconc(tmp,words_dimensions(dims, pdl)),
						    is_safe, false, is_first,
						    in_type_declaration,
						    argument_p,
						    pdl));
      free_type(t1);
      return ret;
    }

  if (derived_type_p(t))
    {
      entity ent = basic_derived(variable_basic(type_variable(t)));
      if(is_first) {
	type t1 = entity_type(ent);
	string n = entity_name(ent);
	pips_debug(9,"Derived type with name = %s\n", list_to_string(name));
	if((strstr(n,ENUM_PREFIX DUMMY_ENUM_PREFIX)==NULL)
	   &&(strstr(n,STRUCT_PREFIX DUMMY_STRUCT_PREFIX)==NULL)
	   &&(strstr(n,UNION_PREFIX DUMMY_UNION_PREFIX)==NULL)) {
	  if(!gen_in_list_p((void *) ent, pdl)) {
	    /* The derived type has been declared explicitly
	       elsewhere: see struct05.c */
	    pc = gen_nconc(pc,words_type(t1, pdl, argument_p));
	    pc = CHAIN_SWORD(pc," ");
	    pc = CHAIN_SWORD(pc,entity_user_name(ent));
	  }
	  else {
	    /* The derived type is declared by itself*/
	    const char* name = entity_user_name(ent);
	    list epc = NIL;
	    /* Do not recurse down if the derived type reference
	       itself */
	    list npdl = gen_copy_seq(pdl);
	    gen_remove(&npdl, (void *) ent);
	    epc =
	      generic_c_words_simplified_entity(t1,
						CHAIN_SWORD(NIL,name),
						is_safe,
						add_dummy_parameter_name_p,
						is_first, in_type_declaration,
						argument_p, npdl);
	    pc = gen_nconc(pc, epc);
	    gen_free_list(npdl);
	  }
	}
	else {
	  //pc = CHAIN_SWORD(pc,"problem!");
	  pc = c_words_entity(t1, pc, pdl);
	}
	pc = CHAIN_SWORD(pc," ");
      }
      return gen_nconc(pc,name);
    }
  if (typedef_type_p(t))
    {
      if(is_first) {
	entity ent = basic_typedef(variable_basic(type_variable(t)));
	pips_debug(9,"Typedef type with name = \"\%s\"\n",
		   list_to_string(name));
	pc = CHAIN_SWORD(pc,entity_user_name(ent));
	if(name!=NIL)
	  pc = CHAIN_SWORD(pc," ");
      }
      return gen_nconc(pc,name);
    }
  if (type_varargs_p(t))
    {
      pips_debug(9,"Varargs type ... with name = %s\n", list_to_string(name));
      pc = CHAIN_SWORD(pc,"...");
      return gen_nconc(pc,name);
    }
  /* This section is derived from c_text_entity() */
  /* it is used for structures, unions and enums which have no names
     because they are part of a more global declaration such as
     typedef s*/
  /* FI: The union and the struct cases could be merged. */
  if(type_struct_p(t))
    {
      list l = type_struct(t);
      string sname = list_to_string(name);
      list cl = list_undefined;

      pips_debug(9,"Struct type ... with name = %s\n", sname);

      pc = CHAIN_SWORD(pc,"struct ");
      // hmmm... name may be an empty list... and the sname test seems true
      if(name && strstr(sname,DUMMY_STRUCT_PREFIX)==NULL) {
	pc = gen_nconc(pc,name);
	pc = CHAIN_SWORD(pc," ");
      }
      free(sname);
      pc = CHAIN_SWORD(pc,"{");

      for(cl = l; !ENDP(cl); POP(cl)) {
	entity sm = ENTITY(CAR(cl));
	type tsm = entity_type(sm);
	pc = gen_nconc(pc,c_words_entity(tsm,CHAIN_SWORD(NIL,entity_user_name(sm)), pdl));
	if(ENDP(CDR(cl)))
	  pc = CHAIN_SWORD(pc,";");
	else
	  pc = CHAIN_SWORD(pc,"; ");
     }
      pc = CHAIN_SWORD(pc,"}");
      return pc;
    }
  if(type_union_p(t))
    {
      list l = type_union(t);
      string sname = list_to_string(name);
      list cl = list_undefined;

      pips_debug(9,"Union type ... with name = %s\n", sname);

      pc = CHAIN_SWORD(pc,"union ");
      //if(strstr(sname,DUMMY_UNION_PREFIX)==NULL) {
      //	pc = gen_nconc(pc,name);
      //	pc = CHAIN_SWORD(pc," ");
      //}
      free(sname);
      pc = CHAIN_SWORD(pc,"{");

      for(cl = l; !ENDP(cl); POP(cl)) {
	entity eu = ENTITY(CAR(cl));
	type tu = entity_type(eu);
	pc = gen_nconc(pc,c_words_entity(tu,CHAIN_SWORD(NIL,entity_user_name(eu)), pdl));
	if(ENDP(CDR(cl)))
	  pc = CHAIN_SWORD(pc,";");
	else
	  pc = CHAIN_SWORD(pc,"; ");
     }
      pc = CHAIN_SWORD(pc,"}");
      return pc;
    }
  if(type_enum_p(t))
    {
      list l = type_enum(t);
      bool first = true;
      string sname = list_to_string(name);
      list cl = list_undefined;
      int cv = 0;

      pips_debug(9,"Enum type ... with name = %s\n", sname);

      pc = CHAIN_SWORD(pc,"enum ");
      //if(strstr(sname,DUMMY_ENUM_PREFIX)==NULL) {
      //	pc = gen_nconc(pc,name);
      //	pc = CHAIN_SWORD(pc," ");
      //}
      free(sname);
      pc = CHAIN_SWORD(pc,"{");

      for(cl = l; !ENDP(cl); POP(cl)) {
	entity em = ENTITY(CAR(cl));
	value emv = entity_initial(em);
	symbolic ems = value_symbolic(emv);
	expression eme = symbolic_expression(ems);
	constant emc = symbolic_constant(value_symbolic(emv));
	int n = constant_int(emc);

	if (!first)
	  pc = CHAIN_SWORD(pc, space_p? ", " : ",");
	pc = CHAIN_SWORD(pc, entity_user_name(em));
	if(n!=cv) {
	  pc = CHAIN_SWORD(pc, "=");
	  pc = gen_nconc(pc, words_expression(eme, pdl));
	  cv = n;
	}
	cv++;
	first = false;
      };
      pc = CHAIN_SWORD(pc,"}");
      return pc;
    }
  pips_internal_error("unexpected case");
  return NIL;
}

/* The declaration list pointer pdl is passed down to determine if an internal
   derived type must be fully expanded within another declaration or
   not. If it is declared by itself, there is no need to expand its
   declaration again. */
list c_words_simplified_entity(type t, list name, bool is_first, bool in_type_declaration, list pdl)
{
  list pc = generic_c_words_simplified_entity(t, name, false, false, is_first,
					      in_type_declaration, false, pdl);

  ifdebug(8) {
    string s = list_to_string(pc);
    pips_debug(8, "End with \"\%s\"\n", s);
  }

  return pc;
}

list c_words_entity(type t, list name, list pdl)
{
  list pc = generic_c_words_entity(t, name, false, false, pdl);

  ifdebug(8) {
    string s = list_to_string(pc);
    pips_debug(8, "End with \"\%s\"\n", s);
  }

  return pc;
}

list safe_c_words_entity(type t, list name)
{
  /* Ignore the parser declared entities? */
  list pdl = NIL;
  list l = generic_c_words_entity(t, name, true, false, pdl);
  return l;
}

/* Generate declarations for a list of entities belonging to the same
   statement declaration

   pdl: derived from the parser declared entity; used to decide if a
   derived type entity de must be declared as a reference to de
   (e.g. "struct s") or as the type definition of de (e.g. "struct s
   {}"). Of course, the type can be defined only once, even if it is
   referenced several times. Hence, pdl is updated in the loop to
   avoid redeclarations.
 */
text c_text_entities(entity module, list ldecl, int margin, list pdl)
{
  text r = make_text(NIL);
  list npdl = gen_copy_seq(pdl); // new parser declaration list

  FOREACH(ENTITY, e, ldecl) {
    text tmp = text_undefined;
    type t = entity_type(e);

    if(!type_area_p(t)
       && ! type_statement_p(t)
       && !type_unknown_p(t)
       && !storage_formal_p(entity_storage(e))
       && !implicit_c_variable_p(e)) {
      string n = entity_name(e);

      /* Dummy enum must be printed sometimes because their members
	 are exposed directly. */
      if(((strstr(n,DUMMY_ENUM_PREFIX)==NULL)
	  || !type_used_in_type_declarations_p(e, ldecl))
	 && (strstr(n,STRUCT_PREFIX DUMMY_STRUCT_PREFIX)==NULL
	     ||strstr(n,MEMBER_SEP_STRING)!=NULL)
	 && (strstr(n,UNION_PREFIX DUMMY_UNION_PREFIX)==NULL
	     ||strstr(n,MEMBER_SEP_STRING)!=NULL) ) {
	type et = ultimate_type(entity_type(e));
	tmp = c_text_entity(module, e, margin, npdl);
	MERGE_TEXTS(r,tmp);

	if(derived_type_p(et)) {
	  entity de= basic_derived(variable_basic(type_variable(et)));
	  gen_remove(&npdl, (void *) de);
	}
      }
    }
  }

  gen_free_list(npdl);

  return r;
}

/* To print out a struct reference, such as "struct s"*/
static list words_struct_reference(const char* name1, list pc)
{
  pc = CHAIN_SWORD(pc,"struct ");
  if(strstr(name1,DUMMY_STRUCT_PREFIX)==NULL) {
    pc = CHAIN_SWORD(pc,name1);
    pc = CHAIN_SWORD(pc," ");
  }
  return pc;
}

/* Prolog to print out a struct definition, such as "struct s { int a;
   int b;}"; for the time being, only the previous function is used. */
/*
static list words_struct(string name1, list pc)
{
  pc = words_struct_reference(name1, pc);
  pc = CHAIN_SWORD(pc,"{");
  return pc;
}
*/

static list words_enum(const char * name1, list l, bool space_p, list pc, list pdl)
{
  bool first = true;
  pc = CHAIN_SWORD(pc,"enum ");
  if(strstr(name1,DUMMY_ENUM_PREFIX)==NULL) {
    pc = CHAIN_SWORD(pc,name1);
    pc = CHAIN_SWORD(pc," ");
  }
  pc = CHAIN_SWORD(pc,"{");
  list cl = list_undefined;
  int cv = 0;

  for(cl = l; !ENDP(cl); POP(cl)) {
    entity em = ENTITY(CAR(cl));
    value emv = entity_initial(em);
    symbolic ems = value_symbolic(emv);
    expression eme = symbolic_expression(ems);
    constant emc = symbolic_constant(value_symbolic(emv));
    int n = constant_int(emc);

    // SG has decided not to evaluate expressions containins a sizeof
    // operator because the result is architecture dependent and
    // because the PIPS user has not specified which architecture
    // should be used.
    //
    // Serge has no idea how many things he has broken in PIPS!!!
    // constant integer expressions are used in many declarations and
    // expected by PIPS components. Unexpected results are going to
    // occur as the blind interpretation of an unknown constant
    // returns 0!
    //
    //if(!constant_int_p(emc))
    //  pips_internal_error("constant expression not evaluated by parser");

    if (!first)
      pc = CHAIN_SWORD(pc, space_p? ", " : ",");
    pc = CHAIN_SWORD(pc, entity_user_name(em));
    if(n!=cv || !constant_int_p(emc)) {
      pc = CHAIN_SWORD(pc, "=");
      pc = gen_nconc(pc, words_expression(eme, pdl));
      cv = n;
    }
    cv++;
    first = false;
  };
  pc = CHAIN_SWORD(pc,"}");
  return pc;
}

static list words_union(const char* name1, list pc)
{
  pc = CHAIN_SWORD(pc,"union ");
  if(strstr(name1,DUMMY_UNION_PREFIX)==NULL) {
    pc = CHAIN_SWORD(pc,name1);
    pc = CHAIN_SWORD(pc," ");
  }
  pc = CHAIN_SWORD(pc,"{");
  return pc;
}

static list words_variable_or_function(entity module, entity e, bool is_first, list pc, bool in_type_declaration, list pdl)
{
  const char* name = entity_user_name(e);
  type t = entity_type(e);
  //storage s = entity_storage(e);
  value val = entity_initial(e);

  pc = gen_nconc(pc,c_words_simplified_entity(t,CHAIN_SWORD(NIL,name),
					      is_first, in_type_declaration, pdl));
  /* This part is for declarator initialization if there is. If the
     entity is declared extern wrt current module, do not add this
     initialization */
  if (/* normal mode: module is defined */
      (!entity_undefined_p(module)
      && !extern_entity_p(module,e)
       && !value_undefined_p(val))
      /* debugging mode: module is often undefined */
      || (entity_undefined_p(module) && !value_undefined_p(val))) {
    if (value_expression_p(val)) {
      expression exp = value_expression(val);
      pc = CHAIN_SWORD(pc," = ");
      if (brace_expression_p(exp))
	pc = gen_nconc(pc,words_brace_expression(exp, pdl));
      else {
	/* */
	pc = gen_nconc(pc,words_subexpression(exp, ASSIGN_OPERATOR_PRECEDENCE, true, pdl));
      }
    }
    else if(value_code_p(val)) {
      if(type_variable_p(t)) {
	/* it must be a pointer to a function. See encoding in
	   set_entity_initial */
	list il = sequence_statements(code_initializations(value_code(val)));
	if(!ENDP(il)) {
	  statement is = STATEMENT(CAR(il));
	  expression exp =
	    instruction_expression(statement_instruction(is));
	  pc = CHAIN_SWORD(pc," = ");
	  pc = gen_nconc(pc,words_expression(exp, pdl));
	}
      }
    }
  }
  return pc;
}

/* Fix the declaration list produced by the parser, which includes
   declared program variables but also derived types used in the
   declaration or the initialization, to accomodate the
   prettyprinter, which must decide which derived types appearing in
   the declaration must be defined, and which one must be referenced:

    - first option: get rid of all derived entities; pro: safe; cons:
      all declarations become one liners

    - second option: get rid of all derived entities but the last
      one, assuming they appear in the right order. Sometimes, you may
      even have to get rid of all derived entities.

   All entities in del are useful to regenerate the corresponding
   declaration, but the key entities are the type used for the
   variables, and the variables. Nested types should not appear in the
   return list.

   FI: I wonder what the parser generates for something like

   int foo[sizeof(struct {int a; int b;})] = {...};

   No new list is allocated, just a pointer to the first relevant chunk
   of del.
*/
static list filtered_declaration_list(list del)
{
  /* Filter out secondary types */
  list el = del;

  ifdebug(8) {
    pips_debug(8, "Input declaration list: ");
    print_entities(del);
    fprintf(stderr, "\n");
  }

  /* FI: initially I assumed > 2 for pattern such as

     struct {struct ..} x;

     but you can also have a struct declaration:

     struct {struct ..};

     with no variable declaration. Or a declaration in the rhs via a
     sizeof or a cast... See below, after the intermediate list.

  */
  if(gen_length(del)>=2) {
    /* Maybe, some type related entities such as substructure should
       not be printed out. Only the last one is useful. */
    entity e_previous = ENTITY(CAR(del));

    if(entity_struct_p(e_previous)
       ||entity_union_p(e_previous)
       ||entity_enum_p(e_previous)) {
      FOREACH(ENTITY, e, CDR(del)) {
	if(entity_struct_p(e)
	   ||entity_union_p(e)
	   ||entity_enum_p(e)) {
	  POP(el);
	}
	else {
	  break;
	}
      }
    }
  }


  ifdebug(8) {
    pips_debug(8, "Intermediate declaration list: ");
    print_entities(el);
    fprintf(stderr, "\n");
  }

  /* Now we still have to deal with "void * p = ... derived type
     definition(s)..."  Maybe, we simply want to make sure that e1
     belongs to type supporting entities of t2... but this is not
     strong enough because it includes types of formal arguments which
     are not relevant
  */
  if(gen_length(el)>=2) {
    entity e1 = ENTITY(CAR(el));

    if(derived_entity_p(e1)) {
      bool match_p = false;
      //type t1 = entity_type(e1);
      entity e2 = ENTITY(CAR(CDR(el)));
      type t2 = entity_type(e2);
      type nt2 = type_undefined;

      /* If t2 is a pointer, get to the final pointed type. */
      nt2 = type_to_final_pointed_type(t2);

      /* If nt2 is a functional type, what is the final pointed type
	 returned? */
      if(type_functional_p(nt2)) {
	functional f2 = type_functional(nt2);
	type rt2 = functional_result(f2);
	nt2 = type_to_final_pointed_type(rt2);
      }

      if(type_variable_p(nt2)) {
	basic b2 = variable_basic(type_variable(nt2));
	if(basic_derived_p(b2) && basic_derived(b2)==e1)
	  match_p = true;
      }
      if(!match_p)
	POP(el);
    }
  }

  ifdebug(8) {
    pips_debug(8, "Output declaration list: ");
    print_entities(el);
    fprintf(stderr, "\n");
  }

  return el;
}

/* It is assumed that all entities in list el can be declared by an
 * unique statement, i.e. their types must be closely related, as in
 *
 * "int i, *pj, foo();".
 *
 * But you can also have:
 *
 * "struct one { struct too {int a;};};"
 *
 * where no variable is declared. And the parser generate a
 * declaration list stating that "struct two" and "struct one"
 * are declared in this statement.
 *
 * In other words, this function prints out a C declaration statement,
 * taking into account the derived entities that have to be defined
 * exactly once, pdl. Of course, pdl can be updated by the caller when
 * a derived entity is declared so as to avoid a redeclaration.
 *
 * At this first level, the declarations of derived types use several
 * lines. If a nested declaration occurs, the nested declaration is
 * packed on a unique line.
 */
text c_text_related_entities(entity module, list del, int margin, int sn, list pdl)
{
  list el = filtered_declaration_list(del);
  text r = make_text(NIL);
  entity e1 = ENTITY(CAR(el)); // Let's use the first declared entity.
  entity e_last = ENTITY(CAR(gen_last(el))); // Let's also use the last declared entity.
  const char* name1 = entity_user_name(e1);
  type t1 = entity_type(e1);
  type t_last = entity_type(e_last);
  //storage s1 = entity_storage(e1);
  storage s_last = entity_storage(e_last);
  //value val1 = entity_initial(e1);
  list pc = NIL;
  bool space_p = get_bool_property("PRETTYPRINT_LISTS_WITH_SPACES");
  bool skip_first_comma_p = true;

  /* overwrite the parser declaration list pdl with del */
  //pdl = del;
  // Not a good idea with recursive calls to this function

  pips_assert("the entity list is not empty", gen_length(el)>0);

  pips_debug(5,"Print declaration for first entity %s in module %s\n",
	     entity_name(e1),
	     entity_undefined_p(module)? "UNDEFINED" : entity_name(module));

  /* A declaration has two parts: declaration specifiers and declarator (even with initializer)
     In declaration specifiers, we can have:
     - storage specifiers : typedef, extern, static, auto, register
     - type specifiers : void, char, short, int, long, float, double, signed, unsigned,
                         struct-or-union specifiers, enum specifier, typedef name
     - type qualifiers : const, restrict, volatile
     - function specifiers : inline */

  /* This part is for storage specifiers */
  if (!entity_undefined_p(module)
      && (explicit_extern_entity_p(module, e_last)
	  || (extern_entity_p(module, e_last) && !type_functional_p(t_last))))
    pc = CHAIN_SWORD(pc,"extern ");

  if (strstr(entity_name(e_last),TYPEDEF_PREFIX) != NULL)
    pc = CHAIN_SWORD(pc,"typedef ");

  /* The global variables stored in static area and in ram but they
     are not static so a condition is needed, which checks if it is not a
     global variable*/
  // entity m = get_current_module_entity();
  if ((storage_ram_p(s_last)
       && static_area_p(ram_section(storage_ram(s_last)))
       && !strstr(entity_name(e_last),TOP_LEVEL_MODULE_NAME))
      || (entity_module_p(e_last) && static_module_p(e_last)))
    pc = CHAIN_SWORD(pc,"static ");


  /* This part is for type specifiers, type qualifiers, function specifiers and declarator
     Three special cases for struct/union/enum definitions are treated here.
     Variable (scalar, array), pointer, function, variables of type struct/union/enum and typedef
     are treated by function c_words_entity */

  bool in_type_declaration = true;
  switch (type_tag(t1)) {
  case is_type_struct:
    {
      pc = words_struct_reference(name1, pc);
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,sn,margin,pc)));
      list npdl = gen_copy_seq(pdl);
      list l = type_struct(t1);
      text fields = text_undefined;

      gen_remove(&npdl, e1); // It is being defined, it must not be
      // redefined if it occurs recursively
      // in its own definition
      fields = c_text_entities(module, l, margin+INDENTATION, npdl);
      pc = CHAIN_SWORD(NIL, "{");
      add_words_to_text(r, pc);
      MERGE_TEXTS(r, fields);
      ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin, "}"));
      gen_free_list(npdl);

      break;
    }
  case is_type_union:
    {
      /* FI->FI: should be updated like struct */
      list l = type_union(t1);
      list npdl = gen_copy_seq(pdl);
      text fields = text_undefined;

      gen_remove(&npdl, e1);
      fields = c_text_entities(module,l,margin+INDENTATION, npdl);
      gen_free_list(npdl);
      pc = words_union(name1, pc);
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,sn,margin,pc)));
      MERGE_TEXTS(r,fields);
      ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"}"));
      break;
    }
  case is_type_enum:
    {
      /* FI->FI: should be updated like struct? */
      list l = type_enum(t1);
      list npdl = gen_copy_seq(pdl);
      gen_remove(&npdl, e1);
      pc = words_enum(name1, l, space_p, pc, npdl);
      gen_free_list(npdl);
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,sn,margin,pc)));
      break;
    }
  case is_type_variable:
  case is_type_functional:
  case is_type_void:
  case is_type_unknown:
    {
      /*pc = words_variable_or_function(module, e1, true, pc,
	in_type_declaration, pdl);*/
      in_type_declaration=false;
      pc = words_variable_or_function(module, e1, true, pc,
				      in_type_declaration, pdl);
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,sn,margin,pc)));
      skip_first_comma_p = false;
      break;
    }
  case is_type_varargs:
  case is_type_statement:
  case is_type_area:
  default:
    pips_internal_error("unexpected type tag");
  }

  /* the word list pc must have been inserted in text r*/
  pc = NIL;



  /* Add the declared variables or more declared variables. */
  list oel = CDR(el); // other entities after e1
  //print_entities(oel);
  FOREACH(ENTITY, e, oel) {
    if(skip_first_comma_p) {
      skip_first_comma_p = false;
      pc = gen_nconc(pc,CHAIN_SWORD(NIL, " "));
    }
    else
      pc = gen_nconc(pc,CHAIN_SWORD(NIL,space_p? ", " : ","));
    pc = words_variable_or_function(module, e, false, pc, in_type_declaration,
				    pdl);
  }
  /* add the asm qualifier if needed */
  string asm_qual = strdup("");
  FOREACH(QUALIFIER, q, entity_qualifiers(e1)) {
    if(qualifier_asm_p(q)) {
            asprintf(&asm_qual,"%s __asm(%s)", asm_qual, qualifier_asm(q));
    }
  }
  pc = CHAIN_SWORD(pc,asm_qual);
  free(asm_qual);
  /* the final semi column,*/
  pc = CHAIN_SWORD(pc,";");

  /* the word list pc must be added to the last sentence of text r */
  if(ENDP(text_sentences(r))) {
    pips_internal_error("Unexpected empty text");
  }
  else {
    add_words_to_text(r, pc);
  }

  return r;
}

/* FI: strange recursion, probably due to Francois...
   c_text_related_entities calls
   c_text_entities calls
   c_text_entity calls back
   c_text_related_entities (!) but with only one element... may call
   words_variable_or_function calls
   c_words_simplified_entity calls
   generic_c_words_simplified_entity

   Note: text when newline are involved, words when everything fits on
   one line.
*/
text c_text_entity(entity module, entity e, int margin, list pdl)
{
  list el = CONS(ENTITY, e, NIL);
  text t =  c_text_related_entities(module, el, margin, 0, pdl);

  return t;
}

text c_text_entity_simple(entity module, entity e, int margin)
{
  list pdl = NIL; // pdl is useless in Fortran or in some debugging situations
  text t = c_text_entity(module, e, margin, pdl);

  return t;
}


/* The fprint_functional() and fprint_environment() functions are
   moved from syntax/declaration.c */

/* C Version of print_common_layout this is called by
   fprint_environment(). This function is much simpler than Fortran
   Version */

list get_common_members(entity common,
			entity __attribute__ ((unused)) module,
			bool __attribute__ ((unused)) only_primary)
{
  list result = NIL;
  //int cumulated_offset = 0;
  pips_assert("entity is a common", type_area_p(entity_type(common)));

  list ld = area_layout(type_area(entity_type(common)));
  entity v = entity_undefined;

  for(; !ENDP(ld); ld = CDR(ld))
    {
      v = ENTITY(CAR(ld));
      storage s = entity_storage(v);
      if(storage_ram_p(s))
	{
	  result = CONS(ENTITY, v, result);
	}
    }
  return gen_nreverse(result);
}

void print_C_common_layout(FILE * fd, entity c, bool debug_p)
{
  entity mod = get_current_module_entity();
  list members = get_common_members(c, mod, false);
  list equiv_members = NIL;

  (void) fprintf(fd, "\nLayout for memory area \"%s\" of size %td: \n",
		 entity_name(c), area_size(type_area(entity_type(c))));

  if(ENDP(members)) {
    pips_assert("An empty area has size 0", area_size(type_area(entity_type(c))) ==0);
    (void) fprintf(fd, "\t* empty area *\n\n");
  }
  else {
    if(area_size(type_area(entity_type(c))) == 0)
      {
	if(debug_p) {
	  user_warning("print_common_layout","Non-empty area %s should have a final size greater than 0\n",
		       entity_module_name(c));
	}
	else {
	  // The non-empty area can have size zero if the entity is extern
	  //pips_internal_error(//     "Non-empty area %s should have a size greater than 0",
	  //     entity_module_name(c));
	}
      }
    MAP(ENTITY, m,
    {
      pips_assert("RAM storage",
		  storage_ram_p(entity_storage(m)));
      int s;
      // There can be a Array whose size is not known (Dynamic Variables)
      SizeOfArray(m, &s);

      pips_assert("An area has no offset as -1",
		  (ram_offset(storage_ram(entity_storage(m)))!= -1));
      if(ram_offset(storage_ram(entity_storage(m))) == DYNAMIC_RAM_OFFSET) {
	(void) fprintf(fd,
		       "\tDynamic Variable %s, \toffset = UNKNOWN, \tsize = DYNAMIC\n",
		       entity_name(m));
      }
      else if(ram_offset(storage_ram(entity_storage(m))) == UNDEFINED_RAM_OFFSET) {

	(void) fprintf(fd,
		       "\tExternal Variable %s,\toffset = UNKNOWN,\tsize = %d\n",
		       entity_name(m),s);
      }
      else {
	(void) fprintf(fd,
		       "\tVariable %s,\toffset = %td,\tsize = %d\n",
		       entity_name(m),
		       ram_offset(storage_ram(entity_storage(m))),
		       s);
      }
      //}
    },
	members);
    (void) fprintf(fd, "\n");
    /* Look for variables aliased with a variable in this common */
    MAP(ENTITY, m,
    {
      list equiv = ram_shared(storage_ram(entity_storage(m)));

      equiv_members = arguments_union(equiv_members, equiv);
    },
	members);

    if(!ENDP(equiv_members)){

      equiv_members = arguments_difference(equiv_members, members);
      if(!ENDP(equiv_members)) {
	sort_list_of_entities(equiv_members);

	(void) fprintf(fd, "\tVariables aliased to this common:\n");

	MAP(ENTITY, m,
	{
	  pips_assert("RAM storage",
		      storage_ram_p(entity_storage(m)));
	  (void) fprintf(fd,
			 "\tVariable %s,\toffset = %td,\tsize = %d\n",
			 entity_name(m),
			 ram_offset(storage_ram(entity_storage(m))),
			 SafeSizeOfArray(m));
	},
	    equiv_members);
	(void) fprintf(fd, "\n");
	gen_free_list(equiv_members);
      }
    }
  }
  gen_free_list(members);
}

/* This function is called from c_parse() via ResetCurrentModule() and
   fprint_environment() */
void fprint_functional(FILE * fd, functional f)
{
  type tr = functional_result(f);
  int count = 0;

  FOREACH(PARAMETER, p, functional_parameters(f)) {
    type ta = parameter_type(p);

    pips_assert("Argument type is variable or varags:variable or functional or void",
		type_variable_p(ta)
		|| (type_varargs_p(ta) && type_variable_p(type_varargs(ta)))
		|| type_functional_p(ta)
		|| type_void_p(ta));

    if(count>0)
      (void) fprintf(fd, " x ");
    count++;

    if(type_functional_p(ta)) {
      functional fa = type_functional(ta);
      /* (void) fprintf(fd, " %s:", type_to_string(ta)); */
      (void) fprintf(fd, "(");
      fprint_functional(fd, fa);
      (void) fprintf(fd, ")");
    }
    else if(type_void_p(ta)) {
      (void) fprintf(fd, "(");
      (void) fprintf(fd, ")");
    }
    else {
      if(type_varargs_p(ta)) {
	(void) fprintf(fd, " %s:", type_to_string(ta));
	ta = type_varargs(ta);
      }
      (void) fprintf(fd, "%s", basic_to_string(variable_basic(type_variable(ta))));
    }
  }

  if(ENDP(functional_parameters(f))) {
    (void) fprintf(fd, " ()");
  }
  (void) fprintf(fd, " -> ");

  if(type_variable_p(tr))
    (void) fprintf(fd, " %s\n", basic_to_string(variable_basic(type_variable(tr))));
  else if(type_void_p(tr))
    (void) fprintf(fd, " %s\n", type_to_string(tr));
  else if(type_unknown_p(tr)){
    /* Well, seems to occur for C compilation units, instead of void... */
    (void) fprintf(fd, " %s\n", type_to_string(tr));
  }
  else if(type_varargs_p(tr)) {
    (void) fprintf(fd, " %s:%s", type_to_string(tr),
		   basic_to_string(variable_basic(type_variable(type_varargs(tr)))));
  }
  else
    /* An argument can be functional, but not (yet) a result. */
    pips_internal_error("Ill. type %d", type_tag(tr));
}

void fprint_environment(FILE *fd, entity m)
{
  fprint_any_environment(fd, m, true);
}

void fprint_C_environment(FILE *fd, entity m)
{
  fprint_any_environment(fd, m, false);
}

void fprint_any_environment(FILE * fd, entity m, bool is_fortran)
{
  list decls = gen_copy_seq(code_declarations(value_code(entity_initial(m))));
  int nth = 0; /* rank of formal parameter */
  entity rv = entity_undefined; /* return variable */

  pips_assert("fprint_environment", entity_module_p(m));

  /* To simplify validation, at the expense of some information about
     the parsing process. */
  gen_sort_list(decls,(gen_cmp_func_t)compare_entities);

  (void) fprintf(fd, "\nDeclarations for module %s with type ",
		 module_local_name(m));
  fprint_functional(fd, type_functional(entity_type(m)));
  (void) fprintf(fd, "\n\n");

  /* In C, no return entity is created (yet). See MakeCurrentModule(). */
  pips_assert("A module storage is ROM or return",
	      storage_rom_p(entity_storage(m))
	      || storage_return_p(entity_storage(m)));

  /* List of implicitly and explicitly declared variables,
     functions and areas */

  (void) fprintf(fd, "%s\n", ENDP(decls)?
		 "* empty declaration list *\n\n": "Variable list:\n\n");

  MAP(ENTITY, e, {
      type t = entity_type(e);

      fprintf(fd, "Declared entity %s\twith type %s ", entity_name(e), type_to_string(t));

      if(type_variable_p(t))
	fprintf(fd, "%s\n", basic_to_string(variable_basic(type_variable(t))));
      else if(type_functional_p(t)) {
	fprint_functional(fd, type_functional(t));
      }
      else if(type_area_p(t)) {
	(void) fprintf(fd, "with size %td\n", area_size(type_area(t)));
      }
      else
	(void) fprintf(fd, "\n");
    },
    decls);

  if(!is_fortran) {
    list edecls = gen_copy_seq(code_externs(value_code(entity_initial(m))));
    /* List of external variables and functions and areas */

    gen_sort_list(edecls, (gen_cmp_func_t)compare_entities);

    (void) fprintf(fd, "%s\n", ENDP(edecls)?
		   "* empty external declaration list *\n\n": "External variable list:\n\n");

    MAP(ENTITY, e, {
	type t = entity_type(e);

	fprintf(fd, "Declared entity %s\twith type %s ", entity_name(e), type_to_string(t));

	if(type_variable_p(t))
	  fprintf(fd, "%s\n", basic_to_string(variable_basic(type_variable(t))));
	else if(type_functional_p(t)) {
	  fprint_functional(fd, type_functional(t));
	}
	else if(type_area_p(t)) {
	  (void) fprintf(fd, "with size %td\n", area_size(type_area(t)));
	}
	else
	  (void) fprintf(fd, "\n");
      },
      edecls);
    gen_free_list(edecls);
  }

  /* Formal parameters */
  nth = 0;
  MAP(ENTITY, v, {
      storage vs = entity_storage(v);

      pips_assert("All storages are defined", !storage_undefined_p(vs));

      if(storage_formal_p(vs)) {
	nth++;
	if(nth==1) {
	  (void) fprintf(fd, "\nLayouts for formal parameters:\n\n");
	}
	(void) fprintf(fd,
		       "\tVariable %s,\toffset = %td\n",
		       entity_name(v), formal_offset(storage_formal(vs)));
      }
      else if(storage_return_p(vs)) {
	pips_assert("No more than one return variable", entity_undefined_p(rv));
	rv = v;
      }
    }, decls);

  /* Return variable */
  if(!entity_undefined_p(rv)) {

    (void) fprintf(fd, "\nLayout for return variable:\n\n");
    (void) fprintf(fd, "\tVariable %s,\tsize = %d\n",
		   entity_name(rv), SafeSizeOfArray(rv));
  }

  /* Structure of each area/common */
  if(!ENDP(decls)) {
    (void) fprintf(fd, "\nLayouts for areas (commons):\n\n");
  }

  MAP(ENTITY, e, {
      void print_common_layout(FILE * fd, entity c, bool debug_p);
      if(type_area_p(entity_type(e))) {
	if(is_fortran)
	  print_common_layout(fd, e, false);
	else
	  print_C_common_layout(fd, e, false);
      }
    },
    decls);

  (void) fprintf(fd, "End of declarations for module %s\n\n",
		 module_local_name(m));

  gen_free_list(decls);
}


/* Transform a declaration with an initialization statement into 2 parts,
   a declaration statement and an initializer statement

   gen_recurse callback on exiting statements. For a declaration to be split:

   - it must be a local declaration

   - the initial value, if any, must be a valid rhs expression or an
     array initialization; struct initialization are not (yet) supported
 */
void split_initializations_in_statement(statement s)
{
  if(!get_bool_property("C89_CODE_GENERATION") && statement_block_p(s)) {
    /* generate C99 code */
    list cs = list_undefined;
    list pcs = NIL;
    list nsl = statement_block(s); // new statement list
    for( cs = statement_block(s); !ENDP(cs); ) {
      statement ls = STATEMENT(CAR(cs));
      if(declaration_statement_p(ls)) {
	list inits = NIL;
	list decls = statement_declarations(ls); // Non-recursive
	//statement sc = statement_undefined; // statement copy

	FOREACH(ENTITY, var, decls) {
	  /* The initialization of a static variable cannot be split */
	  if(entity_static_variable_p(var)) {
	    pips_user_warning("Initialization of variable \"%s\" cannot be "
			      "split from its declaration because \"%s\" "
			      "is a static variable.\n",
			      entity_user_name(var), entity_user_name(var));
	  }
	  else {
	    const char* mn  = entity_module_name(var);
	    const char* cmn = get_current_module_name();
	    if ( same_string_p(mn,cmn)
		 && !value_unknown_p(entity_initial(var))
		 ) {
	      expression ie = variable_initial_expression(var);
	      if (expression_is_C_rhs_p(ie)) {
		statement is = make_assign_statement(entity_to_expression(var), ie);
		inits = gen_nconc(inits, CONS(statement, is, NIL));
		entity_initial(var) = make_value_unknown();
	      }
	      else if(entity_array_p(var)) {
		inits = gen_nconc(inits, brace_expression_to_statements(var,ie));
		brace_expression_to_updated_type(var,ie);
		entity_initial(var) = make_value_unknown();
	      }
	      else if(struct_type_p(entity_basic_concrete_type(var))) {
		inits = gen_nconc(inits, brace_expression_to_statements(var,ie));
		entity_initial(var) = make_value_unknown();
	      }
	      else {
		pips_user_warning("split initializations not implemented yet for structures\n");
	      }
	      /* if this transformation led to an uninitialized const, remove the const qualifier */
	      if(value_unknown_p(entity_initial(var))) {
		list tmp = gen_copy_seq(entity_qualifiers(var));
		FOREACH(QUALIFIER,q,tmp) {
		  if(qualifier_const_p(q))
		    gen_remove_once(&variable_qualifiers(type_variable(entity_type(var))),q);
		}
		gen_free_list(tmp);
	      }
	    }
	  }
	}

	if(!ENDP(inits)) {
	  /* This is not very smart... You do not need pcs in C99
	     since you are going to add the assignment statements
	     just after the current declaration statement... */
	  inits = CONS(STATEMENT, ls, inits);
	  /* Chain the new list within the current statement list */
	  if(ENDP(pcs)) {
	    nsl = inits;
	  }
	  else {
	    CDR(pcs) = inits;
	  }
	  /* Move to the next original element nsl */
	  pcs = gen_last(inits);
	  CDR(pcs) = CDR(cs);
	  POP(cs);
	}
	else {
	  /* Move to the next statement */
	  pcs = cs;
	  POP(cs);
	}
      }
      else {
	/* Move to the next statement */
	pcs = cs;
	POP(cs);
      }
    }
    instruction_block(statement_instruction(s)) = nsl;
  }
  else if(statement_block_p(s)) {
    /* generate C89 code */
    list cs = list_undefined;
    //list pcs = NIL;
    //list nsl = statement_block(s); // new statement list
    list inits = NIL; // list of initialization statements

    for( cs = statement_block(s); !ENDP(cs); POP(cs)) {
      statement ls = STATEMENT(CAR(cs));
      if(declaration_statement_p(ls)) {
	list decls = statement_declarations(ls); // Non-recursive
	//statement sc = statement_undefined; // statement copy

	FOREACH(ENTITY, var, decls) {
        const char* mn  = entity_module_name(var);
        const char* cmn = get_current_module_name();
	  if ( strcmp(mn,cmn) == 0
	       && !value_unknown_p(entity_initial(var))
	       ) {
	    expression ie = variable_initial_expression(var);
            if(expression_undefined_p(ie)) {}
            else if (expression_is_C_rhs_p(ie)) {
	      statement is = make_assign_statement(entity_to_expression(var), ie);
	      inits = gen_nconc(inits, CONS(statement, is, NIL));
	      entity_initial(var) = make_value_unknown();
	    }
	    else if(entity_array_p(var)) {
	      inits=gen_nconc(inits,brace_expression_to_statements(var,ie));
	      entity_initial(var) = make_value_unknown();
	    }
        else {
          pips_user_warning("split initializations not implemented yet for structures\n");
        }
	  }
	}
      }

      if(!ENDP(inits)) {
	list ncs = CDR(cs);
	if(ENDP(ncs) || !declaration_statement_p(STATEMENT(CAR(ncs)))) {
	  list pcs = gen_last(inits);
	  CDR(cs) = inits;
	  CDR(pcs) = ncs;
	  break;
	}
      }
    }
    //instruction_block(statement_instruction(s)) = nsl;
  }
  else {
    /* Do nothing ? */
  }
}
