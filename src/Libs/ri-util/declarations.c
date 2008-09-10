/* Regeneration of declarations from the symbol table
 *
 * $Id$
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

#include "pipsdbm.h"

#include "misc.h"
#include "properties.h"
#include "prettyprint.h"

/*===================== Variables and Function prototypes for C ===========*/

extern bool is_fortran;
text c_text_entity(entity module, entity e, int margin);
list c_words_entity(type t, list name);
static list words_qualifier(list obj);
static list words_dimensions(list dims);

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

static list
words_value(value obj)
{
    list pc;

    if (value_symbolic_p(obj)) {
	pc = words_constant(symbolic_constant(value_symbolic(obj)));
    }
    else if (value_constant(obj)) {
	pc = words_constant(value_constant(obj));
    }
    else {
	pips_internal_error("unexpected tag");
	pc = NIL;
    }

    return(pc);
}

#define LIST_SEPARATOR (is_fortran? ", " : ",")

static list 
words_parameters(entity e)
{
  list pc = NIL;
  type te = entity_type(e);
  functional fe;
  int nparams, i;

  pips_assert("is functionnal", type_functional_p(te));

  fe = type_functional(te);
  nparams = gen_length(functional_parameters(fe));

  for (i = 1; i <= nparams; i++) {
    entity param = find_ith_parameter(e, i);
    if (pc != NIL) {
      pc = CHAIN_SWORD(pc, LIST_SEPARATOR);
    }

    /* If prettyprint alternate returns... Property to be added. */
    if(get_bool_property("PRETTYPRINT_REGENERATE_ALTERNATE_RETURNS")
       && formal_label_replacement_p(param))
      pc = CHAIN_SWORD(pc, "*");
    else
      {
	if(entity_undefined_p(param)) 
	  {
	    parameter p = PARAMETER(gen_nth(i-1,functional_parameters(fe)));
	    type t = parameter_type(p);
	    /* param can be undefined for C language: void foo(void)
	       We do not have an entity corresponding to the 1st argument */
	    if (is_fortran)
	      pips_user_warning("%dth parameter out of %d parameters not found for function %s\n",
				i, nparams, entity_name(e));
	    pc = gen_nconc(pc,words_type(t));
	  }
	else 
	  {
	    if (is_fortran)
	      pc = CHAIN_SWORD(pc, entity_local_name(param));
	    else 
	      {
		/* We have to print variable's type, dimensions, ... with C
		   This can be also a formal function */
		type t = entity_type(param);
		pc = gen_nconc(pc,c_words_entity(t,CHAIN_SWORD(NIL,entity_local_name(param))));
	      }
	  }
      }
  }
  return(pc);
}
  
static list 
words_dimension(dimension obj)
{
  list pc = NIL;
  if (is_fortran) {
    pc = words_expression(dimension_lower(obj));
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(dimension_upper(obj)));
  }
  else {
    /* The lower bound of array in C is always equal to 0, 
       we only need to print (upper dimension + 1) */
    if (unbounded_dimension_p(obj))
      pc = CHAIN_SWORD(pc,"");
    else {
      expression eup = dimension_upper(obj);
      int up;
      if (FALSE && expression_integer_value(eup, &up))
	/* FI: why do you want to change the source code? */
	pc = CHAIN_IWORD(pc,up+1);
      else {
	if(expression_call_p(eup)) {
	  call c = syntax_call(expression_syntax(eup));
	  entity f = call_function(c);
	  if(ENTITY_MINUS_P(f)||ENTITY_MINUS_C_P(f)){
	    expression e1 = EXPRESSION(CAR(call_arguments(c)));
	    expression e2 = EXPRESSION(CAR(CDR(call_arguments(c))));
	    int i;

	    if (expression_integer_value(e2, &i) && i==1)
	      pc = words_expression(e1);
	  }
	}
	if(pc==NIL)
	  /* to be refined here to make more beautiful expression, use normalize ? */
	  /* FI: why would we modify the user C source code?*/
	  pc = words_expression(MakeBinaryCall(CreateIntrinsic("+"),eup,int_to_expression(1)));
      }
    }
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
list 
words_declaration(
    entity e,
    bool prettyprint_common_variable_dimensions_p)
{
    list pl = NIL;

    pl = CHAIN_SWORD(pl, entity_user_name(e));

    if (type_variable_p(entity_type(e)))
    {
	if (prettyprint_common_variable_dimensions_p || 
	    !(variable_in_common_p(e) || variable_static_p(e)))
	{
	    if (variable_dimensions(type_variable(entity_type(e))) != NIL) 
	    {
		list dims = variable_dimensions(type_variable(entity_type(e)));
	
		if (is_fortran)
		  {
		    pl = CHAIN_SWORD(pl, "(");		    
		    MAPL(pd, 
		    {
		      pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd))));
		      if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, LIST_SEPARATOR);
		    }, dims);
		    pl = CHAIN_SWORD(pl, ")");
		  }
		else
		  {
		    MAPL(pd, 
		    {
		      pl = CHAIN_SWORD(pl, "[");
		      pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd))));
		      pl = CHAIN_SWORD(pl, "]");
		    }, dims);
		  }      
	    }
	}
    }
    attach_declaration_to_words(pl, e);
    return(pl);
}

/* what about simple DOUBLE PRECISION, REAL, INTEGER... */
list words_basic(basic obj)
{
    list pc = NIL;
    /* 31/07/2003 Nga Nguyen : add more cases for C*/
    switch (basic_tag(obj)) {
      case is_basic_int: 
	{
	  if (is_fortran)
	    {
	      pc = CHAIN_SWORD(pc,"INTEGER*");
	      pc = CHAIN_IWORD(pc,basic_int(obj));
	    }
	  else 
	    {
	      switch (basic_int(obj)) {
		case 1: pc = CHAIN_SWORD(pc,"char"); 
		  break;
		case 2: pc = CHAIN_SWORD(pc,"short"); 
		  break;
		case 4: pc = CHAIN_SWORD(pc,"int"); 
		  break;
		case 6: pc = CHAIN_SWORD(pc,"long"); 
		  break;
		case 8: pc = CHAIN_SWORD(pc,"long long"); 
		  break;
		case 11: pc = CHAIN_SWORD(pc,"unsigned char");
		  break;
		case 12: pc = CHAIN_SWORD(pc,"unsigned short");
		  break;
		case 14: pc = CHAIN_SWORD(pc,"unsigned int");
		  break;
		case 16: pc = CHAIN_SWORD(pc,"unsigned long");
		  break;
		case 18: pc = CHAIN_SWORD(pc,"unsigned long long");
		  break;
		case 21: pc = CHAIN_SWORD(pc,"signed char");
		  break;
		case 22: pc = CHAIN_SWORD(pc,"signed short");
		  break;
		case 24: pc = CHAIN_SWORD(pc,"signed int");
		  break;
		case 26: pc = CHAIN_SWORD(pc,"signed long");
		  break;
		case 28: pc = CHAIN_SWORD(pc,"signed long long");
		  break;
		}
	    }
	  break;
	}
      case is_basic_float:
	{
	  if (is_fortran)
	    {
	      pc = CHAIN_SWORD(pc,"REAL*");
	      pc = CHAIN_IWORD(pc,basic_float(obj));
	    }
	  else 
	    {
	      switch (basic_float(obj)) {
		case 4: pc = CHAIN_SWORD(pc,"float");
		  break;
		case 8: pc = CHAIN_SWORD(pc,"double");
		  break;
		}
	    }
	  break;
	}
      case is_basic_logical:
	{
	   if (is_fortran)
	     {
	       pc = CHAIN_SWORD(pc,"LOGICAL*");
	       pc = CHAIN_IWORD(pc,basic_logical(obj));
	     }
	   else 
	     pc = CHAIN_SWORD(pc,"int ");
	   break;
	}
      case is_basic_overloaded:
	{
	  /* should be a user error ? */
	  pc = CHAIN_SWORD(pc,is_fortran?"OVERLOADED":"overloaded");
	  break;
	}
      case is_basic_complex:
	{
	  if(is_fortran) {
	    pc = CHAIN_SWORD(pc,"COMPLEX*");
	    pc = CHAIN_IWORD(pc,basic_complex(obj));
	  }
	  else 
	    {
	      switch (basic_complex(obj)) {
		case 8: pc = CHAIN_SWORD(pc,"float _Complex");
		  break;
		case 16: pc = CHAIN_SWORD(pc,"double _Complex");
		  break;
		case 32: pc = CHAIN_SWORD(pc,"long double _Complex");
		  break;
	      default:
		pips_internal_error("Unexpected complex size");
		}
	    }
	  break;
	}
      case is_basic_string:
	{
	  if (is_fortran)
	    {
	      pc = CHAIN_SWORD(pc,"CHARACTER*");
	      pc = gen_nconc(pc, words_value(basic_string(obj)));
	    }
	  else 
	    pc = CHAIN_SWORD(pc,"char");
	  break;
	}
      case is_basic_bit:
	{
	  int i = basic_bit(obj);
	  pips_debug(7,"Bit field basic: %d\n",i);
	  pc = CHAIN_SWORD(pc,"int"); /* ignore if it is signed or unsigned */
	  break;
	}
	/* The following code maybe redundant, because of tests in c_words_entity*/
      case is_basic_pointer:
	{
	  type t = basic_pointer(obj);
	  pips_debug(7,"Basic pointer\n");
	  pc = gen_nconc(pc,words_type(t));
	  pc = CHAIN_SWORD(pc," *");
	  break;
	}
      case is_basic_derived:
	{
	  entity ent = basic_derived(obj);
	  type t = entity_type(ent);
	  pc = gen_nconc(pc,words_type(t));
	  pc = CHAIN_SWORD(pc," ");
	  pc = CHAIN_SWORD(pc,entity_user_name(ent));
	  pc = CHAIN_SWORD(pc," ");
	  break;
	}
      case is_basic_typedef:
	{
	  entity ent = basic_typedef(obj);
	  pc = CHAIN_SWORD(pc,entity_user_name(ent));
	  break;
	}  
      default:
	pips_error("words_basic", "unexpected tag");
    }   
    return(pc);
}

/**************************************************************** SENTENCE */

sentence 
sentence_variable(entity e)
{
    list pc = NIL;
    type te = entity_type(e);

    pips_assert("is a variable", type_variable_p(te));

    pc = gen_nconc(pc, words_basic(variable_basic(type_variable(te))));
    pc = CHAIN_SWORD(pc, " ");

    pc = gen_nconc(pc, words_declaration(e, TRUE));

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
sentence 
sentence_head(entity e)
{
    list pc = NIL;
    type te = entity_type(e);
    functional fe;
    type tr;
    list args = words_parameters(e);

    pips_assert("is functionnal", type_functional_p(te));

    if (static_module_p(e))
      pc = CHAIN_SWORD(pc,"static ");
    
    fe = type_functional(te);
    tr = functional_result(fe);
   
    switch (type_tag(tr)) {
    case is_type_void: 
      {
	if (is_fortran)
	  {
	    if (entity_main_module_p(e))
	      pc = CHAIN_SWORD(pc,"PROGRAM ");
	    else
	      {
		if (entity_blockdata_p(e))
		  pc = CHAIN_SWORD(pc, "BLOCKDATA ");
		else
		  pc = CHAIN_SWORD(pc,"SUBROUTINE ");
	      }
	  }
	else
	  pc = CHAIN_SWORD(pc,"void ");
	break;
      }
    case is_type_variable:
      {
	pc = gen_nconc(pc, words_basic(variable_basic(type_variable(tr))));
	pc = CHAIN_SWORD(pc,is_fortran? " FUNCTION ":" ");
	break;
      }
    case is_type_unknown:
      {
	/* For C functions with no return type. 
	   It can be treated as of type int, but we keep it unknown for 
	   the moment, to make the differences and to regenerate initial code*/
	break;
      }
    default: 
      pips_internal_error("unexpected type for result\n");
    }
      
    pc = CHAIN_SWORD(pc, module_local_name(e));
    
    if (!ENDP(args)) {
      pc = CHAIN_SWORD(pc, "(");
      pc = gen_nconc(pc, args);
      pc = CHAIN_SWORD(pc, ")");
    }
    else if (type_variable_p(tr) || (!is_fortran && (type_unknown_p(tr) || type_void_p(tr)))) {
      pc = CHAIN_SWORD(pc, "()");
    }

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

static bool
empty_static_area_p(entity e)
{
    if (!static_area_p(e)) return FALSE;
    return ENDP(area_layout(type_area(entity_type(e))));
}

/*  special management of empty commons added.
 *  this may happen in the hpfc generated code.
 */
static sentence 
sentence_area(entity e, entity module, bool pp_dimensions)
{
    string area_name = module_local_name(e);
    type te = entity_type(e);
    list pc = NIL, entities = NIL;

    /* FI: POINTER declarations should be generated for the heap area */
    if (dynamic_area_p(e) || heap_area_p(e) || stack_area_p(e)) /* shouldn't get in? */
	return sentence_undefined;

    assert(type_area_p(te));

    if (!ENDP(area_layout(type_area(te))))
    {
	bool pp_hpfc = get_bool_property("PRETTYPRINT_HPFC");

	if (pp_hpfc)
	    entities = gen_copy_seq(area_layout(type_area(te)));
	else
	    entities = common_members_of_module(e, module, TRUE);

	/*  the common is not output if it is empty
	 */
	if (!ENDP(entities))
	{
	    bool comma = FALSE, is_save = static_area_p(e);

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
		 if (comma) pc = CHAIN_SWORD(pc, LIST_SEPARATOR);
		 else comma = TRUE;
		 pc = gen_nconc(pc, 
			words_declaration(ee, !is_save && pp_dimensions));
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
			 make_unformatted(NULL, 0, 0, pc));
}

static sentence 
sentence_basic_declaration(entity e)
{
    list decl = NIL;
    basic b = entity_basic(e);

    pips_assert("b is defined", !basic_undefined_p(b));

    decl = CHAIN_SWORD(decl, basic_to_string(b));
    decl = CHAIN_SWORD(decl, " ");
    decl = CHAIN_SWORD(decl, entity_local_name(e));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, decl)));
}

static sentence 
sentence_external(entity f)
{
    list pc = NIL;

    pc = CHAIN_SWORD(pc, "EXTERNAL ");
    pc = CHAIN_SWORD(pc, entity_local_name(f));

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

static sentence 
sentence_symbolic(entity f)
{
    list pc = NIL;
    value vf = entity_initial(f);
    expression e = symbolic_expression(value_symbolic(vf));

    pc = CHAIN_SWORD(pc, "PARAMETER (");
    pc = CHAIN_SWORD(pc, entity_local_name(f));
    pc = CHAIN_SWORD(pc, " = ");
    pc = gen_nconc(pc, words_expression(e));
    pc = CHAIN_SWORD(pc, ")");

    return(make_sentence(is_sentence_unformatted, 
			 make_unformatted(NULL, 0, 0, pc)));
}

/* why is it assumed that the constant is an int ??? 
 */
static sentence 
sentence_data(entity e)
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
			 make_unformatted(NULL, 0, 0, pc)));
}

/********************************************************************* TEXT */

#define ADD_WORD_LIST_TO_TEXT(t, l)\
  if (!ENDP(l)) ADD_SENTENCE_TO_TEXT(t,\
	        make_sentence(is_sentence_unformatted, \
			      make_unformatted(NULL, 0, 0, l)));

/* if the common is declared similarly in all routines, generate
 * "include 'COMMON.h'", and the file is put in Src. otherwise
 * the full local declarations are generated. That's fun.
 */

static text
include(string file)
{
    return make_text
	(CONS(SENTENCE, 
	      make_sentence(is_sentence_formatted,
		  strdup(concatenate("      include '", file, "'\n", NULL))),
	      NIL));
}

static text
text_area_included(
    entity common /* the common the declaration of which are of interest */,
    entity module /* hte module dealt with */)
{
    string dir, file, local, name;
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

static void 
equiv_class_debug(list l_equiv)
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
 * one comes first; if they have the same lenght, use a lexicographic
 * ordering.
 * author: bc.
 */
static int
equivalent_entity_compare(entity *ent1, entity *ent2)
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
static text
text_equivalence_class(
    list /* of entities */ l_equiv)
{
    text t_equiv = make_text(NIL);
    list lw = NIL;
    list l1, l2;
    entity ent1, ent2;
    int offset1, offset2;
    Value size1, size2, offset_end1;
    boolean first;

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
    
    gen_sort_list(l_equiv,equivalent_entity_compare);
	
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
    first = TRUE;
    /* */
    
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

	    if (first) lw = CHAIN_SWORD(lw, "EQUIVALENCE"), first = FALSE;
	    else lw = CHAIN_SWORD(lw, LIST_SEPARATOR);

	    lw = CHAIN_SWORD(lw, " (");
	    lw = CHAIN_SWORD(lw, entity_local_name(ent1));
	    lw = CHAIN_SWORD(lw, LIST_SEPARATOR);
	    lw = CHAIN_SWORD(lw, entity_local_name(ent2));
	    lw = CHAIN_SWORD(lw, ")");		
	    POP(l2);
	}
	/* Else, we first check that there is an overlap */
	else 
	{
	    pips_assert("the equivalence class has been sorted\n",
			offset1 < offset2);
	    
	    size2 = ValueSizeOfArray(ent2);		
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
		static char buffer[10];
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
		
		if (first) lw = CHAIN_SWORD(lw, "EQUIVALENCE"), first = FALSE;
		else lw = CHAIN_SWORD(lw, LIST_SEPARATOR);

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
		    buffer[0] = '\0';
		    sprintf(buffer+strlen(buffer), "%d", new_decl);		 
		    lw = CHAIN_SWORD(lw,strdup(buffer));			
		    if (current_dim < dim_max)
			lw = CHAIN_SWORD(lw, LIST_SEPARATOR);
		    
		    POP(l_tmp);
		    current_dim++;
		    
		} /* while */
		
		lw = CHAIN_SWORD(lw, ")");	
		lw = CHAIN_SWORD(lw, LIST_SEPARATOR);
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
static text 
text_equivalences(
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
	    
	    if (no_commons && !SPECIAL_COMMON_P(common))
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
		bool found = FALSE;
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
			    found = TRUE;
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
static sentence sentence_data_statement(statement is)
{
  unformatted u =
    make_unformatted
    (NULL,
     STATEMENT_NUMBER_UNDEFINED, 0, 
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
      wl = CHAIN_SWORD(wl, strdup(", "));
    }

    ive = EXPRESSION(CAR(rl));
    ivwl = words_expression(ive);
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

      rfe = EXPRESSION(CAR(call_arguments(vc)));
      rve = EXPRESSION(CAR(CDR(call_arguments(vc))));

      if(!(integer_constant_expression_p(rfe) && expression_to_int(rfe)==1)) {
	/* print out the repeat factor if it is not one */
	rwl = words_expression(rfe);
	wl = gen_nconc(wl, rwl);
	wl = gen_nconc(wl, CONS(STRING, strdup("*"), NIL));
      }
      iwl = words_expression(rve);
      wl = gen_nconc(wl, iwl);
    }
    else {
      iwl = words_expression(ve);
      wl = gen_nconc(wl, iwl);
    }
    if(!ENDP(CDR(al))) {
      wl = gen_nconc(wl, CONS(STRING, strdup(", "), NIL));
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

  MAP(STATEMENT, is, {
    if(!empty_comments_p(statement_comments(is))) {
      ADD_SENTENCE_TO_TEXT(t, make_sentence(is_sentence_formatted, 
					    strdup(statement_comments(is))));
    }
    ADD_SENTENCE_TO_TEXT(t, sentence_data_statement(is));
  }, il);

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

static text
text_of_parameters(
    list /* of entity that are parameters */ lp)
{
    list /* of sentence */ ls = NIL;
    
    /* generate the sentences
     */
    MAP(ENTITY, e,
	ls = CONS(SENTENCE, sentence_basic_declaration(e),
	     CONS(SENTENCE, sentence_symbolic(e), ls)),
	lp);

    return make_text(ls);
}

/********************************************************** ALL DECLARATIONS */

static text 
text_entity_declaration(
    entity module, 
    list /* of entity */ ldecl,
    bool force_common)
{
  string how_common = get_string_property("PRETTYPRINT_COMMONS");
  bool print_commons = !same_string_p(how_common, "none");
  list before = NIL, area_decl = NIL,
    pi1 = NIL, pi2 = NIL, pi4 = NIL, pi8 = NIL,
    ph1 = NIL, ph2 = NIL, ph4 = NIL, ph8 = NIL,
    pf4 = NIL, pf8 = NIL,
    pl = NIL, 
    pc8 = NIL, pc16 = NIL, ps = NIL, lparam = NIL;
  list * ppi=NULL;
  list * pph=NULL;
  text r, t_chars = make_text(NIL), t_area = make_text(NIL); 
  string pp_var_dim = get_string_property("PRETTYPRINT_VARIABLE_DIMENSIONS");
  bool pp_in_type = FALSE, pp_in_common = FALSE, pp_cinc;
  /* Declarations cannot be sorted out because Fortran standard impose at
  least an order on parameters. Fortunately here, PARAMETER are mostly
  integers, defined from other integer parameters... I assume that PIPS
  would fail with an ENTRY referencing an integer array dimensionned with
  a real parameter. But real parameters are not really well processed by
  PIPS...

  list sorted_ldecl = gen_copy_seq(ldecl);

  gen_sort_list(sorted_ldecl, compare_entities); */
     
  /* where to put the dimension information.
   */
  if (same_string_p(pp_var_dim, "type"))
    pp_in_type = TRUE, pp_in_common = FALSE;
  else if (same_string_p(pp_var_dim, "common"))
    pp_in_type = FALSE, pp_in_common = TRUE;
  else 
    pips_internal_error("PRETTYPRINT_VARIABLE_DIMENSIONS=\"%s\""
			" unexpected value\n", pp_var_dim);

  /* prettyprint common in include if possible... 
   */
  pp_cinc = same_string_p(how_common, "include") && !force_common;

  MAP(ENTITY, e,
  {
    type te = entity_type(e);
    bool func = 
      type_functional_p(te) && storage_rom_p(entity_storage(e));
    value v = entity_initial(e);
    bool param = func && value_symbolic_p(v);
    bool external =     /* subroutines won't be declared */
      (func && 
       (value_code_p(v) || value_unknown_p(v) /* not parsed callee */) &&
       !(type_void_p(functional_result(type_functional(te))) ||
	 (type_variable_p(functional_result(type_functional(te))) &&
	  basic_overloaded_p(variable_basic(type_variable
					    (functional_result(type_functional(te))))))));
    bool area_p = type_area_p(te);
    bool var = type_variable_p(te);
    bool in_ram = storage_ram_p(entity_storage(e));
    bool in_common = in_ram &&
      !SPECIAL_COMMON_P(ram_section(storage_ram(entity_storage(e))));
    bool skip_it = same_string_p(entity_local_name(e),
				 entity_local_name(module));
	
    pips_debug(3, "entity name is %s\n", entity_name(e));

    /* Do not declare variables used to replace formal labels */
    if(storage_formal_p(entity_storage(e))
       && get_bool_property("PRETTYPRINT_REGENERATE_ALTERNATE_RETURNS")
       && formal_label_replacement_p(e))
      continue;

    if (!print_commons && area_p && !SPECIAL_COMMON_P(e) && !pp_cinc)
      {
	area_decl = 
	  CONS(SENTENCE, make_sentence(is_sentence_formatted,
				       common_hook(module, e)),
	       area_decl);
      }

    if (skip_it)
      {
	pips_debug(5, "skipping function %s\n", entity_name(e));
      }
    else if (!print_commons && (area_p || (var && in_common && pp_cinc)))
      {
	pips_debug(5, "skipping entity %s\n", entity_name(e));
      }
    else if (param)
      {
	/*        PARAMETER
	 */
	pips_debug(7, "considered as a parameter\n");
	lparam = CONS(ENTITY, e, lparam);
      } 
    else if (external)
      {
	/*        EXTERNAL
	 */
	pips_debug(7, "considered as an external\n");
	before = CONS(SENTENCE, sentence_basic_declaration(e), before);
	before = CONS(SENTENCE, sentence_external(e), before);
      }
    else if (area_p && !dynamic_area_p(e) && !heap_area_p(e) && !stack_area_p(e) && !empty_static_area_p(e))
      {
	/*            AREAS: COMMONS and SAVEs
	 */	     
	pips_debug(7, "considered as a regular common\n");
	if (pp_cinc && !SPECIAL_COMMON_P(e))
	  {
	    text t = text_area_included(e, module);
	    MERGE_TEXTS(t_area, t);
	  }
	else
	  area_decl = CONS(SENTENCE, 
			   sentence_area(e, module, pp_in_common), 
			   area_decl);
      }
    else if (var && !(in_common && pp_cinc))
      {
	basic b = variable_basic(type_variable(te));
	bool pp_dim = pp_in_type || variable_static_p(e);

	pips_debug(7, "is a variable...\n");
	    
	switch (basic_tag(b)) 
	  {
	  case is_basic_int:
	    /* simple integers are moved ahead... */

	    pips_debug(7, "is an integer\n");
	    if (variable_dimensions(type_variable(te)))
	      {
		string s = string_undefined;
		switch (basic_int(b))
		  {
		  case 4: ppi = &pi4;
		    s = "INTEGER ";
		    break;  
		  case 2: ppi = &pi2;
		    s = "INTEGER*2 ";
		    break;
		  case 8: ppi = &pi8;
		    s = "INTEGER*8 ";
		    break;
		  case 1: ppi = &pi1;
		    s = "INTEGER*1 ";
		    break;
		
		  default: pips_internal_error("Unexpected integer size");
		  }
		*ppi = CHAIN_SWORD(*ppi, *ppi==NIL ? s : ", ");
		*ppi = gen_nconc(*ppi, words_declaration(e, pp_dim)); 
	      }
	    else
	      {
		string s = string_undefined;

		switch (basic_int(b))
		  {
		  case 4: pph = &ph4;
		    s = "INTEGER ";
		    break;
		  case 2: pph = &ph2;
		    s = "INTEGER*2 ";
		    break;
		  case 8: pph = &ph8;
		    s = "INTEGER*8 ";
		    break;
		  case 1: pph = &ph1;
		    s = "INTEGER*1 ";
		    break;
		  default: pips_internal_error("Unexpected integer size");
		  }
		*pph = CHAIN_SWORD(*pph, *pph==NIL ? s : LIST_SEPARATOR);
		*pph = gen_nconc(*pph, words_declaration(e, pp_dim)); 
	      }
	    break;
	  case is_basic_float:
	    pips_debug(7, "is a float\n");
	    switch (basic_float(b))
	      {
	      case 4:
		pf4 = CHAIN_SWORD(pf4, pf4==NIL ? "REAL*4 " : LIST_SEPARATOR);
		pf4 = gen_nconc(pf4, words_declaration(e, pp_dim));
		break;
	      case 8:
	      default:
		pf8 = CHAIN_SWORD(pf8, pf8==NIL ? "REAL*8 " : LIST_SEPARATOR);
		pf8 = gen_nconc(pf8, words_declaration(e, pp_dim));
		break;
	      }
	    break;			
	  case is_basic_complex:
	    pips_debug(7, "is a complex\n");
	    switch (basic_complex(b))
	      {
	      case 8:
		pc8 = CHAIN_SWORD(pc8, pc8==NIL ? "COMPLEX*8 " : LIST_SEPARATOR);
		pc8 = gen_nconc(pc8, words_declaration(e, pp_dim));
		break;
	      case 16:
	      default:
		pc16 = CHAIN_SWORD(pc16, pc16==NIL ? "COMPLEX*16 " : LIST_SEPARATOR);
		pc16 = gen_nconc(pc16, words_declaration(e, pp_dim));
		break;
	      }
	    break;
	  case is_basic_logical:
	    pips_debug(7, "is a logical\n");
	    pl = CHAIN_SWORD(pl, pl==NIL ? "LOGICAL " : LIST_SEPARATOR);
	    pl = gen_nconc(pl, words_declaration(e, pp_dim));
	    break;
	  case is_basic_overloaded:
	    /* nothing! some in hpfc I guess...
	     */
	    break; 
	  case is_basic_string:
	    {
	      value v = basic_string(b);
	      pips_debug(7, "is a string\n");
		
	      if (value_constant_p(v) && constant_int_p(value_constant(v)))
		{
		  int i = constant_int(value_constant(v));
		    
		  if (i==1)
		    {
		      ps = CHAIN_SWORD(ps, ps==NIL ? "CHARACTER " : LIST_SEPARATOR);
		      ps = gen_nconc(ps, words_declaration(e, pp_dim));
		    }
		  else
		    {
		      list chars=NIL;
		      chars = CHAIN_SWORD(chars, "CHARACTER*");
		      chars = CHAIN_IWORD(chars, i);
		      chars = CHAIN_SWORD(chars, " ");
		      chars = gen_nconc(chars, 
					words_declaration(e, pp_dim));
		      attach_declaration_size_type_to_words
			(chars, "CHARACTER", i);
		      ADD_WORD_LIST_TO_TEXT(t_chars, chars);
		    }
		}
	      else if (value_unknown_p(v))
		{
		  list chars=NIL;
		  chars = CHAIN_SWORD(chars, "CHARACTER*(*) ");
		  chars = gen_nconc(chars, 
				    words_declaration(e, pp_dim));
		  attach_declaration_type_to_words
		    (chars, "CHARACTER*(*)");
		  ADD_WORD_LIST_TO_TEXT(t_chars, chars);
		}
	      else if (value_symbolic_p(v))
		{
		  list chars = NIL;
		  symbolic s = value_symbolic(v);
		  chars = CHAIN_SWORD(chars, "CHARACTER*(");
		  chars = gen_nconc(chars, 
				    words_expression(symbolic_expression(s)));
		  chars = CHAIN_SWORD(chars, ") ");
		  chars = gen_nconc(chars, words_declaration(e, pp_dim));

		  attach_declaration_type_to_words
		    (chars, "CHARACTER*(*)");
		  ADD_WORD_LIST_TO_TEXT(t_chars, chars);
		}
	      else
		pips_internal_error("unexpected value\n");
	      break;
	    }
	  default:
	    pips_internal_error("unexpected basic tag (%d)\n",
				basic_tag(b));
	  }
      }
  }, /* sorted_ */ ldecl);
    
  /* usually they are sorted in order, and appended backwards,
   * hence the reversion.
   */
  r = make_text(gen_nreverse(before));

  MERGE_TEXTS(r, text_of_parameters(lparam));
  gen_free_list(lparam), lparam = NIL;

  ADD_WORD_LIST_TO_TEXT(r, ph1);
  attach_declaration_type_to_words(ph1, "INTEGER*1");
  ADD_WORD_LIST_TO_TEXT(r, ph2);
  attach_declaration_type_to_words(ph2, "INTEGER*2");
  ADD_WORD_LIST_TO_TEXT(r, ph4);
  attach_declaration_type_to_words(ph4, "INTEGER");
  ADD_WORD_LIST_TO_TEXT(r, ph8);
  attach_declaration_type_to_words(ph8, "INTEGER*8");
  ADD_WORD_LIST_TO_TEXT(r, pi1);
  attach_declaration_type_to_words(pi1, "INTEGER*1");
  ADD_WORD_LIST_TO_TEXT(r, pi2);
  attach_declaration_type_to_words(pi2, "INTEGER*2");
  ADD_WORD_LIST_TO_TEXT(r, pi4);
  attach_declaration_type_to_words(pi4, "INTEGER");
  ADD_WORD_LIST_TO_TEXT(r, pi8);
  attach_declaration_type_to_words(pi8, "INTEGER*8");
  ADD_WORD_LIST_TO_TEXT(r, pf4);
  attach_declaration_type_to_words(pf4, "REAL*4");
  ADD_WORD_LIST_TO_TEXT(r, pf8);
  attach_declaration_type_to_words(pf8, "REAL*8");
  ADD_WORD_LIST_TO_TEXT(r, pl);
  attach_declaration_type_to_words(pl, "LOGICAL");
  ADD_WORD_LIST_TO_TEXT(r, pc8);
  attach_declaration_type_to_words(pc8, "COMPLEX*8");
  ADD_WORD_LIST_TO_TEXT(r, pc16);
  attach_declaration_type_to_words(pc16, "COMPLEX*16");
  ADD_WORD_LIST_TO_TEXT(r, ps);
  attach_declaration_type_to_words(ps, "CHARACTER");
  MERGE_TEXTS(r, t_chars);

  /* all about COMMON and SAVE declarations
   */
  MERGE_TEXTS(r, make_text(area_decl));
  MERGE_TEXTS(r, t_area);

  /* and EQUIVALENCE statements... - BC 
   */
  MERGE_TEXTS(r, text_equivalences(module, /* sorted_ */ ldecl, 
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

  return r;
}

/* exported for hpfc.
 */
text 
text_declaration(entity module)
{
    return text_entity_declaration
	(module, code_declarations(entity_code(module)), FALSE);
}

/* needed for hpfc 
 */
text 
text_common_declaration(
    entity common, 
    entity module)
{
    type t = entity_type(common);
    list l;
    text result;
    pips_assert("indeed a common", type_area_p(t));
    l = CONS(ENTITY, common, common_members_of_module(common, module, FALSE));
    result = text_entity_declaration(module, l, TRUE);
    gen_free_list(l);
    return result;
}

/* ================C prettyprinter functions================= */

static list words_qualifier(list obj)
{
  list pc = NIL;
  MAP(QUALIFIER,q,
  {
    switch (qualifier_tag(q)) {
    case is_qualifier_register:
      pc = CHAIN_SWORD(pc,"register ");
      break; 
    case is_qualifier_const:
      pc = CHAIN_SWORD(pc,"const ");
      break;
    case is_qualifier_restrict:
      pc = CHAIN_SWORD(pc,"restrict ");
      break;  
    case is_qualifier_volatile:
      pc = CHAIN_SWORD(pc,"volatile ");
      break;
    default : 
      pips_error("words_qualifier", "unexpected tag");
    }
  },obj);
  return pc;
}

list words_type(type obj)
{
  list pc = NIL;
  switch (type_tag(obj))
    {
    case is_type_variable:
      {
	basic b = variable_basic(type_variable(obj));
	pc = words_basic(b);
	break;
      }
    case is_type_void:
      {
	pc = CHAIN_SWORD(pc,"void");
	break;
      }
    case is_type_unknown:
      {
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
	string_buffer result = string_buffer_make();
	string rs = string_undefined;

	dump_functional(type_functional(obj), result);
	rs = string_buffer_to_string(result);
	pc = gen_nconc(pc, CONS(STRING, rs, NIL));
	string_buffer_free(&result, TRUE);
	break;
      }
    case is_type_varargs:
      {
	pc = CHAIN_SWORD(pc,"...");
	break;
      }
    default:
      pips_error("words_type", "unexpected tag\n");
    }
  return pc;
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


static list words_brace_expression(expression exp)
{
  list pc = NIL;
  list args = call_arguments(syntax_call(expression_syntax(exp)));
  bool first = TRUE;
  pc = CHAIN_SWORD(pc,"{");
  MAP(EXPRESSION,e,
  {
    if (!first) 
      pc = CHAIN_SWORD(pc,", ");
    if (brace_expression_p(e))
      pc = gen_nconc(pc,words_brace_expression(e));
    else
      pc = gen_nconc(pc,words_expression(e));
    first = FALSE;
  },args);
  pc = CHAIN_SWORD(pc,"}");
  return pc;
}

static list words_dimensions(list dims)
{
  list pc = NIL;
  if (is_fortran)
    {
      pc = CHAIN_SWORD(pc, "(");		    
      MAPL(pd, 
      {
	pc = gen_nconc(pc, words_dimension(DIMENSION(CAR(pd))));
	if (CDR(pd) != NIL) pc = CHAIN_SWORD(pc, ", ");
      }, dims);
      pc = CHAIN_SWORD(pc, ")");
    }
  else
    {
      MAP(DIMENSION,d, 
      {
	pc = CHAIN_SWORD(pc, "[");
	pc = gen_nconc(pc, words_dimension(d));
	pc = CHAIN_SWORD(pc, "]");
      }, dims);
    }  
  return pc; 
}

/* Here is the set of mapping functions, from the RI to C language types*/

/* Returns TRUE if t is one of the following types : 
   void, char, short, int, long, float, double, signed, unsigned, 
   and there is no array dimensions, of course*/

bool basic_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      return ((variable_dimensions(type_variable(t)) == NIL) &&
	      (basic_int_p(b) || basic_float_p(b) || basic_logical_p(b)
	       || basic_overloaded_p(b) || basic_complex_p(b) || basic_string_p(b)
	       || basic_bit_p(b)));
    }
  return (type_void_p(t) || type_unknown_p(t)) ;
}

bool array_type_p(type t)
{
  return (type_variable_p(t) && (variable_dimensions(type_variable(t)) != NIL));
}

bool pointer_type_p(type t)
{
  return (type_variable_p(t) && basic_pointer_p(variable_basic(type_variable(t)))
	  && (variable_dimensions(type_variable(t)) == NIL));
}

/* Returns TRUE if t is of type struct, union or enum. Need to distinguish 
   with the case struct/union/enum in type in RI, these are the definitions 
   of the struct/union/enum themselve, not a variable of this type. 

   Example : struct foo var;*/

bool derived_type_p(type t)
{
  return (type_variable_p(t) && basic_derived_p(variable_basic(type_variable(t))) 
	  && (variable_dimensions(type_variable(t)) == NIL));
}

/* Returns TRUE if t is a typedefED type. 
   Example : Myint i;*/

bool typedef_type_p(type t)
{
  return (type_variable_p(t) && basic_typedef_p(variable_basic(type_variable(t))) 
	  && (variable_dimensions(type_variable(t)) == NIL));
}

/* This recursive function prints a C variable with its type. 
   It can be a simple variable declaration such as "int a"
   or complicated one such as "int (* forces[10])()" (an array of 
   10 pointers, each pointer points to a function 
   with no parameter and the return type is int)  */

list generic_c_words_entity(type t, list name, bool is_safe)
{
  list pc = NIL;

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
      bool first = TRUE;  

      pips_debug(9,"Function type with name = \"%s\" and length %zd\n", list_to_string(name), gen_length(name));
 
      if ((gen_length(name) > 1) || ((gen_length(name) == 1) && (strcmp(STRING(CAR(name)),"*")==0)))
	{
	  /* Function name is an expression like *vfs[] in (*vfs[])() 
	     (syntax = application), or an abstract function type, so parentheses must be added */
	  pc = CHAIN_SWORD(NIL,"(");
	  pc = gen_nconc(pc,name);
	  pc = CHAIN_SWORD(pc,")(");

	}
      else 
	{
	  /* Function name is a simple reference */
	  pc = CHAIN_SWORD(name,"(");
	}
      
      MAP(PARAMETER,p,
      {
	type t1 = parameter_type(p);
	//pips_debug(3,"Parameter type %s\n ",type_undefined_p(t1)? "type_undefined" : words_to_string(words_type(t1)));
	if (!first)
	  pc = gen_nconc(pc,CHAIN_SWORD(NIL, LIST_SEPARATOR));
	/* c_words_entity(t1,NIL) should be replaced by c_words_entity(t1,name_of_corresponding_parameter) */
	pc = gen_nconc(pc,generic_c_words_entity(t1,NIL, is_safe));
	pips_debug(9,"List of parameters \"%s\"\n ",list_to_string(pc));
	first = FALSE;
      },lparams);
      
      pc = CHAIN_SWORD(pc,")");
      return generic_c_words_entity(t2,pc,is_safe);
    }

  if (pointer_type_p(t))
    {
      type t1 = basic_pointer(variable_basic(type_variable(t)));
      pips_debug(9,"Pointer type with name = %s\n", list_to_string(name));

      pc = CHAIN_SWORD(NIL,"*");
      if (variable_qualifiers(type_variable(t)) != NIL)
	pc = gen_nconc(pc,words_qualifier(variable_qualifiers(type_variable(t))));
      pc = gen_nconc(pc,name);
      return generic_c_words_entity(t1,pc,is_safe);
    }

  /* Add type qualifiers if there are */
  if (type_variable_p(t) && variable_qualifiers(type_variable(t)) != NIL)
    pc = words_qualifier(variable_qualifiers(type_variable(t)));

  if (basic_type_p(t))
    {
      string sname = list_to_string(name);
      pips_debug(9,"Basic type with name = \"%s\"\n", sname);
 
      pc = gen_nconc(pc,words_type(t));
      pc = CHAIN_SWORD(pc," ");
      if(!bit_type_p(t) || (strstr(sname,DUMMY_MEMBER_PREFIX)==NULL)) {
	pc = gen_nconc(pc,name);
	}
      free(sname);
      if (bit_type_p(t))
	{
	  int i = basic_bit(variable_basic(type_variable(t)));
	  pc = CHAIN_SWORD(pc,":");
	  pc = CHAIN_IWORD(pc,i);
	}
      return pc;
    }
  if (array_type_p(t))
    {
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
      variable_dimensions(type_variable(t1)) = NIL;
      variable_qualifiers(type_variable(t1)) = NIL;
      return gen_nconc(pc,generic_c_words_entity(t1,gen_nconc(tmp,words_dimensions(dims)),is_safe));
    }
 
  if (derived_type_p(t))
    {
      entity ent = basic_derived(variable_basic(type_variable(t)));
      type t1 = entity_type(ent);
      string n = entity_name(ent);
      pips_debug(9,"Derived type with name = %s\n", list_to_string(name));
      if((strstr(n,DUMMY_ENUM_PREFIX)==NULL)
	 &&(strstr(n,DUMMY_STRUCT_PREFIX)==NULL)
	 &&(strstr(n,DUMMY_UNION_PREFIX)==NULL)) {
	pc = gen_nconc(pc,words_type(t1));
	pc = CHAIN_SWORD(pc," ");
	pc = CHAIN_SWORD(pc,entity_user_name(ent));
      }
      else {
	//pc = CHAIN_SWORD(pc,"problem!");
	pc = c_words_entity(t1, pc);
      }
      pc = CHAIN_SWORD(pc," ");
      return gen_nconc(pc,name);
    }
  if (typedef_type_p(t))
    {
      entity ent = basic_typedef(variable_basic(type_variable(t)));
      pips_debug(9,"Typedef type with name = %s\n", list_to_string(name));
      pc = CHAIN_SWORD(pc,entity_user_name(ent));
      pc = CHAIN_SWORD(pc," ");
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
      //if(strstr(sname,DUMMY_STRUCT_PREFIX)==NULL) {
      //	pc = gen_nconc(pc,name);
      //	pc = CHAIN_SWORD(pc," ");
      //}
      free(sname);
      pc = CHAIN_SWORD(pc,"{");

      for(cl = l; !ENDP(cl); POP(cl)) {
	entity sm = ENTITY(CAR(cl));
	type tsm = entity_type(sm);
	pc = gen_nconc(pc,c_words_entity(tsm,CHAIN_SWORD(NIL,entity_user_name(sm))));
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
	pc = gen_nconc(pc,c_words_entity(tu,CHAIN_SWORD(NIL,entity_user_name(eu))));
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
      bool first = TRUE;
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
	  pc = CHAIN_SWORD(pc, ", ");
	pc = CHAIN_SWORD(pc, entity_user_name(em));
	if(n!=cv) {
	  pc = CHAIN_SWORD(pc, "=");
	  pc = gen_nconc(pc, words_expression(eme));
	  cv = n;
	}
	cv++;
	first = FALSE;
      };
      pc = CHAIN_SWORD(pc,"}");
      return pc;
    }
   pips_internal_error("unexpected case\n");
  return NIL;
}

list c_words_entity(type t, list name)
{
  return generic_c_words_entity(t, name, FALSE);
}

list safe_c_words_entity(type t, list name)
{
  return generic_c_words_entity(t, name, TRUE);
}

text c_text_entities(entity module, list ldecl, int margin)
{
  text r = make_text(NIL);
  list l = ldecl;

  for( ; !ENDP(l); POP(l)) {
    entity e = ENTITY(CAR(l));
    text tmp = text_undefined;
    type t = entity_type(e);

    if(!type_area_p(t) && ! type_statement_p(t) && !type_unknown_p(t) && !storage_formal_p(entity_storage(e))) {
      string n = entity_name(e);
      /* Dummy enum must be printed anyway because their members are exposed directly. */
      if(/*(strstr(n,DUMMY_ENUM_PREFIX)==NULL)&& */
	 (strstr(n,DUMMY_STRUCT_PREFIX)==NULL)
	 &&(strstr(n,DUMMY_UNION_PREFIX)==NULL)) {
	tmp = c_text_entity(module, e, margin);
	MERGE_TEXTS(r,tmp);
      }
    }
  }

  return r; 
}

text c_text_entity(entity module, entity e, int margin)
{
  text r = make_text(NIL);
  string name = entity_user_name(e);
  type t = entity_type(e);
  storage s = entity_storage(e);
  value val = entity_initial(e);
  list pc = NIL;
 
  pips_debug(5,"Print declaration for entity %s in module %s\n",
	     entity_name(e),
	     entity_undefined_p(module)? "UNDEFINED" : entity_name(module));

  /* A declaration has two parts: declaration specifiers and declarator (even with initializer) 
     In declaration specifiers, we can have : 
     - storage specifiers : typedef, extern, static, auto, register
     - type specifiers : void, char, short, int, long, float, double, signed, unsigned, 
                         struct-or-union specifiers, enum specifier, typedef name 
     - type qualifiers : const, restrict, volatile 
     - function specifiers : inline */

  /* This part is for storage specifiers */
  if (!entity_undefined_p(module) && extern_entity_p(module, e))
    pc = CHAIN_SWORD(pc,"extern ");
  
  if (strstr(entity_name(e),TYPEDEF_PREFIX) != NULL)
    pc = CHAIN_SWORD(pc,"typedef ");
 
  /* The global variables stored in static area and in ram but they are not static so a condition is needed which checks if it not a global variable*/
  // entity m = get_current_module_entity();
  if ((storage_ram_p(s) && static_area_p(ram_section(storage_ram(s))) && !strstr(entity_name(e),TOP_LEVEL_MODULE_NAME)) 
      || (entity_module_p(e) && static_module_p(e)))
    pc = CHAIN_SWORD(pc,"static ");

  /* This part is for type specifiers, type qualifiers, function specifiers and declarator
     Three special cases for struct/union/enum definitions are treated here. 
     Variable (scalar, array), pointer, function, variables of type struct/union/enum and typedef 
     are treated by function c_words_entity */

  switch (type_tag(t)) {
  case is_type_struct:
    {
      list l = type_struct(t);
      text fields = c_text_entities(module,l,margin+INDENTATION);
      pc = CHAIN_SWORD(pc,"struct ");
      if(strstr(name,DUMMY_STRUCT_PREFIX)==NULL) {
	pc = CHAIN_SWORD(pc,name);
	pc = CHAIN_SWORD(pc," ");
      }
      pc = CHAIN_SWORD(pc,"{");
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,0,margin,pc)));
      MERGE_TEXTS(r,fields);
      ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"};")); 
      break;
    }
  case is_type_union:
    {
      list l = type_union(t);
      text fields = c_text_entities(module,l,margin+INDENTATION);
      pc = CHAIN_SWORD(pc,"union ");
      if(strstr(name,DUMMY_UNION_PREFIX)==NULL) {
	pc = CHAIN_SWORD(pc,name);
	pc = CHAIN_SWORD(pc," ");
      }
      pc = CHAIN_SWORD(pc,"{");
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,0,margin,pc)));
      MERGE_TEXTS(r,fields);
      ADD_SENTENCE_TO_TEXT(r, MAKE_ONE_WORD_SENTENCE(margin,"};")); 
      break;
    }
  case is_type_enum:
    {
      list l = type_enum(t);
      bool first = TRUE;
      pc = CHAIN_SWORD(pc,"enum ");
      if(strstr(name,DUMMY_ENUM_PREFIX)==NULL) {
	pc = CHAIN_SWORD(pc,name);
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

	if (!first)
	  pc = CHAIN_SWORD(pc, ", ");
	pc = CHAIN_SWORD(pc, entity_user_name(em));
	if(n!=cv) {
	  pc = CHAIN_SWORD(pc, "=");
	  pc = gen_nconc(pc, words_expression(eme));
	  cv = n;
	}
	cv++;
	first = FALSE;
      };
      pc = CHAIN_SWORD(pc,"};");
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,0,margin,pc)));
      break;
    }
  case is_type_variable:
  case is_type_functional:
  case is_type_void:
  case is_type_unknown:
    {
      pc = gen_nconc(pc,c_words_entity(t,CHAIN_SWORD(NIL,name)));
      /* This part is for declarator initialization if there is. 
	 If the entity is declared extern wrt current module, do not add this initialization*/
      if (!entity_undefined_p(module) && !extern_entity_p(module,e) && !value_undefined_p(val))
	{
	  if (value_expression_p(val))
	    {
	      expression exp = value_expression(val);
	      pc = CHAIN_SWORD(pc," = ");
	      if (brace_expression_p(exp))
		pc = gen_nconc(pc,words_brace_expression(exp));
	      else 
		pc = gen_nconc(pc,words_expression(exp));
	    }
	}
      pc = CHAIN_SWORD(pc,";");
      ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_unformatted,
					    make_unformatted(NULL,0,margin,pc)));
      break;
    }
  case is_type_varargs:
  case is_type_statement:
  case is_type_area:
  default:
    pips_error("c_text_entity", "unexpected type tag");
  }
  return r; 
}


/* The fprint_functional() and fprint_environment() functions are moved from syntax/declaration.c */ 

/* C Version of print_common_layout this is called by fprint_environment(). This function is much simpler than Fortran Version */

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
  list members = get_common_members(c, mod, FALSE);
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
	  //pips_error("print_common_layout",
	  //     "Non-empty area %s should have a size greater than 0\n",
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
	extern int SafeSizeOfArray(entity a);
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

/* This function is called from c_parse() via ResetCurrentModule() and fprint_environment() */
void fprint_functional(FILE * fd, functional f)
{
  type tr = functional_result(f);

  MAPL(cp, {
    parameter p = PARAMETER(CAR(cp));
    type ta = parameter_type(p);

    pips_assert("Argument type is variable or varags:variable or functional",
		type_variable_p(ta)
		|| (type_varargs_p(ta) && type_variable_p(type_varargs(ta)))
		|| type_functional_p(ta));

    if(type_functional_p(ta)) {
      functional fa = type_functional(ta);
      /* (void) fprintf(fd, " %s:", type_to_string(ta)); */
      (void) fprintf(fd, "(");
      fprint_functional(fd, fa);
      (void) fprintf(fd, ")");
    }
    else {
      if(type_varargs_p(ta)) {
	(void) fprintf(fd, " %s:", type_to_string(ta));
	ta = type_varargs(ta);
      }
      (void) fprintf(fd, "%s", basic_to_string(variable_basic(type_variable(ta))));
    }
    if(!ENDP(cp->cdr))
      (void) fprintf(fd, " x ");
  },
       functional_parameters(f));

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
    pips_error("fprint_functional", "Ill. type %d\n", type_tag(tr));
}

void fprint_environment(FILE *fd, entity m)
{
  fprint_any_environment(fd, m, TRUE);
}

void fprint_C_environment(FILE *fd, entity m)
{
  fprint_any_environment(fd, m, FALSE);
}

void fprint_any_environment(FILE * fd, entity m, bool is_fortran)
{
    list decls = gen_copy_seq(code_declarations(value_code(entity_initial(m))));
    int nth = 0; /* rank of formal parameter */
    entity rv = entity_undefined; /* return variable */

    pips_assert("fprint_environment", entity_module_p(m));

    /* To simplify validation, at the expense of some information about
       the parsing process. */
    gen_sort_list(decls, compare_entities);

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

      gen_sort_list(edecls, compare_entities);

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
        extern int SafeSizeOfArray(entity a);

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
	  print_common_layout(fd, e, FALSE);
	else
	  print_C_common_layout(fd, e, FALSE);
      }
    }, 
	decls);
    
    (void) fprintf(fd, "End of declarations for module %s\n\n",
		   module_local_name(m));

    gen_free_list(decls);
}
