/*
 * $Id$
 *
 * $Log: declarations.c,v $
 * Revision 1.2  1997/10/30 13:09:14  coelho
 * prettyprint of common/equiv with PRETTYPRINT_COMMONS="include" seems ok.
 *
 * Revision 1.1  1997/10/28 15:01:23  coelho
 * Initial revision
 *
 *
 * Regeneration of declarations...
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "pipsdbm.h"

#include "misc.h"
#include "properties.h"
#include "prettyprint.h"

/********************************************************************* WORDS */

static list 
words_constant(constant obj)
{
    list pc;

    pc=NIL;

    if (constant_int_p(obj)) {
	pc = CHAIN_IWORD(pc,constant_int(obj));
    }
    else {
	pips_internal_error("unexpected tag");
    }

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
	    pc = CHAIN_SWORD(pc, ",");
	}

	pc = CHAIN_SWORD(pc, entity_local_name(param));
    }

    return(pc);
}

static list 
words_dimension(dimension obj)
{
    list pc;
    pc = words_expression(dimension_lower(obj));
    pc = CHAIN_SWORD(pc,":");
    pc = gen_nconc(pc, words_expression(dimension_upper(obj)));
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
    pl = CHAIN_SWORD(pl, entity_local_name(e));

    if (type_variable_p(entity_type(e)))
    {
	if (prettyprint_common_variable_dimensions_p || 
	    !(variable_in_common_p(e) || variable_static_p(e)))
	{
	    if (variable_dimensions(type_variable(entity_type(e))) != NIL) 
	    {
		list dims = variable_dimensions(type_variable(entity_type(e)));
	
		pl = CHAIN_SWORD(pl, "(");

		MAPL(pd, 
		{
		    pl = gen_nconc(pl, words_dimension(DIMENSION(CAR(pd))));
		    if (CDR(pd) != NIL) pl = CHAIN_SWORD(pl, ",");
		}, 
		    dims);
	
		pl = CHAIN_SWORD(pl, ")");
	    }
	}
    }
    
    attach_declaration_to_words(pl, e);

    return(pl);
}

static list 
words_basic(basic obj)
{
    list pc = NIL;

    if (basic_int_p(obj)) {
	pc = CHAIN_SWORD(pc,"INTEGER*");
	pc = CHAIN_IWORD(pc,basic_int(obj));
    }
    else if (basic_float_p(obj)) {
	pc = CHAIN_SWORD(pc,"REAL*");
	pc = CHAIN_IWORD(pc,basic_float(obj));
    }
    else if (basic_logical_p(obj)) {
	pc = CHAIN_SWORD(pc,"LOGICAL*");
	pc = CHAIN_IWORD(pc,basic_logical(obj));
    }
    else if (basic_overloaded_p(obj)) {
	pc = CHAIN_SWORD(pc,"OVERLOADED");
    }
    else if (basic_complex_p(obj)) {
	pc = CHAIN_SWORD(pc,"COMPLEX*");
	pc = CHAIN_IWORD(pc,basic_complex(obj));
    }
    else if (basic_string_p(obj)) {
	pc = CHAIN_SWORD(pc,"STRING*(");
	pc = gen_nconc(pc, words_value(basic_string(obj)));
	pc = CHAIN_SWORD(pc,")");
    }
    else {
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

    fe = type_functional(te);
    tr = functional_result(fe);
    
    if (type_void_p(tr)) 
    {
	if (entity_main_module_p(e))
	    pc = CHAIN_SWORD(pc, "PROGRAM ");
	else
	{
	    if (entity_blockdata_p(e))
		pc = CHAIN_SWORD(pc, "BLOCKDATA ");
	    else
		pc = CHAIN_SWORD(pc, "SUBROUTINE ");
	}
    }
    else if (type_variable_p(tr)) {
	pc = gen_nconc(pc, words_basic(variable_basic(type_variable(tr))));
	pc = CHAIN_SWORD(pc, " FUNCTION ");
    }
    else {
	pips_internal_error("unexpected type for result\n");
    }
    pc = CHAIN_SWORD(pc, module_local_name(e));

    if ( !ENDP(args) ) {
	pc = CHAIN_SWORD(pc, "(");
	pc = gen_nconc(pc, args);
	pc = CHAIN_SWORD(pc, ")");
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

    if (dynamic_area_p(e)) /* shouldn't get in? */
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
		 if (comma) pc = CHAIN_SWORD(pc, ",");
		 else comma = TRUE;
		 pc = gen_nconc(pc, 
			words_declaration(ee, !is_save && pp_dimensions));
	     },
		 entities);

	    gen_free_list(entities);
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
		  strdup(concatenate("      include '", file, "'\n", 0))),
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
    local = strdup(concatenate(name, ".h", 0));
    file = strdup(concatenate(dir, "/", local, 0));
    free(dir);

    if (file_exists_p(file))
    {
	/* the include was generated once before... */
	t = include(local);
    }
    else 
    {
	string nofile = 
	    strdup(concatenate(file, ".sorry_common_not_homogeneous", 0));
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
static string default_common_hook(entity module, entity common)
{
    return strdup(concatenate
        ("common to include: ", entity_local_name(common), "\n", NULL));
}

static string (*common_hook)(entity, entity) = default_common_hook; 
void set_prettyprinter_common_hook(string(*f)(entity,entity)){ common_hook=f;}
void reset_prettyprinter_common_hook(){ common_hook=default_common_hook;}

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
text_equivalence_class(list /* of entities */ l_equiv)
{
    text t_equiv = make_text(NIL);
    list lw = NIL;
    list l1, l2;
    entity ent1, ent2;
    int offset1, offset2;
    Value size1, size2, offset_end1;
    boolean first;

    if (ENDP(l_equiv)) return(t_equiv);

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
	
    /* We are sure that there is at least one equivalence */
    lw = CHAIN_SWORD(lw, "EQUIVALENCE");

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
	    if (! first)
		lw = CHAIN_SWORD(lw, ",");
	    else
		first = FALSE;
	    lw = CHAIN_SWORD(lw, " (");
	    lw = CHAIN_SWORD(lw, entity_local_name(ent1));
	    lw = CHAIN_SWORD(lw, ",");
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
				
		if (! first)
		    lw = CHAIN_SWORD(lw, ",");
		else
		    first = FALSE;
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
			       size, rest, offset, VALUE_TO_INT(val_of(pvlo)));
		    
		    new_decl = VALUE_TO_INT(val_of(pvlo)) + rest;
		    buffer[0] = '\0';
		    sprintf(buffer+strlen(buffer), "%d", new_decl);		 
		    lw = CHAIN_SWORD(lw,strdup(buffer));			
		    if (current_dim < dim_max)
			lw = CHAIN_SWORD(lw, ",");
		    
		    POP(l_tmp);
		    current_dim++;
		    
		} /* while */
		
		lw = CHAIN_SWORD(lw, ")");	
		lw = CHAIN_SWORD(lw, ",");
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
    entity module     /* the module dealt with */, 
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
		pips_debug(1, "considering entity: %s\n", 
			   entity_local_name(e));
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
		}, equiv_classes);
		
		if (found)
		{
		    pips_debug(EQUIV_DEBUG, "already there\n");
		    /* add the entities of shared which are not already in 
		     * the existing equivalence class. Useful ??
		     */
		    MAP(ENTITY, ent,
		    {
			if(!variable_in_list_p(ent, found_equiv_class))
			    found_equiv_class =
				gen_nconc(found_equiv_class,
					  CONS(ENTITY, ent, NIL));
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
			if (!variable_in_list_p(shared_ent, l_tmp))
			    l_tmp = gen_nconc(l_tmp,
					      CONS(ENTITY, shared_ent,
						   NIL));
		    }, 
			l_shared);
		    equiv_classes =
			gen_nconc(equiv_classes, CONS(LIST, l_tmp, NIL));
		}
	    }
	}
    }, ldecl);
    
    ifdebug(EQUIV_DEBUG)
    {
	pips_debug(1, "final equivalence classes:\n");
	MAP(LIST, equiv_class,
	{
	    equiv_class_debug(equiv_class);
	},
	    equiv_classes);	
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

/* returns the DATA initializations.
 * limited to integers, because I do not know where is the value
 * for other types...
 */
static text 
text_data(entity module, list /* of entity */ ldecl)
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

static text 
text_entity_declaration(
    entity module, 
    list /* of entity */ ldecl,
    bool force_common)
{
    string how_common = get_string_property("PRETTYPRINT_COMMONS");
    bool print_commons = !same_string_p(how_common, "none");
    list before = NIL, area_decl = NIL, ph = NIL,
	pi = NIL, pf4 = NIL, pf8 = NIL, pl = NIL, 
	pc8 = NIL, pc16 = NIL, ps = NIL;
    text r, t_chars = make_text(NIL), t_area = make_text(NIL); 
    string pp_var_dim = get_string_property("PRETTYPRINT_VARIABLE_DIMENSIONS");
    bool pp_in_type = FALSE, pp_in_common = FALSE, pp_cinc;
     
    /* where to put the dimensionn information.
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
	
	pips_debug(3, "entity name is %s\n", entity_name(e));

	if (!print_commons && area_p && !SPECIAL_COMMON_P(e) && !pp_cinc)
	{
	    area_decl = 
		CONS(SENTENCE, make_sentence(is_sentence_formatted,
					     common_hook(module, e)),
		     area_decl);
	}
	
	if (!print_commons && (area_p || (var && in_common && pp_cinc)))
	{
	    pips_debug(5, "skipping entity %s\n", entity_name(e));
	}
	else if (param || external)
	{
	    before = CONS(SENTENCE, sentence_basic_declaration(e), before);
	    if (param) {
		/*        PARAMETER
		 */
		pips_debug(7, "considered as a parameter\n");
		before = CONS(SENTENCE, sentence_symbolic(e), before);
	    } else {
		/*        EXTERNAL
		 */
		pips_debug(7, "considered as an external\n");
		before = CONS(SENTENCE, sentence_external(e), before);
	    }
	 }
	else if (area_p && !dynamic_area_p(e) && !empty_static_area_p(e))
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
		 /* simple integers are moved ahead...
		  */
		pips_debug(7, "is an integer\n");
		if (variable_dimensions(type_variable(te)))
		{
		    pi = CHAIN_SWORD(pi, pi==NIL ? "INTEGER " : ",");
		    pi = gen_nconc(pi, words_declaration(e, pp_dim)); 
		}
		else
		{
		    ph = CHAIN_SWORD(ph, ph==NIL ? "INTEGER " : ",");
		    ph = gen_nconc(ph, words_declaration(e, pp_dim)); 
		}
		break;
	    case is_basic_float:
		pips_debug(7, "is a float\n");
		switch (basic_float(b))
		{
		case 4:
		    pf4 = CHAIN_SWORD(pf4, pf4==NIL ? "REAL*4 " : ",");
		    pf4 = gen_nconc(pf4, words_declaration(e, pp_dim));
		    break;
		case 8:
		default:
		    pf8 = CHAIN_SWORD(pf8, pf8==NIL ? "REAL*8 " : ",");
		    pf8 = gen_nconc(pf8, words_declaration(e, pp_dim));
		    break;
		}
		break;			
	    case is_basic_complex:
		pips_debug(7, "is a complex\n");
		switch (basic_complex(b))
		{
		case 8:
		    pc8 = CHAIN_SWORD(pc8, pc8==NIL ? "COMPLEX*8 " : ",");
		    pc8 = gen_nconc(pc8, words_declaration(e, pp_dim));
		    break;
		case 16:
		default:
		    pc16 = CHAIN_SWORD(pc16, pc16==NIL ? "COMPLEX*16 " : ",");
		    pc16 = gen_nconc(pc16, words_declaration(e, pp_dim));
		    break;
		}
		break;
	    case is_basic_logical:
		pips_debug(7, "is a logical\n");
		pl = CHAIN_SWORD(pl, pl==NIL ? "LOGICAL " : ",");
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
			ps = CHAIN_SWORD(ps, ps==NIL ? "CHARACTER " : ",");
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
		else
		    pips_internal_error("unexpected value\n");
		break;
	    }
	    default:
		pips_internal_error("unexpected basic tag (%d)\n",
				    basic_tag(b));
	    }
	}
    }, ldecl);
    
    /* parameters must be kept in order
     * because that may depend one from the other, hence the reversion.
     */ 
    r = make_text(gen_nreverse(before));

    ADD_WORD_LIST_TO_TEXT(r, ph);
    attach_declaration_type_to_words(ph, "INTEGER");
    ADD_WORD_LIST_TO_TEXT(r, pi);
    attach_declaration_type_to_words(pi, "INTEGER");
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

    /* all about COMMON and SAVE declarations */
    MERGE_TEXTS(r, make_text(area_decl));

    MERGE_TEXTS(r, t_area);

    /* and EQUIVALENCE statements... - BC -*/
    MERGE_TEXTS(r, text_equivalences(module, ldecl, 
				     pp_cinc || !print_commons));

    /* what about DATA statements! FC */
    MERGE_TEXTS(r, text_data(module, ldecl));

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
