/*
 * $Id$
 *
 *
 * Functions to handle all declarations, but some related to functional
 * entities
 *
 * Remi Triolet
 *
 * Modifications:
 *  - DeclareVariable() : an implicit type is assumed to be mutable like
 *    an undefined type; Francois Irigoin, 5 April 1991
 *  - FindOrCreateEntity() : see below; numerous trials of various
 *    partial links at compile time; see also procedure.c, 
 *    update_called_modules()
 *  - DeclareVariable() is rewritten from scratch; (Francois Irigoin, 
 *    20 September 1991)
 *  - AnalyzeData() : "too few analyzers" for an exact count; the last 
 *    iteration of pcr = CDR(pcr) is not executed because pcl==NIL;
 *    fix: update of datavar_nbelements(dvr) and modification of the
 *    condition guarding the message emission; FI, 18 February 1992
 *    (this might be no more than a new bug! )
 *  - MakeDataVar() : used to consider that array X was full initialized
 *    each time an array element was initialized; a check for subscripts
 *    was added; FI, 18 February 1992
 *  - check_common_layouts() : added to fix bug 1; FI, 1 December 1993
 * 
 * Bugs:
 *  - layout for commons are wrong if type and/or dimension declarations
 *    follow the COMMON declaration; PIPS Fortran syntax should be modified
 *    to prevent this;
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "misc.h"

#include "syntax.h"

#define IS_UPPER(c) (isascii(c) && isupper(c))

int
SafeSizeOfArray(entity a)
{
    int s;

    if(!SizeOfArray(a, &s)) {
	user_warning("SafeSizeOfArray", "Varying size of array \"%s\"\n", entity_name(a));
	ParserError("SafeSizeOfArray", "Fortran standard prohibit varying size array\n");
    }

    return s;
}

void 
InitAreas()
{
    DynamicArea = FindOrCreateEntity(CurrentPackage, DYNAMIC_AREA_LOCAL_NAME);
    entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(DynamicArea) = MakeStorageRom();
    entity_initial(DynamicArea) = MakeValueUnknown();
    AddEntityToDeclarations(DynamicArea, get_current_module_entity());

    StaticArea = FindOrCreateEntity(CurrentPackage, STATIC_AREA_LOCAL_NAME);
    entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(StaticArea) = MakeStorageRom();
    entity_initial(StaticArea) = MakeValueUnknown();
    AddEntityToDeclarations(StaticArea, get_current_module_entity());
}

/* functions for the SAVE declaration */

void 
save_all_entities()
{
    entity mod = get_current_module_entity();
    list vars = code_declarations(value_code(entity_initial(mod)));

    pips_assert("save_all_entities", StaticArea != entity_undefined);
    pips_assert("save_all_entities", DynamicArea != entity_undefined);

    /* FI: all variables previously allocated should be reallocated */

    MAP(ENTITY, e, {
	storage s;
	if((s=entity_storage(e))!= storage_undefined) {
	    if(storage_ram_p(s) && ram_section(storage_ram(s)) == DynamicArea) {
		free_storage(entity_storage(e));
		entity_storage(e) =
		    make_storage(is_storage_ram,
				 (make_ram(mod,
					   StaticArea, 
					   CurrentOffsetOfArea(StaticArea ,e),
					   NIL)));
	    }
	}
    }, vars);

    /* FI: This is pretty crude... Let's hope it works */
    DynamicArea = StaticArea;
}

/* These two functions transform a dynamic variable into a static one. 
 * They are called to handle SAVE and DATA statements.
 */

void 
SaveEntity(e)
entity e;
{
    entity g = local_name_to_top_level_entity(entity_local_name(e));

    if(!entity_undefined_p(g)
       /* Let's hope functions and subroutines called are listed in the
	* declaration list.
	*/
       && entity_is_argument_p(g, code_declarations(value_code(entity_initial(get_current_module_entity()))))) {
	user_warning("SaveEntity", 
		     "Ambiguity between external %s and local %s forbidden by Fortran standard\n",
		     entity_name(g), entity_name(e));
	ParserError("SaveEntity", "Name conflict\n");
    }

    if (entity_type(e) == type_undefined) {
	DeclareVariable(e, type_undefined, NIL, 
			storage_undefined, value_undefined);
    }

    if (entity_storage(e) != storage_undefined) {
	if (storage_ram_p(entity_storage(e))) {
	    ram r;

	    r = storage_ram(entity_storage(e));

	    if (ram_section(r) == DynamicArea) {
		ram_section(r) = StaticArea;
		ram_offset(r) = CurrentOffsetOfArea(StaticArea, e);
	    }
	    else {
		/* Not much can be said. Maybe it is redundant, but... */
		/* Maybe the standard claims that you are not allowed
		 * to save a common variable?
		 */
		/*
		user_warning("SaveEntity", "Variable %s has already been declared static "
			     "by SAVE, by DATA or by appearing in a common declaration\n",
			     entity_local_name(e));
			     */
	    }
	}
	else {
	    user_warning("SaveEntity",
			 "Cannot save variable %s with non RAM storage (storage tag = %d)\n",
			 entity_local_name(e),
			 storage_tag(entity_storage(e)));
	    ParserError("SaveEntity", "Cannot save this variable");
	}
    }
    else {
	entity_storage(e) =
		make_storage(is_storage_ram,
			     (make_ram(get_current_module_entity(), 
				       StaticArea, 
				       CurrentOffsetOfArea(StaticArea ,e),
				       NIL)));
    }
}

void
ProcessSave(entity v)
{
    if(entity_storage(v) == storage_undefined) {
	SaveEntity(v);
    }
    else if(storage_ram_p(entity_storage(v))) {
	entity a = ram_section(storage_ram(entity_storage(v)));
	    if(a==DynamicArea) {
		SaveEntity(v);
	    }
	    else if(a==StaticArea) {
		/* v may have become static because of a DATA statement (OK)
		 * or because of another SAVE (NOK)
		 */
	    }
	    else {
		user_warning("ProcessSave", "Variable %s has already been declared static "
			     "by appearing in Common %s\n",
			     entity_local_name(v), module_local_name(a));
		ParserError("parser", "SAVE statement incompatible with previous"
			    " COMMON declaration\n");
	    }
    }
    else {
	user_warning("parser", "Variable %s cannot be declared static "
		     "be cause of its storage class (tag=%d)\n",
		     entity_local_name(v), storage_tag(entity_storage(v)));
    }
}

/* this function transforms a dynamic common into a static one.  */

void 
SaveCommon(c)
entity c;
{
    pips_assert("SaveCommon",type_area_p(entity_type(c)));

    Warning("SaveCommon", "common blocks are automatically saved\n");

    return;
}



/* a debugging function, just in case ... */

void 
PrintData(ldvr, ldvl)
cons *ldvr, *ldvl;
{
    cons *pc;

    debug(7, "PrintData", "Begin\n");

    for (pc = ldvr; pc != NIL; pc = CDR(pc)) {
	datavar dvr = DATAVAR(CAR(pc));

	debug(7, "PrintData", "(%s,%d), ", entity_name(datavar_variable(dvr)),
	      datavar_nbelements(dvr));

    }
    debug(7, "PrintData", "\n");

    for (pc = ldvl; pc != NIL; pc = CDR(pc)) {
	dataval dvl = DATAVAL(CAR(pc));

	if (constant_int_p(dataval_constant(dvl))) {
	    debug(7, "PrintData", "(%d,%d), ", constant_int(dataval_constant(dvl)),
		  dataval_nboccurrences(dvl));
	}
	else {
	    debug(7, "PrintData", "(x,%d), ", dataval_nboccurrences(dvl));
	}

    }
    debug(7, "PrintData", "End\n\n");
}



/* this function scans at the same time a list of datavar and a list of
dataval. it tries to match datavar to dataval and to compute initial
values for zero dimension variable of basic type integer. 

ldvr is a list of datavar.

ldvl is a list of dataval. */

void 
AnalyzeData(list ldvr, list ldvl)
{
    list pcr, pcl;
    dataval dvl;

    /* FI: this assertion must be usually wrong!
     * pips_assert("AnalyseData", gen_length(ldvr) == gen_length(ldvl));
     */

    debug(8, "AnalyzeData", "number of reference groups: %d\n", 
	  gen_length(ldvr));

    pcl = ldvl;
    dvl = DATAVAL(CAR(pcl));
    for (pcr = ldvr; pcr != NIL && pcl != NIL; pcr = CDR(pcr)) {
	datavar dvr = DATAVAR(CAR(pcr));
	entity e = datavar_variable(dvr);
	int i = datavar_nbelements(dvr);
	
	debug(8, "AnalyzeData", 
	      "Storage for entity %s must be static or made static\n",
	      entity_name(e));

	if(storage_undefined_p(entity_storage(e))) {
	    entity_storage(e) =
		make_storage(is_storage_ram,
			     (make_ram(get_current_module_entity(),
				       StaticArea, 
				       CurrentOffsetOfArea(StaticArea ,e),
				       NIL)));
	}
	else if(storage_ram_p(entity_storage(e))) {
	    if(dynamic_area_p(ram_section(storage_ram(entity_storage(e))))) {
		SaveEntity(e);
	    }
	    else {
		/* Variable is in static area or in a user declared common */
	    }
	}
	else {
	    user_warning("AnalyzeData",
			 "DATA initialization for non RAM variable %s "
			 "(storage tag = %d)\n",
			entity_name(e), storage_tag(entity_storage(e)));
	    ParserError("AnalyzeData", 
			"DATA statement initializes non RAM variable\n");
	}

	debug(8, "AnalyzeData", "needs %d elements for entity %s\n", 
	      i, entity_name(e));

	pips_assert("AnalyzeData", dataval_nboccurrences(dvl) > 0);

	if (IsIntegerScalar(e)) {
	    pips_assert("AnalyzeData", i == 1);

	    if(constant_int_p(dataval_constant(dvl))) {
	      entity_initial(e) = make_value(is_value_constant, 
					     dataval_constant(dvl));

	      debug(1, "AnalyzeData", "%s %d\n", 
		    entity_name(e), constant_int(dataval_constant(dvl)));
	    }
	    else {
	      Warning("AnalyzeData", 
		      "Integer scalar variable initialized "
		      "with non-integer constant");
	    }
	}

	while (i > 0 && pcl != NIL) {
	    if (i <= dataval_nboccurrences(dvl)) {
		debug(8, "AnalyzeData", "uses %d values out of %d\n",
		      i, dataval_nboccurrences(dvl));
		dataval_nboccurrences(dvl) -= i;
		i = 0;
	    }
	    else {
		debug(8, "AnalyzeData", "satisfies %d references out of %d\n",
		      dataval_nboccurrences(dvl), i);
		i -= dataval_nboccurrences(dvl);
		dataval_nboccurrences(dvl) = 0;
	    }

	    if (dataval_nboccurrences(dvl) == 0) {
		if ((pcl = CDR(pcl)) != NIL) {
			dvl = DATAVAL(CAR(pcl));

			debug(8, "AnalyzeData", "use next dataval\n");
		    }
	    }
	    datavar_nbelements(dvr) = i;
	}
    }

    if (pcl != NIL) {
	Warning("AnalyzeData", "too many initializers\n");
    }

    if (pcr != NIL && 
	(datavar_nbelements(DATAVAR(CAR(pcr))) != 0 || CDR(pcr) != NIL)) {
	ParserError("AnalyzeData", "too few initializers\n");
    }
}


/* type_equal_p -> same_basic_and_scalar_p in latter... FC.
 */
static bool
same_basic_and_scalar_p(type t1, type t2)
{
    variable v1, v2;
    if (!type_variable_p(t1) || !type_variable_p(t2)) return FALSE;
    v1 = type_variable(t1);
    v2 = type_variable(t2);
    if (variable_undefined_p(v1) || variable_undefined_p(v2)) return FALSE;
    if (!basic_equal_p(variable_basic(v1), variable_basic(v2))) return FALSE;
    return variable_dimensions(v1)==NIL && variable_dimensions(v2)==NIL;
}

/* void DeclareVariable(e, t, d, s, v): update entity e description
 * as declaration statements are encountered. Examples of sequences:
 *
 *   INTEGER*4 T
 *   DIMENSION T(10)
 *   SAVE T
 *   
 * or
 *   COMMON /TOTO/X,Y
 *   CHARACTER*30 X
 *   DIMENSION X(10)
 *
 * or
 *   EXTERNAL F
 *   INTEGER F
 *
 * The input code is assumed correct. As the standard states, IMPLICIT
 * statements must occur before *any* declaration.
 *
 * Parameters:
 *  e is an entity which should be either a variable or a funtion; it
 *    may already have a type et, of kind variable or functional;
 *    the type variable may have a dimension; variable or functional
 *    implicit types, as well as undefined type, can be superseded by
 *    the new type t; a NIL type dimension can be superseded by d;
 *    how should area entities be handled ???
 *  t is a type of kind "variable" (functional types are not accepted;
 *    functional declaration are handled by ??? ) or undefined; 
 *    it should have no dimensions;
 *  d is a (possibly) empty list of dimensions; the empty list is
 *    handled as the undefined list; each dimension is an expression
 *  s is the storage, possibly undefined;
 *  v is the initial value, possibly undefined
 *
 * Most problems occur because of the great number of combinations between
 * the entity type et (undefined, variable, functional) and the entity type
 * dimension etd (NIL ot not) giving 7 cases on one hand, the type t and 
 * the dimensions d giving 4 cases on the other hand. That is 28 different
 * behaviors.
 *
 * No sharing is introduced between t and et. However d and s are directly
 * used in e fields.
 */
void 
DeclareVariable(
    entity e,
    type t,
    list d,
    storage s,
    value v)
{
    type et = entity_type(e);
    list etd = list_undefined;
    bool variable_had_implicit_type_p = FALSE;

    debug(8, "DeclareVariable", "%s\n", entity_name(e));
    pips_assert("DeclareVariable", t == type_undefined || type_variable_p(t));

    if(et == type_undefined) {
	if(t == type_undefined) {
	    entity_type(e) = ImplicitType(e);
	    variable_dimensions(type_variable(entity_type(e))) = d;
	}
	else {
	    type nt;
	    nt = MakeTypeVariable
		(copy_basic(variable_basic(type_variable(t))),
		 d);
	    entity_type(e) = nt;
	}
    }
    else
	switch(type_tag(et)) {
	case is_type_functional:
	    if(d!=NIL) {
		user_warning("DeclareVariable",
			     "%s %s between lines %d and % d\n",
			     "Attempt to dimension functional entity",
			     entity_local_name(e), line_b_I, line_e_I);
		ParserError("DeclareVariable", "Likely name conflict\n");
	    }
	    if(t == type_undefined)
		/* no new information: do nothing */
		;
	    else 
		if (implicit_type_p(e)) {
		    /* update functional type */
		    type nt = MakeTypeVariable
			(copy_basic(variable_basic(type_variable(t))),
			 NIL);
		    functional_result(type_functional(et)) = nt;
		    /* the old type should be gen_freed... */
		}
		else {
		    user_warning("DeclareVariable",
				 "%s %s between lines %d and % d\n",
				 "Redefinition of functional type for entity",
				 entity_local_name(e), line_b_I, line_e_I);
		    ParserError("DeclareVariable",
				"Possible name conflict?\n");
		}
	    break;
	case is_type_variable:
	    etd = variable_dimensions(type_variable(et));
	    if(t == type_undefined) {
		/* set dimension etd if NIL */
		if(etd==NIL)
		    variable_dimensions(type_variable(et)) = d;
		else if (d==NIL)
		    ;
		else {
		    user_warning("DeclareVariable",
				 "%s %s between lines %d and % d\n",
				 "Redefinition of dimension for entity",
				 entity_name(e), line_b_I, line_e_I);
		    ParserError("DeclareVariable", "Name conflict?\n");
		}
	    }
	    else {
		pips_assert("DeclareVariable",
			    variable_dimensions(type_variable(t))==NIL);
		if(implicit_type_p(e)){
		    type nt;

		    variable_had_implicit_type_p = TRUE;

		    /* set dimension etd if NIL */
		    if(etd==NIL)
			variable_dimensions(type_variable(et)) = d;
		    else if (d==NIL)
			;
		    else {
			user_warning("DeclareVariable",
				     "%s %s between lines %d and % d\n",
				     "Redefinition of dimension for entity",
				     entity_local_name(e), line_b_I, line_e_I);
			ParserError("DeclareVariable", "Name conflict?\n");
		    }
		    /* update type */
		    nt = MakeTypeVariable
			(copy_basic(variable_basic(type_variable(t))),
			 variable_dimensions(type_variable(et)));
		    
		    if(!same_basic_and_scalar_p(entity_type(e), nt))
		    {
			
			if(/*FI: to check update_common_layout*/ FALSE && 
			   entity_storage(e)!=storage_undefined &&
			   storage_ram_p(entity_storage(e)) &&
			   basic_type_size(variable_basic(type_variable(t)))
	   > basic_type_size(variable_basic(type_variable(entity_type(e))))) 
			{
			    user_warning("DeclareVariable",
     "Storage information for %s is likely to be wrong because its type is "
     "redefined as a larger type\nType is *not* redefined internally to avoid "
     "aliasing\n", entity_local_name(e));
		    /* FI: it should be redefined and the offset be updated,
		     * maybe in check_common_area(); 1 Feb. 1994
		     */
			}
			else {
			    entity_type(e) = nt;
			}
		    }
		    else {
			free_type(nt);
		    }
		}
		else {
		    user_warning("DeclareVariable",
				 "%s %s between lines %d and % d\n",
				 "Redefinition of type for entity",
				 entity_local_name(e), line_b_I, line_e_I);
		    ParserError("DeclareVariable",
				"Name conflict or declaration ordering "
				"not supported by PIPS\n"
				"Late typing of formal parameter and/or "
				"interference with IMPLICIT\n");
		}
	    }
	    break;
	case is_type_area:
	    user_warning("DeclareVariable",
			 "%s %s between lines %d and % d\n%s\n",
			 "COMMON/VARIABLE homonymy for entity name",
			 entity_local_name(e), line_b_I, line_e_I,
			 "Rename your common.");
	    ParserError("DeclareVariable", "Name conflict\n");
	    break;
	default:
	    pips_error("DeclareVariable",
		       "unexpected entity type tag: %d\n",
		       type_tag(et));
	}

    if (s != storage_undefined) {
	if (entity_storage(e) != storage_undefined) {
	    ParserError("DeclareVariable", "storage non implemented\n");
	}
	else {
	    entity_storage(e) = s;
	}
    }

    if (v == value_undefined) {
	if (entity_initial(e) == value_undefined) {
	    entity_initial(e) = MakeValueUnknown();
	}
    }
    else {
	ParserError("DeclareVariable", "value non implemented\n");
    }

    AddEntityToDeclarations(e, get_current_module_entity());

    /* If the return variable is retyped, the function must be retyped */

    if(!type_undefined_p(t) && !storage_undefined_p(entity_storage(e)) 
       && storage_return_p(entity_storage(e))) {
	entity f = get_current_module_entity();
	type tf = entity_type(f);
	functional func = type_functional(tf);
	type tr = functional_result(func);
	basic old = variable_basic(type_variable(tr));
	basic new = variable_basic(type_variable(t));

	pips_assert("Return variable and function must have the same name",
		    strcmp(entity_local_name(e), module_local_name(f)) == 0 );
	pips_assert("Function must have functional type", type_functional_p(tf));
	pips_assert("New type must be of kind variable", type_variable_p(t));

	if(!type_equal_p(tr, t)) {
	    if(variable_had_implicit_type_p) {
		debug(8, "DeclareVariable", " Type for result of function %s "
		      "changed from %s to %s: ", module_local_name(f),
			     basic_to_string(old), basic_to_string(new));
		free_type(functional_result(func));
		old = basic_undefined; /* the pointed area has just been freed! */
		functional_result(func) = copy_type(t);
		ifdebug(8) {
		  fprint_functional(stderr, type_functional(tf));
		  fprintf(stderr, "\n");
		}
	    }
	    else {
		user_warning("DeclareVariable",
			     "Attempt to retype function %s with result of type "
			     "%s with new type %s\n", module_local_name(f),
			     basic_to_string(old), basic_to_string(new));
		ParserError("DeclareVariable", "Illegal retyping");
	    }
	}
	else {
	    /* Meaningless warning when the result variable is declared the first time
	     * with the function itself
	     * user_warning("DeclareVariable",
	     *	     "Attempt to retype function %s with result of type "
	     *	     "%s with very same type %s\n", module_local_name(f),
	     *	     basic_to_string(old), basic_to_string(new));
	     */
	}
    }
}

/*
 * COMMONs are handled as global objects. They may pre-exist when
 * the current module is parsed. They may also be declared in more
 * than one instruction. So we need to keep track of the current
 * size of each common encountered in the current module in
 * mapping from entity to integer, common_size_map.
 *
 * At the end of the declaration phase, common sizes can be
 * either set if yet unknown, or compared for equality with
 * a pre-defined size.
 */
static hash_table common_size_map = hash_table_undefined;

void 
initialize_common_size_map()
{
    pips_assert("initialize_common_size_map", 
		common_size_map == hash_table_undefined);
    common_size_map = hash_table_make(hash_pointer, 0);
}

void
reset_common_size_map()
{
   if (common_size_map != hash_table_undefined) {
      hash_table_free(common_size_map);
      common_size_map = hash_table_undefined;
   }
   else {
       /* Problems:
	*  - this routine may be called from ParserError()... which should not
	*    be called recursively
	*  - but it maight also be called from somewhere else and ParserError()
	*    then should be called
	* A second reset routine must be defined.
	*/
       ParserError("reset_common_size_map", "Resetting a resetted variable!\n");
   }
}

void
reset_common_size_map_on_error()
{
   if (common_size_map != hash_table_undefined) {
      hash_table_free(common_size_map);
      common_size_map = hash_table_undefined;
   }
}

bool
common_to_defined_size_p(entity a)
{
    bool defined = FALSE;

    defined = ( (hash_get(common_size_map,(char *) a))
	!= HASH_UNDEFINED_VALUE );

    return defined;
}

int
common_to_size(entity a)
{
    int size;

    if((size = (int) hash_get(common_size_map,(char *) a))
       == (int) HASH_UNDEFINED_VALUE) {
	    pips_error("common_to_size",
		       "common_size_map uninitialized for common %s\n",
		       entity_name(a));
    }

    return size;
}

void
set_common_to_size(entity a, int size)
{
    (void) hash_put(common_size_map, (char *) a,
		       (char *) (size));
}

void
update_common_to_size(entity a, int new_size)
{
    (void) hash_update(common_size_map, (char *) a,
		       (char *) (new_size));
}

/* updates the common entity if necessary with the common prefix
 */
static entity
make_common_entity(entity c)
{
    if (!entity_common_p(c))
    {
	if (type_undefined_p(entity_type(c)))
	{
	    entity_type(c) = make_type(is_type_area, make_area(0, NIL));
            entity_storage(c) = 
                make_storage(is_storage_ram, 
                             (make_ram(get_current_module_entity(),
                                       StaticArea, 0, NIL)));
            entity_initial(c) = MakeValueUnknown();
            AddEntityToDeclarations(c, get_current_module_entity());
	}
    }

    return c;
}

/* MakeCommon:
 * This function creates a common block. pips creates static common
 * blocks. This is not true in the ANSI standard stricto sensu, but
 * true in most implementations.
 *
 * A common declaration can be made out of several common statements.
 * MakeCommon() is called for each common statement, although it only
 * is useful the first time.
 */
entity 
MakeCommon(entity e)
{
    e = make_common_entity(e);

    /* common e may already exist because it was encountered
     * in another module
     * but not have been registered as known by the current module.
     * It may also already exist because it was encountered in
     * the *same* module, but AddEntityToDeclarations() does not
     * duplicate declarations.
     */
    AddEntityToDeclarations(e, get_current_module_entity());

    /* FI: for a while, common sizes were *always* reset to 0, even when
     * several common statements were encountered in the same module for
     * the same common. This did not matter because offsets in commons are
     * recomputed once variable types and dimensions are all known.
     */
    if(!common_to_defined_size_p(e))
	set_common_to_size(e, 0);

    return e;
}
entity 
NameToCommon(string n)
{
    string c_name = strdup(concatenate(COMMON_PREFIX, n, NULL));
    entity c = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, c_name);
    string prefixes[] = {"", MAIN_PREFIX, BLOCKDATA_PREFIX, NULL};
    string nature[] = {"function or subroutine", "main", "block data"};
    int i = 0;

    c = MakeCommon(c);
    free(c_name);

    /* Check for potential conflicts */
    for(i=0; prefixes[i]!=NULL; i++) {
	string name = strdup(concatenate(prefixes[i], n, NULL));
	entity ce = global_name_to_entity(TOP_LEVEL_MODULE_NAME, name);

	if(!entity_undefined_p(ce)) {
	    user_warning("NameToCommon", "Identifier %s used for a common and for a %s\n",
			 n, nature[i]);
	}

	free(name);
    }

    return c;
}

/* 
 * This function adds a variable v to a common block c. v's storage must be
 * undefined. c's size is indirectly updated by CurrentOffsetOfArea().
 *
 */ 
void 
AddVariableToCommon(c, v)
entity c, v;
{
    if (entity_storage(v) != storage_undefined) {
	FatalError("AddVariableToCommon", "storage already defined\n");
    }

    DeclareVariable(v, 
		    type_undefined, 
		    NIL, 
		    (make_storage(is_storage_ram,
				  (make_ram(get_current_module_entity(), c, 
					    CurrentOffsetOfArea(c ,v), 
					    NIL)))),
		    value_undefined);
}

/* 
 * This function computes the current offset of the area a passed as
 * argument. The length of the variable v is also computed and then added
 * to a's offset. The initial offset is returned to the calling function.
 *
 * Note FI: this function is called too early because a DIMENSION or a Type
 * statement can modify both the basic type and the dimensions of variable v.
 *
 * I do not understand why the Static and Dynamic area sizes are not recorded
 * by a call to update_common_to_size(). Maybe because it is not necessary
 * because they are local to the current procedure and hence area_size can be
 * directly be used. But this is not consistent with other uses of the
 * common_size_map...
 */
int 
CurrentOffsetOfArea(a, v)
entity a, v;
{
    int OldOffset;
    type ta = entity_type(a);
    area aa = type_area(ta);

    if(top_level_entity_p(a)) {
	OldOffset = common_to_size(a);
	(void) update_common_to_size(a, OldOffset+SafeSizeOfArray(v));
    }
    else {
	/* the local areas are StaticArea and DynamicArea */
	OldOffset = area_size(aa);
	area_size(aa) = OldOffset+SafeSizeOfArray(v);
    }

    area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));

    return(OldOffset);
}

void 
update_common_sizes()
{
    list commons = NIL;

    HASH_MAP(k, v,{
	entity c = (entity) k;

	commons = arguments_add_entity(commons, c);
    },
	common_size_map);

    sort_list_of_entities(commons);

    MAP(ENTITY, c, {
	int s = common_to_size(c);
	type tc = entity_type(c);
	area ac = type_area(tc);

	pips_assert("update_common_sizes", s != (int) HASH_UNDEFINED_VALUE);

	if(area_size(ac) == 0) {
	    area_size(ac) = s;
	    debug(1, "update_common_sizes",
		       "set size %d for common %s\n", s, entity_name(c));
	}
	else if (area_size(ac) != s) {
	    /* I'm afraid this wwarning might be printed because area_size is given
	     * a wrong value by CurrentOffsetOfArea()
	     */
	    user_warning("update_common_sizes",
			 "inconsistent size (%d and %d) for common /%s/ in %s\n"
			 "Best results are obtained if all instances of a "
			 "COMMON are declared the same way.\n",
			 area_size(ac), s, module_local_name(c), 
			 CurrentPackage);
	    if(area_size(ac) < s)
		area_size(ac) = s;
	}
	else {
	    debug(1, "update_common_sizes",
		       "reset size %d for common %s\n", s, entity_name(c));
	}
    },
	     commons);
    /* Postpone the resetting because DynamicArea is updated till the EndOfProcedure() */
    /* reset_common_size_map(); */

    gen_free_list(commons);
}

/* local variables for implicit type implementation */
static tag tag_implicit[26];
static int int_implicit[26];

/* this function initializes the data structure used to compute implicit
types */

void 
InitImplicit()
{
    cr_implicit(is_basic_float, DefaultLengthOfBasic(is_basic_float), 'A','H');
    cr_implicit(is_basic_float, DefaultLengthOfBasic(is_basic_float), 'O','Z');
    cr_implicit(is_basic_int, DefaultLengthOfBasic(is_basic_int), 'I','N');
}

/* this function updates the data structure used to compute implicit
types. the implicit type for the range of letters defined by lettre_d
and lettre_f has tag t and length l. tag is_basic_string is temporarely
forbidden. */

void 
cr_implicit(t, l, lettre_d, lettre_f)
tag t;
int l;
int lettre_d, lettre_f;
{
    int i;

    /*
    if (t == is_basic_string)
	    ParserError("cr_implicit",
			"Unsupported implicit character declaration\n");
			*/

    if ((! IS_UPPER(lettre_d)) || (! IS_UPPER(lettre_f)))
	    FatalError("cr_implicit", "bad char\n");	

    for (i = lettre_d-'A'; i <= lettre_f-'A'; i += 1) {
	tag_implicit[i] = t;
	int_implicit[i] = l;
    }
}

/* this function computes the implicit type of entity e. the first
letter of e's name is used. */

type 
ImplicitType(e)
entity e;
{
    int i;
    string s = entity_local_name(e);
    type t = type_undefined;
    value v = value_undefined;

    if (s[0] == '_')
	    s++;

    if (!(IS_UPPER(s[0]))) {
	pips_error("ImplicitType", "[ImplicitType] bad name: %s\n", s);
	FatalError("ImplicitType", "\n");
    }

    i = (int) (s[0] - 'A');

    switch(tag_implicit[i]) {
    case is_basic_int:
    case is_basic_float:
    case is_basic_logical:
    case is_basic_complex:
	t = MakeTypeVariable(make_basic(tag_implicit[i], (void *) int_implicit[i]), NIL);
	break;
    case is_basic_string:
	v = make_value(is_value_constant,
		       make_constant(is_constant_int, (void *) int_implicit[i]));
	t = MakeTypeVariable(make_basic(tag_implicit[i], v), NIL);
	break;
    case is_basic_overloaded:
	FatalError("ImplicitType", "Unsupported overloaded tag for basic\n");
    default:
	FatalError("ImplicitType", "Illegal tag for basic\n");
    }
    /*
    return(MakeTypeVariable(make_basic(tag_implicit[i], int_implicit[i]), NIL));
    */
    return t;
}

/* This function checks that entity e has an undefined or an implicit type
 * which can be superseded by another declaration. The first
 * letter of e's name is used to determine the implicit type.
 * The implicit type of a functional entity is its result type.
 */

bool 
implicit_type_p(entity e)
{
    int i;
    string s = entity_local_name(e);
    type t = entity_type(e);
    basic b;

    if(t == type_undefined)
	return TRUE;

    if(type_functional_p(t))
	t = functional_result(type_functional(t));

    if (s[0] == '_')
	    s++;
    if (!(IS_UPPER(s[0]))) {
	pips_internal_error("bad name: %s\n", s);
	FatalError("implicit_type_p", "\n");
    }
    i = (int) (s[0] - 'A');

    /* ASSERT */
    if (!type_variable_p(t))
	pips_internal_error("expecting a variable for %s, got tag %d\n",
			    entity_name(e), type_tag(t));

    b = variable_basic(type_variable(t));

    if(basic_tag(b) != tag_implicit[i])
	return FALSE;

    switch(basic_tag(b)) {
	case is_basic_int: return basic_int(b)==int_implicit[i];
	case is_basic_float: return basic_float(b)==int_implicit[i];
	case is_basic_logical: return basic_logical(b)==int_implicit[i];
	case is_basic_complex: return basic_complex(b)==int_implicit[i];
	case is_basic_overloaded:
	    pips_internal_error("unexpected overloaded basic tag\n");
	case is_basic_string: 
	    return constant_int(value_constant(basic_string(b)))==
		int_implicit[i];
	default:
	    pips_internal_error("illegal basic tag\n");
	}
    return FALSE; /* to please gcc */
}

/* If an IMPLICIT statement is encountered, it must be applied to
 * the formal parameters, and, if the current module is a function,
 * to the function result type and to the variable used internally
 * when a value is assigned to the function (see MakeCurrentFunction)
 */
void 
retype_formal_parameters()
{
    entity m = get_current_module_entity();
    list vars = code_declarations(value_code(entity_initial(m)));
    type tm = entity_type(m);
    type tr = type_undefined;

    debug(8, "retype_formal_parameters", "Begin for module %s\n",
	  module_local_name(m));

    MAP(ENTITY, v, {
	if(!storage_undefined_p(entity_storage(v)) && formal_parameter_p(v)) {
	    if(!implicit_type_p(v)) {
		free_type(entity_type(v));
		entity_type(v) = ImplicitType(v);

		debug(8, "retype_formal_parameters", "Retype formal parameter %s\n",
		      entity_local_name(v));
	    }
	}
	else if(storage_undefined_p(entity_storage(v))
		|| (storage_ram_p(entity_storage(v)) && variable_entity_p(v))
		|| (storage_rom_p(entity_storage(v)) && entity_function_p(v))) 
	{
	    pips_debug(8, "Cannot retype entity %s: warning!!!\n",
		       entity_local_name(v));
		user_warning("retype_formal_parameters",
			     "Cannot retype variable or function %s."
			     " Move up the implicit statement at the beginning of declarations.\n",
			     entity_local_name(v));
	}
	else {
	    debug(8, "retype_formal_parameters", "Ignore entity %s\n",
		  entity_local_name(v));
	}
    }, vars);

    /* If the current module is a function, its type should be updated. */

    pips_assert("Should be a functional type", type_functional_p(tm));

    /* The function signature is computed later by  UpdateFunctionalType()
     * called from EndOfProcedure: there should be no parameters in the type.
     */
    pips_assert("Parameter type list should be empty",
                ENDP(functional_parameters(type_functional(tm))));

    tr = functional_result(type_functional(tm));
    if(type_variable_p(tr)) {
	if(!implicit_type_p(m)) {
	    entity r = entity_undefined;
	    free_type(tr);
	    functional_result(type_functional(tm)) = ImplicitType(m);
	    debug(8, "retype_formal_parameters", "Retype result of function %s\n",
		      module_local_name(m));

	    /* Update type of internal variable used to store the function result */
	    if((r=global_name_to_entity(module_local_name(m), module_local_name(m)))
	       != entity_undefined) {
		free_type(entity_type(r));
		entity_type(r) = ImplicitType(r);
		pips_assert("Result and function result types should be equal",
			    type_equal_p(functional_result(type_functional(tm)),
					 entity_type(r)));
	    }
	    else {
		pips_error("retype_formal_parameters", "Result entity should exist!\n");
	    }
	}
    }
    else if (type_void_p(tr)) {
	/* nothing to be done: subroutine or main */
    }
    else
	pips_error("retype_formal_parameters", "Unexpected type with tag = %d\n",
		   type_tag(tr));

    pips_assert("Parameter type list should still be empty",
                ENDP(functional_parameters(type_functional(tm))));

    debug(8, "retype_formal_parameters", "End for module %s\n",
	  module_local_name(m));
}

/* this function creates a type that represents a fortran type. its basic
is an int (the length of the fortran type) except in case of strings
where the type might be unknown, as in:

      CHARACTER*(*) PF

t is a tag, eg: INTEGER, REAL, ...

v is a value that represents the length in bytes of the type. */

type 
MakeFortranType(t, v)
tag t;
value v;
{
    basic b;
    int l;

    if (t == is_basic_string) {
	if (v == value_undefined) {
	    l = DefaultLengthOfBasic(t);
	    v = make_value(is_value_constant,
			   make_constant(is_constant_int, (void *) l));
	}
	b = make_basic(t, v);
    }
    else {
	l = (v == value_undefined) ? DefaultLengthOfBasic(t) : 
	constant_int(value_constant(v));
	b = make_basic(t, (void *) l);
    }

    return(MakeTypeVariable(b, NIL));
}

/* this function computes the offset of a variable element from the
begining of the variable. */

int 
OffsetOfReference(r)
reference r;
{
    cons *pi;
    int idim, iindex, pid, o, ilowerbound;

    pi = reference_indices(r);

    for (idim = 0, pid = 1, o = 0; pi != NULL; idim++, pi = CDR(pi)) {
	iindex = ExpressionToInt(EXPRESSION(CAR(pi)));
	ilowerbound = ValueOfIthLowerBound((reference_variable(r)), idim+1);
	/* Use a trick to retrieve the size in bytes of one array element
	 * and use the size of the previous dimension
	 */
	pid *= SizeOfIthDimension((reference_variable(r)), idim);
	o += ((iindex-ilowerbound)*pid);
    }

    return(o);	
}



/* this function returns the size of the ith lower bound of a variable e. */

int 
ValueOfIthLowerBound(e, i)
entity e;
int i;
{
    cons * pc;

    pips_assert("ValueOfIthLowerBound", type_variable_p(entity_type(e)));

    pips_assert("ValueOfIthLowerBound", i >= 1 && i <= 7);

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	    pc = CDR(pc);

    if (pc == NULL)
	ParserError("SizeOfIthLowerBound", "not enough dimensions\n");

    return(ExpressionToInt((dimension_lower(DIMENSION(CAR(pc))))));
}

/* This function computes the size of a range, ie. the number of
 * iterations that would be done by a loop with this range.
 *
 * See also range_count().
 */

int 
SizeOfRange(r)
range r;
{
    int ir, il, iu, ii;

    il = ExpressionToInt(range_lower(r));
    iu = ExpressionToInt(range_upper(r));
    ii = ExpressionToInt(range_increment(r));

    if (ii == 0)
	    FatalError("SizeOfRange", "null increment\n");

    ir = ((iu-il)/ii)+1;

    if (ir < 0)
	    FatalError("SizeOfRange", "negative value\n");

    return(ir);
}



/* FI: should be moved in ri-util;
 * this function returns TRUE if e is a zero dimension variable of basic
 * type integer
 */

int 
IsIntegerScalar(e) 
entity e; 
{ 
    if (type_variable_p(entity_type(e))) {
	variable a = type_variable(entity_type(e));

	if (variable_dimensions(a) == NIL && basic_int_p(variable_basic(a)))
		return(TRUE);
    }

    return(FALSE);
}

/* 
 * Check... and fix, if needed!
 *
 * Only user COMMONs are checked. The two implicit areas, DynamicArea and 
 * StaticArea, have not been initialized yet (see ComputeAddress() and the
 * calls in EndOfProcedure()).
 */

void 
check_common_layouts(m)
entity m;
{
    list decls = NIL;
    list sorted_decls = NIL;

    pips_assert("check_common_layouts", entity_module_p(m));

    decls = code_declarations(value_code(entity_initial(m)));
    sorted_decls = gen_append(decls, NIL);
    sort_list_of_entities(sorted_decls);

    ifdebug(1) {
	pips_debug(1, "\nDeclarations for module %s\n", module_local_name(m));

	/* List of implicitly and explicitly declared variables, 
	   functions and areas */

	pips_debug(1, "%s\n", ENDP(decls)? 
		   "* empty declaration list *\n\n": "Variable list:\n\n");

	MAP(ENTITY, e, 
	    fprintf(stderr, "Declared entity %s\n", entity_name(e)),
	    sorted_decls);

	/* Structure of each area/common */
	if(!ENDP(decls)) {
	    (void) fprintf(stderr, "\nLayouts for areas (commons):\n\n");
	}
    }

    MAP(ENTITY, e, {
	if(type_area_p(entity_type(e))) {
	    ifdebug(1) {
		print_common_layout(stderr, e);
	    }
	    if(update_common_layout(m, e)) {
		ifdebug(1) {
		    print_common_layout(stderr, e);
		}
	    }
	}
    }, sorted_decls);

    gen_free_list(sorted_decls);

    pips_debug(1, "End of declarations for module %s\n\n",
	       module_local_name(m));
}

void 
print_common_layout(FILE * fd, entity c)
{
    entity mod = get_current_module_entity();
    /* list members = area_layout(type_area(entity_type(c))); */
    /* list members = common_members_of_module(c, mod , TRUE); */
    /* for debugging only */
    list members = common_members_of_module(c, mod , FALSE);
    list equiv_members = NIL;

    (void) fprintf(fd,"\nLayout for common /%s/ of size %d:\n",
		   module_local_name(c), area_size(type_area(entity_type(c))));

    if(ENDP(members)) {
	pips_assert("An empty area has size 0", area_size(type_area(entity_type(c)))==0);
	(void) fprintf(fd, "\t* empty area *\n\n");
    }
    else {
	pips_assert("A non-empty area has size greater than 0",
		    area_size(type_area(entity_type(c)))>0 ||
		    common_to_size(c)>0);
	/* Look for variables declared in this common by *some* procedures
	 * which declares it. The procedures involved depend on the ordering
	 * of the parser steps by pipsmake and the user. Maybe, the list should
	 * be filtered and restricted to the current module: YES!
	 */
	MAP(ENTITY, m, 
	    {
		pips_assert("RAM storage",
			    storage_ram_p(entity_storage(m)));
		if(ram_function(storage_ram(entity_storage(m)))==mod)
		    (void) fprintf(fd,
				   "\tVariable %s,\toffset = %d,\tsize = %d\n", 
				   entity_name(m),
				   ram_offset(storage_ram(entity_storage(m))),
				   SafeSizeOfArray(m));
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
	    sort_list_of_entities(equiv_members);

	    (void) fprintf(fd, "\tVariables aliased to this common:\n");

	    MAP(ENTITY, m, 
		{
		    pips_assert("RAM storage",
				storage_ram_p(entity_storage(m)));
		    (void) fprintf(fd,
				   "\tVariable %s,\toffset = %d,\tsize = %d\n", 
				   entity_name(m),
				   ram_offset(storage_ram(entity_storage(m))),
				   SafeSizeOfArray(m));
		}, 
		    equiv_members);
	    (void) fprintf(fd, "\n");
	    gen_free_list(equiv_members);
	}
    }
    gen_free_list(members);
}

/* (Re)compute offests of all variables allocated in common c from module m
 * and update (if necessary) the size of common c for the *whole* program or
 * set of modules in the current workspace. As a consequence, warning messages
 * unfortunately depend on the parsing order.
 *
 * Offsets are computed a first time when the common declaration is
 * encountered, but the variables may be typed or dimensionned *later*.
 */

bool 
update_common_layout(m, c)
entity m;
entity c;
{
    /* It is assumed that:
     *  - each variable appears only once
     *  - variables appears in their declaration order
     *  - all variables that belong to the same module appear contiguously
     *    (i.e. declarations are concatenated on a module basis)
     *  - variables wich are located in the common thru an EQUIVALENCE statement
     *    are *not* (yet) in its layout
     * It also was wrongly assumed that each common would have at least two members.
     */

    list members = area_layout(type_area(entity_type(c)));
    entity previous = entity_undefined;
    bool updated = FALSE;

    /* the layout field does not seem to be filled in for STATIC and DYNAMIC */
    if(!ENDP(members)) {
	/* skip variables which do not belong to the module of interest */
	/*
	for(previous = ENTITY(CAR(members)); !ENDP(members) && !variable_in_module_p(previous, m);
	    POP(members))
	    previous = ENTITY(CAR(members));
	    */
	do {
	    previous = ENTITY(CAR(members));
	    POP(members);
	} while(!ENDP(members) && !variable_in_module_p(previous, m));

	MAPL(cm, {
	    entity current = ENTITY(CAR(cm));
        
	    pips_assert("update_common_layout",
			storage_ram_p(entity_storage(current)));

	    if(!variable_in_module_p(current, m)) {
		/* If c really is a common, check its size because it may have increased.
		 * Note that decreases are not taken into account although they might
		 * occur as well.
		 */
		/* This piece of code might be useful if we were not dealing with the
		 * last parsed module.
		if(top_level_entity_p(c)) {
		    int s = common_to_size(c);
		    int new_s = ram_offset(storage_ram(entity_storage(previous)))
			+SafeSizeOfArray(previous);
		    if(s < new_s) {
			(void) update_common_to_size(c, new_s);
		    }
		}
		updated = TRUE;
		*/
		break;
	    }

	    if(ram_offset(storage_ram(entity_storage(previous)))+SafeSizeOfArray(previous) >
	       ram_offset(storage_ram(entity_storage(current)))) {
		ifdebug(1) {
		    user_warning("update_common_layout", 
				 "entity %s must have been typed after it was allocated in common /%s/\n",
				 entity_local_name(previous), module_local_name(c));
		}
		ram_offset(storage_ram(entity_storage(current))) =
		    ram_offset(storage_ram(entity_storage(previous)))+SafeSizeOfArray(previous);

		/* If c really is a common, check its size because it may have increased.
		 * Note that decreases are not taken into account although they might
		 * occur as well.
		 */
		/* Too late, if the common only contains one element because the MAPL
		 * has not been entered at all if we are dealing wih te last parsed
		 * module... which is always the case up to now!
		 */
		if(top_level_entity_p(c)) {
		    int s = common_to_size(c);
		    int new_s = ram_offset(storage_ram(entity_storage(current)))
			+SafeSizeOfArray(current);
		    if(s < new_s) {
			(void) update_common_to_size(c, new_s);
		    }
		}
		updated = TRUE;
	    }

	    previous = current;
	} , members);


	/* Special case: only one element in the common for the current procedure
	 * (and the current procedure is last one declared - which is not so
	 * special)
	 */
	if(ENDP(members)) {
	  pips_assert("Previous must in declared in the current module",
		      variable_in_module_p(previous, m));
	  /* If c really is a common, check its size because it may have increased.
	   * Note that decreases are not taken into account although they might
	   * occur as well.
	   */
	  if(top_level_entity_p(c)) {
	    int s = common_to_size(c);
	    int new_s = ram_offset(storage_ram(entity_storage(previous)))
	      +SafeSizeOfArray(previous);
	    if(s < new_s) {
	      (void) update_common_to_size(c, new_s);
	      updated = TRUE;
	    }
	  }
	}
    }

    return updated;
}

void
fprint_functional(FILE * fd, functional f)
{
    type tr = functional_result(f);

    MAPL(cp, {
	parameter p = PARAMETER(CAR(cp));
	type ta = parameter_type(p);

	pips_assert("Argument type is variable or varags:variable", type_variable_p(ta)
		    || (type_varargs_p(ta) && type_variable_p(type_varargs(ta))));
	if(type_varargs_p(ta)) {
	    (void) fprintf(fd, " %s:", type_to_string(ta));
	    ta = type_varargs(ta);
	}
	(void) fprintf(fd, "%s", basic_to_string(variable_basic(type_variable(ta))));
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
    else if(type_varargs_p(tr)) {
	(void) fprintf(fd, " %s:%s", type_to_string(tr),
		       basic_to_string(variable_basic(type_variable(type_varargs(tr)))));
    }
    else
	pips_error("fprint_functional", "Ill. type %d\n", type_tag(tr));
}

void 
fprint_environment(FILE * fd, entity m)
{
    list decls = NIL;
    int nth = 0; /* rank of formal parameter */
    entity rv = entity_undefined; /* return variable */

    pips_assert("fprint_environment", entity_module_p(m));

    decls = code_declarations(value_code(entity_initial(m)));

    (void) fprintf(fd, "\nDeclarations for module %s with type ", 
		   module_local_name(m));
    fprint_functional(fd, type_functional(entity_type(m)));
    (void) fprintf(fd, "\n\n");

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
	    (void) fprintf(fd, "with size %d\n", area_size(type_area(t)));
	}
	else
	    (void) fprintf(fd, "\n");
	    },
	decls);

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
			   "\tVariable %s,\toffset = %d\n", 
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
	if(type_area_p(entity_type(e))) {
	    print_common_layout(fd, e);
	}
    }, 
	decls);

    (void) fprintf(fd, "End of declarations for module %s\n\n",
		   module_local_name(m));
}
