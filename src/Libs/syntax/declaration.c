/* 	%A% ($Date: 1997/04/26 11:16:42 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_syntax_declaration[] = "%A% ($Date: 1997/04/26 11:16:42 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */


 /* Functions to handle all declarations, but some related to functional
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "misc.h"

#include "syntax.h"

#define IS_UPPER(c) (isascii(c) && isupper(c))

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

/* this function transforms a dynamic variable into a static one.  */

void 
SaveEntity(e)
entity e;
{
    if (entity_type(e) == type_undefined) {
	DeclareVariable(e, type_undefined, NIL, 
			storage_undefined, value_undefined);
    }

    if (entity_storage(e) != storage_undefined) {
	if (storage_ram_p(entity_storage(e))) {
	    ram r;

	    r = storage_ram(entity_storage(e));

	    if (ram_section(r) != DynamicArea)
		    FatalError("SaveEntity", "cannot save non dynamic variables\n");

	    ram_section(r) = StaticArea;
	    ram_offset(r) = CurrentOffsetOfArea(StaticArea, e);
	}
	else {
	    ParserError("SaveEntity", "cannot save non dynamic variables\n");
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
AnalyzeData(ldvr, ldvl)
cons *ldvr, *ldvl;
{
    cons *pcr, *pcl;
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

	debug(8, "AnalyzeData", "needs %d elements for entity %s\n", 
	      i, entity_name(e));

	pips_assert("AnalyzeData", dataval_nboccurrences(dvl) > 0);

	if (IsIntegerScalar(e)) {
	    pips_assert("AnalyzeData", i == 1);
	    /* pips_assert("AnalyzeData", constant_int_p(dataval_constant(dvl))); */
	    if(constant_int_p(dataval_constant(dvl))) {
	      entity_initial(e) = make_value(is_value_constant, 
					     dataval_constant(dvl));

	      debug(1, "AnalyzeData", "%s %d\n", 
		    entity_name(e), constant_int(dataval_constant(dvl)));
	    }
	    else {
	      Warning("AnalyzeData", 
		      "Integer scalar variable initialized with non-integer constant");
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
DeclareVariable(e, t, d, s, v)
entity e;
type t;
cons *d;
storage s;
value v;
{
    type et = entity_type(e);
    list etd = list_undefined;

    debug(8, "DeclareVariable", "%s\n", entity_name(e));
    pips_assert("DeclareVariable", t == type_undefined || type_variable_p(t));

    if(et == type_undefined)
	if(t == type_undefined) {
	    entity_type(e) = ImplicitType(e);
	    variable_dimensions(type_variable(entity_type(e))) = d;
	}
	else {
	    type nt;
	    nt = MakeTypeVariable
		(gen_copy_tree(variable_basic(type_variable(t))),
		 d);
	    entity_type(e) = nt;
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
			(gen_copy_tree(variable_basic(type_variable(t))),
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
			(gen_copy_tree(variable_basic(type_variable(t))),
			 variable_dimensions(type_variable(et)));
		    if(!type_equal_p(entity_type(e),nt)) {
			if(/*FI: to check update_common_layout*/ FALSE && 
			   entity_storage(e)!=storage_undefined &&
			   storage_ram_p(entity_storage(e)) &&
			   basic_type_size(variable_basic(type_variable(t)))
			   > basic_type_size(variable_basic(type_variable(entity_type(e))))) {
			    user_warning("DeclareVariable",
					 "Storage information for %s is likely to be wrong because its type is redefined as a larger type\nType is *not* redefined internally to avoid aliasing\n", entity_local_name(e));
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
}

/* MakeCommon:
 * This function creates a common block. pips creates static common
 * blocks. this is not true in the ANSI standard.
 */
entity 
MakeCommon(e)
entity e;
{
    if (entity_type(e) == type_undefined) {
	/* Is the common name conflicting with the program name? */
	entity module =
	    gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
					   MODULE_SEP_STRING,
					   MAIN_PREFIX,
					   entity_local_name(e),
					   (char *) NULL),
			       entity_domain);
	if(module == entity_undefined) {
	    entity_type(e) = make_type(is_type_area, make_area(0, NIL));
	    entity_storage(e) = 
		make_storage(is_storage_ram, 
			     (make_ram(get_current_module_entity(),
				       StaticArea, 0, NIL)));
	    entity_initial(e) = MakeValueUnknown();
	    AddEntityToDeclarations(e, get_current_module_entity());
	}
	else {
	    user_warning("MakeCommon", "Conflicting usage of %s\n",
			 entity_local_name(e));
	    ParserError("MakeCommon",
			"Conflicting name between main and common\n");
	}
    }
    else if(!type_area_p(entity_type(e))) {
	/* FI: user_warning is used to display the conflicting name */
	user_warning("MakeCommon",
		     "name conflict for %s between common "
		     "and entity (tag=%d)\n",
		     entity_name(e), type_tag(entity_type(e)));
	ParserError("MakeCommon",
		    "name conflict between common and variable or module\n");
    }
    else {
	/* common e may already exist because it was encountered
	 * in another module
	 * but not have been registered as known by the current module
	 */
	AddEntityToDeclarations(e, get_current_module_entity());
    }

    if(hash_get(common_size_map, (char *) e) == HASH_UNDEFINED_VALUE)
	hash_put(common_size_map, (char *) e, (char *) 0);

    return(e);
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
 */
int 
CurrentOffsetOfArea(a, v)
entity a, v;
{
    int OldOffset;
    type ta = entity_type(a);
    area aa = type_area(ta);

    if(top_level_entity_p(a)) 
	if((OldOffset = (int) hash_get(common_size_map,(char *) a))
	   == (int) HASH_UNDEFINED_VALUE)
	    pips_error("CurrentOffsetOfArea",
		       "common_size_map uninitialized for common %s\n",
		       entity_name(a));
	else
	    /* too bad this will generate a warn on redefinition... */
	    (void) hash_update(common_size_map, (char *) a,
			(char *) (OldOffset+SizeOfArray(v)));
    else {
	/* the local areas are StaticArea and DynamicArea */
	OldOffset = area_size(aa);
	area_size(aa) = OldOffset+SizeOfArray(v);
    }

    area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));

    return(OldOffset);
}

void 
update_common_sizes()
{
    HASH_MAP(k, v,{
	entity c = (entity) k;
	int s = (int) v;
	type tc = entity_type(c);
	area ac = type_area(tc);

	pips_assert("update_common_sizes", s != (int) HASH_UNDEFINED_VALUE);

	if(area_size(ac) == 0) {
	    area_size(ac) = s;
	    debug(1, "update_common_sizes",
		       "set size %d for common %s\n", s, entity_name(c));
	}
	else if (area_size(ac) != s)
/*	    user_warning("update_common_sizes",
		       "inconsistent size (%d and %d) for common %s in %s\n",
		       area_size(ac), s, entity_name(c), 
		       CurrentPackage);
		       ParserError("update_common_sizes", "
*/
	    user_warning("update_common_sizes",
			 "inconsistent size (%d and %d) for common %s in %s\n"
			 "Best results are obtained if all instances of a "
			 "COMMON are declared the same way.",
			 area_size(ac), s, entity_name(c), 
			 CurrentPackage);
	else {
	    debug(1, "update_common_sizes",
		       "reset size %d for common %s\n", s, entity_name(c));
	}
    },
	     common_size_map);
    reset_common_size_map();
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
	t = MakeTypeVariable(make_basic(tag_implicit[i], int_implicit[i]), NIL);
	break;
    case is_basic_string:
	v = make_value(is_value_constant,
		       make_constant(is_constant_int, int_implicit[i]));
	t = MakeTypeVariable(make_basic(tag_implicit[i], v), NIL);
	break;
    case is_basic_overloaded:
    default:
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
implicit_type_p(e)
entity e;
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
	pips_error("implicit_type_p", "[implicit_type_p] bad name: %s\n", s);
	FatalError("implicit_type_p", "\n");
    }
    i = (int) (s[0] - 'A');

    b = variable_basic(type_variable(t));

    if(basic_tag(b) != tag_implicit[i])
	return FALSE;

    switch(basic_tag(b)) {
	case is_basic_int: return basic_int(b)==int_implicit[i];
	case is_basic_float: return basic_float(b)==int_implicit[i];
	case is_basic_logical: return basic_logical(b)==int_implicit[i];
	case is_basic_complex: return basic_complex(b)==int_implicit[i];
	case is_basic_overloaded:
	    pips_error("implicit_type_p", 
		       "[implicit_type_p] unexpected overloaded basic tag\n");
	case is_basic_string: 
	    return constant_int(value_constant(basic_string(b)))==
		int_implicit[i];
	default:
	    pips_error("implicit_type_p", 
		       "[implicit_type_p] illegal basic tag\n");
	}
    return FALSE; /* to please gcc */
}

void 
retype_formal_parameters()
{
    entity m = get_current_module_entity();
    list vars = code_declarations(value_code(entity_initial(m)));

    MAP(ENTITY, v, {
	if(!storage_undefined_p(entity_storage(v)) && formal_parameter_p(v)) {
	    if(!implicit_type_p(v)) {
		free_type(entity_type(v));
		entity_type(v) = ImplicitType(v);
	    }
	}
    }, vars);
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
			   make_constant(is_constant_int, l));
	}
	b = make_basic(t, v);
    }
    else {
	l = (v == value_undefined) ? DefaultLengthOfBasic(t) : 
	constant_int(value_constant(v));
	b = make_basic(t, l);
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
	ilowerbound = ValueOfIthLowerBound((reference_variable(r)), idim);
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

    pc = variable_dimensions(type_variable(entity_type(e)));

    while (pc != NULL && --i > 0)
	    pc = CDR(pc);

    if (pc == NULL)
	ParserError("SizeOfIthLowerBound", "not enough dimensions\n");

    return(ExpressionToInt((dimension_lower(DIMENSION(CAR(pc))))));
}

/* this function computes the size of a range, ie. the number of
iterations that would be done by a loop with this range. */

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

void 
check_common_layouts(m)
entity m;
{
    list decls = NIL;

    pips_assert("check_common_layouts", entity_module_p(m));

    decls = code_declarations(value_code(entity_initial(m)));

    ifdebug(1) {
	pips_debug(1, "\nDeclarations for module %s\n", module_local_name(m));

    /* List of implictly and explicitly declared variables, 
       functions and areas */

	pips_debug(1, "%s\n", ENDP(decls)? 
		   "* empty declaration list *\n\n": "Variable list:\n\n");

	MAP(ENTITY, e, 
	    fprintf(stderr, "Declared entity %s\n", entity_name(e)),
	    decls);

    /* Structure of each area/common */
	if(!ENDP(decls)) {
	    (void) fprintf(stderr, "\nLayouts for areas (commons):\n\n");
	}
	MAP(ENTITY, e, {
	    if(type_area_p(entity_type(e))) {
		ifdebug(1) {
		    print_common_layout(e);
		}
		if(update_common_layout(m, e)) {
		    ifdebug(1) {
		    print_common_layout(e);
		    }
		}
	    }
	}, decls);

	(void) fprintf(stderr, "End of declarations for module %s\n\n",
		       module_local_name(m));
    }
}

void 
print_common_layout(c)
entity c;
{
    list members = area_layout(type_area(entity_type(c)));

    (void) fprintf(stderr,"\nLayout for common %s:\n", entity_name(c));

    if(ENDP(members)) {
	(void) fprintf(stderr, "* empty area *\n\n");
    }
    else {
	MAP(ENTITY, m, 
	     {
		 pips_assert("RAM storage",
			     storage_ram_p(entity_storage(m)));
		 (void) fprintf(stderr,
				"\tVariable %s, offset = %d, size = %d\n", 
				entity_name(m),
				ram_offset(storage_ram(entity_storage(m))),
				SizeOfArray(m));
	     }, 
		 members);
	(void) fprintf(stderr, "\n");
    }
}

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

	    if(!variable_in_module_p(current, m))
		break;

	    if(ram_offset(storage_ram(entity_storage(previous)))+SizeOfArray(previous) >
	       ram_offset(storage_ram(entity_storage(current)))) {
		ifdebug(1) {
		    user_warning("update_common_layout", 
				 "entity %s must have been typed after it was allocated in common %s\n",
				 entity_local_name(previous), entity_local_name(c));
		}
		ram_offset(storage_ram(entity_storage(current))) =
		    ram_offset(storage_ram(entity_storage(previous)))+SizeOfArray(previous);
		updated = TRUE;
	    }

	    previous = current;
	} , members);

    }

    return updated;
}
