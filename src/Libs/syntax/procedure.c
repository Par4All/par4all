/*
 * $Id$
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "string.h"

#include "genC.h"
#include "parser_private.h"
#include "ri.h"
#include "database.h"
#include "resources.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "syntax.h"

#include "syn_yacc.h"

/* list of called subroutines or functions */
static list called_modules = list_undefined;

/* statement of current function */
static statement function_body = statement_undefined;

/*********************************************************** GHOST VARIABLES */

/* list of potential local or top-level variables that turned out to be useless.
 */
static list ghost_variable_entities = list_undefined;

void
init_ghost_variable_entities()
{
    pips_assert("undefined list", list_undefined_p(ghost_variable_entities));
    ghost_variable_entities = NIL;
}

void 
remove_ghost_variable_entities()
{
    pips_assert("defined list", !list_undefined_p(ghost_variable_entities));
    MAP(ENTITY, e, 
    {
	/* The debugging message must use the variable name before it is freed
	 */
	pips_debug(1, "entity '%s'\n", entity_name(e));
	if(entity_in_equivalence_chains_p(e)) {
	    user_warning("remove_ghost_variable_entities",
		     "Entity \"%s\" does not really exist but appears"
		     " in an equivalence chain!\n",
		     entity_name(e));
	    if(!ParserError("remove_ghost_variable_entities",
			    "Cannot remove still accessible ghost variable\n")) {
		/* We already are in ParserError()! Too bad for the memory leak */
		ghost_variable_entities = list_undefined;
		return;
	    }
	}
	else {
	    remove_variable_entity(e);
	}
	pips_debug(1, "destroyed\n");
    }, 
	ghost_variable_entities);

    ghost_variable_entities = list_undefined;
}

void
add_ghost_variable_entity(entity e)
{
    pips_assert("defined list",	!list_undefined_p(ghost_variable_entities));
    ghost_variable_entities = arguments_add_entity(ghost_variable_entities, e);
}

/* It is possible to change one's mind and effectively use an entity which was
 * previously assumed useless
 */
void
reify_ghost_variable_entity(entity e)
{
    pips_assert("defined list",	!list_undefined_p(ghost_variable_entities));
    if(entity_is_argument_p(e, ghost_variable_entities))
	ghost_variable_entities = arguments_rm_entity(ghost_variable_entities, e);
}


/* this function is called each time a new procedure is encountered. */
void 
BeginingOfProcedure()
{
    reset_current_module_entity();
    InitImplicit();
    called_modules = NIL;
}

void 
update_called_modules(e)
entity e;
{
    bool already_here = FALSE;
    string n = entity_local_name(e);
    string nom;
    entity cm = get_current_module_entity();

    /* Self recursive calls are not allowed */
    if(e==cm) {
	pips_user_warning("Recursive call from %s to %s\n",
		     entity_local_name(cm), entity_local_name(e));
	ParserError("update_called_modules", 
		    "Recursive call are not supported\n");
    }

    /* do not count intrinsics; user function should not be named
       like intrinsics */
    nom = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, n, NULL);
    if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined) {
	if(entity_initial(e) == value_undefined) {
	    /* FI, 20/01/92: maybe, initializations of global entities
	       should be more precise (storage, initial value, etc...);
	       for the time being, I choose to ignore the potential
	       problems with other executions of the parser and the linker */
	    /* pips_error("update_called_modules","unexpected case\n"); */
	}
	else if(value_intrinsic_p(entity_initial(e)))
	    return;
    }

    MAPL(ps, {
	if (strcmp(n, STRING(CAR(ps))) == 0) {
	    already_here = TRUE;
	    break;
	}
    }, called_modules);

    if (! already_here) {
	pips_debug(1, "adding %s\n", n);
	called_modules = CONS(STRING, strdup(n), called_modules);
    }
}

/* macros are added, although they should not have been.
 */
void
remove_from_called_modules(entity e)
{
    bool found = FALSE;
    list l = called_modules;
    string name = module_local_name(e);

    if (!called_modules) return;

    if (same_string_p(name, STRING(CAR(called_modules)))) {
	called_modules = CDR(called_modules);
	found = TRUE;
    } else {
	list lp = called_modules;
	l = CDR(called_modules);
	
	for(; !ENDP(l); POP(l), POP(lp)) {
	    if (same_string_p(name, STRING(CAR(l)))) {
		CDR(lp) = CDR(l);
		found = TRUE;
		break;
	    }
	}
    }    

    if (found) {
	pips_debug(3, "removing %s from callees\n", entity_name(e));
	CDR(l) = NIL;
	free(STRING(CAR(l)));
	gen_free_list(l);
    }
}

void 
AbortOfProcedure()
{
    /* get rid of ghost variable entities */
    if (!list_undefined_p(ghost_variable_entities))
	remove_ghost_variable_entities();

    (void) ResetBlockStack() ;
}

/* This function is called when the parsing of a procedure is completed.
 * It performs a few calculations which cannot be done on the fly such
 * as address computations.
 *
 * And it writes the internal representation of the CurrentFunction with a
 * call to gen_free (?). */

void 
EndOfProcedure()
{
    entity CurrentFunction = get_current_module_entity();

    debug(8, "EndOfProcedure", "Begin for module %s\n",
	  entity_name(CurrentFunction));

    /* get rid of ghost variable entities */
    remove_ghost_variable_entities();

    /* we generate the last statement to carry a label or a comment */
    if (strlen(lab_I) != 0 /* || iPrevComm != 0 */ ) {
	LinkInstToCurrentBlock(make_continue_instruction(), FALSE);
    }

    /* we generate statement last+1 to eliminate returns */
    GenerateReturn();

    uses_alternate_return(FALSE);
    ResetReturnCodeVariable();
    SubstituteAlternateReturns("NO");

    /* Check the block stack */
    (void) PopBlock() ;
    if (!IsBlockStackEmpty())
	    ParserError("EndOfProcedure", "bad program structure\n");

    /* are there undefined gotos ? */
    CheckAndInitializeStmt();

    /* The following calls could be located in check_first_statement()
     * which is called when the first executable statement is
     * encountered. At that point, many declaration related
     * problems should be fixed or fixeable. But additional
     * undeclared variables will be added to the dynamic area
     * and their addresses must be computed. At least, ComputeAddresses()
     * must stay here.. so I keep all these calls together.
     */
    UpdateFunctionalType(FormalParameters);

    check_common_layouts(CurrentFunction);

    ComputeEquivalences();
    /* Use equivalence chains to update storages of equivalenced and
       implicitly declared variables */
    ComputeAddresses();

    /* Initialize the shared field in ram storage */
    SaveChains();

    /* Update offsets in commons (and static and dynamic areas?)
     * according to latest type and dimension declarations
     */
    /* check_common_layouts(CurrentFunction); */

    /* Now that retyping and equivalences have been taken into account: */
    update_common_sizes();

    reset_common_size_map();

    code_declarations(EntityCode(CurrentFunction)) =
	    gen_nreverse(code_declarations(EntityCode(CurrentFunction))) ;

    if (get_bool_property("PARSER_DUMP_SYMBOL_TABLE"))
	fprint_environment(stderr, CurrentFunction);

    ifdebug(5){
	fprintf(stderr, "Parser: checking callees consistency = %d\n",
		gen_consistent_p( make_callees( called_modules ))) ;
    }

    /*  remove hpfc special routines if required.
     */
    if (get_bool_property("HPFC_FILTER_CALLEES"))
    {
	list l = NIL;
	string s;

	MAPL(cs,
	 {
	     s = STRING(CAR(cs));

	     if (hpf_directive_string_p(s) && !keep_directive_in_code_p(s))
	     {
		 pips_debug(3, "ignoring %s\n", s);
	     }
	     else
		 l = CONS(STRING, s, l);
	 },
	     called_modules);

	gen_free_list(called_modules);
	called_modules = l;
    }

    /* done here. affects callees and code. FC.
     */
    parser_substitute_all_macros(function_body);
    parser_close_macros_support();

    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, 
			   module_local_name(CurrentFunction), 
			   (char*) make_callees(called_modules));
    
    ifdebug(5) {
	fprintf(stderr, "Parser: checking code consistency = %d\n",
		gen_consistent_p( function_body )) ;
    }

    DB_PUT_MEMORY_RESOURCE(DBR_PARSED_CODE, 
			   module_local_name(CurrentFunction), 
			   (char *)function_body);

    /* the current package is re-initialized */
    CurrentPackage = TOP_LEVEL_MODULE_NAME;
    ResetChains();
    DynamicArea = entity_undefined;
    StaticArea = entity_undefined;
    reset_current_module_entity();

    pips_debug(8, "End for module %s\n", entity_name(CurrentFunction));
}



/* This function analyzes the CurrentFunction formal parameter list to
 * determine the CurrentFunction functional type. l is this list.
 *
 * It is called by EndOfProcedure().
 */

void 
UpdateFunctionalType(l)
cons *l;
{
    cons *pc;
    parameter p;
    functional ft;
    entity CurrentFunction = get_current_module_entity();
    type t = entity_type(CurrentFunction);

    debug(8, "UpdateFunctionalType", "Begin for %s\n",
	  module_local_name(CurrentFunction));

    pips_assert("A module type should be functional", type_functional_p(t));

    ft = type_functional(t);

    /* FI: I do not understand this assert... at least now that
     * functions are typed at call sites. I do not understand why this
     * assert has not made more damage. Only OVL in APSI (Spec-cfp95)
     * generates a core dump. To be studied more!
     *
     * This assert is guaranteed by MakeCurrentFunction() but not by 
     * retype_formal_parameters() which is called in case an intrinsic
     * statement is encountered. It is not guaranteed by MakeExternalFunction()
     * which uses the actual parameter list to estimate a functional type
     */
    pips_assert("Parameter type list should be empty",
		ENDP(functional_parameters(ft)));

    for (pc = l; pc != NULL; pc = CDR(pc)) {
	p = make_parameter((entity_type(ENTITY(CAR(pc)))), 
			   (MakeModeReference()));
	functional_parameters(ft) = 
		gen_nconc(functional_parameters(ft),
			  CONS(PARAMETER, p, NIL));
    }

    debug(8, "UpdateFunctionalType", "End for %s\n",
	  module_local_name(CurrentFunction));
}

/* this function creates one entity cf that represents the function f being
analyzed. if f is a Fortran FUNCTION, a second entity is created; this
entity represents the variable that is used in the function body to
return a value.  both entities share the same name and the type of the
result entity is equal to the type of cf's result.

t is the type of the function result if it has been given by the
programmer as in INTEGER FUNCTION F(A,B,C)

msf indicates if f is a main, a subroutine or a function.

cf is the current function

lfp is the list of formal parameters
*/
void 
MakeCurrentFunction(
    type t,
    int msf,
    string cfn,
    list lfp)
{
    entity cf = entity_undefined; /* current function */
    instruction icf; /* the body of the current function */
    entity result; /* the second entity, used to store the function result */
    /* to split the entity name space between mains, commons, blockdatas and regular modules */
    string prefix = string_undefined;
    string fcfn = string_undefined; /* full current function name */
    entity ce = entity_undefined; /* global entity with conflicting name */

    /* Check that there is no such common: This test is obsolete because
     * the standard does not prohibit the use of the same name for a
     * common and a function. However, it is not a good programming practice
     */
    if (gen_find_tabulated(concatenate
	   (TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING,
	    COMMON_PREFIX, cfn, 0), 
			   entity_domain) != entity_undefined)
    {
	pips_user_warning("global name %s used for a module and for a common\n",
			  cfn);
	/*
	ParserError("MakeCurrentFunction",
		    "Name conflict between a "
		    "subroutine and/or a function and/or a common\n");
		    */
    }

    if(msf==TK_PROGRAM) {
	prefix = MAIN_PREFIX;
    }
    else if(msf==TK_BLOCKDATA) {
	prefix = BLOCKDATA_PREFIX;
    }
    else  {
	prefix = "";
    }
    fcfn = strdup(concatenate(prefix, cfn, NULL));
    cf = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, fcfn);
    free(fcfn);
    ce = global_name_to_entity(TOP_LEVEL_MODULE_NAME, cfn);
    if(!entity_undefined_p(ce) && ce!=cf) {
	user_warning("MakeCurrentFunction", "Global name %s used for a function or subroutine"
		     " and for a %s\n", cfn, msf==TK_BLOCKDATA? "blockdata" : "main");
	ParserError("MakeCurrentFunction", "Name conflict\n");
    }

    /* Let's hope cf is not an intrinsic */
    if( entity_type(cf) != type_undefined
       && intrinsic_entity_p(cf) ) {
	user_warning("MakeCurrentFunction",
		     "Intrinsic %s redefined.\n"
		     "This is not supported by PIPS. Please rename %s\n",
		     entity_local_name(cf), entity_local_name(cf));
	/* Unfortunately, an intrinsics cannot be redefined, just like a user function
	 * or subroutine after editing because intrinsics are not handled like
	 * user functions or subroutines. They are not added to the called_modules
	 * list of other modules, unless the redefining module is parsed FIRST.
	 * There is not mechanism in PIPS to control the parsing order.
	 */
	ParserError("MakeCurrentFunction",
		    "Name conflict between a "
		    "subroutine and/or a function and an intrinsic\n");
    }

    /* set ghost variable entities to NIL */
    init_ghost_variable_entities();

    if (msf == TK_FUNCTION) {
	if (t == type_undefined) {
	    t = ImplicitType(cf);
	}
    }
    else {
	if (t == type_undefined) {
	    t = make_type(is_type_void, UU);
	}
	else {
	    /* the intended result type t for a main or a subroutine should be undefined */
	    FatalError("MakeCurrentFunction", "bad type\n");
	}
    }

    /* clean up existing local entities in case of a recompilation */
    CleanLocalEntities(cf);

    /* the parameters part of cf's functional type is not created
       because the type of formal parameters is not known. this is done by
       UpdateFunctionalType. */
    entity_type(cf) = make_type(is_type_functional, make_functional(NIL, t));

    /* a function has a rom storage */
    entity_storage(cf) = MakeStorageRom();

    /* a function has an initial value 'code' that contains an empty block */
    icf = MakeEmptyInstructionBlock();

    /* FI: This NULL string is a catastrophy for the strcmp used later
     * to check the content of the stack. Any string, including
     * the empty string "", would be better. icf is used to link new
     * instructions/statement to the current block. Only the first
     * block is not pushed for syntactic reasons. The later blocks
     * will be pushed for DO's and IF's.
     */
    /* PushBlock(icf, (string) NULL); */
    PushBlock(icf, "INITIAL");

    function_body = instruction_to_statement(icf);
    entity_initial(cf) = make_value(is_value_code, make_code(NIL, NULL));

    set_current_module_entity(cf);

    /* two global areas are created */
    InitAreas();

    /* No common has yet been declared */
    initialize_common_size_map();

    /* Formal parameters are created. Alternate returns can be ignored
     * or substituted.
     */
    SubstituteAlternateReturns
	(get_string_property("PARSER_SUBSTITUTE_ALTERNATE_RETURNS"));
    ScanFormalParameters(add_formal_return_code(lfp));

    if (msf == TK_FUNCTION) {
	/* a result entity is created */
	/*result = FindOrCreateEntity(CurrentPackage, entity_local_name(cf));*/
	result = make_entity(strdup(concatenate(CurrentPackage, 
						MODULE_SEP_STRING, 
						module_local_name(cf), 
						NULL)), 
			     type_undefined, 
			     storage_undefined, 
			     value_undefined);
	DeclareVariable(result, t, NIL, make_storage(is_storage_return, cf),
			value_undefined);
	AddEntityToDeclarations(result, cf);
    }
}

/* 
 * This function creates an external function. It may happen in
 * Fortran that a function is declared as if it were a variable; example:
 *
 * INTEGER*4 F
 * ...
 * I = F(9)
 *
 * or:
 *
 * SUBROUTINE FOO(F)
 * ...
 * CALL F(9)
 *
 * in these cases, the initial declaration must be updated, 
 * ie. the variable declaration must be
 * deleted and replaced by a function declaration. 
 *
 * See DeclareVariable for other combination based on EXTERNAL
 *
 * Modifications:
 *  - to perform link edition at parse time, returns a new entity when
 *    e is not a TOP-LEVEL entity; this changes the function a lot;
 *    Francois Irigoin, 9 March 1992;
 *  - introduction of fe and tfe to clean up the relationship between e
 *    and the new TOP-LEVEL entity; formal functional parameters were
 *    no more recognized as a bug because of the previous modification;
 *    Francois Irigoin, 11 July 1992;
 *  - remove_variable_entity() added to avoid problems in semantics analysis
 *    with an inexisting variable, FI, June 1993;
 */

entity 
MakeExternalFunction(
    entity e, /* entity to be turned an external function */
    type r /* type of result */)
{
    type te;
    entity fe = entity_undefined;
    type tfe;

    debug(8, "MakeExternalFunction", "Begin for %s\n", entity_name(e));

    te = entity_type(e);
    if (te != type_undefined) {
	if (type_variable_p(te)) {
	    /* e is a function that was implicitly declared as a variable. 
	       this may happen in Fortran. */
	    pips_debug(2, "variable --> fonction\n");
	    pips_assert("undefined type", r == type_undefined);
	    r = te;
	}
    }

    pips_debug(9, "external function %s declared\n", entity_name(e));

    if(!top_level_entity_p(e)) {
	storage s = entity_storage(e);
	if(s == storage_undefined || storage_ram_p(s)) {
	    extern list arguments_add_entity(list a, entity e);

	    fe = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, 
				    entity_local_name(e));

	    pips_debug(1, "external function %s re-declared as %s\n",
		       entity_name(e), entity_name(fe));
	    /* FI: I need to destroy a virtual entity which does not
	     * appear in the program and wich was temporarily created by
	     * the parser when it recognized a name; however, I've no way
	     * to know if the same entity does not appear for good
	     * somewhere else in the code; does the Fortran standard let
	     * you write: LOG = LOG(3.)  If yes, PIPS will core dump...
	     * PIPS also core dumps with ALOG(ALOG(X))... (8 July 1993) 
	     */
	    /* remove_variable_entity(e); */
	    add_ghost_variable_entity(e);
	    pips_debug(1, "entity %s to be destroyed\n", entity_name(e));

	    if(r!=type_undefined) {
		/* r is going to be re-used to build the functional type
		 * which is going to be freed later with e
		 */
		type new_r = type_undefined;
		new_r = gen_copy_tree(r);
		r = new_r;
	    }
	}
	else if(storage_formal_p(s)){
	    pips_user_warning("entity %s is a formal functional parameter\n",
			      entity_name(e));
	    ParserError("MakeExternalFunction",
			"Formal functional parameters are not supported "
			"by PIPS.\n");
	    fe = e;
	}
	else {
	    pips_internal_error("entity %s has an unexpected storage %d\n",
				entity_name(e), storage_tag(s));
	}
    }
    else {
	fe = e;
    }

    /* Assertion: fe is a (functional) global entity and the type of its 
       result is r */

    tfe = entity_type(fe);
    if(tfe == type_undefined) {
	/* this is wrong, because we do not know if we are handling
	   an EXTERNAL declaration, in which case the result type
	   is type_undefined, or a function call appearing somewhere,
	   in which case the ImplicitType should be used;
	   maybe the unknown type should be used? */
	entity_type(fe) = make_type(is_type_functional, 
				   make_functional(NIL, 
						   (r == type_undefined) ?
						   ImplicitType(fe) :
						   r));
    }
    else if (type_functional_p(tfe)) 
    {
	type tr = functional_result(type_functional(tfe));
	if(r != type_undefined && !type_equal_p(tr, r)) {

	    /* a bug is detected here: MakeExternalFunction, as its name
	       implies, always makes a FUNCTION, even when the symbol
	       appears in an EXTERNAL statement; the result type is
	       infered from ImplicitType() - see just above -;
	       let's use implicit_type_p() again, whereas the unknown type
	       should have been used 
	    */
	    if(intrinsic_entity_p(fe)) {
		/* ignore r */
	    } else if (type_void_p(tr)) {
		/* someone used a subroutine as a function.
		 * this happens in hpfc for declaring "pure" routines.
		 * thus I make this case being ignored. warning? FC.
		 */		
	    } else if (implicit_type_p(fe) || overloaded_type_p(tr)) {
		/* memory leak of tr */
		functional_result(type_functional(tfe)) = r;
	    } else  {
		user_warning("MakeExternalFunction",
			     "Type redefinition of result for function %s\n", 
			     entity_name(fe));
		if(type_variable_p(tr)) {
		    user_warning("MakeExternalFunction",
				 "Currently declared result is %s\n", 
				 basic_to_string(variable_basic(type_variable(tr))));
		}
		if(type_variable_p(r)) {
		    user_warning("MakeExternalFunction",
				 "Redeclared result is %s\n", 
				 basic_to_string(variable_basic(type_variable(r))));
		}
		ParserError("MakeExternalFunction",
			    "Functional type redefinition.\n");
	    }
	}
    } else if (type_variable_p(tfe)) {
	pips_internal_error("Fortran does not support global variables\n");
    } else {
	pips_internal_error("Unexpected type for a global name %s\n",
			    entity_name(fe));
    }

    /* a function has a rom storage, except for formal functions */
    if (entity_storage(e) == storage_undefined)
	entity_storage(fe) = MakeStorageRom();
    else
	if (! storage_formal_p(entity_storage(e)))
	    entity_storage(fe) = MakeStorageRom();
	else {
	    pips_user_warning("unsupported formal function %s\n", 
			 entity_name(fe));
	    ParserError("MakeExternalFunction",
			"Formal functions are not supported by PIPS.\n");
	}

    /* an external function has an unknown initial value, else code would be temporarily
     * undefined which is avoided (theoretically forbidden) in PIPS.
     */
    if(entity_initial(fe) == value_undefined)
	entity_initial(fe) = MakeValueUnknown();

    /* fe is added to CurrentFunction's entities */
    AddEntityToDeclarations(fe, get_current_module_entity());

    debug(8, "MakeExternalFunction", "End for %s\n", entity_name(fe));

    return fe;
}

/* This function transforms an untyped entity into a formal parameter. 
 * fp is an entity generated by FindOrCreateEntity() for instance,
 * and nfp is its rank in the formal parameter list.
 *
 * A specific type is used for the return code variable which may be
 * adde by the parser to handle alternate returns. See return.c
 */

void 
MakeFormalParameter(entity fp, int nfp)
{
    pips_assert("type is undefined", entity_type(fp) == type_undefined);

    if(SubstituteAlternateReturnsP() && ReturnCodeVariableP(fp)) {
	entity_type(fp) = MakeTypeVariable(make_basic(is_basic_int, 4), NIL);
    }
    else {
	entity_type(fp) = ImplicitType(fp);
    }

    entity_storage(fp) = 
	make_storage(is_storage_formal, 
		     make_formal(get_current_module_entity(), nfp));
    entity_initial(fp) = MakeValueUnknown();
}



/* this function scans the formal parameter list. each formal parameter
is created with an implicit type, and then is added to CurrentFunction's
declarations. */
void 
ScanFormalParameters(list l)
{
	list pc;
	entity fp; /* le parametre formel */
	int nfp; /* son rang dans la liste */

	FormalParameters = l;

	for (pc = l, nfp = 1; pc != NULL; pc = CDR(pc), nfp += 1) {
		fp = ENTITY(CAR(pc));

		MakeFormalParameter(fp, nfp);

		AddEntityToDeclarations(fp, get_current_module_entity());
	}
}



/* this function creates an intrinsic function. */

entity 
CreateIntrinsic(string name)
{
    entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
    pips_assert("entity is defined", e!=entity_undefined);
    return(e);
}
