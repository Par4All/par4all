 /* PIPS project: syntactic analyzer
  *
  * Remi Triolet
  *
  * Warning: do not use user_error() when processing must be interrupted,
  * but ParserError() which fixes global variables and leaves a consistent
  * symbol table. Use user_warning() to display additional information if
  * ParserError() is too terse.
  *
  * Bugs:
  *  - IO control info list should be checked; undected errors are likely
  *    to induce core dumps in effects computation; Francois Irigoin;
  *  - Type declarations are not enforced thoroughly: the syntax for characters
  *    is extended to other types; for instance REAL*4 X*40 is syntactically
  *    accepted but "*40" is ignored; Francois Irigoin;
  *
  * Modifications:
  *  - bug correction for REWIND, BACKSPACE and ENDFILE; improved error
  *    detection in IO statement; Francois Irigoin;
  *  - add DOUBLE PRECISION as a type; Francois Irigoin;
  *  - update length declaration computation for CHARACTER type; the second
  *    length declaration was ignored; for instance:
  *        CHARACTER*3 X*56, Y
  *    was interpreted as the declaration of two strings of 3 characters;
  *    two variables added: CurrentType and CurrentTypeSize
  *    Francois Irigoin
  *  - bug with EXTERNAL: CurrentType was not reset to type_undefined
  *  - Complex constants were not recognized; the rule for expression
  *    were modified and a new rule, sous-expression, was added, as well
  *    as a rule for complex constants; Francois Irigoin, 9 January 1992
  *  - global variables should be allocated whenever possible in a different
  *    name space than local variables; their package name should be the
  *    top level name; that would let us accept variables and commons
  *    with the same name and that would make the link edit step easier;
  *    lots of things have to be changed:
  *
  *     global_entity_name: should produce a global entity
  *
  *     common_name: should allocate the blank common in the global space
  *
  *     call_inst: has to refer to a global entity
  *
  *     module_name: has to be allocated in the global space
  *
  *     intrinsic_inst: has to require global_entity_name(s) as parameters
  *
  *     external inst: has to require global_entity_name(s) as parameters
  *
  *    This implies that COMMON are globals to all procedure. The SAVE
  *    statement on COMMONs is meaningless
  *  - add common_size_table to handle COMMONs as global variables;
  *    see declarations.c (Francois Irigoin, 22 January 1992)
  *  - remove complex constant detection because this conflicts with
  *    IO statements
  */

%type <chain>	        latom
%type <dimension>	dim_tableau
%type <entity>		icon
%type <entity>		entity_name
%type <entity>		global_entity_name
%type <entity>		functional_entity_name
/* %type <entity>		module_name */
%type <string>		module_name
%type <entity>		common_name
%type <entity>		declaration
%type <entity>		common_inst
%type <entity>		oper_rela
%type <entity>		unsigned_const_simple
%type <expression>	const_simple
%type <expression>	sous_expression
%type <expression>	expression
%type <expression>	io_elem
%type <expression>	io_f_u_id
%type <expression>	opt_expression
%type <instruction>	inst_exec
%type <instruction>	format_inst
%type <instruction>	data_inst
%type <instruction>	assignment_inst
%type <instruction>	goto_inst
%type <instruction>	arithmif_inst
%type <instruction>	logicalif_inst
%type <instruction>	blockif_inst
%type <instruction>	elseif_inst
%type <instruction>	else_inst
%type <instruction>	endif_inst
%type <instruction>	enddo_inst
%type <instruction>	do_inst
%type <instruction>	bdo_inst
%type <instruction>	wdo_inst
%type <instruction>	continue_inst
%type <instruction>	stop_inst
%type <instruction>	pause_inst
%type <instruction>	call_inst
%type <instruction>	return_inst
%type <instruction>	io_inst
%type <integer>		io_keyword
%type <integer>		ival
%type <integer>		opt_signe
%type <integer>		psf_keyword
%type <integer>		iobuf_keyword
%type <integer>		signe
%type <liste>		decl_tableau
%type <liste>		indices 
%type <liste>		parameters
%type <liste>		arguments
%type <liste>		lci
%type <liste>		ci
%type <liste>		ldim_tableau
%type <liste>		ldataval
%type <liste>		ldatavar
%type <liste>		lexpression
%type <liste>		lformalparameter
%type <liste>		licon
%type <liste>		lio_elem
%type <liste>		opt_lformalparameter
%type <liste>		opt_lio_elem
%type <range>           do_plage
%type <string>		label
%type <string>		name
%type <string>		global_name
%type <syntax>          atom
%type <tag>		fortran_basic_type
%type <type> 		fortran_type
%type <type> 		opt_fortran_type
%type <value>		lg_fortran_type
%type <character>       letter
%type <expression>      dataconst
%type <dataval>         dataval
%type <datavar>         datavar
%type <datavar>         dataidl

%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "syntax.h"

extern int yylex(void);
extern void yyerror(char *);

#define YYERROR_VERBOSE 1 /* much clearer error messages with bison */

    /* local variables */
    int ici; /* to count control specifications in IO statements */
    type CurrentType = type_undefined; /* the type in a type or dimension
					  or common statement */
    int CurrentTypeSize; /* number of bytes to store a value of that type */

/* functions for DATA */

/* this function creates a 'dataval'. it takes two expressions as
arguments:

c is a constant.

n is an iteration count, i.e. a positive integer constant. */

static dataval MakeDataVal(expression n, expression c)
{
    int in;
    constant cc;
    value vc;

    in = (n == expression_undefined) ? 1 : ExpressionToInt(n);

    vc = EvalExpression(c);
    if (! value_constant_p(vc)) {
	ParserError("MakeDataVal", "data value must be a constant\n");
    }
    cc = value_constant(vc);
    value_constant(vc) = constant_undefined;
    free_value(vc);

    return make_dataval(cc, in);
}



/* this function creates a 'datavar'. it takes two arguments: an syntax
s that represents the variable being initialized, and a range r that
indicates which elements of s are initalized. the range might be
undefined, in which case the whole variable s is initialized. remember that
a scalar is represented by a zero dimension variable in our internal
representation.  */ 

static datavar MakeDataVar(syntax s, range r)
{
    entity e; 
    reference ref;
    datavar d;

    if (! syntax_reference_p(s))
	    FatalError("MakeDataVar", "bad variable\n");

    ref = syntax_reference(s);
    e = reference_variable(ref);

    if (r == range_undefined) {
	if(reference_indices(ref)==NIL) {
	    int ne = 0;

	    if(!NumberOfElements(variable_dimensions(type_variable(entity_type(e))),
				 &ne)) {
		user_warning("MakeDataVar", "Varying size of array \"%s\"\n", entity_name(e));
		ParserError("MakeDataVar", "Fortran standard prohibit varying size array\n");
	    }

	    d = make_datavar(e, ne);
	}
	else
	    d = make_datavar(e, 1);
    }
    else {
	d = make_datavar(e, SizeOfRange(r));
    }		

    return(d);
}

/* this function is called when a data implied do has more than one
dimension, i.e. the number of nested do loops is greater than one. the
number of elements initialized by the outer loop is calculated by
MakeDataVar, and elements initialized by inner loops are calculated by
ExpandDataVar. 

dvr is the datavar created by a call to MakeDataVar (and possibly
modified by one or more call to ExpandDataVar).

r is the range of the inner loop.  */

static datavar ExpandDataVar(dvr, r)
datavar dvr;
range r;
{
    datavar_nbelements(dvr) *= SizeOfRange(r);

    return(dvr);
}

%}

/* Specify precedences and associativies. */
%left TK_COMMA
%nonassoc TK_COLON
%right TK_EQUALS
%left TK_EQV TK_NEQV
%left TK_OR
%left TK_AND
%left TK_NOT
%nonassoc TK_LT TK_GT TK_LE TK_GE TK_EQ TK_NE
%left TK_CONCAT
%left TK_PLUS TK_MINUS
%left TK_STAR TK_SLASH
%right TK_POWER

%union {
	basic basic;
	chain chain;
        char character;
	cons * liste;
	dataval dataval;
	datavar datavar;
	dimension dimension;
	entity entity;
	expression expression;
	instruction instruction;
	int integer;
	range range;
	string string;
	syntax syntax;
	tag tag;
	type type; 
	value value;
}

%%

lprg_exec: prg_exec
	| lprg_exec prg_exec
            { FatalError("parser", 
			 "Multiple modules in one file! Check fsplit!");}
	;

prg_exec: begin_inst {reset_first_statement();} linstruction { check_first_statement();} end_inst
        | begin_inst {reset_first_statement(); check_first_statement();} end_inst
	;

begin_inst: opt_fortran_type psf_keyword module_name
	       opt_lformalparameter TK_EOS
	    { 
                 MakeCurrentFunction($1, $2, $3, $4);
            }
	;

end_inst: TK_END TK_EOS
            { EndOfProcedure(); }
	;

linstruction: TK_EOS 
        | instruction TK_EOS
	| linstruction instruction TK_EOS
	;

instruction: 
	    data_inst 
	{ /* can appear anywhere in specs and execs! */ 
	    if (first_executable_statement_seen())
	    {
		/* the DATA string to be added to declarations...
		 * however, the information is not really available.
		 * as a hack, I'll try to append the Stmt buffer, but it
		 * has already been processed, thus it is quite far
		 * from the initial statement, and may be incorrect.
		 * I think that this parser is a mess;-) FC.
		 */
		pips_user_warning(
		    "DATA as an executable statement, moved up...\n");
		append_data_current_stmt_buffer_to_declarations();
	    }
	}
        | { check_in_declarations();} inst_spec
	| { check_first_statement();} inst_exec
	    { 
		if ($2 != instruction_undefined)
			LinkInstToCurrentBlock($2, TRUE);
	    }
	|  format_inst
	    { 
		LinkInstToCurrentBlock($1, TRUE);
	    }
	;

inst_spec: parameter_inst
	| implicit_inst
	| dimension_inst
	| equivalence_inst
	| common_inst {}
	| type_inst
	| external_inst
	| intrinsic_inst
	| save_inst
	;

inst_exec: assignment_inst
	    { $$ = $1; }
	| goto_inst
	    { $$ = $1; }
	| arithmif_inst
	    { $$ = $1; }
	| logicalif_inst
	    { $$ = $1; }
	| blockif_inst
	    { $$ = $1; }
	| elseif_inst
	    { $$ = $1; }
	| else_inst
	    { $$ = instruction_undefined; }
	| endif_inst
	    { $$ = instruction_undefined; }
	| enddo_inst
	    { $$ = instruction_undefined; }
	| do_inst
	    { $$ = instruction_undefined; }
	| bdo_inst
	    { $$ = instruction_undefined; }
	| wdo_inst
	    { $$ = instruction_undefined; }
	| continue_inst
	    { $$ = $1; }
	| stop_inst
	    { $$ = $1; }
	| pause_inst
	    { $$ = $1; }
	| call_inst
	    { $$ = $1; }
	| return_inst
	    { $$ = $1; }
	| io_inst
	    { $$ = $1; }
	;

return_inst: TK_RETURN opt_expression
	    { $$ = MakeReturn($2); }
	;

call_inst: tk_call functional_entity_name
	    { $$ = MakeCallInst($2, NIL); reset_alternate_returns();}
        |
	  tk_call functional_entity_name parameters
	    { $$ = MakeCallInst($2, $3); reset_alternate_returns(); }
	;

tk_call: TK_CALL
            { set_alternate_returns();}
        ;

parameters: TK_LPAR TK_RPAR
            { $$ = NULL; }
	| TK_LPAR arguments TK_RPAR
		{ $$ = $2; }
	;

arguments: expression
	    {
		$$ = CONS(EXPRESSION, $1, NIL);
	    }
	| arguments TK_COMMA expression
	    {
		$$ = gen_nconc($1, CONS(EXPRESSION, $3, NIL));
	    }
	| TK_STAR TK_ICON
	    {
		add_alternate_return($2);
		$$ = NIL;
	    }
	| arguments TK_COMMA TK_STAR TK_ICON
	    {
		add_alternate_return($4);
		$$ = $1;
	    }
	;


io_inst:  io_keyword io_f_u_id
	    { 
		expression std, format, unite;
		cons * lci;

		switch($1) {
		case TK_WRITE:
		    FatalError("Syntax","Illegal use of WRITE");
		case TK_READ:
		case TK_PRINT:
		    std = MakeNullaryCall(CreateIntrinsic
					  (LIST_DIRECTED_FORMAT_NAME));
		    unite = MakeCharacterConstantExpression("UNIT=");
		    format = MakeCharacterConstantExpression("FMT=");

		    lci = CONS(EXPRESSION, unite,
			       CONS(EXPRESSION, std,
				    CONS(EXPRESSION, format,
					 CONS(EXPRESSION, $2, NULL))));
		    /* Functionally PRINT is a special case of WRITE */
		    $$ = MakeIoInstA(($1==TK_PRINT)?TK_WRITE:TK_READ,
				     lci, NIL);
		    break;
		case TK_OPEN:
		case TK_CLOSE:
		case TK_INQUIRE:
		    ParserError("Syntax",
				"Illegal syntax in IO statement, "
				"Parentheses and arguments required");
		case TK_BACKSPACE:
		case TK_REWIND:
		case TK_ENDFILE:
		    unite = MakeCharacterConstantExpression("UNIT=");
		    lci = CONS(EXPRESSION, unite,
			       CONS(EXPRESSION, $2, NULL));
		    $$ = MakeIoInstA($1, lci, NIL);
		    break;
		default:
		    ParserError("Syntax","Unexpected token in IO statement");
		}
	    }
        | io_keyword io_f_u_id TK_COMMA opt_lio_elem
            {
		expression std, format, unite;
		cons * lci;

		switch($1) {
		case TK_WRITE:
		    FatalError("Syntax","Illegal use of WRITE");
		case TK_READ:
		case TK_PRINT:
		    std = MakeNullaryCall(CreateIntrinsic
					  (LIST_DIRECTED_FORMAT_NAME));
		    unite = MakeCharacterConstantExpression("UNIT=");
		    format = MakeCharacterConstantExpression("FMT=");

		    lci = CONS(EXPRESSION, unite,
			       CONS(EXPRESSION, std,
				    CONS(EXPRESSION, format,
					 CONS(EXPRESSION, $2, NULL))));
		    $$ = MakeIoInstA(($1==TK_PRINT)?TK_WRITE:TK_READ,
				     lci, $4);
		    break;
		case TK_OPEN:
		case TK_CLOSE:
		case TK_INQUIRE:
		case TK_BACKSPACE:
		case TK_REWIND:
		case TK_ENDFILE:
		  ParserError("Syntax",
		  "Illegal syntax in IO statement, Parentheses are required");
		default:
		    ParserError("Syntax","Unexpected token in IO statement");
		}
	    }
    
	| io_keyword TK_LPAR lci TK_RPAR opt_virgule opt_lio_elem
	    { $$ = MakeIoInstA($1, $3, $6); }
        | iobuf_keyword TK_LPAR io_f_u_id TK_COMMA io_f_u_id TK_RPAR 
                        TK_LPAR expression TK_COMMA expression TK_RPAR
	    { $$ = MakeIoInstB($1, $3, $5, $8, $10); }
	;

io_f_u_id: atom 
        /* Should be an expression, but a conflict results for parentheses 
	 * which may be
	 * part of the expression or part of the IO statement (FI, 1/1/97)
	 */
	    { $$ = make_expression($1, normalized_undefined); }
        | const_simple
	    { $$ = $1; }
	| TK_STAR
	    { $$ = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME)); }
	;

lci: ci
	    { $$ = $1; }
	| lci TK_COMMA ci
	    { 
	      /*
		CDR(CDR($3)) = $1;
		$$ = $3;
		*/
	      $$ = gen_nconc($1, $3);
	    }
	;

/* ci: name TK_EQUALS io_f_u_id */
ci: name TK_EQUALS expression
	    {
		char buffer[20];
		(void) strcpy(buffer, $1);
		free($1);

		if(strcmp(buffer,"END")==0||strcmp(buffer,"ERR")==0) {
		  ;
		}

		(void) strcat(buffer, "=");
		
		$$ = CONS(EXPRESSION, 
			  MakeCharacterConstantExpression(buffer),
			  CONS(EXPRESSION, $3, NULL));
		ici += 1;
	    }
        | name TK_EQUALS TK_STAR
            {
		char buffer[20];
		(void) strcpy(buffer, $1);
		free($1);

		if(strcmp(buffer,"UNIT")!=0 && strcmp(buffer,"FMT")!=0) {
		  ParserError("parser", 
			  "Illegal default option '*' in IO control list\n");
		}

		(void) strcat(buffer, "=");
		
		$$ = CONS(EXPRESSION, 
			  MakeCharacterConstantExpression(buffer),
			  CONS(EXPRESSION,
		 MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME))
			       , NULL));
		ici += 1;
	    }
        | io_f_u_id
	    {
		$$ = CONS(EXPRESSION,
			  MakeCharacterConstantExpression(ici == 1 ? 
			                                  "UNIT=" : 
			                                  "FMT="),
			  CONS(EXPRESSION, $1, NULL));
		ici += 1;
	    }

	;

opt_lio_elem:
	    { $$ = NULL; }
	| lio_elem
	    { $$ = MakeIoList($1); }
        ;

lio_elem:  io_elem
	    { $$ = CONS(EXPRESSION, $1, NULL); }
	| lio_elem TK_COMMA io_elem
	    { $$ = CONS(EXPRESSION, $3, $1); }
	;

io_elem: expression
	    { $$ = $1; }
	| TK_LPAR lio_elem TK_COMMA atom TK_EQUALS do_plage TK_RPAR
	    { $$ = MakeImpliedDo($4, $6, $2); }	;

pause_inst: TK_PAUSE opt_expression
	    { $$ = MakeZeroOrOneArgCallInst("PAUSE", $2); }
	;

stop_inst: TK_STOP opt_expression
	    { $$ = MakeZeroOrOneArgCallInst("STOP", $2); }
	;

continue_inst: TK_CONTINUE
	    { $$ = MakeZeroOrOneArgCallInst("CONTINUE", expression_undefined);}
	;

do_inst: TK_DO label opt_virgule atom TK_EQUALS do_plage
	    { 
		MakeDoInst($4, $6, $2); 
		$$ = instruction_undefined;
	    }
	;

bdo_inst: TK_DO atom TK_EQUALS do_plage
	    { 
		MakeDoInst($2, $4, "BLOCKDO"); 
		$$ = instruction_undefined;
	    }
	;

wdo_inst: TK_DO TK_WHILE TK_LPAR expression TK_RPAR
	    { 
		MakeWhileDoInst($4, "BLOCKDO");
		$$ = instruction_undefined;
	    }
        | TK_DO label TK_WHILE TK_LPAR expression TK_RPAR
	    { 
		MakeWhileDoInst($5, $2);
		$$ = instruction_undefined;
	    }
	;

do_plage: expression TK_COMMA expression
	    { $$ = make_range($1, $3, MakeIntegerConstantExpression("1")); }
	| expression TK_COMMA expression TK_COMMA expression
	    { $$ = make_range($1, $3, $5); }
	;

endif_inst: TK_ENDIF
	    { MakeEndifInst(); }
	;

enddo_inst: TK_ENDDO
	    { MakeEnddoInst(); }
	;

else_inst: TK_ELSE
	    { MakeElseInst(); }
	;

elseif_inst: TK_ELSEIF TK_LPAR expression TK_RPAR TK_THEN
	    {
		int elsifs = MakeElseInst();

		MakeBlockIfInst( $3, elsifs+1 );
		$$ = instruction_undefined;
	    }
	;

blockif_inst: TK_IF TK_LPAR expression TK_RPAR TK_THEN
	    {
		MakeBlockIfInst($3,0);
		$$ = instruction_undefined;
	    }
	;

logicalif_inst: TK_IF TK_LPAR expression TK_RPAR inst_exec
	    {
		$$ = MakeLogicalIfInst($3, $5); 
	    }
	;

arithmif_inst: TK_IF TK_LPAR expression TK_RPAR 
			label TK_COMMA label TK_COMMA label 
	    {
		$$ = MakeArithmIfInst($3, $5, $7, $9);
	    }
	;

goto_inst: TK_GOTO label
	    {
		$$ = MakeGotoInst($2);
	    }
	| TK_GOTO TK_LPAR licon TK_RPAR opt_virgule expression
	    {
		$$ = MakeComputedGotoInst($3, $6);
	    }
	| TK_GOTO entity_name opt_virgule TK_LPAR licon TK_RPAR
	    {
                if(get_bool_property("PARSER_SUBSTITUTE_ASSIGNED_GOTO")) {
		    $$ = MakeAssignedGotoInst($5, $2);
		}
		else {
		    ParserError("parser", "assigned goto statement prohibited"
				" unless property PARSER_SUBSTITUTE_ASSIGNED_GOTO is set\n");
		}
	    }
	| TK_GOTO entity_name
	    {
                if(get_bool_property("PARSER_SUBSTITUTE_ASSIGNED_GOTO")) {
		    ParserError("parser", "assigned goto statement cannot be"
				" desugared without a target list\n");
		}
		else {
		    ParserError("parser", "assigned goto statement prohibited\n");
		}
	    }
	;

licon: label
            {
               $$ = CONS(STRING, $1, NIL);
            }
	| licon TK_COMMA label
            {
               $$ = CONS(STRING, $3, $1);
            }
	;

assignment_inst: TK_ASSIGN icon TK_TO atom
            {
                if(get_bool_property("PARSER_SUBSTITUTE_ASSIGNED_GOTO")) {
		    expression e = entity_to_expression($2);
		    $$ = MakeAssignInst($4, e);
		}
		else {
		    ParserError("parser", "ASSIGN statement prohibited by PIPS"
				" unless property PARSER_SUBSTITUTE_ASSIGNED_GOTO is set\n");
		}
	    }
	| atom TK_EQUALS expression
	    {
		$$ = MakeAssignInst($1, $3);
	    }
	;

format_inst: TK_FORMAT
	    {
		set_first_format_statement();
		$$ = MakeZeroOrOneArgCallInst("FORMAT",
			MakeCharacterConstantExpression(FormatValue));
	    }
	;

save_inst: TK_SAVE
            { save_all_entities(); }
        | TK_SAVE lsavename
	;

lsavename: savename
	| lsavename TK_COMMA savename
	;

savename: entity_name
	    { ProcessSave($1); }
	| common_name
	    { SaveCommon($1); }
	;

intrinsic_inst: TK_INTRINSIC global_entity_name
	    {
		(void) CreateIntrinsic(entity_name($2));
	    }
	| intrinsic_inst TK_COMMA global_entity_name
	    {
		(void) CreateIntrinsic(entity_name($3));
	    }
	;

external_inst: TK_EXTERNAL functional_entity_name
	    {
		CurrentType = type_undefined;
		(void) MakeExternalFunction($2, type_undefined);
	    }
	| external_inst TK_COMMA functional_entity_name
	    {
		(void) MakeExternalFunction($3, type_undefined);
	    }
	;

type_inst: fortran_type declaration
	{}
	| type_inst TK_COMMA declaration
	;

declaration: entity_name decl_tableau lg_fortran_type
	    {
		/* the size returned by lg_fortran_type should be
		   consistent with CurrentType unless it is of type string
		   or undefined */
		type t = CurrentType;

		if(t != type_undefined) {
		    basic b;

		    if(!type_variable_p(CurrentType))
			FatalError("yyparse", "ill. type for CurrentType\n");

		    b = variable_basic(type_variable(CurrentType));

		    /* character [*len1] foo [*len2]:
		     * if len2 is "default" then len1
		     */
		    if(basic_string_p(b))
			t = value_intrinsic_p($3)? /* ??? default */
			    copy_type(t):
				MakeTypeVariable
				    (make_basic(is_basic_string, $3), NIL);

		    DeclareVariable($1, t, $2,
			storage_undefined, value_undefined);

		    if(basic_string_p(b))
			free_type(t);
		}
		else
		    DeclareVariable($1, t, $2,
				    storage_undefined, value_undefined);

		$$ = $1;
	    }
	;

decl_tableau:
	    {
		    $$ = NULL;
	    }
	| TK_LPAR ldim_tableau TK_RPAR
	    {
		    $$ = $2;
	    }
	;

ldim_tableau: dim_tableau
	    {
		    $$ = CONS(DIMENSION, $1, NULL);		    
	    }
	| dim_tableau TK_COMMA ldim_tableau
	    {
		    $$ = CONS(DIMENSION, $1, $3);
	    }
	;

dim_tableau: expression
	    {
		    $$ = make_dimension(MakeIntegerConstantExpression("1"),$1);
	    }
	| TK_STAR
	    {
		$$ = make_dimension(MakeIntegerConstantExpression("1"),
		  MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)));
	    }
	| expression TK_COLON TK_STAR
	    {
		    $$ = make_dimension($1, 
		    MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)));
	    }
	| expression TK_COLON expression
	    {
		    $$ = make_dimension($1, $3);
	    }
	;

common_inst: common declaration
	    { 
		$$ = NameToCommon(BLANK_COMMON_LOCAL_NAME);
	    }
	| common common_name declaration
	    {
		$$ = $2;
		AddVariableToCommon($$, $3);
	    }
	| common_inst TK_COMMA declaration
	    {
		$$ = $1;
		AddVariableToCommon($$, $3);
	    }
	| common_inst opt_virgule common_name declaration
	    {
		$$ = $3;
		AddVariableToCommon($$, $4);
	    }
	;


common: TK_COMMON
	    {
		CurrentType = type_undefined;
	    }

common_name: TK_CONCAT
	    {
		$$ = NameToCommon(BLANK_COMMON_LOCAL_NAME);
	    }
	| TK_SLASH global_name TK_SLASH
	    {
		$$ = NameToCommon($2);
	    }
	;

equivalence_inst: TK_EQUIVALENCE lequivchain
	;

lequivchain: equivchain
	| lequivchain TK_COMMA equivchain
	;

equivchain: TK_LPAR latom TK_RPAR
            { StoreEquivChain($2); }
	;

latom: atom
	    {
		$$ = make_chain(CONS(ATOM, MakeEquivAtom($1), (cons*) NULL));
	    }
	| latom TK_COMMA atom
	    { 
		chain_atoms($1) = CONS(ATOM, MakeEquivAtom($3), 
				     chain_atoms($1));
		$$ = $1;
	    }
	;

dimension_inst: dimension declaration
	    {
	    }
	| dimension_inst TK_COMMA declaration
	    {
	    }
	;

dimension: TK_DIMENSION
	    {
		CurrentType = type_undefined;
	    }
	;

data_inst: TK_DATA ldatavar TK_SLASH ldataval TK_SLASH
	    {
		AnalyzeData($2, $4);
	    }
	| data_inst opt_virgule ldatavar TK_SLASH ldataval TK_SLASH
	    {
		AnalyzeData($3, $5);
	    }
	;

ldatavar: datavar
	    {
		$$ = CONS(DATAVAR, $1, NIL);
	    }
	| datavar TK_COMMA ldatavar
	    {
		$$ = CONS(DATAVAR, $1, $3);
	    }
	;

/* rule reversal because of a stack overflow; bug hit.f */
ldataval: dataval
	    {
		$$ = CONS(DATAVAL, $1, NIL);
	    }
	| ldataval TK_COMMA dataval
	    {
		$$ = gen_nconc($1, CONS(DATAVAL, $3, NIL));
	    }
	;

dataval: dataconst
	    {
		$$ = MakeDataVal(expression_undefined, $1);
	    }
	| dataconst TK_STAR dataconst
	    {
		$$ = MakeDataVal($1, $3);
	    }
	;

dataconst: const_simple
	    {
		$$ = $1;
	    }
	| entity_name
	    {
		/* Cachan bug 4: there should be a check about the entity
		 * returned as $1 because MakeDatVal() is going to try
		 * to evaluate that expression. The entity must be a
		 * parameter.
		 */
		if(symbolic_constant_entity_p($1)) {
		    $$ = make_expression(make_syntax(is_syntax_call,
						     make_call($1, NIL)), 
					 normalized_undefined);
		}
		else {
		    user_warning("gram", "Symbolic constant expected: %s\n",
			       entity_local_name($1));
		    if(strcmp("Z", entity_local_name($1))==0) {
			user_warning("gram",
		       "Might be a non supported hexadecimal constant\n");
		    }
		    ParserError("gram", "Error in initializer");
		}
	    }
	| entity_name TK_LPAR const_simple TK_COMMA const_simple TK_RPAR
	{
	    bool simple = ENTITY_IMPLIED_CMPLX_P($1);
	    pips_assert("is implied complex", 
			simple || ENTITY_IMPLIED_DCMPLX_P($1) );
	    $$ = MakeBinaryCall(CreateIntrinsic
	    (simple? IMPLIED_COMPLEX_NAME: IMPLIED_DCOMPLEX_NAME), $3, $5);
	}
	;

datavar: atom
	    {
		$$ = MakeDataVar($1, range_undefined);
	    }
	| dataidl
	    { $$ = $1; }
	;

dataidl: TK_LPAR atom TK_COMMA entity_name TK_EQUALS do_plage TK_RPAR
	    {
		$$ = MakeDataVar($2, $6);
	    }
	| TK_LPAR dataidl TK_COMMA entity_name TK_EQUALS do_plage TK_RPAR
	    {
		$$ = ExpandDataVar($2, $6);
	    }
	;

implicit_inst: TK_IMPLICIT limplicit
	    {
               /* Formal parameters have inherited default implicit types */
               retype_formal_parameters();
	    }
	;

limplicit: implicit
	    {
	    }
	| limplicit TK_COMMA implicit
	    {
	    }
	;

implicit: fortran_type TK_LPAR l_letter_letter TK_RPAR
	    {
	    }
	;

l_letter_letter: letter_letter
	    {
	    }
	| l_letter_letter TK_COMMA letter_letter
	    {
	    }
	;

letter_letter: letter
	    {
		basic b;

		pips_assert("gram.y", type_variable_p(CurrentType));
		b = variable_basic(type_variable(CurrentType));

		cr_implicit(basic_tag(b), SizeOfElements(b), $1, $1);
	    }
	| letter TK_MINUS letter
	    {
		basic b;

		pips_assert("gram.y", type_variable_p(CurrentType));
		b = variable_basic(type_variable(CurrentType));

		cr_implicit(basic_tag(b), SizeOfElements(b), $1, $3);
	    }
	;

letter:	 TK_NAME
	    {
		$$ = $1[0]; free($1);
	    }
	;

parameter_inst: TK_PARAMETER TK_LPAR lparametre TK_RPAR
	;

lparametre: parametre
	| lparametre TK_COMMA parametre
	;

parametre: entity_name TK_EQUALS expression
	    {
		AddEntityToDeclarations(MakeParameter($1, $3), get_current_module_entity());
	    }
	;

entity_name: name
	    {
                /* malloc_verify(); */
		$$ = FindOrCreateEntity(CurrentPackage, $1);
		free($1);
	    }
	;

name: TK_NAME
	    { $$ = $1; }
        ;

module_name: global_name
            {
		/* $$ = FindOrCreateEntity(CurrentPackage, $1); */
		/* $$ = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, $1); */
		CurrentPackage = strdup($1);
	        BeginingOfProcedure();
		free($1);
		$$ = CurrentPackage;
	    }
        ;

global_entity_name: global_name
	    {
		/* $$ = FindOrCreateEntity(CurrentPackage, $1); */
		$$ = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, $1);
		free($1);
	    }
	;

functional_entity_name: name
	    {
		/* This includes BLOCKDATA modules because of EXTERNAL */
		$$ = NameToFunctionalEntity($1);
		free($1);
	    }
	;

global_name: TK_NAME
	    { $$ = $1; }
        ;

opt_lformalparameter:
	    {
		    $$ = NULL;
	    }
	| TK_LPAR TK_RPAR
	    {
		    $$ = NULL;
	    }
	| TK_LPAR lformalparameter TK_RPAR
	    {
		/* Too early: the current module is still unknown */
		/* $$ = add_formal_return_code($2); */
		$$ = $2
	    }
	;

lformalparameter: entity_name
	    {
		    $$ = CONS(ENTITY, $1, NULL);
	    }
        | TK_STAR
            {
		uses_alternate_return(TRUE);
		$$ = NIL;
            }
	| lformalparameter TK_COMMA entity_name
	    {
		    $$ = gen_nconc($1, CONS(ENTITY, $3, NIL));
	    }
        | lformalparameter TK_COMMA TK_STAR
            {
		uses_alternate_return(TRUE);
		$$ = $1;
            }
	;

opt_fortran_type: fortran_type
            {
		$$ = CurrentType = $1 ;
	    }
    	|
	    {
		$$ = CurrentType = type_undefined;
	    }
	;

fortran_type: fortran_basic_type lg_fortran_type
            {
                if (value_intrinsic_p($2)) /* ??? default! */
                {
		    free_value($2);
		    $2 = make_value(is_value_constant,
		       make_constant(is_constant_int, (void *) CurrentTypeSize));
		}

		$$ = CurrentType = MakeFortranType($1, $2);
	    }
	;

fortran_basic_type: TK_INTEGER
	    {
		    $$ = is_basic_int;
		    CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
	    }
	| TK_REAL
	    {
		    $$ = is_basic_float; 
		    CurrentTypeSize = DEFAULT_REAL_TYPE_SIZE;
	    }
	| TK_DOUBLEPRECISION
	    {
		    $$ = is_basic_float; 
		    CurrentTypeSize = DEFAULT_DOUBLEPRECISION_TYPE_SIZE;
	    }
	| TK_LOGICAL
	    {
		    $$ = is_basic_logical; 
		    CurrentTypeSize = DEFAULT_LOGICAL_TYPE_SIZE;
	    }
	| TK_COMPLEX
	    {
		    $$ = is_basic_complex; 
		    CurrentTypeSize = DEFAULT_COMPLEX_TYPE_SIZE;
	    }
	| TK_CHARACTER
	    {
		    $$ = is_basic_string; 
		    CurrentTypeSize = DEFAULT_CHARACTER_TYPE_SIZE;
	    }
	;

lg_fortran_type:
	    {
                   $$ = make_value(is_value_intrinsic, UU); /* ??? default! */
		/* was: $$ = make_value(is_value_constant,
		 *      make_constant(is_constant_int, CurrentTypeSize)); 
		 * then how to differentiate character*len1 foo[*len2]
		 * if len2 is 1 or whatever... the issue is that 
		 * there should be two lg_..., one for the default that
		 * would change CurrentTypeSize at ival, and the other not...
		 * FC, 13/06/96
		 */
	    }
	| TK_STAR TK_LPAR TK_STAR TK_RPAR /* CHARACTER *(*) */
	    {
		    $$ = MakeValueUnknown();
	    }
	| TK_STAR TK_LPAR expression TK_RPAR
	    {
		    $$ = MakeValueSymbolic($3);
	    }
	| TK_STAR ival
	    {
		    $$ = make_value(is_value_constant, 
				    make_constant(is_constant_int,
						  (void *) $2));
	    }
	;

atom: entity_name
	    {
		$$ = MakeAtom($1, NIL, expression_undefined, 
				expression_undefined, FALSE);
	    }
	| entity_name indices
	    {
		$$ = MakeAtom($1, $2, expression_undefined, 
				expression_undefined, TRUE);
	    }
	| entity_name TK_LPAR opt_expression TK_COLON opt_expression TK_RPAR
	    {
		$$ = MakeAtom($1, NIL, $3, $5, TRUE);
	    }
	| entity_name indices TK_LPAR opt_expression TK_COLON opt_expression TK_RPAR
	    {
		$$ = MakeAtom($1, $2, $4, $6, TRUE);
	    }
	;

indices: TK_LPAR TK_RPAR
		{ $$ = NULL; }
	| TK_LPAR lexpression TK_RPAR
		{ $$ = $2; }
	;

lexpression: expression
	    {
		$$ = CONS(EXPRESSION, $1, NULL);
	    }
	| lexpression TK_COMMA expression
	    {
		$$ = gen_nconc($1, CONS(EXPRESSION, $3, NIL));
	    }
	;

opt_expression: expression
	    { $$ = $1; }
	|
	    { $$ = expression_undefined; }
	;

expression: sous_expression
	    { $$ = $1; }
        | TK_LPAR expression TK_RPAR
            { $$ = $2; }
        ;

sous_expression: atom
	    {
		    $$ = make_expression($1, normalized_undefined);
	    }
	| unsigned_const_simple
	    {
		    $$ = MakeNullaryCall($1);    
	    }
	| signe expression  %prec TK_STAR
	    {
		    if ($1 == -1)
			$$ = MakeUnaryCall(CreateIntrinsic("--"), $2);
		    else
			$$ = $2;
	    }
	| expression TK_PLUS expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic("+"), $1, $3);
	    }
	| expression TK_MINUS expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic("-"), $1, $3);
	    }
	| expression TK_STAR expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic("*"), $1, $3);
	    }
	| expression TK_SLASH expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic("/"), $1, $3);
	    }
	| expression TK_POWER expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic("**"), 
					      $1, $3);
	    }
	| expression oper_rela expression  %prec TK_EQ
	    {
		    $$ = MakeBinaryCall($2, $1, $3);    
	    }
	| expression TK_EQV expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic(".EQV."), 
					      $1, $3);
	    }
	| expression TK_NEQV expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic(".NEQV."), 
					      $1, $3);
	    }
	| expression TK_OR expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic(".OR."), 
					      $1, $3);
	    }
	| expression TK_AND expression
	    {
		    $$ = MakeBinaryCall(CreateIntrinsic(".AND."), 
					      $1, $3);
	    }
	| TK_NOT expression
	    {
		    $$ = MakeUnaryCall(CreateIntrinsic(".NOT."), $2);
	    }
	| expression TK_CONCAT expression
            {
		    $$ = MakeBinaryCall(CreateIntrinsic("//"), 
					      $1, $3);    
	    }
	;

const_simple: opt_signe unsigned_const_simple
	    {
		    if ($1 == -1)
			$$ = MakeUnaryCall(CreateIntrinsic("--"), 
					   MakeNullaryCall($2));
		    else
			$$ = MakeNullaryCall($2);
	    }
	;

unsigned_const_simple: TK_TRUE
	    {
		    $$ = MakeConstant(".TRUE.", is_basic_logical);
	    }
	| TK_FALSE
	    {
		    $$ = MakeConstant(".FALSE.", is_basic_logical);
	    }
	| icon
	    {
		    $$ = $1;
	    }
	| TK_DCON
	    {
		    $$ = make_constant_entity($1, is_basic_float,
					      DEFAULT_DOUBLEPRECISION_TYPE_SIZE);
		    free($1);
	    }
	| TK_SCON
	    {
		    $$ = MakeConstant($1, is_basic_string);
		    free($1);		    
	    }
        | TK_RCON 
	    {
		    $$ = MakeConstant($1, is_basic_float);
		    free($1);
	    }
	;

icon: TK_ICON 
	    {
		    $$ = MakeConstant($1, is_basic_int);
		    free($1);
	    }
	;

label: TK_ICON 
	    {
		    $$ = $1;
	    }
	;

ival: TK_ICON 
	    {
		    $$ = atoi($1);
		    free($1);
	    }
	;

opt_signe:
	    {
		    $$ = 1;
	    }
	| signe
	    {
		    $$ = $1;
	    }
	;

signe:    TK_PLUS
	    {
		    $$ = 1;
	    }
	| TK_MINUS
	    {
		    $$ = -1;
	    }
	;

oper_rela: TK_EQ
	    {
		    $$ = CreateIntrinsic(".EQ.");
	    }
	| TK_NE
	    {
		    $$ = CreateIntrinsic(".NE.");
	    }
	| TK_LT
	    {
		    $$ = CreateIntrinsic(".LT.");
	    }
	| TK_LE
	    {
		    $$ = CreateIntrinsic(".LE.");
	    }
	| TK_GE
	    {
		    $$ = CreateIntrinsic(".GE.");
	    }
	| TK_GT
	    {
		    $$ = CreateIntrinsic(".GT.");
	    }
	;

io_keyword: TK_PRINT
            { $$ = TK_PRINT; ici = 1; }
	| TK_WRITE
            { $$ = TK_WRITE; ici = 1; }
	| TK_READ
            { $$ = TK_READ; ici = 1; }
	| TK_CLOSE
            { $$ = TK_CLOSE; ici = 1; }
	| TK_OPEN
            { $$ = TK_OPEN; ici = 1; }
	| TK_ENDFILE
            { $$ = TK_ENDFILE; ici = 1; }
	| TK_BACKSPACE
            { $$ = TK_BACKSPACE; ici = 1; }
	| TK_REWIND
            { $$ = TK_REWIND; ici = 1; }
	| TK_INQUIRE
            { $$ = TK_INQUIRE; ici = 1; }
        ;

iobuf_keyword: TK_BUFFERIN
            { $$ = TK_BUFFERIN; ici = 1; }
	| TK_BUFFEROUT 
            { $$ = TK_BUFFEROUT ; ici = 1; }
        ;

psf_keyword: TK_PROGRAM 
	    { $$ = TK_PROGRAM; }
	| TK_SUBROUTINE 
	    { $$ = TK_SUBROUTINE; }
	| TK_FUNCTION
	    { $$ = TK_FUNCTION; }
	| TK_BLOCKDATA
	    { $$ = TK_BLOCKDATA; }
	;

opt_virgule:
	| TK_COMMA
	;
%%
