 /* 
  *  Functions for the expressions
  *
  *  Yi-Qing YANG, Lei ZHOU, Francois IRIGOIN
  *
  *  12, Sep, 1991
  */

#include <stdio.h>
#include <string.h>
#include <varargs.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"

#include "arithmetique.h"

#include "ri-util.h"

entity CreateIntrinsic(string name); /* in syntax.h */


/*  a BASIC tag is returned for the expression
 *  this is a preliminary version. should be improved.
 *  was in HPFC.
 */
tag suggest_basic_for_expression(e)
expression e;
{
    tag
	result = basic_tag(expression_basic(e));

    if (result==is_basic_overloaded)
    {
	syntax s = expression_syntax(e);

	/*  must be a call
	 */
	assert(syntax_call_p(s));

	if (ENTITY_RELATIONAL_OPERATOR_P(call_function(syntax_call(s))))
	    result = is_basic_logical;
	else
	{
	    /* else some clever analysis could be done
	     */
	    user_warning("suggest_basic_for_expression",
			 "an overloaded is turned into an int...\n");
	    result = is_basic_int;
	}
    }

    return(result);
}

expression expression_dup(ex)
expression ex;
{
    syntax s = expression_syntax(ex);
    normalized n = expression_normalized(ex);
    return (n == normalized_undefined)?
	make_expression(syntax_dup(s), normalized_undefined):
	make_expression(syntax_dup(s), normalized_dup(n));
}

syntax syntax_dup(s)
syntax s;
{
    syntax new_s = syntax_undefined;

    switch(syntax_tag(s)) {

    case is_syntax_reference: 
    { reference r = syntax_reference(s);
      new_s = make_syntax(is_syntax_reference, reference_dup(r));
      break;
    }

    case is_syntax_range:
    { range r = syntax_range(s);
      new_s = make_syntax(is_syntax_range, range_dup(r));
      break;
    }

    case is_syntax_call: 
    { call c = syntax_call(s);
      new_s = make_syntax(is_syntax_call, call_dup(c));
      break;
    }

    default:
	pips_error("syntax_dup", "ill. tag\n");
    }
    return new_s;
}

normalized normalized_dup(n)
normalized n;
{
    normalized new_n = normalized_undefined;

    switch(normalized_tag(n)) {
    case is_normalized_linear: { 
	Pvecteur v = (Pvecteur) normalized_linear(n);
	new_n = make_normalized(is_normalized_linear, (char *) vect_dup(v));
	break;
    }
    case is_normalized_complex: { 
	new_n = make_normalized(is_normalized_complex, UU);
	break;
    }
    default:
	pips_error("normalized_dup", "ill. tag\n");
    }
    return new_n;
}

reference reference_dup(r)
reference r;
{
    entity v = reference_variable(r);
    list ind = reference_indices(r);
    list new_ind = NIL;

    MAPL(ce, { 
	expression e = EXPRESSION(CAR(ce));
	new_ind = gen_nconc(new_ind, 
			    CONS(EXPRESSION, expression_dup(e), NIL)) ;
    }, ind);

    return make_reference(v, new_ind);
}

range range_dup(r)
range r;
{
    return make_range(expression_dup(range_lower(r)),
		      expression_dup(range_upper(r)),
		      expression_dup(range_increment(r))) ;
}

call call_dup(c)
call c;
{
    entity f = call_function(c);
    list args = call_arguments(c);
    list new_args = NIL;

    MAPL(ce, { 
	expression e = EXPRESSION(CAR(ce));
	new_args = gen_nconc(new_args, 
			     CONS(EXPRESSION, expression_dup(e), NIL)) ;
    }, args);

    return make_call(f, new_args);
}

expression expression_mult(ex)
expression ex;
{
    pips_error("expression_mult", "not implemented\n");
    return ex;
}

/* if v is a constant, returns a constant call.
 * if v is a variable, returns a reference to v.
 */
expression entity_to_expression(e)
entity e;
{
    if (entity_constant_p(e))
	return call_to_expression(make_call(e, NIL));
    else /* should be a scalar variable! */
	return reference_to_expression(make_reference(e, NIL));
}

/*
 * remarks: why is the default to normalized_complex~? 
 * should be undefined, OR normalized if possible.
 * I put normalize_reference... FC 27/09/93 and june 94
 */
expression reference_to_expression(r)
reference r;
{
    expression e;
    syntax s = make_syntax(is_syntax_reference, r);

    e = make_expression(s, normalize_reference(r));

    return e;
}

expression MakeBinaryCall(f, eg, ed)
entity f;
expression eg, ed;
{
    call c =  make_call(f, CONS(EXPRESSION, eg, CONS(EXPRESSION, ed, NIL)));

    return(make_expression(make_syntax(is_syntax_call, c),
			   normalized_undefined));
}

expression call_to_expression(c)
call c;
{
    return(make_expression(make_syntax(is_syntax_call, c),
			   make_normalized(is_normalized_complex, UU)));
}

expression make_call_expression(e, l)
entity e;
list l;
{
    return(call_to_expression(make_call(e, l)));
}

expression MakeTernaryCallExpr(f, e1, e2, e3)
entity f;
expression e1,e2,e3;
{
    return(make_call_expression(f,
	       CONS(EXPRESSION, e1,
	       CONS(EXPRESSION, e2,
	       CONS(EXPRESSION, e3,
		    NULL)))));
}

/* predicates on expressions */

bool expression_call_p(e)
expression e;
{
    return(syntax_call_p(expression_syntax(e)));
}

bool expression_reference_p(e)
expression e;
{
    return(syntax_reference_p(expression_syntax(e)));
}

bool expression_implied_do_p(e)
expression e ;
{
    if (expression_call_p(e)) {
	call c = syntax_call(expression_syntax(e));
	entity e = call_function(c);

	return(strcmp(entity_local_name(e), IMPLIED_DO_NAME) == 0);
    }

    return(FALSE);
}

bool integer_constant_expression_p(e)
expression e;
{
    syntax s = expression_syntax(e);

    if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity cst = call_function(c);
	int i;

	return integer_constant_p(cst, &i);
    }
    else
	return FALSE;
}

bool modulo_expression_p(e)
expression e;
{
    return operator_expression_p(e, MODULO_OPERATOR_NAME);
}

bool divide_expression_p(e)
expression e;
{
    return operator_expression_p(e, DIVIDE_OPERATOR_NAME);
}

bool min0_expression_p(e)
expression e;
{
    return operator_expression_p(e, MIN0_OPERATOR_NAME);
}

bool max0_expression_p(e)
expression e;
{
    return operator_expression_p(e, MAX0_OPERATOR_NAME);
}

bool operator_expression_p(e, op_name)
expression e;
string op_name;
{
    syntax s = expression_syntax(e);

    if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity op = call_function(c);

	return strcmp(op_name, entity_local_name(op)) == 0;
    }
    else
	return FALSE;
}


/* boolean unbounded_dimension_p(dim)
 * input    : a dimension of an array entity.
 * output   : TRUE if the last dimension is unbounded (*),
 *            FALSE otherwise.
 * modifies : nothing
 * comment  : 
 */
boolean unbounded_dimension_p(dim)
dimension dim;
{
    syntax dim_synt = expression_syntax(dimension_upper(dim));
    boolean res = FALSE;
    
    if (syntax_call_p(dim_synt)) {
	string dim_nom = entity_local_name(call_function(syntax_call(dim_synt)));
	
	if (same_string_p(dim_nom, UNBOUNDED_DIMENSION_NAME))
	    res = TRUE;
    }
	
    return(res);
	
}


expression find_ith_argument(args, n)
list args;
int n;
{
    int i;
    pips_assert("find_ith_argument", n > 0);

    for(i=1; i<n && !ENDP(args); i++, POP(args))
	;
    if(i==n && !ENDP(args))
	return EXPRESSION(CAR(args));
    else
	return expression_undefined;
}

/* find_ith_expression() is obsolet; use find_ith_argument() instead */
expression find_ith_expression(le, r)
list le;
int r;
{
    /* the first element is one */
    /* two local variables, useless but for debugging */
    list cle;
    int i;

    pips_assert("find_ith_expression", r > 0);

    for(i=r, cle=le ; i>1 && !ENDP(cle); i--, POP(cle))
	;

    if(ENDP(cle))
	pips_error("find_ith_expression", 
		   "not enough elements in expresion list\n");

    return EXPRESSION(CAR(cle));
}

/* transform an int into an expression and generate the corresponding
   entity if necessary; it is not clear if strdup() is always/sometimes
   necessary and if a memory leak occurs; wait till syntax/expression.c
   is merged with ri-util/expression.c 
*/
expression int_to_expression(i)
int i;
{
    static char constant_name[12];
    expression e;

    (void) sprintf(constant_name,"%d",i);
    e = MakeIntegerConstantExpression(strdup(constant_name));

    return e;
}

/* added interface for linear stuff.
 * it is not ok if Value is not an int, but if Value is changed
 * sometime, I guess code that use this function will not need
 * any change.
 * FC.
 */
expression Value_to_expression(v)
Value v;
{
    return(int_to_expression((int) v));
}

/* conversion of an expression into a list of references; references are
   appended to list lr as they are encountered; array references are
   added before their index expressions are scanned;

   references to functions and constants (which are encoded as null-ary
   functions) are not recorded 
*/
list expression_to_reference_list(e, lr)
expression e;
list lr;
{
    syntax s = expression_syntax(e);

    lr = syntax_to_reference_list(s, lr);

    return lr;
}

list syntax_to_reference_list(s, lr)
syntax s;
list lr;
{
    switch(syntax_tag(s)) {
    case is_syntax_reference:
	lr = gen_nconc(lr, CONS(REFERENCE, syntax_reference(s), NIL));
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = expression_to_reference_list(e, lr);
	    },
	     reference_indices(syntax_reference(s)));
	break;
    case is_syntax_range:
	lr = expression_to_reference_list(range_lower(syntax_range(s)), lr);
	lr = expression_to_reference_list(range_upper(syntax_range(s)), lr);
	lr = expression_to_reference_list(range_increment(syntax_range(s)),
					  lr);
	break;
    case is_syntax_call:
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = expression_to_reference_list(e, lr);
	    },
	     call_arguments(syntax_call(s)));
	break;
    default:
	pips_error("syntax_to_reference_list","illegal tag %d\n", 
		   syntax_tag(s));

    }
    return lr;
}
       
/* no file descriptor is passed to make is easier to use in a debugging
   stage.
   Do not make macros of those printing functions */

void print_expression(e)
expression e;
{
    normalized n;

    if(e==expression_undefined)
	(void) fprintf(stderr,"EXPRESSION UNDEFINED\n");
    else {
	(void) fprintf(stderr,"syntax = ");
	print_syntax(expression_syntax(e));
	(void) fprintf(stderr,"\nnormalized = ");
	if((n=expression_normalized(e))!=normalized_undefined)
	    print_normalized(n);
	else
	    (void) fprintf(stderr,"NORMALIZED UNDEFINED\n");
    }
}

void print_syntax(s)
syntax s;
{
    print_words(stderr,words_syntax(s));
}

void print_reference(r)
reference r;
{
    print_words(stderr,words_reference(r));
}

void print_reference_list(lr)
list lr;
{
    if(ENDP(lr))
	fputs("NIL", stderr);
    else
	MAPL(cr,
	 {
	     reference r = REFERENCE(CAR(cr));
	     entity e = reference_variable(r);
	     (void) fprintf(stderr,"%s, ", entity_local_name(e));
	 },
	     lr);

    (void) putc('\n', stderr);
}

void print_normalized(n)
normalized n;
{
    if(normalized_complex_p(n))
	(void) fprintf(stderr,"COMPLEX\n");
    else
	/* should be replaced by a call to expression_fprint() if it's
	   ever added to linear library */
	vect_dump((Pvecteur)normalized_linear(n));
}

bool expression_equal_p(e1, e2)
expression e1;
expression e2;
{
    /* let's assume that every expression has a correct syntax component */
    syntax s1 = expression_syntax(e1);
    syntax s2 = expression_syntax(e2);

    return syntax_equal_p(s1, s2);
}

bool syntax_equal_p(s1, s2)
syntax s1;
syntax s2;
{
    tag t1 = syntax_tag(s1);
    tag t2 = syntax_tag(s2);

    if(t1!=t2)
	return FALSE;

    switch(t1) {
    case is_syntax_reference:
	return reference_equal_p(syntax_reference(s1), syntax_reference(s2));
	break;
    case is_syntax_range:
	return range_equal_p(syntax_range(s1), syntax_range(s2));
	break;
    case is_syntax_call:
	return call_equal_p(syntax_call(s1), syntax_call(s2));
	break;
    default:
	break;
    }

    pips_error("syntax_equal_p", "ill. tag\n");
    return FALSE;
}

bool reference_equal_p(r1,r2)
reference r1;
reference r2;
{
    entity v1 = reference_variable(r1);
    entity v2 = reference_variable(r2);

    list dims1 = reference_indices(r1);
    list dims2 = reference_indices(r2);

    if(v1 != v2)
	return FALSE;

    if(gen_length(dims1) != gen_length(dims2))
	pips_error("reference_equal_p",
		   "Different dimensions for %s: %d and %d\n",
		   entity_local_name(v1),
		   gen_length(dims1), gen_length(dims2));

    for(; !ENDP(dims1); POP(dims1), POP(dims2))
	if(!expression_equal_p(EXPRESSION(CAR(dims1)), EXPRESSION(CAR(dims2))))
	    return FALSE;

    return TRUE;
}

bool range_equal_p(r1, r2)
range r1;
range r2;
{
    return expression_equal_p(range_lower(r1), range_lower(r2))
	&& expression_equal_p(range_upper(r1), range_upper(r2))
	    && expression_equal_p(range_increment(r1), range_increment(r2));
}

bool call_equal_p(c1, c2)
call c1;
call c2;
{
    entity f1 = call_function(c1);
    entity f2 = call_function(c2);
    list args1 = call_arguments(c1);
    list args2 = call_arguments(c2);

    if(f1 != f2)
	return FALSE;

    if(gen_length(args1) != gen_length(args2))
	return FALSE;

    for(; !ENDP(args1); POP(args1), POP(args2))
	if(!expression_equal_p(EXPRESSION(CAR(args1)), EXPRESSION(CAR(args2))))
	    return FALSE;

    return TRUE;
}

/* expression make_integer_constant_expression(int c)
 *  make expression for integer
 */
expression make_integer_constant_expression(c)
int c;
{
    expression ex_cons;
    entity ce;   
   
    ce = make_integer_constant_entity(c);
    /* make expression for the constant c*/
    ex_cons = make_expression(
			      make_syntax(is_syntax_call,
					  make_call(ce,NIL)), 
			      normalized_undefined);
    return (ex_cons);
}

int integer_constant_expression_value(e)
expression e;
{
    /* could be coded by geting directly the value of the constant entity... */
    /* also available as integer_constant_p() which has *two* arguments */

    normalized n = normalized_undefined;
    int val = 0;

    pips_assert("integer_constant_expression_value", integer_constant_expression_p(e));

    n = NORMALIZE_EXPRESSION(e);
    if(normalized_linear_p(n)) {
	Pvecteur v = (Pvecteur) normalized_linear(n);

	if(vect_constant_p(v)) {
	    val = (int) vect_coeff(TCST, v);
	    }
	else
	    pips_error("integer_constant_expression_value", "non constant expression\n");
    }
    else
	pips_error("integer_constant_expression_value", "non affine expression\n");

    return val;
}


/* expression make_factor_expression(int coeff, entity vari)
 * make the expression "coeff*vari"  where vari is an entity.
 */
expression make_factor_expression(coeff, vari)
int coeff;
entity vari;
{
    expression e1, e2, e3;
    entity operateur_multi;

    e1 = make_integer_constant_expression(coeff);
    if (vari==NULL)
	return(e1);			/* a constant only */
    else {
	e2 = make_expression(make_syntax(is_syntax_reference,
					 make_reference(vari, NIL)),
			     normalized_undefined);
	if (coeff == 1) return(e2);
	else {
	    operateur_multi = gen_find_tabulated("TOP-LEVEL:*",entity_domain);
	    e3 = make_expression(make_syntax(is_syntax_call,
					     make_call(operateur_multi,
				       CONS(EXPRESSION, e1,
				       CONS(EXPRESSION, e2,
					    NIL)))),
				 normalized_undefined);
	    return (e3);
	}
    }
}

/* expression make_vecteur_expression(Pvecteur pv)
 * make expression for vector (Pvecteur)
 */
expression make_vecteur_expression(pv)
Pvecteur pv;
{
    /* sort: to insure a deterministic generation of the expression.
     * note: the initial system is *NOT* touched.
     * ??? Sometimes the vectors are shared, so you cant modify them
     *     that easily. Many cores in Hpfc (deducables), Wp65, and so.
     * ok, I'm responsible for some of them:-)
     *
     *  (c) FC 24/11/94
     */
    Pvecteur
	v_sorted = vect_sort(pv, compare_Pvecteur),
	v = v_sorted;
    expression 	factor1, factor2;
    entity op_add, op_sub;

    op_add = CreateIntrinsic(PLUS_OPERATOR_NAME);
    op_sub = CreateIntrinsic(MINUS_OPERATOR_NAME);

    assert(!entity_undefined_p(op_add) && !entity_undefined_p(op_sub));
    
    if (VECTEUR_NUL_P(v)) 
	return make_integer_constant_expression(0);

    factor1 = make_factor_expression((int) vecteur_val(v), 
				     (entity) vecteur_var(v));

    for (v=v->succ; v!=NULL; v=v->succ)
    {
	factor2 = make_factor_expression(ABS((int) vecteur_val(v)),
					 (entity) vecteur_var(v));
	factor1 = make_expression(make_syntax(is_syntax_call,
		      make_call((vecteur_val(v)>0 ? op_add : op_sub),
		CONS(EXPRESSION, factor1,
		CONS(EXPRESSION, factor2,
		     NIL)))),
				  normalized_undefined);
    }

    vect_rm(v_sorted);

    return factor1;
}

/* generates var = linear expression 
 * from the Pvecteur. var is removed if necessary.
 * ??? should manage an (positive remainder) integer divide ?
 */
statement
Pvecteur_to_assign_statement(
    entity var,
    Pvecteur v)
{
    statement result;
    Pvecteur vcopy;
    int coef;
    
    coef = vect_coeff((Variable) var, v);
    assert(abs(coef)<=1);

    vcopy = vect_dup(v);
	
    if (coef) vect_erase_var(&vcopy, (Variable) var);
    if (coef==1) vect_chg_sgn(vcopy);
	
    result = make_assign_statement(entity_to_expression(var),
				   make_vecteur_expression(vcopy));
    vect_rm(vcopy);

    return result;
}

reference expression_reference(e)
expression e;
{
    return syntax_reference(expression_syntax(e));
}

/* predicates on references */

bool array_reference_p(r)
reference r;
{
    /* two possible meanings:
     * - the referenced variable is declared as an array
     * - the reference is to an array element
     *
     * This makes a difference in procedure calls and IO statements
     *
     * The second interpretation is chosen.
     */

    return reference_indices(r) != NIL;
}

/*
 *    Utils from hpfc on 15 May 94, FC
 *
 */
expression expression_list_to_binary_operator_call(l, op)
list l;
entity op;
{
    int
	len = gen_length(l);
    expression
	result = expression_undefined;

    pips_assert("list_to_binary_operator_call", len!=0);

    result = EXPRESSION(CAR(l));

    MAPL(ce,
     {
	 result = MakeBinaryCall(op, EXPRESSION(CAR(ce)), result);
     },
	 CDR(l));

    return(result);
}

/*
 * expression list_to_conjonction(l)
 */
expression expression_list_to_conjonction(l)
list l;
{
    int
	len = gen_length(l);
    entity
	and = CreateIntrinsic(AND_OPERATOR_NAME);

    return(len==0? 
	   MakeNullaryCall(CreateIntrinsic(".TRUE.")):
	   expression_list_to_binary_operator_call(l, and));
}

/*============================================================================*/
/* bool expression_intrinsic_operation_p(expression exp): Returns TRUE
 * if "exp" is an expression with a call to an intrinsic operation.
 */
bool expression_intrinsic_operation_p(exp)
expression exp;
{
    entity e;
    syntax syn = expression_syntax(exp);

    if (syntax_tag(syn) != is_syntax_call)
	return (FALSE);

    e = call_function(syntax_call(syn));

    return(value_tag(entity_initial(e)) == is_value_intrinsic);
}

/*============================================================================*/
/* bool call_constant_p(call c): Returns TRUE if "c" is a call to a constant,
 * that is, a constant number or a symbolic constant.
 */
bool call_constant_p(c)
call c;
{
    value cv = entity_initial(call_function(c));

    return( (value_tag(cv) == is_value_constant) ||
	   (value_tag(cv) == is_value_symbolic)   );
}

/*
 *   that is all
 */
