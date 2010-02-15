
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "properties.h"

static bool has_address_of_operator_walker_p(call c,bool *panswer)
{
    return !(*panswer|=ENTITY_ADDRESS_OF_P(call_function(c)));
}

/* look for an & in the expression*/
static bool has_address_of_operator_p(expression exp)
{
    bool answer = false;
    gen_context_recurse(exp,&answer,call_domain,has_address_of_operator_walker_p,gen_null);
    return answer;
}

/* the walker*/
static void simplify_subscript(subscript ss)
{
    statement parent_statement = (statement)gen_get_ancestor(statement_domain,ss);
    expression ss_array = subscript_array(ss);
    /* do nothing if an adress-of is involved - over cautious indeed */
    if( !has_address_of_operator_p(ss_array) )
    {
        /* create atomized_expression */
        basic exp_basic = basic_of_expression(ss_array);
        entity new_entity = make_new_scalar_variable(get_current_module_entity(),exp_basic);
        AddEntityToCurrentModule(new_entity);
        statement new_statement = make_assign_statement(entity_to_expression(new_entity),ss_array);

        /* create replacement reference */
        reference new_ref= make_reference(new_entity,subscript_indices(ss));
        /* do the replacement */
        expression parent_expression = (expression) gen_get_ancestor(expression_domain,ss);
        { /*free stuffs */
            free_normalized(expression_normalized(parent_expression));
            subscript_array(ss)=expression_undefined;
            subscript_indices(ss)=NIL;
            free_syntax(expression_syntax(parent_expression));
        }
        expression_syntax(parent_expression)=make_syntax_reference(new_ref);
        expression_normalized(parent_expression)=normalized_undefined;
        /* we must do this now and not before */
        insert_statement(parent_statement,new_statement,true);
    }
}

/* atomize subscript expressions so that thay can be reprsetned as references*/
bool simplify_subscripts(string module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* do the job */
    gen_recurse(get_current_module_statement(),
            subscript_domain,gen_true,simplify_subscript);

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}

/**************************************************************
 * SIMPLIFY_COMPLEX
 */
expression make_float_constant_expression(float);

static entity THE_I = entity_undefined;
static void set_the_i() {
    THE_I = make_new_scalar_variable_with_prefix("IM",get_current_module_entity(), make_basic_complex(DEFAULT_COMPLEX_TYPE_SIZE));
    free_value(entity_initial(THE_I));
    entity_initial(THE_I)=make_value_expression(MakeComplexConstantExpression(make_float_constant_expression(0),make_float_constant_expression(1)));
    AddEntityToCurrentModule(THE_I);
}

static void reset_the_i() {
    THE_I = entity_undefined;
}

static void (*complex_translation)(list*,list)=NULL;
void complex_translation_array_of_structs(list *pl,list l)
{
    gen_append(*pl,l);
}
void complex_translation_struct_of_array(list *pl,list l)
{
    gen_append(l,*pl);
    *pl=l;
}

static void set_translation_mode()
{
    if(get_bool_property("SIMPLIFY_COMPLEX_USE_ARRAY_OF_STRUCTS"))
        complex_translation=complex_translation_array_of_structs;
    else
        complex_translation=complex_translation_struct_of_array;
}
static void reset_translation_mode(){ complex_translation=NULL;}


/* adds a dimension to all entity whose basic is complex , and change their basic to float */
static void simplify_complex_entity(entity c)
{
    if(entity_variable_p(c) && basic_complex_p(entity_basic(c)))
    {
        variable v = type_variable(entity_type(c));
        complex_translation(
                &variable_dimensions(v),
                CONS(DIMENSION,make_dimension(int_to_expression(0),int_to_expression(1)),NIL)
                );
        intptr_t old_basic = basic_complex(variable_basic(v));
        free_basic(variable_basic(v));
        variable_basic(v)=make_basic_float(old_basic);
    }
}
static void simplify_complex_declaration(entity e)
{
    FOREACH(ENTITY,c,entity_declarations(e))
        simplify_complex_entity(c);
    FOREACH(ENTITY,c,code_declarations(value_code(entity_initial(e))))
        simplify_complex_entity(c);
    FOREACH(PARAMETER,p,functional_parameters(type_functional(entity_type(e))))
    {
        dummy d = parameter_dummy(p);
        if(dummy_identifier_p(d))
            simplify_complex_entity(dummy_identifier(d));
    }
    free(code_decls_text(entity_code(e)));
    code_decls_text(entity_code(e))=strdup("");
}

static expression split_complex_expression(expression e)
{
    reference r = expression_reference(e);
    list ri = reference_indices(r);

    list ri0 = gen_full_copy_list(ri);
    complex_translation(&ri0,make_expression_list(make_expression_0()));
    list ri1 = gen_full_copy_list(ri);
    complex_translation(&ri1,make_expression_list(make_expression_1()));

    expression new = MakeBinaryCall(
            CreateIntrinsic(PLUS_OPERATOR_NAME),
            reference_to_expression(make_reference(reference_variable(r),ri0)),
            MakeBinaryCall(
                CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                entity_to_expression(THE_I),
                reference_to_expression(make_reference(reference_variable(r),ri1))
                )
            );
    return new;
}

static bool simplify_complex_expression(expression e)
{
    basic b = basic_of_expression(e);
    if(expression_call_p(e))
    {
        call c = expression_call(e);
        entity op = call_function(c);
        expression arg = ENDP(call_arguments(c))?expression_undefined:EXPRESSION(CAR(call_arguments(c)));
        /* replace |.| by sqrt(re^2+im^2) */
        if(ENTITY_CABS_P(op))
        {
            call_function(c)=CreateIntrinsic(SQRT_OPERATOR_NAME);
            gen_free_list(call_arguments(c));
            call_arguments(c)=make_expression_list(
                    MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME),
                        MakeUnaryCall(CreateIntrinsic(REAL_GENERIC_CONVERSION_NAME),copy_expression(arg)),
                        MakeUnaryCall(CreateIntrinsic(AIMAG_CONVERSION_NAME),arg)
                        )
                    );
        }
        /* replace REAL(.) by .[0] */
        else if(same_entity_p( op, CreateIntrinsic(REAL_GENERIC_CONVERSION_NAME) ))
        {
            if(expression_reference_p(arg)) /* ok let's do something */
            {
                reference r = expression_reference(arg);
                complex_translation(&reference_indices(r),make_expression_list(make_expression_0()));
                /* replace the call by a reference */
                syntax_reference(expression_syntax(arg))=reference_undefined;
                free_syntax(expression_syntax(e));
                free_normalized(expression_normalized(e));
                expression_syntax(e)=make_syntax_reference(r);
                expression_normalized(e)=normalized_undefined;
            }
            else
                pips_user_warning("case unhandled, this would require an atomization in fortran i guess\n");
        }
        /* replace AIMAG(.) by .[1] */
        else if(same_entity_p( op, CreateIntrinsic(AIMAG_CONVERSION_NAME) ))
        {
            if(expression_reference_p(arg)) /* ok let's do something */
            {
                reference r = expression_reference(arg);
                complex_translation(&reference_indices(r),make_expression_list(make_expression_1()));
                /* replace the call by a reference */
                syntax_reference(expression_syntax(arg))=reference_undefined;
                free_syntax(expression_syntax(e));
                free_normalized(expression_normalized(e));
                expression_syntax(e)=make_syntax_reference(r);
                expression_normalized(e)=normalized_undefined;
            }
            else
                pips_user_warning("case unhandled, this would require an atomization in fortran i guess\n");
        }
        else if(basic_complex_p(b) && ENTITY_FIVE_OPERATION_P(op) )
        {
            expression e0 = binary_call_lhs(c),
                       e1 = binary_call_rhs(c);
            basic b0 = basic_of_expression(e0),
                  b1=basic_of_expression(e1);
            if(basic_complex_p(b0) && expression_reference_p(e0))
            {
                *e0=*split_complex_expression(e0);
                gen_recurse_stop(e0);
            }
            if(basic_complex_p(b1) && expression_reference_p(e1))
            {
                *e1=*split_complex_expression(e1);
                gen_recurse_stop(e1);
            }
            free_basic(b0);
            free_basic(b1);

        }
    }
    free_basic(b);
    /* more to come */
    return true;

}


static void simplify_complex_statements(statement s)
{
    /* do the job */
    gen_recurse(s,expression_domain,simplify_complex_expression,gen_null);
}

/* split complexes into an array with first element <> real part and second element <> imaginary part */
bool simplify_complex(string module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    set_the_i();
    set_translation_mode();

    /* change call to complex intrinsics */
    /* change other complex references*/
    simplify_complex_statements(get_current_module_statement());
    /* and all their declarations */
    simplify_complex_declaration(get_current_module_entity());

    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /*postlude*/
    reset_translation_mode();
    reset_the_i();
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}
