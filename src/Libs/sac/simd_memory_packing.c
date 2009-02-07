#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "sac-local.h" /* needed because sac.h may not exist when 
                        * simdizer.c is compiled */
#include "sac.h"

#include "semantics.h"
#include "transformations.h"
#include "control.h"

#include "effects-convex.h"

static
bool simd_replace_parameters( hash_table array_to_vector )
{
    bool failed = true;
    functional f = type_functional( entity_type( get_current_module_entity() ) );
    FOREACH(PARAMETER, p, functional_parameters(f) )
    {
        dummy d = parameter_dummy(p);
        if( dummy_identifier_p(d) )
        {
            /* recover the parameter name */
            entity dummy_e = dummy_identifier(d);
            string parameter_name = entity_local_name(dummy_e);
            /* get associated entity */
            entity real_e = FindOrCreateEntity(entity_local_name(get_current_module_entity()),parameter_name);
            /* find associated vector if any */
            entity found = hash_get(array_to_vector,real_e);
            if( ! entity_undefined_p(found) )
            {
                /* gather information concerning the vector */
                variable v = type_variable(entity_type(found));
                int n = ValueNumberOfElements( variable_dimensions(v) );

                basic b = variable_basic(v);
                pips_assert( "found vector is a vector of int", !basic_undefined_p(b) && basic_int_p(b) );

                /* create a new basic : increased int size*/
                /* substitute this basic to the old one*/
                variable new_var = copy_variable( type_variable(entity_type(real_e)) );

                basic new_var_basic = variable_basic(new_var);
                while( basic_pointer_p(new_var_basic))
                {
                    new_var_basic = variable_basic(type_variable( basic_pointer(new_var_basic)));
                }
                pips_assert("is a basic int",basic_int_p(new_var_basic));
                basic_int(new_var_basic)=basic_int(b) * n ;

                /* update dummy parameter */
                free_variable(type_variable(entity_type(dummy_e)));
                entity_type(dummy_e)= make_type_variable(new_var);
                free_variable(type_variable(parameter_type(p)));
                parameter_type(p)= make_type_variable(copy_variable(new_var));

                /* update declaration */
                FOREACH(ENTITY, decl_ent,
                        code_declarations(value_code(entity_initial(get_current_module_entity()))))
                {
                    if( same_entity_lname_p(decl_ent, real_e))
                    {
                        free_variable(type_variable(entity_type(decl_ent)));
                        entity_type(decl_ent)= make_type_variable(copy_variable(new_var));
                    }
                }

                failed=false; 
            }
        }
    }
    return !failed;

}

static
void simd_trace_call(statement s, hash_table array_to_vector)
{
    /* we only look for load - save statement */
    if( simd_loadsave_stat_p(s) )
    {
        list args = call_arguments(statement_call(s));
        int nb_args = gen_length(args);
        /* and only of the form SIMD_(...)(vect,array) */
        if( nb_args == 2 )
        {
            expression e = EXPRESSION(CAR(CDR(args)));
            syntax s = expression_syntax(e);
            bool through_address_of = false;
            if( through_address_of=(syntax_call_p(s) && ENTITY_ADDRESS_OF_P(call_function(syntax_call(s)))))
                s = expression_syntax(EXPRESSION(CAR(call_arguments(syntax_call(s)))));
            pips_assert("parameter is a reference", syntax_reference_p(s));
            reference r = syntax_reference(s);
            pips_assert("parameter is a variable",type_variable_p(entity_type(reference_variable(r))));
            variable v = type_variable(entity_type(reference_variable(r)));
            pips_assert("parameter is an array", (through_address_of&&reference_indices(r)) || ! ENDP(variable_dimensions(v)));
            hash_put(array_to_vector,
                    reference_variable(r),
                    reference_variable(expression_reference(EXPRESSION(CAR(args))))
                    );
        }
    }

}

/** 
 * @brief pack load / store from char or short array
 * into load / store from int array
 * a new module is generated
 * @param mod_name name of the module to convert
 * 
 * @return false if something went wrong
 */
bool simd_memory_packing(char *mod_name)
{
    bool failed = false;
        /* get the resources */
        statement mod_stmt = (statement)db_get_memory_resource(DBR_CODE, mod_name, TRUE);
        set_current_module_statement(mod_stmt);
        set_current_module_entity(module_name_to_entity(mod_name));

        /* first step : create a vector <> array table */
        hash_table array_to_vector = hash_table_make(hash_pointer,0);
        gen_context_recurse(mod_stmt, array_to_vector, statement_domain, gen_true, simd_trace_call);
        if(failed=hash_table_empty_p(array_to_vector))
        {
            pips_user_warning("I did not find any simd load / store operation :'(\n");
            goto simd_memory_packing_end;
        }

        /* second step : look for any array ``vectorized'' in the parameter
         * and modify their type to fit the real array size
         */
        bool replaced_something = simd_replace_parameters(array_to_vector);
        if( failed = !replaced_something )
        {
            pips_user_warning("I did not find any vectorized array in module parameters :'(\n");
            goto simd_memory_packing_end;
        }
        entity ee =module_name_to_entity(mod_name);
        add_new_module_from_text( mod_name,
                text_module(get_current_module_entity(),get_current_module_statement()),
                fortran_module_p(get_current_module_entity())
        );
        /* update/release resources */
        hash_table_free(array_to_vector);

        reset_current_module_statement();
        reset_current_module_entity();

simd_memory_packing_end:
    return !failed;
}
