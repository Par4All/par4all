/*

  $Id: outlining.c -1   $

  Copyright 1989-2010 MINES ParisTech

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
/**
 * @file outlining.c
 * @brief add outlining support to pips, with two flavors
 *
 * @author Serge Guelton <serge.guelton@enst-bretagne.fr>
 * @date 2009-01-07
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "resources.h"
#include "properties.h"
#include "misc.h"
#include "control.h"
#include "callgraph.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "preprocessor.h"
#include "expressions.h"
#include "text-util.h"
#include "parser_private.h"
#include "accel-util.h"
/**
 * @name outlining
 * @{ */

#define STAT_ORDER "PRETTYPRINT_STATEMENT_NUMBER"

static
void patch_outlined_reference(expression x, entity e)
{
    if(expression_reference_p(x))
    {
        reference r =expression_reference(x);
        entity e1 = reference_variable(r);
        if(same_entity_p(e,e1))
        {
			variable v = type_variable(entity_type(e));
			if( (!basic_pointer_p(variable_basic(v))) && (ENDP(variable_dimensions(v))) ) /* not an array / pointer */
			{
				expression X = make_expression(expression_syntax(x),normalized_undefined);
				expression_syntax(x)=make_syntax_call(make_call(
							CreateIntrinsic(DEREFERENCING_OPERATOR_NAME),
							CONS(EXPRESSION,X,NIL)));
			}
        }
    }

}

static
void patch_outlined_reference_in_declarations(statement s, entity e)
{
    FOREACH(ENTITY, ent, statement_declarations(s))
    {
        value v = entity_initial(ent);
        if(!value_undefined_p(v) && value_expression_p(v))
            gen_context_recurse(value_expression(v),e,expression_domain,gen_true,patch_outlined_reference);
    }
}

static
void bug_in_patch_outlined_reference(loop l , entity e)
{
    if( same_entity_p(loop_index(l), e))
    {
        statement parent = (statement)gen_get_ancestor(statement_domain,l);
        pips_assert("child's parent child is me",statement_loop_p(parent) && statement_loop(parent)==l);
        statement body =loop_body(l);
        range r = loop_range(l);
        instruction new_instruction = make_instruction_forloop(
                    make_forloop(
                        make_assign_expression(
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_lower(r)),
                        MakeBinaryCall(entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_upper(r)),
                        MakeBinaryCall(entity_intrinsic(PLUS_UPDATE_OPERATOR_NAME),
                            MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(e)),
                            range_increment(r)),
                        body
                        )
                    );

        statement_instruction(parent)=instruction_undefined;/*SG this is a small leak */
        update_statement_instruction(parent,new_instruction);
        pips_assert("statement consistent",statement_consistent_p(parent));
    }
}

static
void get_private_entities_walker(loop l, set s)
{
    set_append_list(s,loop_locals(l));
}

static
set get_private_entities(void *s)
{
    set tmp = set_make(set_pointer);
    gen_context_recurse(s,tmp,loop_domain,gen_true,get_private_entities_walker);
    return tmp;
}

static
void sort_entities_with_dep(list *l)
{
    set params = set_make(set_pointer);
    FOREACH(ENTITY,e,*l)
    {
        set e_ref = get_referenced_entities(e);
        set_del_element(e_ref,e_ref,e);
        set_union(params,params,e_ref);
        set_free(e_ref);
    }

    set base = set_make(set_pointer);
    set_assign_list(base,*l);
    set_difference(base,base,params);

    list l_params = set_to_sorted_list(params,(gen_cmp_func_t)compare_entities);
    list l_base = set_to_sorted_list(base,(gen_cmp_func_t)compare_entities);

    set_free(base);set_free(params);
    gen_free_list(*l);

    *l=gen_nconc(l_params,l_base);
}

struct cpv {
    entity e;
    bool rm;
};


static
void check_private_variables_call_walker(call c,struct cpv * p)
{
    set s = get_referenced_entities(c);
    if(set_belong_p(s,p->e)){
        p->rm=true;
        gen_recurse_stop(0);
    }
    set_free(s);
}
static
bool check_private_variables_loop_walker(loop l, struct cpv * p)
{
    return !has_entity_with_same_name(p->e,loop_locals(l));
}

static
list private_variables(statement stat)
{
    set s = get_private_entities(stat);
    list l =NIL;
    SET_FOREACH(entity,e,s) {
        struct cpv p = { .e=e, .rm=false };
        gen_context_multi_recurse(stat,&p,
                call_domain,gen_true,check_private_variables_call_walker,
                loop_domain,check_private_variables_loop_walker,gen_null,
                0);
        if(!p.rm)
            l=CONS(ENTITY,e,l);
    }
    set_free(s);

    return l;
}

typedef struct {
    entity old;
    entity new;
    size_t nb_dims;
} ocontext_t;
static void do_outliner_smart_replacment(reference r, ocontext_t * ctxt)
{
    if(same_entity_p(ctxt->old,reference_variable(r)))
    {
        size_t nb_dims = ctxt->nb_dims;
        list indices = reference_indices(r);
        while (nb_dims--) POP(indices); 
        indices=gen_full_copy_list(indices);
        expression parent = (expression)gen_get_ancestor(expression_domain,r);
        unnormalize_expression(parent);
        if(basic_pointer_p(entity_basic(ctxt->new))) /*sg:may cause issues if basic_pointer_p(old) ? */
        {
            pips_assert("parent exist",parent);
            //free_syntax(expression_syntax(parent)); /* sg a small leak is better than a crash :) */
            if(!ENDP(indices))
                expression_syntax(parent)=
                    make_syntax_subscript(
                            make_subscript(
                                MakeUnaryCall(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),entity_to_expression(ctxt->new)),
                                indices)
                            );
            else
                expression_syntax(parent)=
                    make_syntax_call(
                            make_call(entity_intrinsic(DEREFERENCING_OPERATOR_NAME),make_expression_list(entity_to_expression(ctxt->new)))
                            );

        }
        else {
            reference_variable(r)=ctxt->new;
            gen_full_free_list(reference_indices(r));
            reference_indices(r)=indices;
        }
    }
}

static void outliner_smart_replacment(statement in, entity old, entity new,size_t nb_dims)
{
    ocontext_t ctxt = { old,new,nb_dims };
    gen_context_recurse(in,&ctxt,reference_domain,gen_true,do_outliner_smart_replacment);
}

/**
 * purge the list of referenced entities by replacing calls to a[i][j] where i is a constant in statements
 * outlined_statements by a call to a single (new) variable
 */
static hash_table outliner_smart_references_computation(list referenced_entities, list outlined_statements,entity new_module, statement new_body)
{
    /* this will hold new referenced_entities list */
    hash_table entity_to_init = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    /* first check candidates, that is array entities accessed by a constant index */
    FOREACH(ENTITY,e,referenced_entities)
    {
        FOREACH(STATEMENT,st,outlined_statements)
        {
            list regions = load_rw_effects_list(st);
            list the_constant_indices = NIL;
            action mode = action_undefined;
            FOREACH(REGION,reg,regions)
            {
                reference rr = region_any_reference(reg);
                if(same_entity_p(e,reference_variable(rr)))
                {
                    list constant_indices = NIL;
                    Psysteme sc = region_system(reg);
                    FOREACH(EXPRESSION,index,reference_indices(rr))
                    {
                        Variable phi = expression_to_entity(index);
                        expression index_value = expression_undefined;
                        /* we are looking for constant index, so only check equalities */
                        for(Pcontrainte iter = sc_egalites(sc);iter;iter=contrainte_succ(iter))
                        {
                            Pvecteur cvect = contrainte_vecteur(iter);
                            Value phi_coeff = vect_coeff(phi,cvect);
                            if(phi_coeff != VALUE_ZERO )
                            {
                                pips_assert("phi coeff sould be one",phi_coeff == VALUE_ONE);
                                Pvecteur lhs_vect = vect_del_var(cvect,phi);
                                vect_chg_sgn(lhs_vect);
                                pips_assert("phi coef should be mentionned only once",expression_undefined_p(index_value));
                                index_value = VECTEUR_NUL_P(lhs_vect) ? int_to_expression(0) : Pvecteur_to_expression(lhs_vect);
                                vect_rm(lhs_vect);
                            }
                        }
                        if(!expression_undefined_p(index_value))
                        {
                            /* it's ok, we can keep on finding constant indices */
                            constant_indices=CONS(EXPRESSION,index_value,constant_indices);
                        }
                        else break;
                    }
                    constant_indices=gen_nreverse(constant_indices);
                    /* check for clashes */
                    if(!ENDP(the_constant_indices) && ! gen_equals(constant_indices,the_constant_indices,(gen_eq_func_t)same_expression_p))
                    {
                        /* abort there , we could be smarter */
                        gen_full_free_list(the_constant_indices);
                        gen_full_free_list(constant_indices);
                        the_constant_indices=constant_indices=NIL;
                        break;
                    }
                    else if( ENDP(the_constant_indices) )
                    {
                        the_constant_indices=constant_indices;
                        mode=region_action(reg);
                    }
                    else if( action_read_p(mode) && action_write_p(region_action(reg)))
                    {
                        mode =region_action(reg);
                    }
                }
            }
            /* we have gathered a sub array of e that is constant and we know its mode
             * get ready for substitution in the statement */
            if(!ENDP(the_constant_indices))
            {
                size_t nb_constant_indices = gen_length(the_constant_indices);
                list entity_dimensions = variable_dimensions(type_variable(entity_type(e)));
                size_t nb_dimensions = gen_length(entity_dimensions);

                /* compute new dimensions */
                list new_dimensions = NIL;
                size_t count_dims = 0;
                for(list iter = entity_dimensions;!ENDP(iter);POP(iter))
                {
                    ++count_dims;
                    if(count_dims==nb_constant_indices) { new_dimensions=gen_full_copy_list(CDR(iter));break; }
                }


                basic new_basic;
                if(action_read_p(mode)&&nb_constant_indices==nb_dimensions)
                    new_basic=copy_basic(entity_basic(e));
                else
                {
                    type new_type = make_type_variable(
                            make_variable(
                                copy_basic(entity_basic(e)),
                                new_dimensions,
                                gen_full_copy_list(variable_qualifiers(type_variable(entity_type(e))))
                                )
                            );
                    new_basic=make_basic_pointer(new_type);
                }

                entity new_entity;
                if(action_read_p(mode)&&nb_constant_indices==nb_dimensions)
                {
                    new_entity = make_new_array_variable_with_prefix(
                            entity_user_name(e),
                            new_module,
                            new_basic,
                            new_dimensions);
                }
                else
                {
                    new_entity = make_new_scalar_variable_with_prefix(
                            entity_user_name(e),
                            new_module,
                            new_basic);
                }
                outliner_smart_replacment(st,e,new_entity,nb_constant_indices);
                expression effective_parameter = reference_to_expression(make_reference(e,the_constant_indices));
                if(!(action_read_p(mode)&&nb_constant_indices==nb_dimensions))
                    effective_parameter=MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),effective_parameter);
                hash_put(entity_to_init,new_entity,effective_parameter);
            }
        }
    }
    return entity_to_init;
}
static void statements_localize_declarations(list statements,entity module,statement module_statement)
{
    list sd = statements_to_declarations(statements);
    FOREACH(STATEMENT, s, statements)
    {
        /* We want to declare private variables as locals, but it may not
           be valid */
        list private_ents = private_variables(s);
        gen_sort_list(private_ents,(gen_cmp_func_t)compare_entities);
        FOREACH(ENTITY,e,private_ents)
        {
            if(gen_chunk_undefined_p(gen_find_eq(e,sd))) {
                AddLocalEntityToDeclarations(e,module,module_statement);
            }
        }
        gen_free_list(private_ents);
    }
    gen_free_list(sd);
}
static list statements_referenced_entities(list statements)
{
    list referenced_entities = NIL;
    set sreferenced_entities = set_make(set_pointer);

    FOREACH(STATEMENT, s, statements)
    {
        set tmp = get_referenced_entities(s);
        set_union(sreferenced_entities,tmp,sreferenced_entities);
        set_free(tmp);
    }
    /* set to list */
    referenced_entities=set_to_list(sreferenced_entities);
    set_free(sreferenced_entities);
    return referenced_entities;
}
static void outliner_extract_loop_bound(statement sloop, hash_table entity_to_effective_parameter)
{
    loop l =statement_loop(sloop);
    range r = loop_range(l);
    expression upper = range_upper(r);
    if(!expression_scalar_p(upper))
    {
        basic b = basic_of_expression(upper);
        if(basic_overloaded_p(b)) { free_basic(b); b = make_basic_int(DEFAULT_INTEGER_TYPE_SIZE); }
        entity holder = make_new_scalar_variable(get_current_module_entity(),b);
        hash_put(entity_to_effective_parameter,holder,upper);
        range_upper(r)=entity_to_expression(holder);
    }
}
static void convert_pointer_to_array_aux(expression exp,entity e) {
    if(expression_reference_p(exp)) {
        reference r = expression_reference(exp);
        if(same_entity_p(reference_variable(r),e)) {
            syntax syn = expression_syntax(exp);
            expression_syntax(exp)=syntax_undefined;
            syntax syn2=make_syntax_call(
                    make_call(
                        entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                        CONS(EXPRESSION,
                            make_expression(
                                syn,normalized_undefined
                                ),
                            NIL)
                        )
                    );
            update_expression_syntax(exp,syn2);
        }
    }
}
static void convert_pointer_to_array_aux2(statement s, entity e){
    FOREACH(ENTITY,en,statement_declarations(s))
    gen_context_recurse(entity_initial(en),e,expression_domain,gen_true,convert_pointer_to_array_aux);
}

static void convert_pointer_to_array(entity e,entity re, expression x, list statements) {
    type t =entity_type(e);
    type pointed_type = basic_pointer(
            variable_basic(
                type_variable(t)
                )
            );
    basic_pointer(
            variable_basic(
                type_variable(t)
                )
            )=type_undefined;
    free_type(t);
    entity_type(e)=pointed_type;
    FOREACH(STATEMENT,s,statements) {
        gen_context_multi_recurse(s,re,expression_domain,gen_true,convert_pointer_to_array_aux,
                statement_domain,gen_true,convert_pointer_to_array_aux2,
                NULL);
        cleanup_subscripts(s);
    }

    /* crado */
    syntax syn = expression_syntax(x);
    expression_syntax(x)=syntax_undefined;
    syntax syn2=make_syntax_call(
            make_call(
                entity_intrinsic(DEREFERENCING_OPERATOR_NAME),
                CONS(EXPRESSION,
                    make_expression(
                        syn,normalized_undefined
                        ),
                    NIL)
                )
            );
    update_expression_syntax(x,syn2);
}

/**
 * outline the statements in statements_to_outline into a module named outline_module_name
 * the outlined statements are replaced by a call to the newly generated module
 * statements_to_outline is modified in place to represent that call
 *
 * @param outline_module_name name of the new module

 * @param statements_to_outline is a list of consecutive statements to
 * outline into outline_module_name
 *
 * @return pointer to the newly generated statement (already inserted in statements_to_outline)
 */
statement outliner(string outline_module_name, list statements_to_outline)
{
    pips_assert("there are some statements to outline",!ENDP(statements_to_outline));
    entity new_fun = make_empty_subroutine(outline_module_name,copy_language(module_language(get_current_module_entity())));
    statement new_body = instruction_to_statement(make_instruction_sequence(make_sequence(statements_to_outline)));


    /* Retrieve referenced entities */
    list referenced_entities = statements_referenced_entities(statements_to_outline);
    /* try to be smart concerning array references */
    hash_table entity_to_effective_parameter = hash_table_undefined;
    if(get_bool_property("OUTLINE_SMART_REFERENCE_COMPUTATION"))
    {
        entity_to_effective_parameter = outliner_smart_references_computation(referenced_entities,statements_to_outline,new_fun,new_body);
        /*and recompute referenced entities*/
        gen_free_list(referenced_entities);
        referenced_entities = statements_referenced_entities(statements_to_outline);
    }
    else
        entity_to_effective_parameter = hash_table_make(hash_pointer,1);
    /* pass loop bounds as parameters if required */
    string loop_label = get_string_property("OUTLINE_LOOP_BOUND_AS_PARAMETER");
    statement theloop = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),loop_label);
    if(!statement_undefined_p(theloop) && statement_loop(theloop))
    {
        outliner_extract_loop_bound(theloop,entity_to_effective_parameter);
        /*and recompute referenced entities*/
        gen_free_list(referenced_entities);
        referenced_entities = statements_referenced_entities(statements_to_outline);
    }

    /* Retrieve declared entities */
    statements_localize_declarations(statements_to_outline,new_fun,new_body);
    list declared_entities = statements_to_declarations(statements_to_outline);
    declared_entities=gen_nconc(declared_entities,statement_to_declarations(new_body));
    
    /* get the relative complements and create the parameter list*/
    gen_list_and_not(&referenced_entities,declared_entities);
    gen_free_list(declared_entities);


    /* purge the functions from the parameter list, we assume they are declared externally
     * also purge the formal parameters from other modules, gathered by get_referenced_entities but wrong here
     * also purge memebers, not relevant
     */
    list tmp_list=NIL;
    FOREACH(ENTITY,e,referenced_entities)
    {
        /* function should be added to compilation unit */
        if(entity_function_p(e))
            ;//AddEntityToModuleCompilationUnit(e,get_current_module_entity());
        else if( (!entity_constant_p(e) ) && (!entity_field_p(e)) &&
                !( entity_formal_p(e) && (!same_string_p(entity_module_name(e),get_current_module_name()))) )
            tmp_list=CONS(ENTITY,e,tmp_list);
    }
    gen_free_list(referenced_entities);
    referenced_entities=tmp_list;


    /* remove global variables if needed */
    if(get_bool_property("OUTLINE_ALLOW_GLOBALS"))
    {
        string cu_name = compilation_unit_of_module(get_current_module_name());
        entity cu = module_name_to_entity(cu_name);
        list cu_decls = entity_declarations(cu);

        tmp_list=NIL;

        FOREACH(ENTITY,e,referenced_entities)
        {
            if( !top_level_entity_p(e) && gen_chunk_undefined_p(gen_find_eq(e,cu_decls) ) )
                tmp_list=CONS(ENTITY,e,tmp_list);
            else if (gen_chunk_undefined_p(gen_find_eq(e,cu_decls)))
            {
                AddLocalEntityToDeclarations(e,new_fun,new_body);
            }
        }
        gen_free_list(referenced_entities);
        referenced_entities=tmp_list;
    }

    /* sort list, and put parameters first */
    sort_entities_with_dep(&referenced_entities);




    intptr_t i=0;

	/* all variables are promoted parameters */
    list effective_parameters = NIL;
    list formal_parameters = NIL;
    FOREACH(ENTITY,e,referenced_entities)
    {
        type t = entity_type(e);
        bool is_parameter_p = /* != formal parameter */ (entity_symbolic_p(e) && storage_rom_p(entity_storage(e)) && type_functional_p(t));
        if( type_variable_p(t) || is_parameter_p )
        {
            /* this create the dummy parameter */
            entity dummy_entity = FindOrCreateEntity(
                    outline_module_name,
                    entity_user_name(e)
                    );
            entity_type(dummy_entity)=is_parameter_p?
                make_type_variable(make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL)):
                copy_type(t);
            entity_storage(dummy_entity)=make_storage_formal(make_formal(dummy_entity,++i));


            formal_parameters=CONS(PARAMETER,make_parameter(
                        is_parameter_p?make_type_variable(make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL)):copy_type(t),
                        fortran_module_p(get_current_module_entity())?make_mode_reference():make_mode_value(),
                        make_dummy_identifier(dummy_entity)),formal_parameters);

            /* this adds the effective parameter */
            expression effective_parameter = (expression)hash_get(entity_to_effective_parameter,e);
            if(effective_parameter == HASH_UNDEFINED_VALUE)
                effective_parameter = entity_to_expression(e);

            effective_parameters=CONS(EXPRESSION,effective_parameter,effective_parameters);
        }
        /* this is a constant variable */
        else if(entity_constant_p(e)) {
            AddLocalEntityToDeclarations(e,new_fun,new_body);
        }

    }
    formal_parameters=gen_nreverse(formal_parameters);
    effective_parameters=gen_nreverse(effective_parameters);
    hash_table_free(entity_to_effective_parameter);


    /* we need to patch parameters , effective parameters and body in C
     * because of by copy function call
	 * it's not needed if
	 * - the parameter is only read
	 * - it's an array / pointer
     */
    if(c_module_p(get_current_module_entity()))
    {
		list iter = effective_parameters,
             riter = referenced_entities;
        FOREACH(PARAMETER,p,formal_parameters)
        {
			expression x = EXPRESSION(CAR(iter));
            entity re = ENTITY(CAR(riter));
            entity ex = entity_undefined;
            if(expression_reference_p(x))
                ex = reference_variable(expression_reference(x));

            entity e = dummy_identifier(parameter_dummy(p));
            if(!entity_undefined_p(ex)&& type_variable_p(entity_type(ex)) ) {
                variable v = type_variable(entity_type(ex));
                bool entity_written=false;
                FOREACH(STATEMENT,stmt,statements_to_outline)
                    entity_written|=find_write_effect_on_entity(stmt,ex);

                if( (!basic_pointer_p(variable_basic(v))) &&
                        ENDP(variable_dimensions(v)) &&
                        entity_written
                  )
                {
                    entity_type(e)=make_type_variable(
                            make_variable(
                                make_basic_pointer(copy_type(entity_type(e))),
                                NIL,
                                NIL
                                )
                            );
                    parameter_type(p)=copy_type(entity_type(e));
                    syntax s = expression_syntax(x);
                    expression X = make_expression(s,normalized_undefined);
                    expression_syntax(x)=make_syntax_call(make_call(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),CONS(EXPRESSION,X,NIL)));
                    gen_context_multi_recurse(new_body,ex,
                            statement_domain,gen_true,patch_outlined_reference_in_declarations,
                            loop_domain,gen_true,bug_in_patch_outlined_reference,
                            expression_domain,gen_true,patch_outlined_reference,
                            0);
                }
            }
            if(type_variable_p(entity_type(re))) {
                variable v = type_variable(entity_type(re));
                if( basic_pointer_p(variable_basic(v)) &&
                        array_type_p(basic_pointer(variable_basic(v)))) {
                    convert_pointer_to_array(e,re,x,statements_to_outline);

                }
            }
			POP(iter);
            POP(riter);
        }
		pips_assert("no effective parameter left", ENDP(iter));
    }

    /* prepare parameters and body*/
    functional_parameters(type_functional(entity_type(new_fun)))=formal_parameters;
	FOREACH(PARAMETER,p,formal_parameters) {
		code_declarations(value_code(entity_initial(new_fun))) =
			gen_nconc(
					code_declarations(value_code(entity_initial(new_fun))),
					CONS(ENTITY,dummy_identifier(parameter_dummy(p)),NIL));
	}

    /* we can now begin the outlining */
    bool saved = get_bool_property(STAT_ORDER);
    set_bool_property(STAT_ORDER,false);
    text t = text_named_module(new_fun, new_fun /*get_current_module_entity()*/, new_body);
    add_new_module_from_text(outline_module_name, t, fortran_module_p(get_current_module_entity()), compilation_unit_of_module(get_current_module_name()) );
    set_bool_property(STAT_ORDER,saved);
	/* horrible hack to prevent declaration duplication
	 * signed : Serge Guelton
	 */
	gen_free_list(code_declarations(EntityCode(new_fun)));
	code_declarations(EntityCode(new_fun))=NIL;

    /* we need to free them now, otherwise recompilation fails */
    FOREACH(PARAMETER,p,formal_parameters) {
        entity e = dummy_identifier(parameter_dummy(p));
        if(entity_variable_p(e)) {
            free_type(entity_type(e));
            entity_type(e)=type_undefined;
        }
    }

    /* and return the replacement statement */
    instruction new_inst =  make_instruction_call(make_call(new_fun,effective_parameters));
    statement new_stmt = statement_undefined;

    /* perform substitution :
     * replace the original statements by a single call
     * and patch the remaining statement (yes it's ugly)
     */
    FOREACH(STATEMENT,old_statement,statements_to_outline)
    {
        free_instruction(statement_instruction(old_statement));
        if(statement_undefined_p(new_stmt))
        {
            statement_instruction(old_statement)=new_inst;
            new_stmt=old_statement;
        }
        else
            statement_instruction(old_statement)=make_continue_instruction();
        gen_free_list(statement_declarations(old_statement));
        statement_declarations(old_statement)=NIL;
        /* trash any extensions, they may not be valid now */
        free_extensions(statement_extensions(old_statement));
        statement_extensions(old_statement)=empty_extensions();

    }
    free_text(t);
    return new_stmt;
}



/**
 * @brief entry point for outline module
 * outlining will be performed using either comment recognition
 * or interactively
 *
 * @param module_name name of the module containg the statements to outline
 */
bool
outline(char* module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
 	set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
 	set_rw_effects((statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, TRUE));

    /* retrieve name of the outiled module */
    string outline_module_name = get_string_property_or_ask("OUTLINE_MODULE_NAME","outline module name ?\n");

    /* retrieve statement to outline */
    list statements_to_outline = find_statements_with_pragma(get_current_module_statement(),OUTLINE_PRAGMA) ;
    if(ENDP(statements_to_outline)) {
        string label_name = get_string_property("OUTLINE_LABEL");
        if( empty_string_p(label_name) ) {
            statements_to_outline=find_statements_interactively(get_current_module_statement());
        }
        else  {
            statement statement_to_outline = find_statement_from_label_name(get_current_module_statement(),get_current_module_name(),label_name);
            if(statement_loop_p(statement_to_outline) && get_bool_property("OUTLINE_LOOP_STATEMENT"))
                statement_to_outline=loop_body(statement_loop(statement_to_outline));
            statements_to_outline=make_statement_list(statement_to_outline);
        }
    }

    /* apply outlining */
    (void)outliner(outline_module_name,statements_to_outline);


    /* validate */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,
			   get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
	reset_cumulated_rw_effects();
    reset_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();

    return true;
}
/**  @} */

