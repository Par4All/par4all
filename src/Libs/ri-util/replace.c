#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"
#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

/** @defgroup entity replacement in statements
 * it is not unusual to generate a new entity,
 * this helper functions will take care of substituing all referenced to
 * an old entity by reference in new entities
 * @{
 */

struct entity_pair
{
    entity old;
    entity new;
};

/** @brief substitute `thecouple->new' to `thecouple->old' in `exp'
 *  only works if `exp' is a reference
 */
static void
replace_entity_expression_walker(expression exp, struct entity_pair* thecouple)
{
    if( expression_reference_p(exp) )
    {
        reference ref = syntax_reference(expression_syntax(exp));
        entity referenced_entity = reference_variable(ref);
		string emn_r = strdup(entity_module_name(referenced_entity));
		string emn_o = entity_module_name(thecouple->old);
		string eun_r = entity_user_name(referenced_entity);
		string eun_o = entity_user_name(thecouple->old);
        if( same_string_p(emn_r,emn_o) && same_string_p(eun_r,eun_o))
        {
            if( entity_constant_p(thecouple->new) )
            {
                expression_syntax(exp) = make_syntax_call(make_call(thecouple->new,NIL));
            }
            else
            {
                reference_variable(ref) = thecouple->new;
            }
        }
		free(emn_r);
    }
}


static void
replace_entity_declaration_walker(statement s, struct entity_pair* thecouple)
{
    FOREACH(ENTITY,decl_ent,statement_declarations(s))
    {
        value v = entity_initial(decl_ent);
        if( !value_undefined_p(v) && value_expression_p( v ) )
            gen_context_recurse( v, thecouple, expression_domain, gen_true, replace_entity_expression_walker);
    }
}

static void replace_entity_loop_walker(loop l, struct entity_pair* thecouple)
{
	string emn_l = entity_module_name(loop_index(l));
	string emn_o = entity_module_name(thecouple->old);
	string eun_l = entity_user_name(loop_index(l));
	string eun_o = entity_user_name(thecouple->old);
	if( same_string_p(emn_l,emn_o) && same_string_p(eun_l,eun_o))
	{
		loop_index(l) = thecouple->new;
	}

}

/** 
 * @brief recursievly substitute new to old in s
 * 
 * @param s newgen type where the substitution must be done
 * @param old old entity
 * @param new new entity
 */
void
replace_entity(void* s, entity old, entity new)
{
    struct entity_pair thecouple = { old, new };

    gen_context_multi_recurse( s, &thecouple, expression_domain, gen_true, replace_entity_expression_walker,
            statement_domain,gen_true, replace_entity_declaration_walker,
			loop_domain,gen_true, replace_entity_loop_walker,
			NULL);

}

void
replace_entities(void* s, hash_table ht)
{
    HASH_MAP(k, v, replace_entity(s,(entity)k,(entity)v);, ht);
}

void
replace_reference(void* s, reference old, entity new)
{
    /* if the reference is a scalar, it's similar to replace_entity, otherwise, it's replace_entity_by_expression */
    if( ENDP(reference_indices(old)) )
        replace_entity(s,reference_variable(old),new);
    else {
        expression e = make_expression(make_syntax_reference(copy_reference(old)),normalized_undefined);
        replace_entity_by_expression(s,e,new);
        free_expression(e);
    }
}

struct param { entity ent; expression exp; };
static
void replace_entity_by_expression_expression_walker(expression e, struct param *p)
{
    if( expression_reference_p(e) )
    {
        reference r = expression_reference(e);
        if(same_entity_p(p->ent, reference_variable(r) ))
        {
            free_syntax(expression_syntax(e));
            expression_syntax(e) = copy_syntax(expression_syntax(p->exp));
            unnormalize_expression(e);
        }
    }
}

static
void replace_entity_by_expression_entity_walker(entity e, struct param *p)
{
    value v = entity_initial(e);
    if( value_expression_p(v) )
        gen_context_recurse(value_expression(v),p,expression_domain,gen_true,replace_entity_by_expression_expression_walker);
}


static
void replace_entity_by_expression_declarations_walker(statement s, struct param *p)
{
    FOREACH(ENTITY,e,statement_declarations(s))
        replace_entity_by_expression_entity_walker(e,p);

}

static
void replace_entity_by_expression_loop_walker(loop l, struct param *p)
{
    replace_entity_by_expression_entity_walker(loop_index(l),p);
}

/** 
 * replace all reference to entity @a ent by expression @e exp
 * in @a s. @s can be any newgen type !
 */
void
replace_entity_by_expression(void* s, entity ent, expression exp)
{
    struct param p = { ent, exp };
    gen_context_multi_recurse(s,&p,
            expression_domain,gen_true,replace_entity_by_expression_expression_walker,
            statement_domain,gen_true,replace_entity_by_expression_declarations_walker,
			loop_domain,gen_true, replace_entity_by_expression_loop_walker,
            NULL);
}
void
replace_entities_by_expression(void* s, hash_table ht)
{
    HASH_MAP(k, v, replace_entity_by_expression(s,(entity)k,(expression)v);, ht);
}
/** @} */
