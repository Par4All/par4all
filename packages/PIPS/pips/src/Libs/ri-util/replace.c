#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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

static void
replace_entities_declaration_walker(statement s, hash_table ht)
{
  // FI: this does not seem to replace the declaration itself
  // I need a side-effect on the statement_declarations list when
  // decl_ent == thecouple->old
    FOREACH(ENTITY,decl_ent,statement_declarations(s))
        replace_entities(decl_ent,ht);
}

static void
replace_entities_call_walker(call c, hash_table ht) {
    entity new = hash_get(ht, call_function(c));
    if( new != HASH_UNDEFINED_VALUE) {
        call_function(c)=new;
    }
}

static void
replace_entities_expression_walker(expression parent, hash_table ht)
{
    if(expression_reference_p(parent)) {
        reference r = expression_reference(parent);
        entity new = hash_get(ht,reference_variable(r));
        if(new!= HASH_UNDEFINED_VALUE) {
            if(entity_constant_p(new)) {
                update_expression_syntax(parent,
                        make_syntax_call(make_call(new,NIL))
                        );
            }
            else
                reference_variable(r)=new;
            expression next=parent,root=parent;
            /* we need to unormalize the uppermost parent of this expression
             * otherwise its normalized field gets incorrect */
            while((next=(expression) gen_get_ancestor(expression_domain,next)))
                root=next;
            unnormalize_expression(root); /* otherwise field normalized get wrong */
        }
    }
}

static void replace_entities_loop_walker(loop l, hash_table ht) {
    entity new = hash_get(ht,loop_index(l));
    if(new!= HASH_UNDEFINED_VALUE) {
        // We replace the loop index
        loop_index(l) = new;
    }

    // Handle loop locals, in-place replacement
    for(list ll = loop_locals(l); !ENDP(ll); POP(ll)) {
        entity local = ENTITY(CAR(ll));
        entity new = hash_get(ht,local);
        if(new!= HASH_UNDEFINED_VALUE) {
            CAR(ll).p = (gen_chunkp)new;
        }
    }
}

/**
 * @brief Recursively substitute a set of entities in a statement
 *
 * @param s newgen type where the substitution must be done
 * @param ht table of old->new pairs
 *
 * Warning: if the intersection between the input and output sets is not empty, the 
 * behavior is undefined
 */
void
replace_entities(void* s, hash_table ht)
{
  if( INSTANCE_OF(entity,(gen_chunkp)s) ) {
      replace_entities(entity_type((entity)s),ht);
      value v = entity_initial((entity)s);
      if( !value_undefined_p(v) && value_expression_p( v ) )
          replace_entities(v,ht);
  }
  else {
      gen_context_multi_recurse(s, ht,
              expression_domain, gen_true, replace_entities_expression_walker,
              statement_domain, gen_true, replace_entities_declaration_walker,
              call_domain, gen_true, replace_entities_call_walker,
              loop_domain, gen_true, replace_entities_loop_walker,
              NULL);
  }
}

/* per variable version of replace_entities.
 * Much slower if you have an hash table of old/new pairs than the replace_entities version
 */
void
replace_entity(void* s, entity old, entity new) {
    hash_table ht = hash_table_make(hash_pointer,1);
    hash_put(ht,old,new);
    replace_entities(s,ht);
    hash_table_free(ht);
}


/** Replace an old reference by a reference to a new entity in a statement
 */
void
replace_reference(void* s, reference old, entity new) {
  /* If the reference is a scalar, it's similar to replace_entity,
     otherwise, it's replace_entity_by_expression */
  if (ENDP(reference_indices(old)))
    replace_entity(s, reference_variable(old), new);
  else {
    pips_internal_error("not implemented yet");
  }
}


struct param { entity ent; expression exp; set visited_entities;};
static
void replace_entity_by_expression_expression_walker(expression e, struct param *p)
{
    if( expression_reference_p(e) )
    {
        reference r = expression_reference(e);
        if(same_entity_p(p->ent, reference_variable(r) ))
        {
            syntax syn = copy_syntax(expression_syntax(p->exp));
            if(!ENDP(reference_indices(r))) {
                list indices = gen_full_copy_list(reference_indices(r));
                /* if both are references , concatenante indices */
                if(expression_reference_p(p->exp))
                {
                    reference pr = expression_reference(p->exp);
                    syn = make_syntax_reference(
                            make_reference(
                                reference_variable(pr),
                                gen_nconc(gen_full_copy_list(reference_indices(pr)),indices)
                                )
                            );
                }
                /* else generate a subscript */
                else {
                    syn = make_syntax_subscript(make_subscript(make_expression(syn,normalized_undefined),indices));
                }
            }
            update_expression_syntax(e,syn);
        }
    }
    else if(expression_call_p(e))
    {
        call c = expression_call(e);
        if(call_constant_p(c) && same_entity_p(p->ent,call_function(c)))
            update_expression_syntax(e,copy_syntax(expression_syntax(p->exp)));

    }
}

static
void replace_entity_by_expression_entity_walker(entity e, struct param *p)
{
    if( ! set_belong_p(p->visited_entities,e)) {
        set_add_element(p->visited_entities,p->visited_entities,e);
        value v = entity_initial(e);
        if( value_expression_p(v) )
            gen_context_recurse(value_expression(v),p,expression_domain,gen_true,replace_entity_by_expression_expression_walker);
        gen_context_recurse(entity_type(e),p,expression_domain,gen_true,replace_entity_by_expression_expression_walker);
    }
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
void
replace_entity_by_expression_with_filter(void* s, entity ent, expression exp,bool (*filter)(expression))
{
    struct param p = { ent, exp, set_make(set_pointer)};
    gen_context_multi_recurse(s,&p,
            expression_domain,filter,replace_entity_by_expression_expression_walker,
            statement_domain,gen_true,replace_entity_by_expression_declarations_walker,
			loop_domain,gen_true, replace_entity_by_expression_loop_walker,
            NULL);
    set_free(p.visited_entities);
}

/** 
 * replace all reference to entity @a ent by expression @e exp
 * in @a s. @s can be any newgen type !
 */
void
replace_entity_by_expression(void* s, entity ent, expression exp)
{
    replace_entity_by_expression_with_filter(s,ent,exp,(bool(*)(expression))gen_true);
}
void
replace_entities_by_expression(void* s, hash_table ht)
{
    HASH_MAP(k, v, replace_entity_by_expression(s,(entity)k,(expression)v);, ht);
}
/** @} */
