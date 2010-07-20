/*

  $Id$

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "text-util.h"
#include "effects-simple.h"

#include "sac.h"
#include "patterns.h"

#include "properties.h"

#include "misc.h"
#include <ctype.h>
#include <stdlib.h>

#define MAX_PACK 16
#define VECTOR_POSTFIX "_vec"


static float gSimdCost;

static entity _padding_entity_ = entity_undefined;

void init_padding_entities() {
    pips_assert("no previously defined table",entity_undefined_p(_padding_entity_));
}
void reset_padding_entities() {
    _padding_entity_=entity_undefined;
}

static entity get_padding_entity() {
    if(entity_undefined_p(_padding_entity_))
    {
        _padding_entity_=make_scalar_entity(SAC_PADDING_ENTITY_NAME,get_current_module_name(),make_basic_overloaded());
        AddEntityToCurrentModule(_padding_entity_);
    }
    return _padding_entity_;
}

static hash_table vector_to_expressions = hash_table_undefined;
void init_vector_to_expressions()
{
    pips_assert("reset was called",hash_table_undefined_p(vector_to_expressions));
    vector_to_expressions=hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
}
void reset_vector_to_expressions()
{
    pips_assert("init was called",!hash_table_undefined_p(vector_to_expressions));
    hash_table_free(vector_to_expressions);
    vector_to_expressions=hash_table_undefined;
}

static entity expressions_to_vector(list expressions)
{
    void * hiter = NULL;
    entity key;
    list value;
    while( (hiter = hash_table_scan(vector_to_expressions,hiter,(void**)&key,(void**)&value) ))
    {
        if(gen_equals(value,expressions,(gen_eq_func_t)expression_equal_p))
            return key;
    }
    return entity_undefined;
}
static void update_vector_to_expressions(entity e, list exps)
{
    pips_assert("entity is ok",entity_consistent_p(e));
    FOREACH(EXPRESSION,exp,exps) pips_assert("expressions are ok",expression_consistent_p(exp));
    hash_put_or_update(vector_to_expressions,e,gen_full_copy_list(exps));
}

void invalidate_expressions_in_statement(statement s)
{
    list effects = load_cumulated_rw_effects_list( s );
    void * hiter = NULL;
    entity key;
    list value;
    list purge = NIL;
    while( (hiter = hash_table_scan(vector_to_expressions,hiter,(void**)&key,(void**)&value) ))
    {
        bool invalidate=false;
        FOREACH(EXPRESSION,v,value)
        {
            if(expression_reference_p(v))
            {
                FOREACH(EFFECT,eff,effects)
                    if((invalidate=references_may_conflict_p(expression_reference(v),effect_any_reference(eff))))
                    {
                        gen_full_free_list(value);
                        break;
                    }
            }
            if(invalidate) break;
        }
        if(invalidate) purge=CONS(ENTITY,key,purge);
    }
    FOREACH(ENTITY,e,purge)
        hash_del(vector_to_expressions,e);
    gen_free_list(purge);
}

bool expression_reference_or_field_p(expression e)
{
    return expression_reference_p(e) || expression_field_p(e);
}

bool simd_vector_entity_p(entity e)
{
    return 
        strstr(entity_user_name(e),VECTOR_POSTFIX)
        && type_variable_p(entity_type(e))
        && !entity_scalar_p(e);
    ;
}
/* expression is an simd vector
 * if it is a reference array
 * containing VECTOR_POSTFIX in its name
 */
static
bool simd_vector_expression_p(expression e)
{
    syntax s = expression_syntax(e);
    return syntax_reference_p(s) && simd_vector_entity_p(reference_variable(syntax_reference(s)));
}


/* a padding statement as all its expression set to 1, a constant, 
 * event the fist one, that is the lhs of the assignement */

static list padded_simd_statement_p(simdstatement ss)
{
    expression * args = simdstatement_arguments(ss);
    /* we have to check the assignment expression, if it is a constant, then it results from padding */
    list padded_statements = NIL;
    int vs = opcode_vectorSize(simdstatement_opcode(ss));
    for(int i=vs*simdstatement_nbArgs(ss)-1;i>=0;i-=vs)
    {
        expression arg = args[i];
        padded_statements=CONS(BOOL,
                expression_constant_p(arg) ||
                (expression_reference_p(arg)&&same_string_p(entity_user_name(reference_variable(expression_reference(arg))),SAC_PADDING_ENTITY_NAME)),
                padded_statements);
    }
    pips_assert("processed all arguments",gen_length(padded_statements)==(size_t)simdstatement_nbArgs(ss));
    return padded_statements;
}

static bool all_padded_p(list padding)
{
    for(list iter=padding;!ENDP(iter);POP(iter))
    {
        bool b = BOOL(CAR(iter));
        if(!b)
            return false;
    }
    return true;
}

static void patch_all_padded_simd_statements(simdstatement ss)
{
    list paddings = padded_simd_statement_p(ss);
    if(all_padded_p(paddings))
    {
        expression * args = simdstatement_arguments(ss);
        int vs = opcode_vectorSize(simdstatement_opcode(ss));
        for(int i=vs*simdstatement_nbArgs(ss)-1;i>=0;i-=vs)
        {
            free_syntax(expression_syntax(args[i]));
            expression_syntax(args[i])=make_syntax_reference(make_reference(get_padding_entity(),NIL));
        }

    }
    gen_free_list(paddings);
}


/*
   This function return the basic corresponding to the argNum-th
   argument of opcode oc
   */
static enum basic_utype get_basic_from_opcode(opcode oc, int argNum)
{
    int type = INT(gen_nth(argNum, opcode_argType(oc)));

    switch(type)
    {
        case QI_REF_TOK:
        case HI_REF_TOK:
        case SI_REF_TOK:
        case DI_REF_TOK:
            return is_basic_int;
        case SF_REF_TOK:
        case DF_REF_TOK:
            return is_basic_float;
        case SC_REF_TOK:
        case DC_REF_TOK:
            return is_basic_complex;
        default:
            pips_internal_error("subword size unknown.\n");
    }

    return is_basic_int;
}


int get_subwordSize_from_opcode(opcode oc, int argNum)
{
    int type = INT(gen_nth(argNum, opcode_argType(oc)));

    switch(type)
    {
        case QI_REF_TOK:
            return 8;
        case HI_REF_TOK:
            return 16;
        case SI_REF_TOK:
            return 32;
        case DI_REF_TOK:
            return 64;
        case SF_REF_TOK:
            return 32;
        case DF_REF_TOK:
            return 64;
        case SC_REF_TOK:
            return 64;
        case DC_REF_TOK:
            return 128;
        default:
            pips_internal_error("subword size unknown.\n");
    }

    return 8;
}

/* auto-guess vector size */
opcode generate_opcode(string name, list types, float cost)
{
    intptr_t elem_size=0,vector_size = get_int_property("SAC_SIMD_REGISTER_WIDTH");
    opcode oc =  make_opcode(name,vector_size,types,cost);
    int n = gen_length(types);
    for(int i=0; i<n;i++)
    {
        int curr = get_subwordSize_from_opcode(oc,i);
        if(curr > elem_size) elem_size=curr;
    }
    opcode_vectorSize(oc)/=elem_size;
    if( opcode_vectorSize(oc) * elem_size !=get_int_property("SAC_SIMD_REGISTER_WIDTH"))
        pips_user_warning("SAC_SIMD_REGISTER_WIDTH and description of %s leads "
            "to partially filled register\n", name);
    return oc;
}
#if 0
static int get_subwordSize_from_vector(entity vec)
{
    char * name = entity_local_name(vec);

    switch(name[2])
    {
        case 'q':
            return 8;
        case 'h':
            return 16;
        case 's':
            return 32;
        case 'd':
            return 64;
        default:
            pips_internal_error("subword size unknown.\n");
    }

    return 8;
}
#endif

/* Computes the optimal opcode for simdizing 'argc' statements of the
 * 'kind' operation, applied to the 'args' arguments
 * it is a greedy matchinbg, so it supposes the list of args is in the best order for us
 */
static opcode get_optimal_opcode(opcodeClass kind, int argc, list* args)
{
    int i;
    opcode best;
    /* Based on the available implementations of the operation, decide
     * how many statements to pack together
     */
    best = opcode_undefined;
    FOREACH(OPCODE,oc,opcodeClass_opcodes(kind))
    {
        bool bTagDiff = FALSE;

        for(i = 0; i < argc; i++)
        {
            int count = 0;
            int width = 0;

            FOREACH(EXPRESSION,arg,args[i])
            {

                if (!expression_reference_or_field_p(arg))
                {
                    count++;
                    continue;
                }

                basic bas = basic_of_expression(arg);

                if(!basic_overloaded_p(bas) && get_basic_from_opcode(oc, count)!=basic_tag(bas))
                {
                    bTagDiff = TRUE;
                    free_basic(bas);
                    break;
                }

                switch(basic_tag(bas))
                {
                    case is_basic_int: width = 8 * basic_int(bas); break;
                    case is_basic_float: width = 8 * basic_float(bas); break;
                    case is_basic_complex: width= 8 * basic_complex(bas); break;
                    case is_basic_logical: width = 1; break;
                    default: pips_user_error("basic %d not supported",basic_tag(bas));
                }
                free_basic(bas);

                if(width > get_subwordSize_from_opcode(oc, count))
                {
                    bTagDiff = TRUE;
                    break;
                }

                count++;

            }
        }

        if ( (!bTagDiff) &&
                (opcode_vectorSize(oc) <= argc) &&
                ((best == opcode_undefined) || 
                 (opcode_vectorSize(oc) > opcode_vectorSize(best))) )
            best = oc;
    }

    return best;
}

entity get_function_entity(string name)
{
    entity e = local_name_to_top_level_entity(name); 
    if ( entity_undefined_p( e ) )
    {
        pips_user_warning("entity %s not defined, sac is likely to crash soon\n"
                "Please feed pips with its definition and source\n",name);
    }

    return e;
}


/** 
 * computes the offset between an entity and its reference
 * 
 * @param r 
 * 
 * @return an expression of the offset, in octets
 */
static
expression reference_offset(const reference r)
{
    if( ENDP(reference_indices(r)) ) return int_to_expression(0);
    else {
        expression offset = 
            copy_expression(
                    get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION") ?
                    EXPRESSION(CAR(reference_indices(r))):
                    EXPRESSION(CAR(gen_last(reference_indices(r)))));
        list dims = get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION") ?
            CDR(variable_dimensions(type_variable(entity_type(reference_variable(r))))):
            variable_dimensions(type_variable(entity_type(reference_variable(r))));
        list indices = NIL;
        if( get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION"))
            indices=gen_copy_seq(CDR(reference_indices(r)));
        else {
            /* list without last element */
            indices=gen_copy_seq(reference_indices(r));
            gen_remove(&indices,EXPRESSION(CAR(gen_last(indices))));
        }
        FOREACH(EXPRESSION,exp,indices)
        {
            dimension d = DIMENSION(CAR(dims));
            expression d_exp = SizeOfDimension(d);
            offset = make_op_exp(PLUS_OPERATOR_NAME,
                    make_op_exp(MULTIPLY_OPERATOR_NAME,
                        copy_expression(exp),
                        d_exp),
                    offset
                    );
        }
        gen_free_list(indices);
        expression result = make_op_exp(MULTIPLY_OPERATOR_NAME,
                int_to_expression(SizeOfElements(variable_basic(type_variable(entity_type(reference_variable(r)))))),
                offset);
        return result;
    }
}

static expression offset_of_struct(entity e)
{

    entity parent_struct = entity_field_to_entity_struct(e);
    list fields = type_struct(entity_type(parent_struct));
    expression offset = int_to_expression(0);
    FOREACH(ENTITY,field,fields)
        if(same_entity_p(field,e)) break;
        else {
            offset=make_op_exp(PLUS_OPERATOR_NAME,copy_expression(offset),
                        make_expression(
                            make_syntax_sizeofexpression(make_sizeofexpression_type(copy_type(entity_type(field)))),
                            normalized_undefined)
                    );
        }
    return offset;
}

static expression distance_between_entity(const entity e0, const entity e1)
{
    if(same_entity_p(e0,e1))
        return int_to_expression(0);
    else if( entity_field_p(e0) && entity_field_p(e1) && same_struct_entity_p(e0,e1) )
        return make_op_exp(MINUS_OPERATOR_NAME,offset_of_struct(e1),offset_of_struct(e0));
    else
        return expression_undefined;
}

/** 
 * computes the distance betwwen two expression
 * 
 * @return expression_undefined if not comparable, an expression of the distance otherwise
 */
expression distance_between_expression(const expression exp0, const expression exp1)
{
    bool eval_sizeof = get_bool_property("EVAL_SIZEOF");
    set_bool_property("EVAL_SIZEOF",true);
    expression result = expression_undefined;
    if( expression_reference_p(exp0) && expression_reference_p(exp1))
    {
        reference r0 = expression_reference(exp0),
                  r1 = expression_reference(exp1);

        /* reference with different variables have an infinite distance */
        entity e0 = reference_variable(r0),
               e1 = reference_variable(r1);
        expression edistance = distance_between_entity(e0,e1);
        if(!expression_undefined_p(edistance))
        {
            list indices0 = gen_full_copy_list(reference_indices(r0)),
                 indices1 = gen_full_copy_list(reference_indices(r1));
            if(get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION"))
            {
                indices0=gen_nreverse(indices0);
                indices1=gen_nreverse(indices1);
            }
            while(!ENDP(indices0) && !ENDP(indices1))
            {
                expression i0 = EXPRESSION(CAR(indices0)),
                           i1 = EXPRESSION(CAR(indices1));
                if(!same_expression_p(i0,i1))
                    break;
                POP(indices0);
                POP(indices1);
            }
            if(get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION"))
            {
                indices0=gen_nreverse(indices0);
                indices1=gen_nreverse(indices1);
            }
            reference fake0 = make_reference(e0,indices0),
                      fake1 = make_reference(e1,indices1);
            expression offset0 = reference_offset(fake0),
                       offset1 = reference_offset(fake1);
            expression distance =  make_op_exp(MINUS_OPERATOR_NAME,offset1,offset0);
            free_reference(fake0);
            free_reference(fake1);

            result= make_op_exp(PLUS_OPERATOR_NAME,distance,edistance);
        }
    }
    else if (expression_field_p(exp0) && expression_field_p(exp1))
    {
        expression str0 = binary_call_lhs(expression_call(exp0)),
                   str1 = binary_call_lhs(expression_call(exp1));
        expression lhs_distance = distance_between_expression(str0,str1);
        expression rhs_distance = distance_between_expression(binary_call_rhs(expression_call(exp1)),binary_call_rhs(expression_call(exp0)));
        result = make_op_exp(MINUS_OPERATOR_NAME,lhs_distance,rhs_distance);
    }
    set_bool_property("EVAL_SIZEOF",eval_sizeof);
    return result;
}

/*
   This function returns TRUE if e0 and e1 have a distance of 1 + lastOffset
   */
static bool consecutive_expression_p(expression e0, int lastOffset, expression e1)
{
    bool result=false;
    basic b = basic_of_expression(e0);
    int ref_offset = SizeOfElements(b);
    free_basic(b);
    expression distance = distance_between_expression(e0,e1);
    if( !expression_undefined_p(distance) )
    {
        intptr_t idistance;
        NORMALIZE_EXPRESSION(distance);
        if((result=expression_integer_value(distance,&idistance)))
        {
            pips_debug(3,"distance between %s and %s is %"PRIdPTR"\n",words_to_string(words_expression(e0,NIL)),words_to_string(words_expression(e1,NIL)),idistance);
            result= idistance == ref_offset+ref_offset*lastOffset;
        }
        else {
            pips_debug(3,"distance between %s and %s is too complexd\n",words_to_string(words_expression(e0,NIL)),words_to_string(words_expression(e1,NIL)));
        }
        free_expression(distance);
    }
    return result;
}

/*
   This function returns the vector type string by reading the name of the first 
   element of lExp
   */
static string get_simd_vector_type(list lExp)
{
    string result = NULL;

    FOREACH(EXPRESSION, exp,lExp)
    {
        type t = entity_type(reference_variable(
                    syntax_reference(expression_syntax(
                            exp))));

        if(type_variable_p(t))
        {
            //        basic bas = variable_basic(type_variable(t));

            result = strdup(entity_name(
                        reference_variable(syntax_reference(expression_syntax(exp)))
                        ));
            string c = strrchr(result,'_');
            pips_assert("searching in a sec - encoded variable name",c);
            *c='\0';
            /* switch to upper cases... */
            result=strupper(result,result);

            break;

        }
    }

    return result;
}

/*
   This function returns the name of a vector from the data inside it
   */
static string get_vect_name_from_data(int argc, expression exp)
{
    char prefix[5];
    string result;
    basic bas;
    int itemSize;

    bas = basic_of_expression(exp);

    prefix[0] = 'v';
    prefix[1] = '0'+argc;

    switch(basic_tag(bas))
    {
        case is_basic_int:
            prefix[3] = 'i';
            itemSize = 8 * basic_int(bas);
            break;

        case is_basic_float:
            prefix[3] = 'f';
            itemSize = 8 * basic_float(bas);
            break;

        case is_basic_logical:
            prefix[3] = 'i';
            itemSize = 8 * basic_logical(bas);
            break;
        case is_basic_complex:
            prefix[3] = 'c';
            itemSize = 8 * basic_complex(bas);
            break;

        default:
            return strdup("");
            break;
    }
    free_basic(bas);

    switch(itemSize)
    {
        case 8:  prefix[2] = 'q'; break;
        case 16: prefix[2] = 'h'; break;
        case 32: prefix[2] = 's'; break;
        case 64: prefix[2] = 'd'; break;
    }

    prefix[4] = 0;

    result = strdup(prefix);

    /* switch to upper cases... */
    result=strupper(result,result);

    return result;
}

static
void replace_subscript(expression e)
{
    if( !simd_vector_expression_p(e) && ! expression_constant_p(e) )
    {
        if(!expression_call_p(e) || expression_field_p(e))
        {
            unnormalize_expression(e);
            expression_syntax(e) = make_syntax_call(
                    make_call(
                        entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                        make_expression_list(
                            make_expression(
                                expression_syntax(e),
                                normalized_undefined
                                )
                            )
                        )
                    );
        }
    }

}

static statement make_exec_statement_from_name(string ename, list args)
{
    /* SG: ugly patch to make sure fortran's parameter passing and c's are respected */
    entity exec_function = module_name_to_entity(ename);
    if( c_module_p(exec_function) )
    {
        if( strstr(ename,SIMD_GEN_LOAD_NAME) )
        {
            replace_subscript( EXPRESSION(CAR(args)));
        }
        else
        {
            FOREACH(EXPRESSION,e,args) replace_subscript(e);
        }
    }
    return call_to_statement(make_call(get_function_entity(ename), args));
}
static statement make_exec_statement_from_opcode(opcode oc, list args)
{
    return make_exec_statement_from_name( opcode_name(oc) , args );
}

#define SAC_ALIGNED_VECTOR_NAME "aligned"
bool sac_aligned_entity_p(entity e)
{
    return same_stringn_p(entity_user_name(e),SAC_ALIGNED_VECTOR_NAME,sizeof(SAC_ALIGNED_VECTOR_NAME)-1);
}

static bool sac_aligned_expression_p(expression e)
{
    if(expression_reference_p(e))
    {
        reference r = expression_reference(e);
        return sac_aligned_entity_p(reference_variable(r));
    }
    return false;
}



static statement make_loadsave_statement(int argc, list args, bool isLoad, list padded)
{
    enum {
        CONSEC_REFS,
        MASKED_CONSEC_REFS,
        CONSTANT,
        OTHER
    } argsType;
    const char funcNames[4][2][20] = {
        { SIMD_SAVE_NAME"_",            SIMD_LOAD_NAME"_"},
        { SIMD_MASKED_SAVE_NAME"_",     SIMD_MASKED_LOAD_NAME"_"},
        { SIMD_CONS_SAVE_NAME"_",       SIMD_CONS_LOAD_NAME"_"},
        { SIMD_GEN_SAVE_NAME"_",        SIMD_GEN_LOAD_NAME"_"}
    };
    int lastOffset = 0;
    char *functionName;

    string lsType = local_name(get_simd_vector_type(args));
    bool all_padded= all_padded_p(padded);
    bool all_scalar = false;
    bool all_same_aligned_ref = false;

    /* the function should not be called with an empty arguments list */
    assert((argc > 1) && (args != NIL));

    /* first, find out if the arguments are:
     *    - consecutive references to the same array
     *    - all constant
     *    - or any other situation
     */
    expression fstExp = expression_undefined;

    /* classify according to the second element
     * (first one should be the SIMD vector) */
    expression exp = EXPRESSION(CAR(CDR(args)));
    expression real_exp = expression_field_p(exp)?binary_call_rhs(expression_call(exp)):exp;
    if (expression_constant_p(real_exp))
    {
        argsType = CONSTANT;
        all_scalar = true;
    }
    // If e is a reference expression, let's analyse this reference
    else if (expression_reference_p(real_exp))
    {
        argsType = CONSEC_REFS;
        fstExp = exp;
        all_scalar = expression_scalar_p(exp); /* and not real_exp ! */
        all_same_aligned_ref = sac_aligned_expression_p(exp);
    }
    else
        argsType = OTHER;

    /* now verify the estimation on the first element is correct, and update
     * parameters needed later */
    for(list iter = CDR(CDR(args));!ENDP(iter);POP(iter))
    {
        expression e = EXPRESSION(CAR(iter));
        expression real_e = expression_field_p(e)?binary_call_rhs(expression_call(e)):e;
        if (argsType == OTHER)
        {
            all_scalar = all_scalar && expression_scalar_p(e);
            continue;
        }
        else if (argsType == CONSTANT)
        {
            if (!expression_constant_p(real_e))
            {
                argsType = OTHER;
                all_scalar = all_scalar && expression_scalar_p(e);
                continue;
            }
        }
        else if (argsType == CONSEC_REFS)
        {
            // If e is a reference expression, let's analyse this reference
            // and see if e is consecutive to the previous references
            if ( (expression_reference_p(real_e)) &&
                    (consecutive_expression_p(fstExp, lastOffset, e)) )
            {
                ++lastOffset;
                all_scalar=false;
            }
            /* if all arguments are padded, we cas safely load an additionnal reference,
             * it will not be used anyway */
            else if(ENDP(CDR(iter)) && all_padded )
            {
                /* it is safe to load from anywhere (even a non existing data) but not to write anywhere
                 * that's why we use a mask !
                 */
                if(!isLoad) argsType=MASKED_CONSEC_REFS;
            }
            else
            {
                argsType = OTHER;
                all_same_aligned_ref = all_same_aligned_ref && same_expression_p(e,fstExp);
                all_scalar = all_scalar && expression_scalar_p(e);
                continue;
            }
        }
    }

    /* first pass of analysis is done
     * we may have found that we have no consecutive references
     * but a set of scalar
     * if so, we should replace those scalars by appropriate array
     */
    if(all_scalar)
    {
        list new_statements = NIL;
        size_t nbargs=gen_length(CDR(args));
        basic shared_basic = basic_undefined;
        FOREACH(EXPRESSION,e,CDR(args))
        {
            shared_basic=basic_of_expression(e);
            if(basic_overloaded_p(shared_basic))
                free_basic(shared_basic);
            else
                break;
        }
        entity scalar_holder = make_new_array_variable_with_prefix(
                SAC_ALIGNED_VECTOR_NAME,get_current_module_entity(),shared_basic,
                CONS(DIMENSION,make_dimension(int_to_expression(0),int_to_expression(nbargs-1)),NIL)
                );
        AddLocalEntityToDeclarations(scalar_holder,get_current_module_entity(),sac_current_block);
        int index=0;
        list inits = NIL;
        list replacements = NIL;
            for(list iter = CDR(args); !ENDP(iter) ; POP(iter) )
            {
                expression e = EXPRESSION(CAR(iter));
                bool formal_p=expression_reference_p(e) && formal_parameter_p(reference_variable(expression_reference(e)));
                if(expression_constant_p(e) || formal_p)
                {
                    /* no support for array inital value in fortran */
                    if(fortran_module_p(get_current_module_entity()))
                    {
                        new_statements=CONS(STATEMENT,
                                make_assign_statement(
                                    reference_to_expression(
                                        make_reference(scalar_holder,CONS(EXPRESSION,int_to_expression(index),NIL))
                                        ),
                                    copy_expression(e)),new_statements);
                    }
                    else
                    {
                        inits=CONS(EXPRESSION,copy_expression(e),inits);
                    }
                    if(formal_p)
                    {
                        entity current_scalar = expression_to_entity(e);
                        expression replacement = make_entity_expression(scalar_holder,make_expression_list(int_to_expression(index)));
                        replacements=gen_cons(current_scalar,gen_cons(replacement,replacements));
                    }
                    else
                    {
                        expression replacement = make_entity_expression(scalar_holder,make_expression_list(int_to_expression(index)));
                        *REFCAR(iter) = (gen_chunkp)replacement;
                    }
                }
                else
                {
                    entity current_scalar = expression_to_entity(e);
                    inits=CONS(EXPRESSION,int_to_expression(0),inits);
                    expression replacement = make_entity_expression(scalar_holder,make_expression_list(int_to_expression(index)));
                    replacements=gen_cons(current_scalar,gen_cons(replacement,replacements));
                }
                index++;
            }
            /* manage replacements in the end, otherwise it disturbs the whole process */
            replacements=gen_nreverse(replacements);
        for(list iter = replacements;!ENDP(iter);POP(iter))
        {
            expression r = EXPRESSION(CAR(iter));
            POP(iter);
            entity e = ENTITY(CAR(iter));
            replace_entity_by_expression(sac_real_current_instruction,e,r);
        }
        gen_free_list(replacements);
        if(!fortran_module_p(get_current_module_entity()))
        {
            free_value(entity_initial(scalar_holder));
            entity_initial(scalar_holder) = make_value_expression(
                    call_to_expression(
                        make_call(entity_intrinsic(BRACE_INTRINSIC),
                            gen_nreverse(inits)
                            )
                        )
                    );
        }
        else if(!ENDP(new_statements))
            insert_statement(get_current_module_statement(),make_block_statement(new_statements),true);
        argsType=CONSEC_REFS;
        fstExp = EXPRESSION(CAR(CDR(args)));

    }
    if(all_same_aligned_ref)
    {
        argsType=CONSEC_REFS;
    }



    /* Now that the analyze is done, we can generate an "optimized"
     * load instruction.
     */
    list current_args = NIL;
    switch(argsType)
    {
        case CONSEC_REFS:
        case MASKED_CONSEC_REFS:
            {

                string realVectName = get_vect_name_from_data(argc, EXPRESSION(CAR(CDR(args))));

                if(strcmp(strchr(realVectName, MODULE_SEP)?local_name(realVectName):realVectName, lsType))
                {
                    /*string temp = local_name(lsType);*/
                    asprintf(&lsType,"%s_TO_%s",realVectName,lsType);
                }
                if(get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION"))
                {
                    current_args = gen_make_list( expression_domain, 
                            EXPRESSION(CAR(args)),
                            fstExp,
                            NULL);
                }
                else
                {
                    expression addr = fstExp;
                    current_args = make_expression_list( copy_expression(EXPRESSION(CAR(args))), copy_expression(addr));
                }

                gSimdCost += - argc + 1;

                break;
            }

        case CONSTANT:
            {
                gSimdCost += - argc + 1;

                break;
            }

        case OTHER:
            current_args=args;
        default:
            break;
    }
    /* make the new binding available to everybody */
    update_vector_to_expressions(expression_to_entity(EXPRESSION(CAR(args))),CDR(args));

    asprintf(&functionName, "%s%s", funcNames[argsType][isLoad], lsType);
    statement es = make_exec_statement_from_name(functionName, current_args);
    free(functionName);
    return es;

}

static statement make_load_statement(int argc, list args, list padded)
{
    return make_loadsave_statement(argc, args, TRUE, padded);
}

static statement make_save_statement(int argc, list args, list padded)
{
    return make_loadsave_statement(argc, args, FALSE, padded);
}


/*
   This function creates a simd vector.
   */
static entity make_new_simd_vector(int itemSize, int nbItems, enum basic_utype basicTag)
{
    //extern list integer_entities, real_entities, double_entities;

    basic simdVector;

    entity new_ent, mod_ent;
    char prefix[6]={ 'v', '0', '\0', '\0', '\0', '\0' },
         num[1 + sizeof(VECTOR_POSTFIX) + 3 ],
         name[sizeof(prefix)+sizeof(num)+1];
    static int number = 0;

    mod_ent = get_current_module_entity();

    /* build the variable prefix code, which is in fact also the type */
    prefix[1] += nbItems;
    switch(itemSize)
    {
        case 8:  prefix[2] = 'q'; break;
        case 16: prefix[2] = 'h'; break;
        case 32: prefix[2] = 's'; break;
        case 64: prefix[2] = 'd'; break;
    }


    switch(basicTag)
    {
        case is_basic_int:
            simdVector = make_basic_int(itemSize/8);
            prefix[3] = 'i';
            break;

        case is_basic_float:
            simdVector = make_basic_float(itemSize/8);
            prefix[3] = 'f';
            break;

        default:
            simdVector = make_basic_int(itemSize/8);
            prefix[3] = 'i';
            break;
    }

    pips_assert("buffer doesnot overflow",number<10000);
    sprintf(name, "%s%s%u",prefix,VECTOR_POSTFIX,number++);
    list lis=CONS(DIMENSION, make_dimension(int_to_expression(0),int_to_expression(nbItems-1)), NIL);  

    new_ent = make_new_array_variable_with_prefix(name, mod_ent , simdVector, lis);

#if 0
    string type_name = strdup(concatenate(prefix,"_struct", (char *) NULL));
    entity str_type = FindOrCreateEntity(entity_local_name(mod_ent), type_name);
    entity_type(str_type) =make_type_variable(make_variable(simdVector,NIL,NIL)); 

    entity str_dec = FindOrCreateEntity(entity_local_name(mod_ent), name);
    entity_type(str_dec) = entity_type(str_type);
#endif

    AddLocalEntityToDeclarations(new_ent,get_current_module_entity(),sac_current_block);

    return new_ent;
}




static simdstatement make_simd_statement(opcodeClass kind, opcode oc, list* args)
{
    simdstatement ss;
    size_t nbargs;

    /* find out the number of arguments needed */
    nbargs = opcodeClass_nbArgs(kind);

    /* allocate memory */
    ss = make_simdstatement(oc, 
            nbargs,
            (entity *)malloc(sizeof(entity)*nbargs),
            (expression*)malloc(sizeof(expression) * nbargs * opcode_vectorSize(oc)));


    /* create the simd vector entities */
    hash_table expression_to_entity = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    int j=nbargs-1;
    pips_assert("nb parameters match nb dummy parameters",nbargs==gen_length(*args));
    FOREACH(EXPRESSION,exp,*args)
    {
        enum basic_utype basicTag = get_basic_from_opcode(oc, nbargs-1-j);
        simdstatement_vectors(ss)[j] = entity_undefined;
        register void * hiter = NULL;
        expression key;
        entity value;
        while ((hiter = hash_table_scan(expression_to_entity, hiter, (void**)&key, (void**)&value))) { 
            if( expression_equal_p(exp,key ) ) {
                simdstatement_vectors(ss)[j] = value;
                break;
            }
        }

        if( entity_undefined_p(simdstatement_vectors(ss)[j]) )
        {
            simdstatement_vectors(ss)[j] = 
                make_new_simd_vector(get_subwordSize_from_opcode(oc, j),
                        opcode_vectorSize(oc),
                        basicTag);
            hash_put(expression_to_entity,exp,simdstatement_vectors(ss)[j]);
        }
        j--;
    }
    hash_table_free(expression_to_entity);

    /* Fill the matrix of arguments */
    for(int j=0; j<opcode_vectorSize(oc); j++)
    {

        list l = args[j];

        for(int i=nbargs-1; i>=0; i--)
        {
            expression e = EXPRESSION(CAR(l));

            //Store it in the argument's matrix
            simdstatement_arguments(ss)[j + opcode_vectorSize(oc) * i] = e;

            l = CDR(l);
        }
    }

    // substitute reference padding to constant padding when possible
    patch_all_padded_simd_statements(ss);

    return ss;
}
#if 0
static
void free_simd_statement_info(simdstatement s)
{
    free(simdstatement_vectors(s));
    free(simdstatement_arguments(s));
    free_simdstatement(s);
}

static int compare_statements(const void * v0, const void * v1)
{
    statement s0 = *(statement*)v0;
    statement s1 = *(statement*)v1;
    if (statement_ordering(s0) > statement_ordering(s1)) return 1;
    if (statement_ordering(s0) < statement_ordering(s1)) return -1;
    return 0;
}
#endif

simdstatement make_simd_statements(set opkinds, list statements)
{
    list args[MAX_PACK];
    /* there should be a better order than this */
    list first = statements;
    list last = gen_last(first);
    list kinds = set_to_list(opkinds);

    pips_debug(3,"make_simd_statements 1\n");

    if (first == last)
        return simdstatement_undefined;

    pips_debug(3,"make_simd_statements 2\n");

    list i = first;
    simdstatement instr = simdstatement_undefined;

    opcodeClass type = OPCODECLASS(CAR(kinds));
    {
        int index;
        opcode oc;
        list j;

        /* get the variables */
        for( index = 0, j = i;
                (index < MAX_PACK) && (j != CDR(last));
                index++, j = CDR(j) )
        {
            match m = get_statement_match_of_kind(STATEMENT(CAR(j)), type);
            args[index] = match_args(m);
        }

        /* if index is a power of 2 less 1 , we may want to complete it with a fake statement 
         * this would enlarge the matching at the cost of less efficiency
         * indeed this is a kind of array padding !
         */
        bool padding_added=false;
        if(get_bool_property("SIMDIZER_ALLOW_PADDING") && 
                (padding_added=(index == 3 || index == 7 || index == 15 || index == 31)) )
        {
            args[index]=NIL;
            for(int i=0;i<=index;i++)
                args[index]=CONS(EXPRESSION,int_to_expression(0),args[index]);
            ++index;
        }

        /* compute the opcode to use */
        oc = get_optimal_opcode(type, index, args);

        if (!opcode_undefined_p(oc))
        {
            /* now that we know the optimal opcode, we can change the padding to the neutral element
            */
            if(padding_added)
            {
                statement sfirst = STATEMENT(CAR(first));
                if(assignment_statement_p(sfirst))
                {
                    expression neutral_element= entity_to_expression(operator_neutral_element(
                                call_function(expression_call(binary_call_rhs(statement_call(sfirst))))
                                ));
                    bool first=true;
                    FOREACH(EXPRESSION,e,args[index-1])
                    {
                        free_syntax(expression_syntax(e));
                        if(first)
                        {
                            /* we always use the same padding entity, it proves to be usefull later on */
                            entity pe = get_padding_entity();
                            expression_syntax(e)=make_syntax_reference(make_reference(pe,NIL));
                            first=false;
                        }
                        else
                        {
                            expression_syntax(e)=copy_syntax(expression_syntax(neutral_element));
                        }
                    }
                    free_expression(neutral_element);
                }
                else
                    pips_user_warning("wrong padding may have been added\n");
            }
            /* update the pointer to the next statement to be processed */
            for(index = 0; 
                    (index<opcode_vectorSize(oc)) && (i!=CDR(last)); 
                    index++)
            {
                i = CDR(i);
            }

            /* generate the statement information */
            pips_debug(3,"make_simd_statements : simd\n");
            instr =  make_simd_statement(type, oc, args);
        }
    }

    pips_debug(3,"make_simd_statements 3\n");
    return instr;
}

static statement generate_exec_statement(simdstatement ss)
{
    list args = NIL;
    int i;

    for(i = 0; i < simdstatement_nbArgs(ss); i++)
    {
        args = CONS(EXPRESSION,
                entity_to_expression(simdstatement_vectors(ss)[i]),
                args);
    }

    gSimdCost += opcode_cost(simdstatement_opcode(ss));

    return make_exec_statement_from_opcode(simdstatement_opcode(ss), args);
}

static 
statement generate_load_statement(simdstatement si, int line)
{
    list args = NIL;
    int offset = line * opcode_vectorSize(simdstatement_opcode(si));
    /* is this statement padded ? if so, we are allowded to do more ... */
    list padded = padded_simd_statement_p(si);
    {
        //Build the arguments list
        for(int i = opcode_vectorSize(simdstatement_opcode(si))-1; 
                i >= 0; 
                i--)
        {
            args = CONS(EXPRESSION,
                    simdstatement_arguments(si)[i + offset],
                    args);
        }
        entity vector = expressions_to_vector(args);
        if(!entity_undefined_p(vector)){
            simdstatement_vectors(si)[line] = vector;
            return statement_undefined;
        }
        else {
            args = CONS(EXPRESSION,
                    entity_to_expression(simdstatement_vectors(si)[line]),
                    args);

            //Make a load statement
            return make_load_statement(
                    opcode_vectorSize(simdstatement_opcode(si)), 
                    args, padded);
        }
    }
}

static statement generate_save_statement(simdstatement si)
{
    list args = NIL;
    int i;
    int offset = opcode_vectorSize(simdstatement_opcode(si)) * 
        (simdstatement_nbArgs(si)-1);

    /* is this statement padded ? if so, we are allowded to do more ... */
    list padded = padded_simd_statement_p(si);


    for(i = opcode_vectorSize(simdstatement_opcode(si))-1; 
            i >= 0; 
            i--)
    {
        args = CONS(EXPRESSION,
                (
                    simdstatement_arguments(si)[i + offset]),
                args);
    }

    args = CONS(EXPRESSION,
            entity_to_expression(simdstatement_vectors(si)[simdstatement_nbArgs(si)-1]),
            args);

    return make_save_statement(opcode_vectorSize(simdstatement_opcode(si)), args, padded);
}


list generate_simd_code(simdstatement ssi, float * simdCost)
{
    list out = NIL;
    gSimdCost = 0;

    pips_debug(3,"generate_simd_code 1\n");

    /* this is the classical generation process:
     * several load, an exec and a store
     */
    /* SIMD statement (will generate more than one statement) */
    int i;
    
    //First, the load statement(s)
    for(i = 0; i < simdstatement_nbArgs(ssi)-1; i++)
    {
        statement s = generate_load_statement(ssi, i);
        if (! statement_undefined_p(s))
            out = CONS(STATEMENT, s, out);
    }

    //Then, the exec statement
    out=CONS(STATEMENT, generate_exec_statement(ssi), out);

    //Finally, the save statement (always generated. It is up to 
    //latter phases (USE-DEF elimination....) to remove it, if needed
    out= CONS(STATEMENT, generate_save_statement(ssi), out);

    *simdCost += gSimdCost;
    pips_debug(3,"generate_simd_code 2 that costs %lf\n",gSimdCost);
    /* the order is reversed, but we use it */
    return out;
}
