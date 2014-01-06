/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#define VECTOR_POSTFIX "vec"


static float gSimdCost;

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

static entity do_expressions_to_vector(list expressions) {
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
    /* create the enw vector */
    FOREACH(EXPRESSION,exp,exps) pips_assert("expressions are ok",expression_consistent_p(exp));
    /* remove the old ones */
    HASH_FOREACH(entity,v,list,expressions,vector_to_expressions) {
        FOREACH(EXPRESSION,exp,exps) {
            for(list iter=expressions;!ENDP(iter);POP(iter)) {
                expression * piter = (expression*) REFCAR(iter);
                if(expression_equal_p(exp,*piter)) /* invalidate */ {
                    free_expression(*piter);
                    *piter=expression_undefined;
                }
            }
        }
    } 
    hash_put_or_update(vector_to_expressions,e,gen_full_copy_list(exps));
}

/* This function will create the permutations indexes that will allow
   the creation of exp_final from exp_loaded with a shuffle */
static void make_permutations_indexes(list exp_final, list exp_loaded, int* perms)
{
	int index_loaded = 0;
	FOREACH(EXPRESSION,ef,exp_loaded) {
		int index_perm = 0;
		FOREACH(EXPRESSION,el,exp_final) {
			if (expression_equal_p(ef,el)) {
				perms[index_loaded] = index_perm;
				break;
			}
			index_perm++;
		}
		index_loaded++;
	}
}

static entity try_all_permutations(list expressions, list remainder, list exp_org, statement* shuffle,int *perms)
{
	FOREACH(EXPRESSION,exp,remainder){
		list tmp = gen_copy_seq(expressions);
		list rtmp = gen_copy_seq(remainder);
		gen_remove_once(&rtmp,exp);
		list newexp = CONS(EXPRESSION,exp,tmp);
		entity out = try_all_permutations(newexp,rtmp,exp_org,shuffle,perms);
		gen_free_list(newexp);
		gen_free_list(rtmp);
		if(!entity_undefined_p(out))
			return out;
	}
	if(ENDP(remainder)) {
		entity out = do_expressions_to_vector(expressions);
		if(!entity_undefined_p(out))
		{
			make_permutations_indexes(exp_org, expressions, perms);
			*shuffle=make_shuffle_statement(&out,expressions,perms);
		}
		return out;
	}
	return entity_undefined;
}

static entity expressions_to_vector(list expressions,statement* shuffle)
{
    /* try all possible permutations, well only invert from now on */
    entity out = do_expressions_to_vector(expressions);
    size_t nb_expressions = gen_length(expressions);
    if(!entity_undefined_p(out)) return out;
    /* give up if there are two may permutations to try out */
    else if(nb_expressions <= 8 ) {
        int perm [1+nb_expressions];
        perm[nb_expressions]=0;
        return try_all_permutations(NIL,expressions,expressions,shuffle,perm);
    }
    else
        return entity_undefined;
}

void invalidate_expressions_in_statement(statement s)
{
    list effects = load_proper_rw_effects_list( s );
    void * hiter = NULL;
    entity key;
    list value;
    list purge = NIL;
    while( (hiter = hash_table_scan(vector_to_expressions,hiter,(void**)&key,(void**)&value) ))
    {
        bool invalidate=false;
        FOREACH(EXPRESSION,v,value)
        {
            if(!expression_undefined_p(v) && expression_reference_p(v))
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
            pips_internal_error("subword size unknown.");
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
            pips_internal_error("subword size unknown.");
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

/* Computes the optimal opcode for simdizing 'argc' statements of the
 * 'kind' operation, applied to the 'args' arguments
 * it is a greedy matching, so it supposes the list of args is in the best order for us
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
        bool bTagDiff = false;
        int mwidth;

        for(i = 0; i < argc; i++)
        {
            int count = 0;
            int width = mwidth=0;

            FOREACH(EXPRESSION,arg,args[i])
            {

                if (!expression_reference_or_field_p(arg))
                {
                    count++;
                    continue;
                }

                basic bas = basic_of_expression(arg);
                enum basic_utype bas_pattern = get_basic_from_opcode(oc, count);

                if(!basic_overloaded_p(bas) && bas_pattern!=basic_tag(bas)
				/*&& !(bas_pattern == is_basic_float && basic_int_p(bas))*/) /* allow the promotion of an int
											    to a float */

                {
                    bTagDiff = true;
                    free_basic(bas);
                    break;
                }

                width = SizeOfElements(bas);
                if(width>mwidth)mwidth=width;
                free_basic(bas);

                if(width > get_subwordSize_from_opcode(oc, count))
                {
                    bTagDiff = true;
                    break;
                }

                count++;

            }
        }

        if ( (!bTagDiff) &&
                (opcode_vectorSize(oc) <= argc) &&
                 (opcode_vectorSize(oc)*mwidth*8 == get_int_property("SAC_SIMD_REGISTER_WIDTH"))) {
            best = oc;
        }
    }

    return best;
}


/** 
 * computes the offset between an entity and its reference
 * 
 * @param r 
 * 
 * @return an expression of the offset, in octets
 */
static
expression sreference_offset(reference r)
{
    if(get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION")) {
        reference cr = copy_reference(r);
        reference_indices(cr)=gen_nreverse(reference_indices(cr));
        expression out = reference_offset(cr);
        free_reference(cr);
        return out;
    }
    else
        return reference_offset(r);
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
            basic b = basic_of_reference(r0);
            expression offset0 = sreference_offset(r0),
                       offset1 = sreference_offset(r1);
            expression distance =  make_op_exp(MINUS_OPERATOR_NAME,offset1,offset0);
            distance = make_op_exp(MULTIPLY_OPERATOR_NAME,
                    distance,int_to_expression(SizeOfElements(b)));
            free_basic(b);
            result= make_op_exp(PLUS_OPERATOR_NAME,distance,edistance);
        }
    }
    else if (expression_field_p(exp0) && expression_field_p(exp1))
    {
        expression str0 = binary_call_lhs(expression_call(exp0)),
                   str1 = binary_call_lhs(expression_call(exp1));
        expression lhs_distance = distance_between_expression(str0,str1);
        expression rhs_distance = distance_between_expression(binary_call_rhs(expression_call(exp1)),binary_call_rhs(expression_call(exp0)));
        if(!expression_undefined_p(lhs_distance) && !expression_undefined_p(rhs_distance))
            result = make_op_exp(MINUS_OPERATOR_NAME,lhs_distance,rhs_distance);
    }
    set_bool_property("EVAL_SIZEOF",eval_sizeof);
    //maxima_simplify(&result);
    simplify_expression(&result);
    return result;
}

/*
   This function returns true if e0 and e1 have a distance of 1 + lastOffset
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
            basic bas = variable_basic(type_variable(t));
            pips_assert("searching in a sac vector",basic_typedef_p(bas));

            result = strdup(entity_name(basic_typedef(bas)));
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
static string get_vect_name_from_data(int argc, list exps)
{
    char prefix[5];
    string result;
    int itemSize;

    basic bas = basic_of_expressions(exps,true);

    prefix[0] = 'v';
    prefix[1] = '0'+argc;

    switch(basic_tag(bas))
    {
        case is_basic_int:
            prefix[3] = 'i';
            break;

        case is_basic_float:
            prefix[3] = 'f';
            break;

        case is_basic_logical:
            prefix[3] = 'i';
            break;
        case is_basic_complex:
            prefix[3] = 'c';
            break;

        default:
            free_basic(bas);
            return strdup("");
            break;
    }
    itemSize=8*SizeOfElements(bas);
    free_basic(bas);

    switch(itemSize)
    {
        case 8:  prefix[2] = 'q'; break;
        case 16: prefix[2] = 'h'; break;
        case 32: prefix[2] = 's'; break;
        case 64: prefix[2] = 'd'; break;
        default:pips_internal_error("case not handled");
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
            syntax syn = expression_syntax(e);
            expression_syntax(e)=syntax_undefined;
            update_expression_syntax(e,make_syntax_call(
                        make_call(
                            entity_intrinsic(ADDRESS_OF_OPERATOR_NAME),
                            make_expression_list(
                                make_expression(syn,normalized_undefined)
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
    entity exec_function = module_name_to_runtime_entity(ename);
    if( c_module_p(exec_function) )
    {
        string pattern0;
        asprintf(&pattern0,"%s" SIMD_GENERIC_SUFFIX,get_string_property("ACCEL_LOAD"));
        string pattern1;
        asprintf(&pattern1,"%s" SIMD_BROADCAST_SUFFIX,get_string_property("ACCEL_LOAD"));
        if( strstr(ename,pattern0) )
            replace_subscript( EXPRESSION(CAR(args)));
        else if( strstr(ename,pattern1) )
            replace_subscript( EXPRESSION(CAR(args)));
        else {
            FOREACH(EXPRESSION,e,args) replace_subscript(e);
        }
        free(pattern0);
        free(pattern1);
    }
    return call_to_statement(make_call(module_name_to_runtime_entity(ename), args));
}
static statement make_exec_statement_from_opcode(opcode oc, list args)
{
    return make_exec_statement_from_name( opcode_name(oc) , args );
}

static basic get_typedefed_array(const char * type_name, basic b, list dims) {
    entity e = FindEntity(TOP_LEVEL_MODULE_NAME,type_name);
    if(entity_undefined_p(e)) {
        e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,type_name);
        entity_type(e)=make_type_variable(
                make_variable(
                    b,
                    dims,NIL
                    )
                );
        entity_storage(e)=make_storage_rom();
        entity_initial(e)=make_value_unknown();
    }
    return make_basic_typedef(e);
}

#define SAC_ALIGNED_VECTOR_NAME "pdata"
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
/*
   This function creates a simd vector.
   */
static entity make_new_simd_vector_with_prefix(int itemSize, int nbItems, enum basic_utype basicTag, const char *vname)
{
    //extern list integer_entities, real_entities, double_entities;

    basic simdVector;

    entity new_ent, mod_ent;
    char prefix[6]={ same_string_p(vname,VECTOR_POSTFIX)?'v':'a', '0', '\0', '\0', '\0', '\0' },
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
    sprintf(name, "%s%u", vname,number++);
    list lis=CONS(DIMENSION, make_dimension(int_to_expression(0),int_to_expression(nbItems-1)), NIL);  

    basic typedefedSimdVector = get_typedefed_array(prefix,simdVector,lis);

    new_ent = make_new_scalar_variable_with_prefix(name, mod_ent , typedefedSimdVector);

#if 0
    string type_name = strdup(concatenate(prefix,"_struct", (char *) NULL));
    entity str_type = FindOrCreateEntity(entity_local_name(mod_ent), type_name);
    entity_type(str_type) =make_type_variable(make_variable(simdVector,NIL,NIL)); 

    entity str_dec = FindOrCreateEntity(entity_local_name(mod_ent), name);
    entity_type(str_dec) = entity_type(str_type);
#endif

    if( same_string_p(vname, VECTOR_POSTFIX))
        AddLocalEntityToDeclarations(new_ent,get_current_module_entity(),sac_current_block);
    else
        AddEntityToCurrentModule(new_ent);

    return new_ent;
}
static entity make_new_simd_vector(int itemSize, int nbItems, enum basic_utype basicTag)
{
    return make_new_simd_vector_with_prefix(itemSize, nbItems, basicTag, VECTOR_POSTFIX);
}

/* This function change the "load/store type" to XX_TO_XX if a conversion
 * is needed. It returns true if a change is made, and the pointer
 * *lsType must be freed after being used. */
static bool loadstore_type_conversion_string(int argc, list args, string* lsType, bool isLoad)
{
    string lsTypeTmp = (char*)local_name(get_simd_vector_type(args));
    string realVectName = get_vect_name_from_data(argc, CDR(args));
    if (!same_string_p(realVectName, lsTypeTmp))
    {
        entity forged_array = entity_undefined;
        FOREACH(EXPRESSION,arg,CDR(args)) {
            if(expression_reference_p(arg)) {
                reference r = expression_reference(arg);
                entity var = forged_array=reference_variable(r);
                if(strncmp(entity_user_name(var),
                            SAC_ALIGNED_VECTOR_NAME,
                            sizeof(SAC_ALIGNED_VECTOR_NAME)-1)!=0) {
                    forged_array=entity_undefined;
                    break;
                }
                else if(entity_undefined_p(forged_array)) {
                    forged_array = var;
                }
                else if(!same_entity_p(forged_array,var)) {
                    forged_array=entity_undefined;
                    break;
                }
            }
        }
        /* we can handle this situation :p */
        if(!entity_undefined_p(forged_array)) {
            expression exp = EXPRESSION(CAR(args));
            reference r = expression_reference(exp);
            type t = ultimate_type(entity_type(reference_variable(r)));
            variable v = type_variable(t);
            entity new = make_new_simd_vector_with_prefix(
                    8*basic_type_size(variable_basic(v)),
                    dimension_size(DIMENSION(CAR(variable_dimensions(v)))),
                    basic_tag(variable_basic(v)),
                    SAC_ALIGNED_VECTOR_NAME
                    );
            free_value(entity_initial(new));
            entity_initial(new) = copy_value(entity_initial(forged_array));
            FOREACH(EXPRESSION,ex,CDR(args))
                replace_entity(ex,forged_array,new);
        }
        else {
            asprintf(lsType,"%s_TO_%s",
                    isLoad?realVectName:lsTypeTmp,
                    isLoad?lsTypeTmp:realVectName);
            return true;
        }
    }
    return false;
}

static statement make_loadsave_statement(int argc, list args, bool isLoad)
{
    enum {
        CONSEC_REFS,
        MASKED_CONSEC_REFS,
        CONSTANT,
        OTHER,
        BROADCAST
    } argsType;
    static  char *funcNames[6][2] = { { NULL,NULL},{NULL,NULL},{NULL,NULL},{NULL,NULL}};
    if(!funcNames[0][0]) {
        asprintf(&funcNames[0][0],"%s_",get_string_property("ACCEL_STORE"));
        asprintf(&funcNames[0][1],"%s_",get_string_property("ACCEL_LOAD"));
        asprintf(&funcNames[1][0],"%s"SIMD_MASKED_SUFFIX"_",get_string_property("ACCEL_STORE"));
        asprintf(&funcNames[1][1],"%s"SIMD_MASKED_SUFFIX"_",get_string_property("ACCEL_LOAD"));
        asprintf(&funcNames[2][0],"%s"SIMD_CONSTANT_SUFFIX"_",get_string_property("ACCEL_STORE"));
        asprintf(&funcNames[2][1],"%s"SIMD_CONSTANT_SUFFIX"_",get_string_property("ACCEL_LOAD"));
        asprintf(&funcNames[3][0],"%s"SIMD_GENERIC_SUFFIX"_",get_string_property("ACCEL_STORE"));
        asprintf(&funcNames[3][1],"%s"SIMD_GENERIC_SUFFIX"_",get_string_property("ACCEL_LOAD"));
        asprintf(&funcNames[4][0],"%s"SIMD_BROADCAST_SUFFIX"_",get_string_property("ACCEL_STORE"));
        asprintf(&funcNames[4][1],"%s"SIMD_BROADCAST_SUFFIX"_",get_string_property("ACCEL_LOAD"));
    }
    int lastOffset = 0;
    char *functionName;

    string lsType = (char*)local_name(get_simd_vector_type(args));
    bool all_scalar = false;
    bool all_same_aligned_ref = false;
    bool all_same_ref = true;

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
        if(expression_undefined_p(fstExp) ||! same_expression_p(fstExp,real_e))
            all_same_ref=false;

        if (argsType == OTHER)
        {
            all_scalar = all_scalar && expression_scalar_p(e) &&
                (!isLoad || (expression_constant_p(e) || formal_parameter_p(expression_to_entity(e))));
            continue;
        }
        else if (argsType == CONSTANT)
        {
            if (!expression_constant_p(real_e))
            {
                argsType = OTHER;
                all_scalar = all_scalar && expression_scalar_p(e) &&
                    (!isLoad || (expression_constant_p(e) || formal_parameter_p(expression_to_entity(e))));
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
            else
            {
                argsType = OTHER;
                all_same_aligned_ref = all_same_aligned_ref && same_expression_p(e,fstExp);
                all_scalar = all_scalar && expression_scalar_p(e) &&
                    (!isLoad || (expression_constant_p(e) || formal_parameter_p(expression_to_entity(e))));
                continue;
            }
        }
    }
    /* eventually, we have four times the same reference loaded.
     * In that case, use a broadcast call */
    if(all_same_ref && !expression_undefined_p(fstExp) ) {
    pips_assert("all reference are the same, so it's an 'other' kind\n",argsType==OTHER);
        argsType = BROADCAST;
    }
    /* first pass of analysis is done
     * we may have found that we have no consecutive references
     * but a set of scalar
     * if so, we should replace those scalars by appropriate array
     */
    else if(all_scalar)
    {
        list new_statements = NIL;
        size_t nbargs=gen_length(CDR(args));
        basic shared_basic = basic_of_expressions(CDR(args),true);
        char * tname =strdup(get_vect_name_from_data(nbargs,CDR(args)));
        tname=strlower(tname,tname);
        tname[0]='a';/*array*/

        entity scalar_holder = make_new_scalar_variable_with_prefix(
                SAC_ALIGNED_VECTOR_NAME,get_current_module_entity(),
                get_typedefed_array(tname,shared_basic, CONS(DIMENSION,make_dimension(int_to_expression(0),int_to_expression(nbargs-1)),NIL)
                    )
                );
        free(tname);
        AddEntityToCurrentModule(scalar_holder);
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
            replace_entity_by_expression(get_current_module_statement(),e,r);
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
    char* tofree = NULL;
    switch(argsType)
    {
        case CONSEC_REFS:
        case MASKED_CONSEC_REFS:
            {

                if (loadstore_type_conversion_string(argc, args, &lsType, isLoad))
                    tofree=lsType;

                if(get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION"))
                {
                    current_args = make_expression_list(
                            copy_expression(EXPRESSION(CAR(args))),
                            copy_expression(fstExp));
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
            {
                if (loadstore_type_conversion_string(argc, args, &lsType, isLoad))
                    tofree=lsType;
                current_args=gen_full_copy_list(args);
            } break;
        case BROADCAST:
            current_args = make_expression_list(
                    copy_expression(EXPRESSION(CAR(args))),
                    copy_expression(fstExp)
                    );
        default:
            break;
    }
    /* make the new binding available to everybody */
    update_vector_to_expressions(expression_to_entity(EXPRESSION(CAR(args))),CDR(args));

    asprintf(&functionName, "%s%s", funcNames[argsType][isLoad], lsType);
    if(tofree) free(tofree);
    statement es = make_exec_statement_from_name(functionName, current_args);
    free(functionName);
    return es;

}
statement make_shuffle_statement(entity *from,list expressions,int *perms) {
	list shuffle=NIL;
    FOREACH(EXPRESSION,_,expressions)
		shuffle=CONS(EXPRESSION,int_to_expression(*perms++),shuffle);
	shuffle=gen_nreverse(shuffle);
	entity out = make_entity_copy(*from);
	AddEntityToCurrentModule(out);
	string sshuffle;
	statement outs;
	/* If the permutation indexes are equivalent to an invertion,
	 * generate an SIMD_INVERT function (useful for some MIS). */
	if (perms[0] == 4 && perms[1] == 3 && perms[2] == 2 && perms[3] == 1) {
		asprintf(&sshuffle,"SIMD_INVERT_%s",
				get_vect_name_from_data(gen_length(expressions),expressions));
		list args = NIL;
		args = CONS(EXPRESSION,entity_to_expression(*from),args);
		args = CONS(EXPRESSION,entity_to_expression(out),args);

		outs =  instruction_to_statement(
				make_instruction_call(
					make_call(
						module_name_to_runtime_entity(sshuffle),
						args)
					)
				);
	}
	else {
		asprintf(&sshuffle,"SIMD_SHUFFLE_%s",
				get_vect_name_from_data(gen_length(expressions),expressions));

		outs =  instruction_to_statement(
				make_instruction_call(
					make_call(
						module_name_to_runtime_entity(sshuffle),
						CONS(EXPRESSION,entity_to_expression(out),
							CONS(EXPRESSION,entity_to_expression(*from), shuffle)))
					)
				);
	}
	*from=out;
	return outs;
}


static statement make_load_statement(int argc, list args)
{
    return make_loadsave_statement(argc, args, true);
}

static statement make_save_statement(int argc, list args)
{
    return make_loadsave_statement(argc, args, false);
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

    return ss;
}

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

        /* compute the opcode to use */
        oc = get_optimal_opcode(type, index, args);

        if (!opcode_undefined_p(oc))
        {
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
        statement shuffle=statement_undefined;
        entity vector = expressions_to_vector(args,&shuffle);
        if(!entity_undefined_p(vector)){
            simdstatement_vectors(si)[line] = vector;
            return shuffle;
        }
        else {
            args = CONS(EXPRESSION,
                    entity_to_expression(simdstatement_vectors(si)[line]),
                    args);

            //Make a load statement
            return make_load_statement(
                    opcode_vectorSize(simdstatement_opcode(si)), 
                    args);
        }
    }
}

static statement generate_save_statement(simdstatement si)
{
    list args = NIL;
    int i;
    int offset = opcode_vectorSize(simdstatement_opcode(si)) * 
        (simdstatement_nbArgs(si)-1);

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

    return make_save_statement(opcode_vectorSize(simdstatement_opcode(si)), args);
}


list generate_simd_code(simdstatement ssi, float * simdCost)
{
    gSimdCost = 0;

    pips_debug(3,"generate_simd_code 1\n");

    /* this is the classical generation process:
     * several load, an exec and a store
     */
    //First, the load statement(s)
    list loads=NIL;
    for(int i = 0; i < simdstatement_nbArgs(ssi)-1; i++)
    {
        statement s = generate_load_statement(ssi, i);
        if (! statement_undefined_p(s))
            loads = CONS(STATEMENT, s, loads);
    }

    //Then, the exec statement
    statement exec= generate_exec_statement(ssi);

    //Finally, the save statement (always generated. It is up to 
    //latter phases (USE-DEF elimination....) to remove it, if needed
    statement save = generate_save_statement(ssi);

    list out = NIL;
        out=CONS(STATEMENT,save,CONS(STATEMENT,exec,loads));
    *simdCost += gSimdCost;
    pips_debug(3,"generate_simd_code 2 that costs %lf\n",gSimdCost);
    /* the order is reversed, but we use it */
    return out;
}
