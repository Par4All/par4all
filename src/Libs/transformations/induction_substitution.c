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
/* package induction_substitution
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformations.h"

/* Context used for substitution with gen_context_recurse */
typedef struct {
    entity to_substitute; /* The induction variable */
    expression substitute_by; /* The substitution expression */
} substitute_ctx;


/** \fn static bool is_modified_entity_in_transformer( transformer T, entity ent )
 *  \brief Check in transformer if the entity ent is not (potentially) modified
 *  \param T the transformer that will be checked
 *  \param ent the entity we are looking for
 *  \return true if entity ent has been found in the transformer T
 */
static bool is_modified_entity_in_transformer( transformer T, entity ent ) {
    bool is_modified = FALSE;

    list entities = transformer_arguments( T );

    for ( list el = entities; !ENDP( el ); POP( el ) ) {
        if ( ENTITY( CAR( el ) ) == ent ) {
            is_modified = TRUE;
            break;
        }
    }
    return is_modified;
}


/**
 * Use transformer associated to the loop to check that variable v is invariant
 */
static bool loop_invariant_p( Variable v, list /* of statements */ loops ) {
  bool result = false;
  FOREACH(statement, s, loops) {
    transformer t = load_statement_transformer( s );
    if( !is_modified_entity_in_transformer(t,(entity)v ) ) {
      result = true;
    }
  }

  if(result) pips_debug(4,"%s is a loop invariant !\n",entity_name((entity)v));
  else pips_debug(4,"%s is not a loop invariant !\n",entity_name((entity)v));

  return result;
}


/**
 * Check if variable v is an index for an enclosing loop
 */
static bool index_of_a_loop_p( Variable v, list /* of statements */ loops ) {
  bool result = false;
  FOREACH(statement, s, loops) {
    loop l = statement_loop(s);
    pips_debug(5,"Enclosed loop with index : %s\n",entity_name(loop_index(l)));
    if( same_entity_p((entity)v,loop_index(l) ) ) {
      result = true;
    }
  }

  if(result) pips_debug(4,"%s is a loop index !\n",entity_name((entity)v));
  else pips_debug(4,"%s is not a loop index !\n",entity_name((entity)v));

  return result;
}

/** \fn static bool inc_or_de_crement_instruction_p( instruction i ) {
 * \brief check if the instruction is a ++ or --
 */
static bool inc_or_de_crement_instruction_p( instruction i ) {
  return  native_instruction_p( i, POST_INCREMENT_OPERATOR_NAME ) //
    || native_instruction_p( i, PRE_INCREMENT_OPERATOR_NAME ) //
    || native_instruction_p( i, POST_DECREMENT_OPERATOR_NAME ) //
    || native_instruction_p( i, PRE_DECREMENT_OPERATOR_NAME ) //
    ;
}


/** \fn static bool reference_substitute( expression e, substitute_ctx *ctx )
 *  \brief callback for gen_recurse
 *  called during bottom-up phase on expression,
 *  in fact we aim to filter reference expressions only
 *  It will used context ctx to substitute references
 *  to ctx->to_substitute by the expression ctx->substitute_by
 *  \param e the expression that may contain a reference to the induction variable
 *  \param ctx the context contains the induction variable, and the substitute expression.
 *  \return always true
 */
static bool reference_substitute( expression e, substitute_ctx *ctx ) {
    /* We filter expression that are references */
    if ( syntax_reference_p( expression_syntax( e ) ) ) {
        /* Check if the reference is the variable we aim to substitute */
        if ( reference_variable( syntax_reference ( expression_syntax( e ) ) ) == ctx->to_substitute ) {

            ifdebug(6) {
                pips_debug( 6, "Find substitution :" );
                print_expression( e );
                pips_debug( 6, "Will replace by :" );
                print_expression( ctx->substitute_by );
            }

            /* Delete the old syntax (no leak ;-))*/
            free_syntax( expression_syntax( e ) );
            /* Substitute by new one */
            expression_syntax( e ) = copy_syntax( expression_syntax( ctx->substitute_by ) );
            expression_normalized( e ) = normalized_undefined;
        }
    }
    return true;
}

/** \fn static bool is_left_part_of_assignment( instruction i, entity v )
 *  \brief Check if a variable is the left part of an assignment
 *  \param i the instruction to check
 *  \param v the variable we are looking for
 *  \return true if v has been found in the left part of i, else false
 */
static bool is_left_part_of_assignment( instruction i, entity v ) {
    bool result = FALSE;
    /* Instruction must be an assignment or a self modifying operator : += /= *= -= */
    if ( instruction_assign_p( i ) //
            || native_instruction_p( i, MULTIPLY_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, DIVIDE_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, PLUS_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, MINUS_UPDATE_OPERATOR_NAME )
            || inc_or_de_crement_instruction_p(i) //
            ) {
        /* Assignment are represented as function call */
        call c = call_undefined;
        if ( instruction_call_p(i) ) {
            c = instruction_call(i);
        } else if ( instruction_expression_p(i) ) {
            syntax s = expression_syntax(instruction_expression(i));
            c = syntax_call( s );
        }

        /* Left part of assignment is the first argument of the call */
        if ( !call_undefined_p(c) ) {
            expression e = EXPRESSION( CAR( call_arguments( c ) ) );
            if ( expression_reference_p( e ) ) {
                if ( is_expression_reference_to_entity_p( e, v ) ) {
                    result = TRUE;
                }
            }
        }
    }
    return result;
}

/** \fn static expression get_right_part_of_assignment( instruction i )
 *  \brief Return an expression corresponding to the right part of an assignment
 *  \param i the instruction from which we want the right part
 *  \return the expression corresponding to right part of an assignment,
 *  or expression_undefined if instruction i is not an assignment.
 */
static expression get_right_part_of_assignment( instruction i ) {
    expression e = expression_undefined;
    /* Instruction must be an assignment or a self modifying operator : += /= *= -= */
    if ( instruction_assign_p( i ) //
            || native_instruction_p( i, MULTIPLY_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, DIVIDE_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, PLUS_UPDATE_OPERATOR_NAME ) //
            || native_instruction_p( i, MINUS_UPDATE_OPERATOR_NAME ) ) {
        /* Assignment are represented as function call */
        call c = call_undefined;
        if ( instruction_call_p(i) ) {
            c = instruction_call(i);
        } else if ( instruction_expression_p(i) ) {
            syntax s = expression_syntax(instruction_expression(i));
            c = syntax_call( s );
        }

        /* Left part of assignment is the arguments of the call but the first one (CDR jump it) */
        if ( !call_undefined_p(c) ) {
            e = EXPRESSION( CAR( CDR( call_arguments( c ) ) ) );
        }
    }
    return e;
}


/** \fn static bool subtitute_induction_statement_in( statement s )
 *  \brief Call during top-down phase while recursing on statements
 *  Will use precondition on each assignment statement to construct
 *  substitution expression when possible
 *  \param s the statement that will be checked
 *  \return always true
 */
static bool subtitute_induction_statement_in( statement s ) {
    bool result = TRUE;
    
    ifdebug( 1 ) {
        pips_debug( 1, "Statement:\n" );
        print_statement( s );
    }

    // Get enclosing loops
    list loops = load_statement_enclosing_loops(s);
    if ( !ENDP( loops ) && !statement_loop_p(s) ) {
        /* s is not a loop, but we must be somewhere inside a loop since we have some (useful) loop_indices */

        /* For the moment, we will only work on assignment */
        if ( instruction_assign_p( statement_instruction( s ) ) //
                || native_instruction_p( statement_instruction( s ), MULTIPLY_UPDATE_OPERATOR_NAME ) //
                || native_instruction_p( statement_instruction( s ), DIVIDE_UPDATE_OPERATOR_NAME ) //
                || native_instruction_p( statement_instruction( s ), PLUS_UPDATE_OPERATOR_NAME ) //
                || native_instruction_p( statement_instruction( s ), MINUS_UPDATE_OPERATOR_NAME )
                || inc_or_de_crement_instruction_p(statement_instruction( s ) )
            ) {

            /* The precondition will be used for building a substitution expression */
            transformer prec = load_statement_precondition( s );
            
            ifdebug( 5 ) {
                pips_debug( 5, "Preconditions:\n" );
                text tmp = text_transformer( prec );
                dump_text( tmp );
                free_text( tmp );
                pips_debug( 5, "Transformers:\n" );
                tmp = text_transformer( load_statement_transformer( s ) );
                dump_text( tmp );
                free_text( tmp );
                print_statement( s );
                pips_debug( 5, "Nb of enclosing loops : %d\n", (int)gen_length(loops) );
            }

            /* Get the set of preconditions equations from which we'll build substitution expressions  */
            transformer trans_range = transformer_range( prec );
            Psysteme D = predicate_system( transformer_relation( transformer_range( prec ) ) );
            Pcontrainte eqs = sc_egalites( D );
            Pcontrainte c;
            
            /* Loop over equations
             for each one we will evaluate if it can be used to produce a substitution expression,
             and if applicable, we'll use this expression to make the substitution. */
            for ( c = eqs; !CONTRAINTE_UNDEFINED_P( c ); c = contrainte_succ( c ) ) {
                Pvecteur vec = contrainte_vecteur( c );
                Pvecteur p;

                /* Loop over variable in this equation
                 simple induction variables are detected
                 The condition is that the equation has to involve only loop
                 indices, a constant, and the variable itself */

                bool found_loop_index = FALSE; /* flag to keep track if we have find at least one loop index */
                expression substitute = expression_undefined; /* the substitution expression */
                Variable induction_variable_candidate = entity_undefined; /* the variable to substitute */
                int induction_variable_candidate_coeff = 0; /* the coefficient associated to the variable */

                for ( p = vec; !VECTEUR_NUL_P( p ); p = vecteur_succ( p ) ) {
                    Variable v = vecteur_var( p ); /* Current variable */
                    Value coeff = vecteur_val( p ); /* Associate coefficient */
                    expression local_expr = expression_undefined; /* will be used to compute "coeff*v" */

                    if ( v == TCST ) {
                        /* This is the constant term in the equation */
                        local_expr = int_to_expression( coeff );
                    } else {
                        /* We have a variable */

                        if(index_of_a_loop_p(v, loops) || loop_invariant_p(v, loops)) {
                            /* We have found a loop index */
                            found_loop_index = TRUE;
                            /* We build an expression "coeff*v" */
                            if ( coeff == -1 ) {
                                /* Coeff is -1, we want "-v" instead of "-1 * v" */
                                local_expr = MakeUnaryCall( entity_intrinsic( UNARY_MINUS_OPERATOR_NAME ), //
                                entity_to_expression( v ) );
                            } else if ( coeff == 1 ) {
                                /* Coeff is 1, we want "v" instead of "1 * v" */
                                local_expr = entity_to_expression( v );
                            } else {
                                /* General case : produce "coeff*v" */
                                local_expr = MakeBinaryCall( entity_intrinsic( MULTIPLY_OPERATOR_NAME ), //
                                int_to_expression( coeff ), //
                                entity_to_expression( v ) );
                            }
                        } else if ( !entity_undefined_p( induction_variable_candidate ) ) {
                            /* We have a variable that is not a loop index,
                             * but we already have one, so we abort substitution */
                            induction_variable_candidate = entity_undefined;
                            break;
                        } else {
                            /* We have a variable that is not a loop index,
                             * and it's the first one, so record it */
                            induction_variable_candidate = v;
                            /* we have to take the opposite (-coeff) because we want
                             * to go from coeff*x+b=0 to x=b/(-coeff) */
                            induction_variable_candidate_coeff = -coeff;
                        }
                    }

                    if ( !expression_undefined_p( local_expr ) ) {
                        /* we have a local_expr (loop index or constant) */
                        if ( expression_undefined_p( substitute ) ) {
                            /* it's the first one */
                            substitute = local_expr;
                        } else {
                            /* it's not the first one, we chain it with a "+" operation */
                            substitute = MakeBinaryCall( entity_intrinsic( PLUS_OPERATOR_NAME ), //
                            local_expr, //
                            substitute );
                        }
                    }

                    ifdebug( 5 ) {
                        if ( !expression_undefined_p( substitute ) ) {
                            pips_debug( 5, "Expression : " );
                            print_syntax( expression_syntax( substitute ) );
                            fprintf( stderr, "\n" );
                        }
                    }
                }

                if ( found_loop_index // The expression depends on loop index
                        // We have found an induction variable
                        && !entity_undefined_p( induction_variable_candidate ) //
                        // The variable is really active (this check shouldn't be useful ?)
                        && induction_variable_candidate_coeff != 0 //
                        // We have a substitution expression to use :-)
                        && !expression_undefined_p( substitute ) //
                        // Variable is modified by this statement
                        && is_modified_entity_in_transformer( load_statement_transformer( s ), //
                        induction_variable_candidate ) //
                        // variable is modified on left part of assignment (no side effects )
                        // or is a simple side-effects statement like "k++;"
                        && is_left_part_of_assignment( statement_instruction( s ), //
                                                         induction_variable_candidate )
                ) {

                    /* Add final division using the coeff */
                    if ( induction_variable_candidate_coeff == -1 ) {
                        /* Instead of dividing by -1, we rather use
                         * unary minus in front of expression */
                        substitute = MakeUnaryCall( entity_intrinsic( UNARY_MINUS_OPERATOR_NAME ), substitute );
                    } else if ( induction_variable_candidate_coeff != 1 ) {
                        /* General case */
                        substitute = MakeBinaryCall( entity_intrinsic( DIVIDE_OPERATOR_NAME ), //
                        substitute, //
                        int_to_expression( induction_variable_candidate_coeff ) );
                    }

                    ifdebug( 1 ) {
                        pips_debug( 1, "Induction substitution : %s => ", //
                                entity_local_name( induction_variable_candidate ) );
                        print_syntax( expression_syntax( substitute ) );
                        fprintf( stderr, "\n" );
                    }

                    expression unsugarized = expression_undefined;
                    if( inc_or_de_crement_instruction_p(statement_instruction( s ) ) ) {
                      /* In the case of unary operator, prepare the assignment
                       * It's a separate case since there's no right hand side
                       */
                      if( native_instruction_p( statement_instruction( s ), POST_INCREMENT_OPERATOR_NAME )
                          || native_instruction_p( statement_instruction( s ), PRE_INCREMENT_OPERATOR_NAME ) ) {
                        unsugarized = MakeBinaryCall( entity_intrinsic( PLUS_OPERATOR_NAME ), //
                                                      substitute, //
                                                      int_to_expression(1) );
                      } else if( native_instruction_p( statement_instruction( s ), POST_DECREMENT_OPERATOR_NAME )
                                 || native_instruction_p( statement_instruction( s ), PRE_DECREMENT_OPERATOR_NAME ) ) {
                        unsugarized = MakeBinaryCall( entity_intrinsic( MINUS_OPERATOR_NAME ), //
                                                      substitute, //
                                                      int_to_expression(1) );
                      }

                    } else {
                      /* Now recurse on expressions in current statement and
                       * replace induction_variable_candidate by substitute expression */
                      substitute_ctx ctx;
                      ctx.to_substitute = induction_variable_candidate;
                      ctx.substitute_by = substitute;

                      expression substitute_on = get_right_part_of_assignment( statement_instruction( s ) );
                      gen_context_recurse ( substitute_on, &ctx, expression_domain, gen_true, reference_substitute );

                      /* Force to generate again the normalized field of the expression */
                      expression_normalized(substitute_on) = NormalizeExpression( substitute_on );

                      /* Handle "update" affection (+=, -= , ...)
                       * Transform z += 1 in z = induction + 1
                       * */
                      if ( native_instruction_p( statement_instruction( s ), MULTIPLY_UPDATE_OPERATOR_NAME ) ) {
                          unsugarized = MakeBinaryCall( entity_intrinsic( MULTIPLY_OPERATOR_NAME ), //
                          substitute, //
                          copy_expression( get_right_part_of_assignment( statement_instruction( s ) ) ) );
                      } else if ( native_instruction_p( statement_instruction( s ), DIVIDE_UPDATE_OPERATOR_NAME ) ) {
                          unsugarized = MakeBinaryCall( entity_intrinsic( DIVIDE_OPERATOR_NAME ), //
                          substitute, //
                          copy_expression( get_right_part_of_assignment( statement_instruction( s ) ) ) );
                      } else if ( native_instruction_p( statement_instruction( s ), PLUS_UPDATE_OPERATOR_NAME ) ) {
                          unsugarized = MakeBinaryCall( entity_intrinsic( PLUS_OPERATOR_NAME ), //
                          substitute, //
                          copy_expression( get_right_part_of_assignment( statement_instruction( s ) ) ) );
                      } else if ( native_instruction_p( statement_instruction( s ), MINUS_UPDATE_OPERATOR_NAME ) ) {
                          unsugarized = MakeBinaryCall( entity_intrinsic( MINUS_OPERATOR_NAME ), //
                          substitute, //
                          copy_expression( get_right_part_of_assignment( statement_instruction( s ) ) ) );
                      }
                    }


                    if ( !expression_undefined_p( unsugarized ) ) {
                        /* substitute expression is no longer needed, but we used
                         * it when building unsugarized and we don't want it to
                         * be freed later, so unreference it now */
                        substitute = expression_undefined;

                        ifdebug( 1 ) {
                            pips_debug( 1, "Unsugar update assignment : " );
                            print_statement( s );
                        }

                        /* we will replace instruction inside statement, so free it first */
                        free_instruction( statement_instruction( s ) );

                        /* Force to generate again the normalized field of the expression */
                        expression_normalized(unsugarized) = NormalizeExpression( unsugarized );

                        /* Construct the unsugarized instruction */
                        statement_instruction( s ) = make_call_instruction( entity_intrinsic( ASSIGN_OPERATOR_NAME ), //
                        CONS( EXPRESSION, //
                                entity_to_expression( induction_variable_candidate ), //
                                CONS(EXPRESSION, unsugarized, NIL ) ) );

                        ifdebug( 1 ) {
                            pips_debug( 1, "Unsugar update assignment : " );
                            print_statement( s );
                        }
                    }
                }

                /* Free substitute expression */
                if ( !expression_undefined_p( substitute ) ) {
                    free_expression( substitute );
                }
            }
            /* Try to avoid memory leak */
            empty_transformer( trans_range );
            transformer_free( trans_range );
        }
    }
    return result;
}

/** \fn bool induction_substitution( char * module_name )
 *  \brief This pass implement the detection and the substitution
 *  of induction variable in loop
 *  \param module_name the name of the module to work on
 *  \return currently always true
 */
bool induction_substitution( char * module_name ) {
    entity module;
    statement module_stat;

    set_current_module_entity( module_name_to_entity( module_name ) );
    module = get_current_module_entity( );

    set_current_module_statement( (statement) db_get_memory_resource( DBR_CODE, module_name, TRUE ) );
    module_stat = get_current_module_statement( );

    set_cumulated_rw_effects( (statement_effects) db_get_memory_resource( DBR_CUMULATED_EFFECTS, module_name, TRUE ) );
    module_to_value_mappings( module );

    set_precondition_map( (statement_mapping) db_get_memory_resource( DBR_PRECONDITIONS, module_name, TRUE ) );

    set_transformer_map( (statement_mapping) db_get_memory_resource( DBR_TRANSFORMERS, module_name, TRUE ) );

    debug_on( "INDUCTION_SUBSTITUTION_DEBUG_LEVEL" );
    pips_debug( 1, "begin\n" );


    pips_debug(6, "finding enclosing loops ...\n");
    set_enclosing_loops_map( loops_mapping_of_statement( module_stat ) );

    /* We now traverse our module's statements. */
    gen_recurse( module_stat, statement_domain, subtitute_induction_statement_in,
            gen_true );

    /* Sanity checks */
    pips_assert( "Consistent final check\n", gen_consistent_p( (gen_chunk *)module_stat ) );

    pips_debug( 1, "end\n" );
    debug_off();

    /* Save modified code to database */
    module_reorder( module_stat );
    DB_PUT_MEMORY_RESOURCE( DBR_CODE, strdup( module_name ), module_stat );

    clean_enclosing_loops( );

    reset_current_module_entity( );
    reset_current_module_statement( );

    reset_cumulated_rw_effects( );
    reset_precondition_map( );
    reset_transformer_map( );

    free_value_mappings( );

    /* Return value */
    bool good_result_p = TRUE;

    return ( good_result_p );

}

/* strength reduction context */
typedef struct {
    hash_table entity_to_coeff; ///< mapping between an entity and the value of the involved induction variable
    hash_table entity_to_entity; ///< mapping between an entity and its strength_reduced value
    entity index;               ///< induction variable
    expression increment;       ///< original loop increment
    expression init;            ///< original loop increment initial value
    list header_statements;     ///< assignments from external to internal var
    list incr_statements;       ///< the internal vars increments
} strength_reduction_context_t;

/* the big stuff is there
 * for each candidate expression, we examine its linear field and look for something
 * link a0.{variable}+a1.TCST+a2.{induction variable}+a3.{other symbolic constant}
 * we normalize this, and if it's ok, we can transform this into
 * a0'.{variable}+a1'.TCST+a3'.{other symbolic constant}
 * and increment by @p ctxt->increment * a2'
 * a new variable is created for each different increment
 */
static bool do_strength_reduction_gather(expression exp, strength_reduction_context_t *ctxt) {
    /* ensure the normalized field is filled */
    NORMALIZE_EXPRESSION(exp);
    normalized n = expression_normalized(exp);
    /* focus on the linear problem */
    if(normalized_linear_p(n) && VALUE_ZERO != vect_coeff(ctxt->index,normalized_linear(n)) ) {
        /* we look for a linear form involving constants and our index */
        entity other = entity_undefined;
        for(Pvecteur iter=normalized_linear(n);
                !VECTEUR_NUL_P(iter);iter=vecteur_succ(iter)) {
            entity var = (entity)vecteur_var(iter);
            /* constant terms and the index are ignored  */
            if(term_cst(iter) ||
                same_entity_p(ctxt->index,var) ||
                entity_constant_p(var)) {
                continue;
            }
            /* the others are stored, but only one per expression
             * eg: what to do with a + b +i ?
             * we also have to pay attention to 2a-6i: this one is ok,
             * it will lead to 2a; a-=3;
             * but not 3a-4i
             */
            else if(entity_undefined_p(other)) {
                Pvecteur pv = vect_copy(normalized_linear(n));
                vect_normalize(pv);
                Value v = vect_coeff(var,pv);
                if(value_one_p(v))
                    other= var;
                else break;//do not manage this case as of now
            }
            else {
                other=entity_undefined;// cannot decide between two not constant entities
                break;
            }
        }
        /* we only take care of scalar variables */
        if(!entity_undefined_p(other) && entity_scalar_p(other) ) {
            /* look for an entity that olds the same increment as ours */
            entity already_there = entity_undefined;
            HASH_FOREACH(entity,e,intptr_t,v,ctxt->entity_to_coeff) {
                if(((intptr_t)vect_coeff(ctxt->index,normalized_linear(n))==v) &&
                        same_entity_p(other,(entity)hash_get(ctxt->entity_to_entity,e)))
                {
                    already_there=e;
                    break;
                }
            }
            Pvecteur pv = vect_copy(normalized_linear(n));
            if(entity_undefined_p(already_there)) {
                /* create a new induction variable */
                already_there=make_new_scalar_variable_with_prefix(
                        entity_user_name(other),get_current_module_entity(),
                        copy_basic(entity_basic(other))
                        );
                /* memorize it for further use:
                 * *(a+i)=(*a+1)+1;
                 * should lead to *a0=*a0+1; a0++;
                 * and not *a1=*a1+1; a1++;
                 */
                Value coeff = vect_coeff(ctxt->index,pv);
                hash_put(ctxt->entity_to_coeff,already_there,(void*)(intptr_t)coeff);
                hash_put(ctxt->entity_to_entity,already_there,other);
                AddEntityToCurrentModule(already_there);
                /* and fill the header / footer / increment */
                intptr_t v;
                ctxt->header_statements=CONS(
                        STATEMENT,
                        make_assign_statement(
                            entity_to_expression(already_there),
                            expression_integer_value(ctxt->init,&v) && v == 0 ?
                            entity_to_expression(other):
                            MakeBinaryCall(
                                entity_intrinsic(PLUS_C_OPERATOR_NAME),
                                entity_to_expression(other),
                                copy_expression(ctxt->init)
                                )
                            ),
                        ctxt->header_statements);
                /* compute the value of the new increment */
                expression new_increment=int_to_expression((int)coeff);
                ctxt->incr_statements=CONS(
                        STATEMENT,
                        call_to_statement(
                            make_call(
                                entity_intrinsic(PLUS_UPDATE_OPERATOR_NAME),
                                make_expression_list(
                                    entity_to_expression(already_there),
                                    make_op_exp(MULTIPLY_OPERATOR_NAME,
                                        copy_expression(ctxt->increment),
                                        new_increment)
                                    )
                                )
                            ),
                        ctxt->incr_statements);

            }
            /* either way regenerate the expression from the patched linear field*/
            vect_erase_var(&pv,ctxt->index);
            vect_chg_var(&pv,other,already_there);
            expression p = Pvecteur_to_expression(pv);
            update_expression_syntax(exp,
                    expression_syntax(p)
                    );
            expression_syntax(p)=syntax_undefined;
            free_expression(p);
            return false;
        }
    }
    return true;
}

/* looks for expression in @p l's body that are linear combination of @p l's index
 * those expressions are strength reduced
 */
static void do_strength_reduction_in_loop(loop l) {
    // parent statement
    statement s = (statement) gen_get_ancestor(statement_domain,l);
    // context
    strength_reduction_context_t ctxt = {
        hash_table_make(hash_pointer,HASH_DEFAULT_SIZE),
        hash_table_make(hash_pointer,HASH_DEFAULT_SIZE),
        loop_index(l),
        range_increment(loop_range(l)),
        range_lower(loop_range(l)),
        NIL,NIL
    };
    // find all possible & relevant cases and fill the context
    gen_context_recurse(loop_body(l),&ctxt,
            expression_domain,do_strength_reduction_gather,gen_null);
    // insert prelude and postlude that take care of the assignment to the iterator
    // plus the increments
    insert_statement(s,
            make_block_statement(ctxt.header_statements),
            true);
    insert_statement(loop_body(l),
            make_block_statement(ctxt.incr_statements),
            false);

    hash_table_free(ctxt.entity_to_coeff);
    hash_table_free(ctxt.entity_to_entity);
}

/* dispatch over all loops */
static
void do_strength_reduction(entity module, statement module_statement) {
    gen_recurse(module_statement,loop_domain,
            gen_true,do_strength_reduction_in_loop);
}

/* this phase is the opposite of induction substitution:
 * it generates induction variables
 * It is a lame implementation without much smart things in it:
 * works only for loops, generates a lot of copy ...
 * But it does the job for the simple case I (SG) need in Terapix
 *
 *
 * after a talk with FI, it appears that transformers should be used
 * to detect induction variable
 *
 * deriving the preconditions with respect to induction variable should
 * also give insightful informations about the strength reduction pattern
 *
 * see paper from Robert Paije
*/
bool strength_reduction(const char *module_name) {
    /* prelude */
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement(
            (statement) db_get_memory_resource(DBR_CODE, module_name, true)
            );
    /* To set up the hash table to translate value into value names */
    set_cumulated_rw_effects((statement_effects)
            db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true)
            );
    /* do the job */
    do_strength_reduction(get_current_module_entity(),get_current_module_statement());
    // we may have done bad things, such has inserting empty statements
    clean_up_sequences(get_current_module_statement());

    /* some declaration statements may have been added */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    reset_semantic_map();
    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();
    return true;
}
