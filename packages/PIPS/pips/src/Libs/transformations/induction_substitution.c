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
#include "database.h"
#include "makefile.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "pipsmake.h"
#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "transformations.h"

/* We use a set to accumulate loop indices during loop traversal */
static set loop_indices_b = HASH_UNDEFINED_VALUE;

/* Context used for substitution with gen_context_recurse */
typedef struct {
    entity to_substitute; /* The induction variable */
    expression substitute_by; /* The substitution expression */
} substitute_ctx;





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

/** \fn static bool subtitute_induction_statement_in( statement s )
 *  \brief Call during top-down phase while recursing on statements
 *  Will push loop indices in global static variable loop_indices_b
 *  and will use precondition on each assignment statement to construct
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

    if ( statement_loop_p( s ) ) {
        /* We have a loop, we keep track of the index */
        entity i = loop_index( statement_loop( s ) );
        
        pips_debug( 3, "Entering level-%d loop, index=%s\n", set_size( loop_indices_b ), entity_name( i ) );
        
        // Add current loop index
        set_add_element( loop_indices_b, loop_indices_b, (Variable) i );
        
    } else if ( !set_empty_p( loop_indices_b ) ) {
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

                        if ( set_belong_p( loop_indices_b, v ) ) {
                            /* We have found a loop index */
                            found_loop_index = TRUE;
                            /* We build an expression "coeff*v" */
                            if ( coeff == -1 ) {
                                /* Coeff is -1, we want "-v" instead of "-1 * v" */
                                local_expr = MakeUnaryCall( entity_intrinsic( UNARY_MINUS_OPERATOR_NAME ), //
                                make_expression_from_entity( v ) );
                            } else if ( coeff == 1 ) {
                                /* Coeff is 1, we want "v" instead of "1 * v" */
                                local_expr = make_expression_from_entity( v );
                            } else {
                                /* General case : produce "coeff*v" */
                                local_expr = MakeBinaryCall( entity_intrinsic( MULTIPLY_OPERATOR_NAME ), //
                                int_to_expression( coeff ), //
                                make_expression_from_entity( v ) );
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
                                make_expression_from_entity( induction_variable_candidate ), //
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

/** \fn static bool subtitute_induction_statement_out( statement s )
 *  \brief During Bottom-up we pop out loop indices
 *  \param s the statement that will be checked
 *  \return always true
 */
static void subtitute_induction_statement_out( statement s ) {
    if ( statement_loop_p( s ) ) {
        entity i = loop_index( statement_loop( s ) );

        pips_debug( 3, "Exiting loop with index %s, size=%d\n",
                entity_name( i ), set_size( loop_indices_b ) );

        /* Sanity check */
        pips_assert( "Current loop indices was not in the base", set_belong_p( loop_indices_b, (Variable)i ) );

        set_del_element( loop_indices_b, loop_indices_b, (Variable) i );
    }
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

    /* We now traverse our module's statements. */
    loop_indices_b = set_make( set_pointer );
    gen_recurse( module_stat, statement_domain, subtitute_induction_statement_in,
            subtitute_induction_statement_out );

    /* Sanity checks */
    pips_assert("Loop index Pbase is empty", set_empty_p(loop_indices_b));
    pips_assert( "Consistent final check\n", gen_consistent_p( (gen_chunk *)module_stat ) );

    pips_debug( 1, "end\n" );
    debug_off();

    /* Save modified code to database */
    module_reorder( module_stat );
    DB_PUT_MEMORY_RESOURCE( DBR_CODE, strdup( module_name ), module_stat );

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
