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
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "ri-util.h"
#include "effects-util.h"

#include "sac.h"

static list transformations = NIL; /* <transformation> */

void insert_transformation(char * name, int vectorLengthOut, int subwordSizeOut, int vectorLengthIn, int subwordSizeIn, int nbArgs, list mapping)
{
    int i;
    list l;
    transformation t = make_transformation(strdup(name),
            vectorLengthOut,
            subwordSizeOut,
            vectorLengthIn,
            subwordSizeIn,
            nbArgs,
            (int*)malloc(sizeof(int)*vectorLengthOut));

    for(i = vectorLengthOut-1, l=mapping; (i>=0) && (l!=NIL); i--, l=CDR(l))
        transformation_mapping(t)[i] = INT(CAR(l));

    transformations = CONS(TRANSFORMATION, t, transformations);
}

#if 0
static statement make_transformation_statement(transformation t, entity vdest, entity vsrc1, entity vsrc2)
{
    list l=NIL;

    if (transformation_nbArgs(t) == 3)
        l = CONS(EXPRESSION, entity_to_expression(vsrc2), l);
    l = CONS(EXPRESSION, entity_to_expression(vsrc1), l);
    l = CONS(EXPRESSION, entity_to_expression(vdest), l);

    return make_exec_statement_from_name(transformation_name(t), l);
}

static bool best_transformation_p(transformation t1, transformation t2)
{
    return ((t2 == transformation_undefined) || 
            (transformation_nbArgs(t1) < transformation_nbArgs(t2)));
}
static statement generate_transformation_statement(simdStatementInfo si, int line)
{
    int vectorLength = opcode_vectorSize(simdStatementInfo_opcode(si));
    //MOIint subwordSize = opcode_subwordSize(simdStatementInfo_opcode(si));
    int offset = line * vectorLength;

    list t;
    transformation bestTr = transformation_undefined;
    entity bestFirstArg = entity_undefined;
    entity bestSecArg = entity_undefined;

    //Look up all the registered transformations to find the 
    //best one
    for(t = transformations; t != NIL; t = CDR(t))
    {
        int i;
        list firstArg;
        list secArg;
        list l;
        transformation tr = TRANSFORMATION(CAR(t));

        //Dismiss transformation that do not have the right kind of
        //arguments
        if ((transformation_vectorLengthOut(tr) != vectorLength) /*MOI||
                                                                   (transformation_subwordSizeOut(tr) != subwordSize)*/)
            continue;

        //Find out if some vectors may be used, based on the first element
        firstArg = secArg = NIL;
        for(l = statementArgument_dependances(simdStatementInfo_arguments(si)[offset]);
                l != NIL; 
                l = CDR(l))
        {
            vectorElement ve = VECTORELEMENT(CAR(l));

            if ( (vectorElement_vectorLength(ve) != transformation_vectorLengthIn(tr)) //||
                    //MOI	      (vectorElement_subwordSize(ve) != transformation_subwordSizeIn(tr)) 
               )
                continue;

            if (vectorElement_element(ve) == transformation_mapping(tr)[0])
                firstArg = CONS(ENTITY, vectorElement_vector(ve), firstArg);
            if ((vectorElement_element(ve) | 0x100) == transformation_mapping(tr)[0])
                secArg = CONS(ENTITY, vectorElement_vector(ve), secArg);
        }

        //Now verify these vector can actually be used, by looking at all
        //the other elements
        for( i = 1; 
                (i < vectorLength) && (firstArg != NIL) && 
                ((transformation_nbArgs(tr) == 2) || (secArg != NIL)); 
                i ++ )
        {
            statementArgument ssa = simdStatementInfo_arguments(si)[i + offset];
            list first = NIL;
            list sec = NIL;

            //Build the list of the vectors that match for the current element
            for(l = statementArgument_dependances(ssa); l != NIL; l = CDR(l))
            {
                vectorElement ve = VECTORELEMENT(CAR(l));

                if (vectorElement_element(ve) == transformation_mapping(tr)[i])
                    first = CONS(ENTITY, vectorElement_vector(ve), first);
                if ((vectorElement_element(ve) | 0x100) == transformation_mapping(tr)[i])
                    sec = CONS(ENTITY, vectorElement_vector(ve), sec);
            }

            //Merge the lists to get vectors that match on all elements tested 
            //up to now
            gen_list_and(&firstArg, first);
            gen_free_list(first);

            gen_list_and(&secArg, sec);
            gen_free_list(sec);
        }

        //If the transformation can be used, test if it is better than some
        //other we may have found
        if ( (firstArg != NIL) &&
                ((transformation_nbArgs(tr) == 2) || (secArg != NIL)) &&
                (best_transformation_p(tr, bestTr)) )
        {
            bestTr = tr;
            bestFirstArg = ENTITY(CAR(firstArg));
            if (secArg != NIL)
                bestSecArg = ENTITY(CAR(secArg));
        }

        gen_free_list(firstArg);
        gen_free_list(secArg);
    }

    //No appliable transformation found, so exit now: can't do anything!
    if (bestTr == transformation_undefined)
        return statement_undefined;

    //We found out if there is some appliable transformation, and we even 
    //have chosen the best one. Now is time to generate the actual 
    //statement.
    return make_transformation_statement(bestTr, 
            simdStatementInfo_vectors(si)[line],
            bestFirstArg,
            bestSecArg);
}
#endif
