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
/*{{{  File banner*/
/*
 * Project : Research
 * Author  : Manjunathaiah
 *           University of Southampton
 *  
 *   Title : routines for complementary sections computation
 *  System : PIPS compiler
 *Filename : ss.h
 * Version : 1
 *    Date : 18/4/96 
 */
/*}}}*/

#ifndef _DAD
#define _DAD

#include "all.h"

/*{{{  defines*/
/* classifying subscript types for reference template 
 * done in newgen now
 */
#define LIN_INVARIANT		2
#define LIN_VARIANT			1
#define NON_LINEAR		 -1

/* accessing DAD components */
#define LSEC(x,i)	 	GetBoundary(x,i,1)
#define USEC(x,i)	 	GetBoundary(x,i,0)


/*}}}*/

enum BoundType {LOWER, UPPER};
enum RefType {READ, WRITE};
enum NestType {ZERO, SINGLE, MULTI};

/* used for merging linear expressions */
typedef enum { PLUS, MINUS} OpFlag;

/* only used for in print routines. The analysis does not
 * pose any limit on the number of array dimensions
 */
#define MAX_RANK 8

/*{{{  Dad definition*/
  /* data structures for data access descriptor */
  
  /* Reference Template part of DAD, an array of integers allocated dynamically */
  typedef unsigned int tRT;

  /*  A linear expression in Pips ; Pvecteur is a pointer */
  typedef Svecteur *LinExpr;  

  /* bounds are retained as high level tree structures to accommodate symbolic
     information in boundary expressions. When all the symbolic information
     gets resolved then the tree nodes are collapsed into a single instruction
     holding the constant value  
  */

  typedef struct sSimpBound 
  {
    LinExpr lb; /* lower bound */
    LinExpr ub; /* upper bound */
  } tSS;

  /* Simple Section part of DAD
   An array of type SimpBound struct allocated 
   dynamically based on rank of array
  */

  typedef struct DadComponent 
  {
    tRT  *RefTemp;
    tSS  *SimpSec;
  } DadComp;


/*}}}*/
/*{{{  Data structures required for computing Dads*/

/*{{{  structures for TranslateToLoop*/
/* structure to hold both Old and New variants */
typedef struct Variants  {
  list Old;
  list New;
}tVariants;

/*}}}*/

/*}}}*/

typedef simple_section tDad;

/* function prototypes */
/* ss.c */
extern tag GetRefTemp(simple_section Dad, int DimNo);
extern void PutRefTemp(simple_section Dad, int DimNo, tag Val);
extern LinExpr GetBoundary(simple_section Dad, int DimNo, unsigned Low);
extern void PutBoundPair(simple_section Dad, int DimNo, LinExpr Low, LinExpr Up);
extern LinExpr MinBoundary(LinExpr Lin1, LinExpr Lin2);
extern LinExpr MaxBoundary(LinExpr Lin1, LinExpr Lin2);
extern LinExpr MergeLinExprs(LinExpr Expr1, LinExpr Expr2, OpFlag Op);
extern unsigned int ComputeIndex(unsigned int I, unsigned int J, unsigned int Rank);
extern LinExpr CopyAccVec(LinExpr Expr);
extern bool IsExprConst(LinExpr Expr);
extern bool DivExists(loop Loop, LinExpr Lin);
extern expression GetAccVec(unsigned No, const reference ref);
extern unsigned int CardinalityOf(list gl);
extern dad_struct AllocateDadStruct(int Rank);
extern simple_section AllocateSimpleSection(reference ref);
extern void ScanAllDims(reference ref, simple_section Dad);
extern void ComputeRTandSS(expression Sub, unsigned DimNo, simple_section Dad);
extern void TranslateRefsToLoop(loop ThisLoop, list ListOfComps);
extern void TranslateToLoop(loop ThisLoop, comp_desc Desc);
extern tVariants *TransRefTemp(loop ThisLoop, comp_desc Desc);
extern void ComputeBoundaries(simple_section Dad, loop Loop, LinExpr lbExpr, LinExpr ubExpr, unsigned Offset);
extern void TransSimpSec(simple_section Dad, loop Loop, tVariants *Vars);
extern void Lbound(loop Loop, LinExpr Lin);
extern void Ubound(loop Loop, LinExpr Lin);
extern simple_section SimpUnion(simple_section S1, simple_section S2);

#endif

