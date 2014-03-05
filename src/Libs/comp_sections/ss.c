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
#include "all.h"


/*{{{ print_loop */

/*========================================================================*/
/* void fprint_loop(fp, lp): print a loop.
 * 
 * AC 94/06/07
 */

/* copied from reindexing/reindexing_utils.c for debugging */

void printf_loop(lp)
 loop   lp;
{
/*{{{ about */

 FILE *fp = stdout;

 fprintf(fp,"\nLoop index information :");
 fprint_entity_list(fp, CONS(ENTITY, loop_index(lp), NIL));
 fprintf(fp,"\nDomaine (lower, upper, inc):");
 fprint_list_of_exp(fp,
		    CONS(EXPRESSION, range_lower(loop_range(lp)), NIL));
 fprintf(fp,", ");
 fprint_list_of_exp(fp, 
		    CONS(EXPRESSION, range_upper(loop_range(lp)), NIL));
 fprintf(fp,", ");
 fprint_list_of_exp(fp, 
		    CONS(EXPRESSION, range_increment(loop_range(lp)), NIL));
 fprintf(fp,"\n");

/*}}}*/
}

/*}}}*/
/*{{{ my vect var subst function */

/* mimicked this function from paf-tuil/utils.c 
   because it was misbehaving !!
   */
Pvecteur my_vect_var_subst(Pvecteur vect, Variable var, Pvecteur new_vect)
{
 Pvecteur    vect_aux;
 Value       val;

 if ((val = vect_coeff(var,vect)) != 0)
    {
     vect_erase_var(&vect,var);
     vect_aux = vect_multiply(new_vect,val);
     vect = vect_add(vect, vect_aux);
    }

 return(vect);
}

Pvecteur my_vect_substract (pvec1,pvec2)
Pvecteur pvec1,pvec2;
{
        int coeff;
	Pvecteur dvec = vect_dup (pvec1);
/*	Pvecteur var_val;
	for (var_val = pvec2; var_val!= NIL; 
	    vect_add_elem (&dvec,var_of(var_val),-val_of(var_val)),var_val=var_val->succ);
	    */

        Pvecteur v2 = pvec2;
        for ( ; v2!= NULL; v2=v2->succ) {
          coeff = 0-(val_of(v2));
	  vect_add_elem (&dvec,var_of(v2), coeff);
        }
        
	return (dvec);
}

/*}}}*/

/*{{{  PrintSimpleSection */
/*{{{  PrintCompRegions */
void 
PrintCompRegions(list CompList)
{
  if (CompList != NIL) {
    MAP(COMP_DESC, Cdesc,
	{
      PrintSimp(Cdesc);
    }, CompList);
  }
}

/*}}}*/
/*{{{  PrintLinExpr */
void 
PrintLinExpr(LinExpr Lin)
{
  /* call the approprite print routine : check later */
  if (Lin != NULL)
    vect_debug(Lin);
  else
    fprintf(stderr, "NULL");
}

/*}}}*/
/*{{{  PrintSimp */
void 
PrintSimp(comp_desc Dad)
{
  fprintf(stderr, "*********************************************************************\n");
  DisplayDad(Dad);
  fprintf(stderr, "---------------------------------------------------------------------\n");
}

void 
DisplayDad(comp_desc TheDad)
{
  entity Var = reference_variable(comp_desc_reference(TheDad));
  simple_section Dad = comp_sec_hull(comp_desc_section(TheDad));

  fprintf(stderr, "\nData Access Descriptor for %s \n", entity_minimal_name(Var));

  DisplayRefTemp(Dad);
  DisplaySimpleSection(Dad);
  fprintf(stderr, "\n");

}

/*}}}*/
/*{{{  DisplayRefTemp */
void 
DisplayRefTemp(simple_section Dad)
{
  unsigned Rank;
  unsigned int i;
  tag RefType;

  Rank = context_info_rank(simple_section_context(Dad));

  fprintf(stderr, "\nReference Template :: \n[ ");
  for (i = 0; i < Rank; i++) {
    RefType = GetRefTemp(Dad, i);
    switch (RefType) {
    case is_rtype_lininvariant:
      fprintf(stderr, "INV");
      break;
    case is_rtype_linvariant:
      fprintf(stderr, "VAR");
      break;
    case is_rtype_nonlinear:
      fprintf(stderr, "NON");
      break;
    default:
      fprintf(stderr, "NUL");
      break;
    }

    if (i < (Rank - 1))
      fprintf(stderr, ", ");
    else
      fprintf(stderr, " ]\n");
  }
  fprintf(stderr, "\n");
}

/*}}}*/
/*{{{  DisplaySimpSec */

void 
DisplaySimpleSection(simple_section Dad)
{
  unsigned int i, I, J, K;
  unsigned Rank = context_info_rank(simple_section_context(Dad));

  fprintf(stderr, "\nBoundary Pairs ::\n");
  /*{{{  xi = c */

  for (i = 0; i < Rank; i++) {
    PrintLinExpr(LSEC(Dad, i));
    fprintf(stderr, "                <= X%d         <= ", i);
    PrintLinExpr(USEC(Dad, i));
    fprintf(stderr, "\n");
  }
  /*}}}*/

  /*{{{  Xi+Xj = c */
  I = 0;
  K = Rank - I - 2;
  J = I + 1;
  for (; i < ((Rank * (Rank + 1)) / 2); i++) {
    /*{{{  print */
    PrintLinExpr(LSEC(Dad, i));
    fprintf(stderr, "            <= X%d + X%d       <= ", I, J);
    PrintLinExpr(USEC(Dad, i));
    fprintf(stderr, "\n");
    /*}}}*/
    /*{{{  index */
    if (K < 1) {
      I += 1;
      J = I + 1;
      K = Rank - I - 2;
    } else {
      K -= 1;
      J += 1;
    }
    /*}}}*/
  }
  /*}}}*/
  /*{{{  Xi-Xj */
  I = 0;
  K = Rank - I - 2;
  J = I + 1;
  for (; i < (Rank * Rank); i++) {
    PrintLinExpr(LSEC(Dad, i));
    fprintf(stderr, "            <= X%d - X%d       <= ", I, J);
    PrintLinExpr(USEC(Dad, i));
    fprintf(stderr, "\n");
    if (K < 1) {
      I += 1;
      J = I + 1;
      K = Rank - I - 2;
    } else {
      K -= 1;
      J += 1;
    }
  }
  /*}}}*/

  fprintf(stderr, "\n");

}

/*}}}*/
/*}}}*/
/*{{{  auxilliary functions */
/*{{{  GetRefTemp */
tag 
GetRefTemp(simple_section Dad, _int DimNo)
{
  list Rtemp = dad_struct_rtemps(simple_section_dad(Dad));
  tag ret_tag;

  MAP(REF_TEMP, Ref,
      {
	  if (ref_temp_index(Ref) == DimNo) {
             ret_tag = rtype_tag(ref_temp_rtype(Ref));
             break;
	  }
  }, Rtemp);

  return(ret_tag);
}

/*}}}*/
/*{{{  PutRefTemp */
void 
PutRefTemp(simple_section Dad, int DimNo, tag Val)
{
  list Rtemp = dad_struct_rtemps(simple_section_dad(Dad));

  MAP(REF_TEMP, rt,
      {
    if (ref_temp_index(rt) == DimNo) {
      rtype_tag(ref_temp_rtype(rt)) = Val;
      return;
    }
  }, Rtemp);

}

/*}}}*/
/*{{{  GetBoundaryPair */
/* return the lower or upper  boundary */
LinExpr 
GetBoundary(simple_section Dad, int DimNo, unsigned Low)
{
  list BoundPair = dad_struct_shape(simple_section_dad(Dad));
  LinExpr Lin = NULL;

  MAP(BOUND_PAIR, bp,
  {
    if (bound_pair_index(bp) == DimNo) {
      if (Low == 1) {
	Lin = bound_pair_lb(bp);
      } else {
	Lin = bound_pair_ub(bp);
      }
      break;
    }
  }, BoundPair);

  return(Lin);
}

/*}}}*/
/*{{{  PutBoundryPair */
/* substitute with new boundary */
void
PutBoundPair(simple_section Dad, _int DimNo, LinExpr Low, LinExpr Up)
{
  list BoundPair = dad_struct_shape(simple_section_dad(Dad));

  ifdebug(5) {
    pips_debug(5, "PutBoundPair : dump Low and Up vectors \n");
    PrintLinExpr(Low);
    PrintLinExpr(Up);
  }

  MAP(BOUND_PAIR, bp,
      {
    if (bound_pair_index(bp) == DimNo) {
      if (Low != NULL) {
	  /* check later vect_rm(bound_pair_lb(bp));  */
	  bound_pair_lb_(bp) = (Pvecteur) Low;
      }
      if (Up != NULL) {
	  /* check later vect_rm(bound_pair_ub(bp)); */
	  bound_pair_ub_(bp) = (Pvecteur) Up;
      }
      return;
    }
  }, BoundPair);
}

/*}}}*/
/*{{{  MinBoundary */
/* substitute with new boundary */
LinExpr 
MinBoundary(LinExpr Lin1, LinExpr Lin2)
{
  int Result;

  Result = vect_compare(&Lin1, &Lin2);
  if (Result < 0)
    return (vect_dup(Lin1));
  else
    return (vect_dup(Lin2));
}

/*}}}*/
/*{{{  MaxBoundary */
/* substitute with new boundary */
LinExpr 
MaxBoundary(LinExpr Lin1, LinExpr Lin2)
{
  int Result;

  Result = vect_compare(&Lin1, &Lin2);
  if (Result > 0)
    return (vect_dup(Lin1));
  else
    return (vect_dup(Lin2));
}

/*}}}*/
/*{{{  MergeLinExprs */
LinExpr 
MergeLinExprs(LinExpr Expr1, LinExpr Expr2, OpFlag Op)
{
  /* must free Expr1 because vect_add will return a copy : check later */
  if (Op == PLUS)
    return (vect_add(Expr1, Expr2));
  else
    return (my_vect_substract(Expr1, Expr2));
}

/*}}}*/
/*{{{  ComputeIndex */
/* compute index of tSS array for a boundary */
unsigned int 
ComputeIndex(unsigned int I, unsigned int J, unsigned int Rank)
{
  unsigned int LoIndex, HiIndex, Index, i;

  LoIndex = (I < J) ? I : J;
  HiIndex = (I < J) ? J : I;
  Index = 0;
  for (i = 1; i < (LoIndex + 1); i++)
    Index = Index + (Rank - i);
  Index = Index + (HiIndex - LoIndex) - 1;
  return (Index);
}

/*}}}*/
/*{{{  CopyAccVec */
LinExpr 
CopyAccVec(LinExpr Expr)
{
  /* re-used function from alloc.c */
  return (vect_dup(Expr));
}

/*}}}*/

/* check whether a given expresion is a constant */
bool 
IsExprConst(LinExpr Expr)
{
  return (vect_constant_p(Expr));
}


/* check whether loop index variable var is contained in LinExpr */
bool 
DivExists(loop Loop, LinExpr Lin)
{
  entity ent;
  Pvecteur Vec;
  bool Val;

  ent = loop_index(Loop);
  Vec = (Pvecteur) Lin;
  Val = vect_contains_variable_p(Vec, (Variable) ent);
  return (Val);

}

/* return the n'th subscript expression */
expression 
GetAccVec(unsigned No, const reference ref)
{
  unsigned Cur = 0;
  list inds = reference_indices(ref);

  MAP(EXPRESSION, index,
      {
    if (Cur == No)
      return (index);
    Cur++;
  }, inds);

  return (NULL);

}

unsigned int 
CardinalityOf(list gl)
{
  /* re-used Pips functional programs !!! */
  return gen_length(gl);
}



/*}}}*/
/*{{{  allocate simple section */
/* allocate a structure to hold Dad */
dad_struct 
AllocateDadStruct(int Rank)
{
  dad_struct Dad;
  bound_pair Bounds;
  int cnt;
  list RefTempList = NIL;
  list BoundList = NIL;
  rtype Ref = make_rtype(is_rtype_nonlinear, UU);

  /* initialize the reference template */
  for (cnt = 0; cnt < Rank; cnt++) {
    RefTempList = gen_nconc(RefTempList, 
                  CONS(REF_TEMP,make_ref_temp(cnt, Ref), NIL));
  }

  /* initialize the shape */
  for (cnt = 0; cnt < Rank * Rank; cnt++) {
    Bounds = make_bound_pair(cnt, NULL, NULL);
    BoundList = gen_nconc(BoundList, CONS(BOUND_PAIR, Bounds, NIL));
  }

  Dad = make_dad_struct(RefTempList, BoundList);

  return (Dad);
}

simple_section 
AllocateSimpleSection(reference ref)
{
#define LINE 0
  unsigned int Rank;
  context_info Context;
  dad_struct DadStruct;
  simple_section Dad;
  variable TmpVar;

  TmpVar = type_variable(entity_type(reference_variable(ref)));

  /*{{{  intialize the data structure */
  Rank = CardinalityOf(variable_dimensions(TmpVar));
  Context = make_context_info(LINE, Rank, ZERO);
  DadStruct = AllocateDadStruct(Rank);
  Dad = make_simple_section(Context, DadStruct);
  /*}}}*/
  return (Dad);
}

/*}}}*/
/*{{{  translate inner most references */
void 
ScanAllDims(loop Loop, comp_desc Desc)
{
  /*
   * unfortunately an access vector cannot tell to which dimension it
   * corresponds hence this round about of explicitly tracking the dimension
   * number
   */
  unsigned int DimNo;
  expression exp_ind;
  simple_section Dad;
  reference ref;
  list inds;

  pips_debug(3, "begin\n");

  Dad = comp_sec_hull(comp_desc_section(Desc));
  ref = comp_desc_reference(Desc);

  inds = reference_indices(ref);

  for (DimNo = 0; inds != NIL; DimNo++, inds = CDR(inds)) {
    /*{{{  init RT and SS */
    exp_ind = EXPRESSION(CAR(inds));
    ComputeRTandSS(exp_ind, DimNo, Dad, Loop);
    /*}}}*/
  }

  pips_debug(3, "end\n");

}

/* initialise reference template and shapes for inner most references */
/* for an access vector */
void 
ComputeRTandSS(expression Sub, unsigned DimNo, simple_section Dad, loop Loop)
{
  normalized Nexpr;

  Nexpr = NORMALIZE_EXPRESSION(Sub);

  if (!normalized_linear_p(Nexpr)) {
    /*{{{  non-linear */
    /*
     * non linear subscripts if it is non-linear we don't bother to
     * distinguish between invariant and variant ; they are just non-linear
     */
    PutRefTemp(Dad, DimNo, is_rtype_nonlinear);
    /*}}}*/
  } 
  else {
    LinExpr Low, Up;

    /*{{{  linear */
    /* copy the subscript expression into the shape descriptor */
    Low = CopyAccVec(normalized_linear(Nexpr));
    Up = CopyAccVec(normalized_linear(Nexpr));

    PutBoundPair(Dad, DimNo, Low, Up);

    if(IsExprConst(normalized_linear(Nexpr)) == true ) {
      PutRefTemp(Dad, DimNo, is_rtype_linvariant);
    }
    else {
    /* linear subscripts : update if DIV exists */
      if (DivExists(Loop, normalized_linear(Nexpr))) {
        Low = Lbound(Loop, LSEC(Dad, DimNo));
        Up = Ubound(Loop, USEC(Dad, DimNo));
        PutBoundPair(Dad, DimNo, Low, Up);
        PutRefTemp(Dad, DimNo, is_rtype_linvariant);
      }
      else {
         PutRefTemp(Dad, DimNo, is_rtype_lininvariant);
      }
    }
    /*}}}*/
  }
}

/*}}}*/
/*{{{  translate to outer loop  comp list */
/*
 * Input a list of complement descriptors and translate the sections to the
 * enclosing loop context modifies : ListOfComplements
 */

/*{{{  translate the set to the outer loop */
void 
TranslateRefsToLoop(loop ThisLoop, list ListOfComps)
{
  MAP(COMP_DESC, Desc,
      {
    TranslateToLoop(ThisLoop, Desc);
  }, ListOfComps);

}

/*}}}*/
/*{{{  translate to loop */
void 
TranslateToLoop(loop ThisLoop, comp_desc Desc)
{
  tVariants Vars;
  simple_section Dad;

  if (GET_NEST(Desc) == ZERO)  {
    ScanAllDims(ThisLoop, Desc);
    PUT_NEST(Desc,SINGLE);
  }
  else  {
    Vars  = TransRefTemp(ThisLoop, Desc);
    TransSimpSec(Desc, ThisLoop, &Vars);
    PUT_NEST(Desc,MULTI);
  }

  Dad = comp_sec_hull(comp_desc_section(Desc));
  UpdateUnresolved(Dad, ThisLoop); 

}

/*}}}*/
/*{{{  TransRefTemp */
tVariants 
TransRefTemp(loop ThisLoop, comp_desc Desc)
{
  tVariants Vars;
  simple_section Dad;
  _int DimNo, Rank;
  normalized Nexpr;
  expression TmpExpr;
  list OldList, NewList, NewEle;

  
  comp_sec Csec = comp_desc_section(Desc);
  reference Ref = comp_desc_reference(Desc);

  Dad = comp_sec_hull(Csec);
  Rank = context_info_rank(simple_section_context(Dad));

  pips_debug(3, "begin\n");

  OldList = NIL;
  NewList = NIL;

  /* iterate through all entries of reference template */
  for (DimNo = 0; DimNo < Rank; DimNo++) {
    /* process only linear elements */
    if (!(GetRefTemp(Dad, DimNo) == is_rtype_nonlinear)) {
      /* Pass only normalized expression into Divexists */
      TmpExpr = GetAccVec(DimNo, Ref);
      Nexpr = NORMALIZE_EXPRESSION(TmpExpr);

      if (DivExists(ThisLoop,  normalized_linear(Nexpr))) {
	/* variant w.r.t this loop index */
        NewEle = CONS(INT, DimNo, NIL);
	NewList = gen_nconc(NewList, NewEle);
	PutRefTemp(Dad, DimNo, is_rtype_linvariant);
      } else {
	/*
	 * invariant w.r.t. this loop index; but variant w.r.t. previous loop
	 * constant subscripts will also get included in the old list
	 */
	if (GetRefTemp(Dad, DimNo) == is_rtype_linvariant)
	  OldList =  gen_nconc(OldList, CONS(INT, DimNo, NIL));
      }
    }
  }
 
  Vars.Old = OldList;
  Vars.New = NewList;

  pips_debug(3, "end\n");
  return (Vars);
}

/*}}}*/
/*{{{  TransSimpSec */
void 
UpdateUnresolved(simple_section Dad, loop Loop)
{
  /*{{{  declarations */
  unsigned Index, I, J;
  LinExpr lbExpr, ubExpr;
  unsigned ZhiExprNo;
  unsigned Rank = context_info_rank(simple_section_context(Dad));
  /*}}}*/
  /*{{{  Update unresolved boundaries */
  /*{{{  update parallel or diagonal boundaries with induction variable */
  if (Loop != NULL) {
    /*{{{  scan all dimensions */
    for (ZhiExprNo = 0; ZhiExprNo < Rank; ZhiExprNo++) {
      lbExpr = LSEC(Dad, ZhiExprNo);
      ubExpr = USEC(Dad, ZhiExprNo);
      if (DivExists(Loop, lbExpr))
	lbExpr = Lbound(Loop, lbExpr);
      if (DivExists(Loop, ubExpr))
	ubExpr = Ubound(Loop, ubExpr);
      PutBoundPair(Dad, ZhiExprNo, lbExpr, ubExpr);

      /* ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, ZhiExprNo); */

    }
    /*}}}*/
  }
  /*}}}*/
  /*{{{  update diagonal boundaries whose parallel components are constants */
  for (I = 0; I < Rank; I++) {
    for (J = I + 1; J < Rank; J++) {
      if ((GetRefTemp(Dad, I) != is_rtype_nonlinear) && (GetRefTemp(Dad, J) != is_rtype_nonlinear)) {
	/*{{{  process only linear subscripts */
	/*{{{  compute index */
	int PlusOffset, MinusOffset;
	Index = ComputeIndex(I, J, Rank);
	PlusOffset = Rank + Index;
	MinusOffset = ((Rank * (Rank + 1)) / 2) + Index;
	/*}}}*/
	/*{{{  update diagonals */
	/*{{{  update Xi + Xj */
	/* lower boundary */
	/* bug fix : check boundary otherwise it will get overwritten */
	if (LSEC(Dad, PlusOffset) == NULL) {
	  /*{{{  update */
	  if (IsExprConst(LSEC(Dad, I)) && IsExprConst(LSEC(Dad, J))) {
	    /* copy done in MergeLinExpr lbExpr = CopyAccVec(LSEC(Dad, I)); */
	    lbExpr = MergeLinExprs(LSEC(Dad, I), LSEC(Dad, J), PLUS);
	    /* LSEC(Dad,PlusOffset) = lbExpr; */
	    PutBoundPair(Dad, PlusOffset, lbExpr, NULL);
	  }
	  /*}}}*/
	}
	/* upper boundary */
	if (USEC(Dad, PlusOffset) == NULL) {
	  /*{{{  update */
	  if (IsExprConst(USEC(Dad, I)) && IsExprConst(USEC(Dad, J))) {
	    ubExpr = MergeLinExprs(USEC(Dad, I), USEC(Dad, J), PLUS);
	    /* USEC(PlusOffset) = ubExpr; */
	    PutBoundPair(Dad, PlusOffset, NULL, ubExpr);
	  }
	  /*}}}*/
	}
	/*}}}*/
	/*{{{  update Xi - Xj */
	/* lower boundary */
	if (LSEC(Dad, MinusOffset) == NULL) {
	  /*{{{  upate */
	  if (IsExprConst(LSEC(Dad, J)) && IsExprConst(USEC(Dad, I))) {
	    /* lbExpr = CopyAccVec(LSEC(J)); */
	    lbExpr = MergeLinExprs(LSEC(Dad, J), USEC(Dad, I),  MINUS);
	    /* LSEC(MinusOffset) = lbExpr; */
	    PutBoundPair(Dad, MinusOffset, lbExpr, NULL);
	  }
	  /*}}}*/
	}
	/* upper boundary */
	if (USEC(Dad, MinusOffset) == NULL) {
	  /*{{{  update */
	  if (IsExprConst(USEC(Dad, J)) && IsExprConst(LSEC(Dad, I))) {
	    /* ubExpr = CopyAccVec(USEC(J)); */
	    ubExpr = MergeLinExprs(USEC(Dad, J), LSEC(Dad, I), MINUS);
	    /* USEC(MinusOffset) = ubExpr; */
	    PutBoundPair(Dad, MinusOffset, NULL, ubExpr);
	  }
	  /*}}}*/
	}
	/*}}}*/
	/*}}}*/
	/*}}}*/
      }
    }
  }
  /*}}}*/
  /*}}}*/
}

/* compute both boundary expression and store in the tSS array */
void 
ComputeBoundaries(simple_section Dad, loop Loop, LinExpr lbExpr, LinExpr ubExpr, unsigned Offset)
{
  LinExpr Low, Up;

  Low = lbExpr;
  Up = ubExpr;

  /*{{{  about */
  if (DivExists(Loop, lbExpr))
    Low = Lbound(Loop, lbExpr);
  if (DivExists(Loop, ubExpr))
    Up = Ubound(Loop, ubExpr);
  PutBoundPair(Dad, Offset, Low, Up);
  /*}}}*/
}

void 
TransSimpSec(comp_desc Desc, loop Loop, tVariants * Vars)
{
  /*{{{  declarations */
  unsigned Index, Offset;
  LinExpr lbExpr, ubExpr, TmpExpr1, TmpExpr2;

  simple_section Dad = comp_sec_hull(comp_desc_section(Desc));
  unsigned Rank = context_info_rank(simple_section_context(Dad));
  list Old = Vars->Old;
  list New = Vars->New;
  /*}}}*/

  pips_debug(3, "begin\n");

  /*{{{  Diagonal Boundaries for old and new */
  MAP(INT, I,
      {
    MAP(INT, J,
	{
      /*{{{  compute offset in the tSS array  to store diagonal boundaries */
      Index = ComputeIndex(I, J, Rank);
      /*}}}*/
      /*{{{  compute diagonal boundaries Xi + Xj */
      /* compute index for tSS array */
      Offset = Rank + Index;
      /* copying done inside MergeLinExprs 
      lbExpr = CopyAccVec(LSEC(Dad, I));
      TmpExpr1 = CopyAccVec(LSEC(Dad, J));
      */
      lbExpr = LSEC(Dad, I);
      TmpExpr1 = LSEC(Dad, J);
      lbExpr = MergeLinExprs(lbExpr, TmpExpr1, PLUS);
      /* 
      ubExpr = CopyAccVec(USEC(Dad, I));
      TmpExpr2 = CopyAccVec(USEC(Dad, J));
      */
      ubExpr = USEC(Dad, I);
      TmpExpr2 = USEC(Dad, J);
      ubExpr = MergeLinExprs(ubExpr, TmpExpr2, PLUS);
      ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, Offset);
      /*}}}*/
      /*{{{  compute diagonal boundaries Xi - Xj */
      /* compute index for tSS array */
      Offset = (Rank * (Rank + 1)) / 2 + Index;
      /*
      lbExpr = CopyAccVec(LSEC(Dad, I));
      TmpExpr1 = CopyAccVec(USEC(Dad, J));
      */
      lbExpr = LSEC(Dad, I);
      TmpExpr1 = USEC(Dad, J);
      lbExpr = MergeLinExprs(lbExpr, TmpExpr1, MINUS);
      /*
      ubExpr = CopyAccVec(USEC(Dad, I));
      TmpExpr2 = CopyAccVec(LSEC(Dad, J));
      */
      ubExpr = USEC(Dad, I);
      TmpExpr2 = LSEC(Dad, J);
      ubExpr = MergeLinExprs(ubExpr, TmpExpr2, MINUS);
      ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, Offset);
      /*}}}*/
    }, New);
  }, Old);
  /*}}}*/
  /*{{{  Compute parallel boundaries only for new variants */
  MAP(INT, J,
      {
    /*{{{  compute Xi = c */
	  /* bug unsigned Offset = J; */
    lbExpr = CopyAccVec(LSEC(Dad, J));
    ubExpr = CopyAccVec(USEC(Dad, J));
    ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, J);
    /*}}}*/
  }, New);
  /*}}}*/
  /*{{{  Compute diagonal boundaries for new variants */
  /* set size must be atleast 2 */
  if (CardinalityOf(New) > 1) {
    int I, J;

    /*{{{  compute diagonals */
    /* set an  iterator for the newvar */
    list New_iter1;
    list New_iter2;
    New_iter1 = Vars->New;
    for (; !ENDP(New_iter1); (New_iter1 = CDR(New_iter1))) {
      /*{{{  iterators 1 and 2 */
      I = INT(CAR(New_iter1));
      /* set this iterator to next element in the first list */
      New_iter2 = CDR(New_iter1);
      /*}}}*/
      for (; !ENDP(New_iter2); New_iter2 = CDR(New_iter2)) {
	J = INT(CAR(New_iter2));
	/*{{{  compute index for storing in tSS array */
	Index = ComputeIndex(I, J, Rank);
	/*}}}*/
	/*{{{  compute diagonal boundary Xi + Xj */
	Offset = Rank + Index;
        TmpExpr1 = LSEC(Dad, J);
	lbExpr = MergeLinExprs(LSEC(Dad, I), TmpExpr1, PLUS);
        TmpExpr2 = USEC(Dad, J);
	ubExpr = MergeLinExprs(USEC(Dad, I), TmpExpr2, PLUS);
	ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, Offset);
	/*}}}*/
	/*{{{  compute diagonal boundary Xi - Xj */
	Offset = ((Rank * (Rank + 1)) / 2) + Index;
        TmpExpr1 = USEC(Dad, J);
	lbExpr = MergeLinExprs(LSEC(Dad, I), TmpExpr1, MINUS);
        TmpExpr2 = LSEC(Dad, J);
	ubExpr = MergeLinExprs(USEC(Dad, I), TmpExpr2, MINUS);
	ComputeBoundaries(Dad, Loop, lbExpr, ubExpr, Offset);
	/*}}}*/
      }
    }
    /*}}}*/
  }
  /*}}}*/

  pips_debug(3, "end\n");
}

/*}}}*/
/*}}}*/
/*{{{  lbound and ubound */
/*
 * compute lower bound input : a loop header and a linear expression output :
 * updated  boundary expression modifies : the input linear expression
 */
LinExpr 
Lbound(loop Loop, LinExpr Lin)
{
  /*
   * preconditions : lbound is invoked only if DivExists and if only on
   * LinExprs
   */

  expression TmpExpr;
  normalized Nexpr;
  LinExpr NewLin, NewVect;
  entity Var = loop_index(Loop);

  /* only one step in pips ! it provides direct substitution functions */
  /*{{{  substitution step */
  /*
   * use the lower/upper bound of the index variable depending on the sign of
   * the coefficient
   */
  /*
   * this needs to be changed if the representation for linear expression is
   * changed to accommodate symbolic coefficients
   */

  int Val = vect_coeff((Variable) Var, Lin);
  if ((Val > 0)) {
    /* substitute with lower bound */
    TmpExpr = range_lower(loop_range(Loop));
  } else {
    /* substitute with upper bound */
    TmpExpr = range_upper(loop_range(Loop));
  }

  Nexpr  = NORMALIZE_EXPRESSION(TmpExpr);
  if (normalized_linear_p(Nexpr)) {
      /* make a copy because vect_var_subst modifies NewVect */
     NewVect = CopyAccVec(normalized_linear(Nexpr));     
     NewLin = my_vect_var_subst(Lin, (Variable) Var, NewVect);
  }
  else {
     NewLin = NULL;
  }

  /*}}}*/
  return (NewLin);
}

/*
 * compute upper bound input : a loop header and a linear expression output :
 * updated  boundary expression modifies : the input linear expression
 */
LinExpr 
Ubound(loop Loop, LinExpr Lin)
{
  /*
   * preconditions : Ubound is invoked only if DivExists and only on LinExprs
   */
  expression TmpExpr;
  LinExpr NewLin, NewVect;  
  normalized Nexpr;

  entity Var = loop_index(Loop);

  /* only one step in pips ! it provides direct substitution functions */
  /*{{{  substitution step */
  /*
   * use the lower/upper bound of the index variable depending on the sign of
   * the coefficient
   */
  /*
   * this needs to be changed if the representation for linear expression is
   * changed to accommodate symbolic coefficients
   */

  int Val = vect_coeff((Variable) Var, Lin);
  if ((Val < 0)) {
    /* substitute with lower bound */
    TmpExpr = range_lower(loop_range(Loop));
  } else {
    /* substitute with upper bound */
    TmpExpr = range_upper(loop_range(Loop));
  }

  Nexpr  = NORMALIZE_EXPRESSION(TmpExpr);
  if (normalized_linear_p(Nexpr) ) {
     /* make a copy because vect_var_subst modifies NewVect */
     NewVect = CopyAccVec(normalized_linear(Nexpr));     
     NewLin = my_vect_var_subst(Lin, (Variable) Var, NewVect);
  }
  else {
     NewLin = NULL;
  }

  /*}}}*/
  return (NewLin);
}

/*}}}*/
/*{{{  SimpUnion */
/*
 * input : two simple descriptors output : the union descriptor modifies :
 * nothing
 */
simple_section 
SimpUnion(simple_section S1, simple_section S2)
{
/* SG: i am unsure this is a valid init */
  simple_section UnionDad = make_simple_section(context_info_undefined,make_dad_struct(NIL,NIL));
  tag Variant;
  size_t i;
  LinExpr Low, Up;
  size_t Rank = context_info_rank(simple_section_context(S1));

  /* allocate a simple_section */

  /*{{{  update reference template */
  /* update reference template */
  /* if a subscript is invariant or boundary expression is not constant */
  /* then whole dimension is assumed to be used */
  for (i = 0; i < Rank; i++) {
    Variant = ((GetRefTemp(S1, i) == is_rtype_linvariant) &&
	       (GetRefTemp(S2, i) == is_rtype_linvariant)) ?
      is_rtype_linvariant : is_rtype_nonlinear;
    PutRefTemp(UnionDad, i, Variant);
  }
  /*}}}*/
  /*{{{  update simple sections */
  /* scan all boundary pairs */
  for (i = 0; i < (Rank * Rank); i++) {
    /* later insert code for releasing the space of LSEC and USEC */
    if ((LSEC(S1, i) != NULL) && (LSEC(S2, i) != NULL)) {
      /* compute new lower boundary in the union */
      Low = MinBoundary(LSEC(S1, i), LSEC(S2, i));
    }
    if ((USEC(S1, i) != NULL) && (USEC(S2, i) != NULL)) {
      /* compute new upper boundary in the union */
      Up = MaxBoundary(USEC(S1, i), USEC(S2, i));
    }
    PutBoundPair(UnionDad, i, Low, Up);
  }
  /*}}}*/

  return (UnionDad);
}

/*}}}*/
