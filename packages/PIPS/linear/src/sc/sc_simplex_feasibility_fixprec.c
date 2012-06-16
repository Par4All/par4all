/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* test du simplex :
 * Si on compile grace a` "make simp" dans le repertoire
 * /projects/C3/Linear/Development/polyedre/Tests
 * alors on peut tester l'execution dans le meme directory
 * en faisant : make test_simp
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* To replace #define NB_EQ and #define NB_INEQ - BC, 2/4/96 - 
 * NB_EQ and NB_INEQ are initialized at the beginning of the main subroutine.
 * they represent the number of non-NULL constraints in sc. This is useful
 * to allocate the minimum amount of memory necessary.  */
static int NB_EQ = 0;
static int NB_INEQ = 0;


/************************************************************* DEBUG MACROS */
/* debug macros may be trigered with -DDEBUG{,1,2}
 */
#ifndef DEBUG
#define DEBUG(code) { }
#else 
#undef DEBUG
#define DEBUG(code) { code }
#endif

#ifndef DEBUG1
#define DEBUG1(code) { }
#else 
#undef DEBUG1
#define DEBUG1(code) { code }
#endif

#ifndef DEBUG2
#define DEBUG2(code) { }
#else 
#undef DEBUG2
#define DEBUG2(code) { code }
#endif

#ifndef DEBUG3
#define DEBUG3(code) { }
#else 
#undef DEBUG3
#define DEBUG3(code) { code }
#endif

#ifdef CONTROLING

#include <signal.h>
#define CONTROLING_TIMEOUT_SIMPLEX 

static void 
control_catch_alarm_Simplex (int sig)
{
  alarm(0); /*clear the alarm */
}

#endif

/*************************************************************** CONSTANTS */

/* Hmmm. To be compatible with some weird old 16-bit constants... RK */
enum {
  PTR_NIL = INTPTR_MIN+767,
  INFINI = INTPTR_MAX-767,
  MAX_VAR = 1971, /* nombre max de variables */
  /* seuil au dela duquel on se mefie d'un overflow
   */
#if defined(LINEAR_VALUE_IS_LONGLONG)
  MAXVAL = 576,
#else
  MAXVAL = 24  
#endif
};


#define DIMENSION sc->dimension
#define NUMERO hashtable[h].numero
#define SOLUBLE(N) soluble=N;goto FINSIMPLEX ;
#define CREVARVISIBLE variables[compteur-3]=compteur-2;
#define CREVARCACHEE { variablescachees[nbvariables]=nbvariables + MAX_VAR ; \
			 if (nbvariables ++ >= MAX_VAR) abort(); }

/* for tracing macros after expansion: 
 */
#define tag(x) /* printf(x); */

/*************************************************** MACROS FOR FRACTIONS */
/* maybe most of them should be functions?
 */

/* G computes j=gcd(a,b) assuming b>0 and better with a>b.
 * there can be no artihmetic errors.
 */
#define G(j,a,b)					\
{tag("G")						\
    j=b;						\
    if (value_gt(b,VALUE_ONE))				\
    {							\
	Value i=a, k;					\
	while(value_notzero_p(k=value_mod(i,j)))	\
	    i=j, j=k;					\
    }							\
    if (value_neg_p(j))					\
	value_oppose(j);				\
}

#define GCD(j,a,b)				\
{tag("GCD")					\
    if (value_gt(a,b)) 				\
      { G(j,a,b); } 				\
    else 					\
      { G(j,b,a); }				\
}
/*************************************************** Replacement of macro GCD to GCD_ZERO_CTRL: DN.
Explication:
- Negative numbers (e.g: lines 238-239, variables x.num, x.den, see older version) may be given to 
macro GCD(j,a,b). This macro calls G(j,a,b), which assumes b > 0 and better with a>b (means a>b>0).
(note that G(j,a,b) works well with 0 < a < b)
- If a < 0 then G(j,a,b) still works fine, but GCD(j,a,b) will be wrong:
  if b > 0, a < 0, then we alwayls have GCD(a,b) = abs(a). 
  moreover, when b < 0, if we put G(a,b) = b => This is confusing!
Proposition:
- macro GCDZZN(j,a,b): ZxZ)->N+: 
  GCDZZN(a,b) = GCD(value_absolute(a),value_absolute(b)). 
- Remark, is that the Greatest Commom Divisor of a and b where a = 0 or b = 0 makes no sense.
  So we should test if a = 0 or b = 0 by macro GCD_ZERO_CTRL(j,a,b) before send it to GCDNNZ(j,a,b).
  Then we can assume a != 0 and b != 0. If a = 0 or b = 0 then GCD_ZERO_CTRL(a,b) = 1.
****************************************************/

#define GCDZZN(j,a,b)					\
{tag("GCDZZN")						\
    Value i,k;						\
    i = (value_absolute(a)), k = (value_absolute(b));	\
    while(value_notzero_p(j=value_mod(i,k)))	\
	    i=k, k=j;					\
    j = k;						\
}

#define GCD_ZERO_CTRL(j,a,b)				\
{tag("GCD_ZERO_CTRL")					\
    if ((value_notzero_p(a)) && (value_notzero_p(b)))	\
        { GCDZZN(j,a,b);}				\
    else \
   	{fprintf(stderr,"********************************************************************Error : GCD of number zero !");	\
   	j = (VALUE_ONE);}			\
}


/* SIMPL normalizes rational a/b (b<>0):
 *   divides by gcd(a,b) and returns with b>0
 * note that there should be no arithmetic exceptions within this macro:
 * (well, only uminus may have trouble for VALUE_MIN...)
 * ??? a==0 ? then a = a/b = 0 and b = b/b = 1.
 */
#define SIMPL(a,b)					\
{							\
    if (value_notone_p(b) && value_notone_p(a) && 	\
	value_notmone_p(b) && value_notmone_p(a))	\
    {							\
	register Value i=a, j=b, k;			\
	while (value_notzero_p(k=value_mod(i,j)))	\
	    i=j, j=k;					\
	value_division(a,j), value_division(b,j);	\
    }							\
    if (value_neg_p(b))					\
	value_oppose(a), value_oppose(b);		\
}

/* SIMPLIFIE normalizes fraction f
 */
#define SIMPLIFIE(f) 				\
{tag("SIMPLIFIE") 				\
     if (value_zero_p(f.den))			 \
       { MET_ZERO(f);}				 \
     else {					 \
       if (value_zero_p(f.num))			\
	 f.den = VALUE_ONE;			\
     else					\
       SIMPL(f.num,f.den);	}		\
}

#define AFF(x,y) {x.num=y.num; x.den=y.den;} /* x=y should be ok:-) */
#define AFF_PX(x,y) {x->num=y.num; x->den=y.den;}
#define INV(x,y) {x.num=y.den; x.den=y.num;} /* x=1/y */
/* ??? value_zero_p(y.num)? 
Then x.num != VALUE_ZERO and x.den = VALUE_ZERO, it's not good at all.(assuming y.den != VALUE_ZERO) 
This means : test if  y = 0 then x = 0 else x = 1/y
Change in line 286 : DN.*/

#define INV_ZERO_CTRL(x,y) {if (value_zero_p(y.num)) {fprintf(stderr,"ERROR : inverse of fraction zero !"); x.num = VALUE_ZERO; x.den = VALUE_ONE;} else {INV(x,y)}}


#define METINFINI(f) {f.num=VALUE_MAX;  f.den=VALUE_ONE;}
#define MET_ZERO(f)  {f.num=VALUE_ZERO; f.den=VALUE_ONE;}
#define MET_UN(f)    {f.num=VALUE_ONE;  f.den=VALUE_ONE;}

#define EGAL1(x) (value_eq(x.num,x.den))
#define EGAL0(x) (value_zero_p(x.num))
#define NUL(x) (value_zero_p(x.num))

#define NEGATIF(x)					\
  ((value_neg_p(x.num) && value_pos_p(x.den)) || 	\
   (value_pos_p(x.num) && value_neg_p(x.den)))

#define POSITIF(x)					\
  ((value_pos_p(x.num) && value_pos_p(x.den)) || 	\
   (value_neg_p(x.num) && value_neg_p(x.den)))

#define SUP1(x)								   \
  ((value_pos_p(x.num) && value_pos_p(x.den) &&  value_gt(x.num,x.den)) || \
   (value_neg_p(x.num) && value_neg_p(x.den) &&  value_gt(x.den,x.num)))

#define EGAL_MACRO(x,y,mult)					\
  ((value_zero_p(x.num) && value_zero_p(y.num)) || 		\
   (value_notzero_p(x.den) && value_notzero_p(y.den) && 	\
    value_eq(mult(x.num,y.den),mult(x.den,y.num))))

/*#define INF_MACRO(x,y,mult) (value_lt(mult(x.num,y.den),mult(x.den,y.num)))
  DN: c'est pas assez pour la comparaison entre deux fractions, qu'est-ce qui se passe
  s'il y a seulement un denominateur negatif??? Ca donnera un resultat faux.
  a/b < c/d <=> if b*d > 0 then a*d < b*c else a*d < b*c
*/

#define INF_MACRO(x,y,mult) ((value_pos_p(mult(x.den,y.den)) && value_lt(mult(x.num,y.den),mult(x.den,y.num))) || (value_neg_p(mult(x.den,y.den)) && value_gt(mult(x.num,y.den),mult(x.den,y.num))))

/* computes x = simplify(y/z)
 */

/*#define DIV_MACRO(x,y,z,mult)			\
{tag("DIV_MACRO")				\
    if (value_zero_p(y.num))			\
    {						\
	MET_ZERO(x);				\
    }						\
    else					\
    {						\
	x.num=mult(y.num,z.den);		\
	x.den=mult(y.den,z.num);		\
	SIMPLIFIE(x);				\
    }						\
}
*/

/* DN: This macro DIV_MACRO doesn't test if z = 0, 
then x = y/z means x.num = y.num*z.den and x.den = 0. 
We better add the test : if z = 0 then x.num = 0 and x.den = 1
We try to avoid the denominator equal to 0.
*/

#define DIV_MACRO(x,y,z,mult)			\
{tag("DIV_MACRO")				\
    if (value_zero_p(y.num))			\
    {						\
	MET_ZERO(x);				\
    }						\
    else					\
    {					\
           if (value_zero_p(z.num))		\
           {					\
	      fprintf(stderr,"ATTENTION : divided by zero number!");	\
	      MET_ZERO(x);			\
           }					\
           else					\
{						\
	      x.num=mult(y.num,z.den);		\
	      x.den=mult(y.den,z.num);		\
	      SIMPLIFIE(x);			\
}						\
    }						\
}


/* computes x = simplify(y*z)
 */
#define MUL_MACRO(x,y,z,mult) 				\
{tag("MUL_MACRO")					\
    if(value_zero_p(y.num) || value_zero_p(z.num))	\
	MET_ZERO(x)					\
    else 						\
    {							\
	x.num=mult(y.num,z.num);			\
        x.den=mult(y.den,z.den);			\
	SIMPLIFIE(x);					\
    }							\
}

/* computes X = simplify(A-B)
 */
#define SUB_MACRO(X,A,B,mult)						      \
{ tag("SUB_MACRO")							      \
    if (value_zero_p(A.num))						      \
	X.num = value_uminus(B.num),					      \
	X.den = B.den;							      \
    else if (value_zero_p(B.num))					      \
    { AFF(X, A); }							      \
    else if (value_eq(A.den,B.den))					      \
    {									      \
	X.num = value_minus(A.num,B.num);				      \
	X.den = A.den;							      \
	if (value_notone_p(A.den))					      \
	    { SIMPLIFIE(X);}						      \
    }									      \
    else /* must compute the stuff: */					      \
    {									      \
	Value ad=A.den, bd=B.den, gd, v;				      \
	GCD_ZERO_CTRL(gd,ad,bd);							      \
	if (value_notone_p(gd)) value_division(ad,gd), value_division(bd,gd); \
        X.num = mult(A.num,bd);						      \
        v = mult(B.num,ad);						      \
	value_substract(X.num,v);					      \
	v = mult(ad,bd);						      \
	X.den = mult(v,gd);						      \
	SIMPLIFIE(X);							      \
    }									      \
}

/* computes X = A - B*C/D, trying to avoid arithmetic exceptions...
 */
#define FULL_PIVOT_MACRO_SIOUX(X,A,B,C,D,mult) 				\
{									\
    frac u,v,w; tag("FULL_PIVOT_SIOUX")					\
    AFF(u,B); AFF(v,C); INV_ZERO_CTRL(w,D); /* u*v*w == B*C/D */			\
    SIMPL(u.num,v.den); SIMPL(u.num,w.den);				\
    SIMPL(v.num,u.den); SIMPL(v.num,w.den);				\
    SIMPL(w.num,u.den); SIMPL(w.num,v.den);				\
    u.num = mult(u.num,v.num); /* u*=v */				\
    u.den = mult(u.den,v.den);						\
    u.num = mult(u.num,w.num); /* u*=w */				\
    u.den = mult(u.den,w.den);						\
    SUB_MACRO(X,A,u,mult);						\
}

/* computes X = A - B*C/D, but does not try to avoid arithmetic exceptions
 */
#define FULL_PIVOT_MACRO_DIRECT(X,A,B,C,D,mult)				  \
{									  \
    Value v; tag("FULL_PIVOT_DIRECT")					  \
    X.num = mult(A.num,B.den);						  \
    X.num = mult(X.num,C.den);						  \
    X.num = mult(X.num,D.num);						  \
    v = mult(A.den,B.num);						  \
    v = mult(v,C.num);							  \
    v = mult(v,D.den);							  \
    value_substract(X.num,v);						  \
    X.den = mult(A.den,B.den);						  \
    X.den = mult(X.den,C.den);						  \
    X.den = mult(X.den,D.num);						  \
    SIMPLIFIE(X);							  \
}

#define direct_p(v) (value_lt(v,MAXVAL))

/* computes X = A - B*C/D, with a switch to use SIOUX or DIRECT
 * thae rationale for the actual condition is quite fuzzy.
 */
#define FULL_PIVOT_MACRO(X,A,B,C,D,mult)				\
{ tag("FULL_PIVOT")							\
    if (direct_p(A.den) && direct_p(B.den) &&				\
	direct_p(C.den) && direct_p(value_abs(D.num)))			\
    {									\
	FULL_PIVOT_MACRO_DIRECT(X,A,B,C,D,mult);			\
    }									\
    else								\
    {									\
	FULL_PIVOT_MACRO_SIOUX(X,A,B,C,D,mult);				\
    }									\
} 

/* idem if A==0
 */
#define PARTIAL_PIVOT_MACRO_SIOUX(X,B,C,D,mult)		\
{ tag("PARTIAL_PIVOT_SIOUX")				\
    frac u;						\
    MUL_MACRO(u,B,C,mult); /* u=simplify(b*c) */	\
    DIV_MACRO(X,u,D,mult); /* x=simplify(u/d) */	\
    value_oppose(X.num);   /* x=-x */			\
}

#define PARTIAL_PIVOT_MACRO_DIRECT(X,B,C,D,mult)	\
{ tag("PARTIAL_PIVOT_DIRECT")				\
    X.num = mult(B.num,C.num);				\
    X.num = mult(X.num,D.den);				\
    value_oppose(X.num);				\
    X.den = mult(B.den,C.den);				\
    X.den = mult(X.den,D.num);				\
    SIMPLIFIE(X);					\
}

#define PARTIAL_PIVOT_MACRO(X,B,C,D,mult)			\
{								\
    if (direct_p(B.den) && direct_p(C.den) && direct_p(D.num))	\
    {								\
	PARTIAL_PIVOT_MACRO_DIRECT(X,B,C,D,mult);		\
    }								\
    else							\
    {								\
	PARTIAL_PIVOT_MACRO_SIOUX(X,B,C,D,mult);		\
    }								\
}

/* Pivot :  x = a - b c / d
 * mult is used for multiplying values.
 * the macro has changed a lot, for indentation and so... FC.
 * DN: Why don't we test d.num = 0 ? (meanwhile, we do test d.den = 0)
 */
	     
#define PIVOT_MACRO(X,A,B,C,D,mult)					      \
{ if (value_zero_p(D.num)) fprintf(stderr,"division of zero!!!");	      \
    DEBUG3(fprintf(stdout, "pivot on: ");				      \
	   printfrac(A); printfrac(B); printfrac(C); printfrac(D));	      \
   if (value_zero_p(A.num))/* a==0? */					      \
   {									      \
       if (value_zero_p(B.num) || value_zero_p(C.num) || value_zero_p(D.den)) \
	   { MET_ZERO(X);}						      \
       else /* b*c/d != 0, calculons! */				      \
	   { PARTIAL_PIVOT_MACRO(X,B,C,D,mult);}			      \
   }									      \
   else /* a!=0 */							      \
      if (value_zero_p(B.num) || value_zero_p(C.num) || value_zero_p(D.den))  \
	  { AFF(X,A);}							      \
      else /*  b*c/d != 0, calculons! */				      \
	  if (value_one_p(D.num) && value_one_p(A.den) &&		      \
	      value_one_p(B.den) && value_one_p(C.den))			      \
	  { /* no den to compute */					      \
	      Value v = mult(B.num,C.num);				      \
              v = mult(v,D.den);					      \
	      X.num=value_minus(A.num,v);				      \
	      X.den=VALUE_ONE;						      \
	  }								      \
	  else /* well, we must compute the full formula! */		      \
	      { FULL_PIVOT_MACRO(X,A,B,C,D,mult);}			      \
    DEBUG3(fprintf(stdout, " = "); printfrac(X); fprintf(stdout, "\n"));      \
}

/* multiplies two Values of no arithmetic overflow, or throw exception.
 * this version is local to the simplex. 
 * note that under some defined macros value_mult can expand to 
 * value_protected_mult, which would be ok.
 */
#undef value_protected_mult
#define value_protected_mult(v,w) \
    value_protected_multiply(v,w,THROW(simplex_arithmetic_error))

/* Version with and without arithmetic exceptions...
 */
#define MULT(RES,A,B) RES=value_mult(A,B)
#define MULTOFL(RES,A,B) RES=value_protected_mult(A,B)

#define DIV(x,y,z) DIV_MACRO(x,y,z,value_mult)
#define DIVOFL(x,y,z) DIV_MACRO(x,y,z,value_protected_mult)

#define MUL(x,y,z) MUL_MACRO(x,y,z,value_mult)
#define MULOFL(x,y,z) MUL_MACRO(x,y,z,value_protected_mult)

#define SUB(X,A,B) SUB_MACRO(X,A,B,value_mult)
#define SUBOFL(X,A,B) SUB_MACRO(X,A,B,value_protected_mult)

#define PIVOT(X,A,B,C,D) PIVOT_MACRO(X,A,B,C,D,value_mult)
#define PIVOTOFL(X,A,B,C,D) PIVOT_MACRO(X,A,B,C,D,value_protected_mult)

#define EGAL(x,y) EGAL_MACRO(x,y,value_mult)
#define EGALOFL(x,y) EGAL_MACRO(x,y,value_protected_mult)

#define INF(x,y) INF_MACRO(x,y,value_mult)
#define INFOFL(x,y) INF_MACRO(x,y,value_protected_mult)
static frac frac0={(Value)0,(Value)1,0} ;

/* this is already too much...
 */

typedef struct
{
    Variable nom;
    int numero; 
    int hash ;
    Value val ;
    intptr_t succ ;
} hashtable_t;


void frac_init(frac *f, int n)
{
  int i;
  for (i=0;i<n;i++) {
    f[i].num =  (Value)0;
    f[i].den =  (Value)1;
    f[i].numero =  0;
  }
}

void frac_simpl(Value  *a,Value *b)
{
  if (value_notone_p(*b) && value_notone_p(*a) &&
      value_notmone_p(*b) && value_notmone_p(*a))
    {
      register long long int lli = ABS(VALUE_TO_LONG(*a)), llj= ABS(VALUE_TO_LONG(*b)),k;
      if (lli<llj)  {
	long long int tmp = lli;
	lli=llj;llj=tmp;}

      while ((k=lli%llj) !=0)
	lli=llj, llj=k;
      value_division(*a,llj), value_division(*b,llj);
    }
  if (value_neg_p(*b))
    value_oppose(*a), value_oppose(*b);
}

/* simplifie normalizes fraction f
 */
void frac_simplifie(frac *f)
{
  tag("frac_simplifie")
    if (value_zero_p(f->den))
      {
	MET_ZERO((*f));}
    else {
      if (value_zero_p(f->num))
	f->den = VALUE_ONE;
      else
	frac_simpl(&(f->num),&(f->den));
    }
}

void frac_div(frac *x,frac y,frac z, bool ofl_ctrl)
{tag("FRAC_DIV")
    if (value_zero_p(y.num))
      {
	MET_ZERO((*x));
      }
    else
      {
	if (value_zero_p(z.num))
	  {
	    fprintf(stderr,"ATTENTION : divided by zero number!");
	    MET_ZERO((*x));
	  }
	else
	  {
	    if (ofl_ctrl == FWD_OFL_CTRL) {
	    x->num=value_protected_mult(y.num,z.den);
	    x->den=value_protected_mult(y.den,z.num);
	    }
	    else {
	      x->num=value_mult(y.num,z.den);
	      x->den=value_mult(y.den,z.num);
	    }
	    frac_simplifie(x);
	  }
      }
}


/* computes x = simplify(y*z)
 */
void  frac_mul(frac *x,frac y,frac z, bool ofl_ctrl)
{tag("FRAC_MUL")
    if(value_zero_p(y.num) || value_zero_p(z.num))
      MET_ZERO((*x))
      else
	{
	  if (ofl_ctrl == FWD_OFL_CTRL) {
	  x->num=value_protected_mult(y.num,z.num);
	  x->den=value_protected_mult(y.den,z.den);
	  }
	  else {
	    x->num=value_mult(y.num,z.num);
	    x->den=value_mult(y.den,z.den);
	  }
	  frac_simplifie(x);
	}
}
void frac_sub(frac *X,frac A,frac B, bool ofl_ctrl)
{ tag("FRAC_SUB")
    if (value_zero_p(A.num))
      X->num = value_uminus(B.num),
	X->den = B.den;
    else if (value_zero_p(B.num))
      { AFF_PX(X, A); }
    else if (value_eq(A.den,B.den))
      {
	X->num = value_minus(A.num,B.num);
	X->den = A.den;
	if (value_notone_p(A.den))
	  { frac_simplifie(X);}
      }
    else /* must compute the stuff: */
      {
	Value ad=A.den, bd=B.den, gd, v;
	GCD_ZERO_CTRL(gd,ad,bd);
	if (value_notone_p(gd)) value_division(ad,gd), value_division(bd,gd);
	if (ofl_ctrl == FWD_OFL_CTRL) {
	  X->num = value_protected_mult(A.num,bd);
	  v = value_protected_mult(B.num,ad);
	  value_substract(X->num,v);
	  v =value_protected_mult(ad,bd);
	  X->den = value_protected_mult(v,gd);
	}
	else {
	  X->num = value_mult(A.num,bd);
	  v = value_mult(B.num,ad);
	  value_substract(X->num,v);
	  v =value_mult(ad,bd);
	  X->den = value_mult(v,gd);
	}
	frac_simplifie(X);
      }
}

void full_pivot_sioux(frac *X,frac A,frac B,frac C,frac D,bool ofl_ctrl)
{
  frac u,v,w;
  tag("FULL_PIVOT_SIOUX")
    AFF(u,B); AFF(v,C); INV_ZERO_CTRL(w,D); /* u*v*w == B*C/D */
  frac_simpl(&(u.num),&(v.den));
  frac_simpl(&(u.num),&(w.den));
  frac_simpl(&(v.num),&(u.den));
  frac_simpl(&(v.num),&(w.den));
  frac_simpl(&(w.num),&(u.den));
  frac_simpl(&(w.num),&(v.den));
  if (ofl_ctrl == FWD_OFL_CTRL) {
    u.num = value_protected_mult(u.num,v.num); /* u*=v */
    u.den = value_protected_mult(u.den,v.den);
    u.num =value_protected_mult(u.num,w.num); /* u*=w */
    u.den =value_protected_mult(u.den,w.den);
  }
  else {
    u.num = value_mult(u.num,v.num); /* u*=v */
    u.den = value_mult(u.den,v.den);
    u.num =value_mult(u.num,w.num); /* u*=w */
    u.den =value_mult(u.den,w.den);
  }
  frac_sub(X,A,u,ofl_ctrl);

}

/* computes X = A - B*C/D, but does not try to avoid arithmetic exceptions
 */
void full_pivot_direct(frac *X,frac A,frac B,frac C,frac D,bool ofl_ctrl)
{
  Value v; tag("FULL_PIVOT_DIRECT") 
	     if (ofl_ctrl == FWD_OFL_CTRL) {
	       X->num = value_protected_mult(A.num,B.den);
	       X->num = value_protected_mult(X->num,C.den);
	       X->num = value_protected_mult(X->num,D.num);
	       v = value_protected_mult(A.den,B.num);
	       v = value_protected_mult(v,C.num);
	       v = value_protected_mult(v,D.den);
	       value_substract(X->num,v);
	       X->den = value_protected_mult(A.den,B.den);
	       X->den = value_protected_mult(X->den,C.den);
	       X->den = value_protected_mult(X->den,D.num);
	     }
	     else {
	       X->num = value_mult(A.num,B.den);
	       X->num = value_mult(X->num,C.den);
	       X->num = value_mult(X->num,D.num);
	       v = value_mult(A.den,B.num);
	       v = value_mult(v,C.num);
	       v = value_mult(v,D.den);
	       value_substract(X->num,v);
	       X->den = value_mult(A.den,B.den);
	       X->den = value_mult(X->den,C.den);
	       X->den = value_mult(X->den,D.num);
	     }
  frac_simplifie(X);
}
void  full_pivot(frac *X,frac A,frac B,frac C,frac D,bool ofl_ctrl)
{ tag("FULL_PIVOT")

    if (direct_p(A.den) && direct_p(B.den) &&
	direct_p(C.den) && direct_p(value_abs(D.num)))
      {
	full_pivot_direct(X,A,B,C,D,ofl_ctrl);
      }
    else
      {
	full_pivot_sioux(X,A,B,C,D,ofl_ctrl);
      }
}

/* idem if A==0
 */
void partial_pivot_sioux(frac *X,frac B,frac C,frac D,bool ofl_ctrl)
{
  tag("PARTIAL_PIVOT_SIOUX")
    frac u =  {(Value)0,(Value)1,0};

  frac_mul(&u,B,C,ofl_ctrl); /* u=simplify(b*c) */
  frac_div(X,u,D,ofl_ctrl); /* x=simplify(u/d) */
  value_oppose(X->num);   /* x=-x */
}

void partial_pivot_direct(frac *X,frac B,frac C,frac D,bool ofl_ctrl)
{ tag("PARTIAL_PIVOT_DIRECT") 
    if (ofl_ctrl == FWD_OFL_CTRL) {
      X->num = value_protected_mult(B.num,C.num);
      X->num = value_protected_mult(X->num,D.den);
      value_oppose(X->num);
      X->den =value_protected_mult(B.den,C.den);
      X->den = value_protected_mult(X->den,D.num);
    }
    else {
      X->num = value_mult(B.num,C.num);
      X->num = value_mult(X->num,D.den);
      value_oppose(X->num);
      X->den =value_mult(B.den,C.den);
      X->den = value_mult(X->den,D.num);
    }
  frac_simplifie(X);
}

void  partial_pivot(frac *X,frac B,frac C,frac D,bool ofl_ctrl)
{

  if (direct_p(B.den) && direct_p(C.den) && direct_p(D.num))
    {
      partial_pivot_direct(X,B,C,D,ofl_ctrl);
    }
  else
    {
      partial_pivot_sioux(X,B,C,D,ofl_ctrl);
    }
}



void  pivot(frac *X,frac A,frac B,frac C,frac D,bool ofl_ctrl)
{
  if (value_zero_p(D.num))
    fprintf(stderr,"division of zero!!!");
  DEBUG3(fprintf(stdout, "pivot on: ");
	 printfrac(A);
	 printfrac(B);
	 printfrac(C);
	 printfrac(D));
  if (value_zero_p(A.num))/* a==0? */
    {
      if (value_zero_p(B.num) || value_zero_p(C.num) || value_zero_p(D.den))
	{ MET_ZERO((*X)); }
      else /* b*c/d != 0, calculons! */
	{ partial_pivot(X,B,C,D,ofl_ctrl);}
    }
  else /* a!=0 */
    if (value_zero_p(B.num) || value_zero_p(C.num) || value_zero_p(D.den))
      { AFF_PX(X,A);}
    else /*  b*c/d != 0, calculons! */
      if (value_one_p(D.num) && value_one_p(A.den) &&
	  value_one_p(B.den) && value_one_p(C.den))
	{ /* no den to compute */
	  Value v;
	  if (ofl_ctrl == FWD_OFL_CTRL) {
	    v= value_protected_mult(B.num,C.num);
	    v = value_protected_mult(v,D.den);
	  }
	  else {
	    v= value_mult(B.num,C.num);
	    v = value_mult(v,D.den);
	  }
	  X->num=value_minus(A.num,v);
	  X->den=VALUE_ONE;
	}
      else /* well, we must compute the full formula! */
	{ full_pivot(X,A,B,C,D,ofl_ctrl);}
  DEBUG3(fprintf(stdout, " = ");
	 printfrac(X); fprintf(stdout, "\n"));
}

/* For debugging: */
static void  __attribute__ ((unused))
dump_hashtable(hashtable_t hashtable[]) {
  int i;
  for(i=0;i<MAX_VAR;i++)
    if(VARIABLE_DEFINED_P(hashtable[i].nom))
      printf("%s %d ", (char *) hashtable[i].nom, hashtable[i].numero),
	print_Value(hashtable[i].val),
	printf("\n");
}

/* Le nombre de variables visibles est : compteur-2
 * La i-eme variable visible a le numero : variables[i+1]=i
 *   (0 <= i < compteur-2)
 * Le nombre de variables cachees est : nbvarables
 * La i-eme variable cachee a le numero : variablescachees[i+1]=MAX_VAR+i-1
 *   (0 <= i < nbvariables)
 */
/* utilise'es par dump_tableau ; a rendre local */
static int nbvariables, variablescachees[MAX_VAR], variables[MAX_VAR] ; 

static void printfrac(frac x) {
    printf(" "); print_Value(x.num);
    printf("/"); print_Value(x.den);
}

/* For debugging: */
static void  __attribute__ ((unused))
dump_tableau(char *msg, tableau *t, int colonnes) {
    int i,j, k, w;
    int max=0;
    for(i=0;i<colonnes;i++) 
      if(t[i].colonne[t[i].taille-1].numero>max)
	  max=t[i].colonne[t[i].taille-1].numero ; 
    printf("\nTableau (%s): %d colonnes  %d lignes\n",msg,colonnes,max) ;
    printf("%d Variables  visibles :",colonnes-2) ;
    for(i=0;i<colonnes-2;i++) printf(" %d",variables[i]) ;
    printf("\n%d Variables cachees :",nbvariables);
    for(i=0;i<nbvariables;i++) printf(" %d",variablescachees[i]) ;
    printf("\n") ;
    for(j=0;j<=max;j++) {
	printf("\nLigne %d ",j) ;
	for(i=0;i<colonnes;i++) {
	    w=1 ;
	    for(k=0;k<t[i].taille;k++)
		if(t[i].colonne[k].numero==j)
		  printfrac(t[i].colonne[k]) , w=0 ;
	    if(w!=0)printfrac(frac0) ;
	}
    }
    printf("\n\n");
} /* dump_tableau */


/* calcule le hashcode d'un pointeur
   sous forme d'un nombre compris entre 0 et  MAX_VAR */
static int hash(Variable s) 
{ long l ;
  l=(long)s % MAX_VAR ;
  return l ;
}

/* fonction de calcul de la faisabilite' d'un systeme
 * d'equations et d'inequations
 * Auteur : Robert Mahl, Date : janvier 1994
 *
 * Retourne : 
 *   1 si le systeme est soluble (faisable) en rationnels,
 *   0 s'il n'y a pas de solution.
 *
 * overflow control :
 *  ofl_ctrl == NO_OFL_CTRL  => no overflow control
 *  ofl_ctrl == FWD_OFL_CTRL  
 *           => overflow control is made THROW(overflow_error,5)
 * BC, 13/12/94
 */
bool 
sc_simplex_feasibility_ofl_ctrl_fixprec(
    Psysteme sc, 
    int ofl_ctrl)
{
    Pcontrainte pc, pc_tmp ;
    Pvecteur pv ;
    /* All the folowing automatic variables are used when coming back from
     * longjmp (i.e. in a CATCH block) so they need to be declared volatile as
     * specified by the documentation*/      
    intptr_t volatile premier_hash = PTR_NIL; /* tete de liste des noms de variables */
    /* Necessaire de declarer "hashtable" static 
     *  pour initialiser tout automatiquement a` 0.
     * Necessaire de chainer les enregistrements
     *  pour reinitialiser a 0
     *  en sortie de la procedure.
     */
    static hashtable_t volatile hashtable[MAX_VAR];
    Pbase volatile saved_base;
    int volatile saved_dimension;
    // tableau * volatile eg = NULL; /* tableau des egalite's  */
    tableau * volatile t = NULL; /* tableau des inegalite's  */
    frac * volatile nlle_colonne = NULL;
    /* les colonnes 0 et 1 sont reservees au terme const: */
    int compteur = 2 ;
    intptr_t i, j, k, h, trouve, ligne, i0, i1, jj, ii ;
    Value poidsM, valeur, tmpval;
    intptr_t w ;
    int soluble; /* valeur retournee par feasible */
    frac* colo;
    frac objectif[2] ; /* objectif de max pour simplex : 
			  somme des (b2,c2) termes constants "inferieurs" */
    frac rapport1, rapport2, min1, min2, piv, cc ;

    
    DEBUG(static int simplex_sc_counter = 0;)
      /* count number of calls of this function (at the beginning)       */

      soluble=1;/* int soluble = 1;*/

    rapport1 =frac0, rapport2 =frac0, min1 =frac0, min2 =frac0, piv=frac0, cc =frac0 ;
    objectif[0] = frac0, objectif[1] = frac0;
    i=-1, j=-1, k=-1, h=-1, trouve=-1, ligne=-1, i0=-1, i1=-1, jj=-1, ii=-1;
    poidsM =-1, valeur=-1, tmpval=-1,w=-1;/*DN.*/

    /* recompute the base so as to only allocate necessary columns
     * some bases are quite large although all variables do not appear in
     * actual contraints. The base is used to store all variants in
     * preconditions for instance.
     */
  
    saved_base = sc_base(sc);
    saved_dimension = sc_dimension(sc);
    sc_base(sc) = BASE_NULLE;

    sc_creer_base(sc);


    /* Allocation a priori du tableau des egalites.
     * "eg" : tableau a "nb_eq" lignes et "dimension"+2 colonnes.
     */
    if (ofl_ctrl!=FWD_OFL_CTRL)
	fprintf(stderr, "[sc_simplexe_feasibility_ofl_ctrl] "
		"should not (yet) be called with control %d...\n", ofl_ctrl);

    /* DEBUG(fprintf(stdout, "\n\n IN sc_simplexe_feasibility_ofl_ctrl:\n");
       sc_fprint(stdout, sc, default_variable_to_string);)*/

    DEBUG(simplex_sc_counter ++;
		  fprintf(stderr,"BEGIN SIMPLEX : %d th\n",simplex_sc_counter);
		  sc_default_dump(sc);/*sc_default_dump_to_file(); print to file	  */
	  );

    /* the input Psysteme must be consistent; this is not the best way to
     * do this; array bound checks should be added instead in proper places;
     * no time to do it properly for the moment. BC.
     */
	linear_assert("sc is weakly consistent", sc_weak_consistent_p(sc));

    /* Do not allocate place for NULL constraints */
    NB_EQ = 0;
    NB_INEQ = 0;
    for(pc_tmp = sc->egalites; pc_tmp!= NULL; pc_tmp=pc_tmp->succ)
    {
	if (pc_tmp->vecteur != NULL)
	    NB_EQ++;
    }
    for(pc_tmp = sc->inegalites; pc_tmp!= NULL; pc_tmp=pc_tmp->succ)
    {
	if (pc_tmp->vecteur != NULL)
	    NB_INEQ++;
    }

    CATCH(simplex_arithmetic_error|timeout_error|overflow_error)
    {
      /*      ifscdebug(2) {
	fprintf(stderr,"[sc_simplexe_feasibility_ofl_ctrl] arithmetic error\n");
	}
      */
      DEBUG(fprintf(stderr, "arithmetic error or timeout in simplex\n"););

	for(i = premier_hash ; i != PTR_NIL; i = hashtable[i].succ) {
	  hashtable[i].nom =  VARIABLE_UNDEFINED ;
	  hashtable[i].numero = 0 ;
	  hashtable[i].hash = 0 ;
	  hashtable[i].val = (Value)0 ;
	}

      for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++)
	free(t[i].colonne);
      free(t);
      free(nlle_colonne);

      /* restore initial base */
      base_rm(sc_base(sc));
      sc_base(sc) = saved_base;
      sc_dimension(sc) = saved_dimension;

#ifdef CONTROLING
      alarm(0); /*clear the alarm*/
#endif

      if (ofl_ctrl == FWD_OFL_CTRL) {
	RETHROW(); /*rethrow whatever the exception is*/
      }
      /*THROW(user_exception_error);*/
      /* need CATCH(user_exception_error) before calling sc_simplexe_feasibility)*/
      ifscdebug(5) {fprintf(stderr,"DNDNDN WARNING: Exception not treated, return feasible!");}
      return true; /* if don't catch exception, then default is feasible */

      /*if (ofl_ctrl == FWD_OFL_CTRL)  */
      /*THROW(overflow_error);*/

      /*return true;  default is feasible */
    }/* of CATCH(simplex_arithmetic_error)*/

    /*begin of TRY*/

#ifdef CONTROLING
    /*start the alarm*/
    if (CONTROLING_TIMEOUT_SIMPLEX) {
      signal(SIGALRM, controling_catch_alarm_Simplex);
      alarm(CONTROLING_TIMEOUT_SIMPLEX);
    } /*else nothing*/
#endif



    /* Allocation a priori du tableau du simplex "t" par
     * colonnes. Soit
     * "dimension" le nombre de variables, la taille maximum
     * du tableau est de (1 + nb_ineq) lignes
     * et de (2 + dimension + nb_ineq + nb_eq) colonnes
     * On y ajoute en fait le double du nombre d'egalite's.
     * Ce tableau sera rempli de la facon suivante :
     * - ligne 0 : critere d'optimisation
     * - lignes 1 a nb_ineq : les inequations
     * - colonne 0 : le terme constant (composante de poids 1)
     * - colonne 1 : le terme constant (composante de poids M)
     * - colonnes 2 et suivantes : les elements initiaux
     *   et les termes d'ecart
     * Le tableau a une derniere colonne temporaire pour
     *  pivoter un vecteur unitaire.
     *     */

    t = (tableau*)malloc((3 + NB_INEQ + NB_EQ + DIMENSION)*sizeof(tableau));
    for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++) {
        t[i].colonne= (frac*) malloc((4 + 2*NB_EQ + NB_INEQ)*sizeof(frac)) ;
	frac_init(t[i].colonne,4 + 2*NB_EQ + NB_INEQ);
        t[i].existe = 0 ;
        t[i].taille = 1 ;
        t[i].colonne[0].numero = 0 ;
        t[i].colonne[0].num = VALUE_ZERO ;
	t[i].colonne[0].den = VALUE_ONE ;
    }
    nbvariables= 0 ;
    /* Initialisation de l'objectif */

    for(i=0;i<=1;i++)
	objectif[i].num=VALUE_ZERO, objectif[i].den=VALUE_ONE;

    DEBUG2(dump_hashtable(hashtable);)

    /* Entree des inegalites dans la table */

    for(pc=sc->inegalites, ligne=1; pc!=0; pc=pc->succ, ligne++)
    {
	pv=pc->vecteur;
	if (pv!=NULL) /* skip if empty */
	{
	    valeur = VALUE_ZERO ;
	    poidsM = VALUE_ZERO ;
	    for(; pv !=0 ; pv=pv->succ)
		if(vect_coeff(pv->var,sc_base(sc)))
		    value_addto(poidsM,pv->val) ;
		else
		    valeur = value_uminus(pv->val) ; /* val terme const */

	    for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
		if(value_notzero_p(vect_coeff(pv->var,sc_base(sc)))) {
		    h = hash((Variable)  pv->var) ; trouve=0 ;
		    while (VARIABLE_DEFINED_P(hashtable[h].nom))  {
			if (hashtable[h].nom==pv->var) {
			    trouve=1 ;
			    break ;
			}
			else { h = (h+1) % MAX_VAR ; }
		    }
		    if(!trouve) {
			hashtable[h].succ=premier_hash ;
			premier_hash = h ;
			hashtable[h].val = VALUE_NAN ;
			hashtable[h].numero=compteur++ ;
			hashtable[h].nom=pv->var ;
			CREVARVISIBLE ;
		    }
		    linear_assert("current NUMERO in bound",
						  (NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
		    if (value_neg_p(poidsM) || 
			(value_zero_p(poidsM) && value_neg_p(valeur)))
			{value_addto(t[NUMERO].colonne[0].num,pv->val),
			   t[NUMERO].colonne[0].den = VALUE_ONE ;}
		    t[NUMERO].existe = 1 ;
		    t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
		    if(value_neg_p(poidsM) || 
		       (value_zero_p(poidsM) && value_neg_p(valeur)))
			tmpval = value_uminus(pv->val) ; 
		    else tmpval = pv->val ;
		    t[NUMERO].colonne[t[NUMERO].taille].num = tmpval ;
		    t[NUMERO].colonne[t[NUMERO].taille].den = VALUE_ONE ;
		    t[NUMERO].taille++ ;
		}
	    }

	    /* Creation de variable d'ecart ? */
	    if(value_neg_p(poidsM) ||
	       (value_zero_p(poidsM) && value_neg_p(valeur))) {
		DEBUG1(dump_tableau("cre var ec", t, compteur);)
		i=compteur++ ;
		CREVARVISIBLE ;
		t[i].existe = 1 ; t[i].taille = 2 ;
		t[i].colonne[0].num = VALUE_ONE ;
		t[i].colonne[0].den = VALUE_ONE ;
		DEBUG1(printf("ligne ecart = %ld, colonne %ld\n",ligne,i);)
		t[i].colonne[1].numero = ligne ;
		t[i].colonne[1].num = VALUE_MONE ;
		t[i].colonne[1].den = VALUE_ONE ;
		value_oppose(poidsM);
		value_oppose(valeur);
		value_addto(objectif[0].num,valeur) ; 
		value_addto(objectif[1].num,poidsM) ;
	    }

	    /* Mise a jour des colonnes 0 et 1 */
	    t[0].colonne[t[0].taille].numero = ligne ;
	    t[0].colonne[t[0].taille].den = VALUE_ONE ;
	    t[0].colonne[t[0].taille].num = valeur ;
	    t[0].existe = 1 ;
	    t[0].taille++ ;
	    /* Element de poids M en 1ere colonne */
	    t[1].colonne[t[1].taille].numero = ligne ;
	    t[1].colonne[t[1].taille].num = poidsM ;
	    t[1].colonne[t[1].taille].den = VALUE_ONE ;
	    t[1].existe = 1 ;
	    t[1].taille++ ;
	    /* Creation d'une colonne cachee */
	    CREVARCACHEE ;
	    DEBUG1(dump_tableau("cre col cach", t, compteur);)
		}
	else
	    ligne--;
    }

    DEBUG1(dump_hashtable(hashtable);)
    DEBUG1(dump_tableau("avant sol prov", t, compteur);)
      
    /* NON IMPLEMENTE' */
    
    /* Elimination de Gauss-Jordan dans le tableau "eg"
     *  Chaque variable a` eliminer est marquee
     *  eg[ ].existe = 2
     *  Si le processus d'elimination ne revele pas
     *  d'impossibilite', il est suivi du processus
     *  d'elimination dans les inegalites.
     */
    /* FIN DE NON IMPLEMENTE' */
    
    /* SOLUTION PROVISOIRE
     *  Pour chaque egalite on introduit
     *  2 inequations complementaires
     */
    
    for(pc=sc->egalites ; pc!=0; pc=pc->succ, ligne++)
    {
	/* Added by bc: do nothing for nul equalities */
	if (pc->vecteur == NULL) continue;

        valeur = VALUE_ZERO ;
        poidsM = VALUE_ZERO ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                value_addto(poidsM,pv->val) ;
            else valeur = value_uminus(pv->val); /* val terme const */
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if(vect_coeff(pv->var,sc_base(sc))) {
                h = hash((Variable) pv->var) ; trouve=0 ;
                while (VARIABLE_DEFINED_P(hashtable[h].nom))  {
                    if (hashtable[h].nom==pv->var) {
                        trouve=1 ;
                        break ;
                    }
                    else { h = (h+1) % MAX_VAR ; }
                }
                if(!trouve) {
                    hashtable[h].succ=premier_hash ;
                    premier_hash = h ;
                    hashtable[h].val = VALUE_NAN ;
                    hashtable[h].numero=compteur++ ;
                    CREVARVISIBLE ;
                    hashtable[h].nom=pv->var ;
                }
				linear_assert("current NUMERO in bound",
				  (NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
                if(value_neg_p(poidsM) ||
		   (value_zero_p(poidsM) && value_neg_p(valeur)))
                    {value_addto(t[NUMERO].colonne[0].num,pv->val),
		       t[NUMERO].colonne[0].den = VALUE_ONE ;}
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(value_neg_p(poidsM) ||
		   (value_zero_p(poidsM) && value_neg_p(valeur)))
                    tmpval = value_uminus(pv->val);
		else tmpval = pv->val ;
                t[NUMERO].colonne[t[NUMERO].taille].num = tmpval ;
                t[NUMERO].colonne[t[NUMERO].taille].den = VALUE_ONE ;
                t[NUMERO].taille++ ;
            }
        }
	/* Creation de variable d'ecart ? */
        if(value_neg_p(poidsM) ||
	   (value_zero_p(poidsM) && value_neg_p(valeur))) {
            i=compteur++ ;
            CREVARVISIBLE ;
            t[i].existe = 1 ; t[i].taille = 2 ;
            t[i].colonne[0].num = VALUE_ONE ;
            t[i].colonne[0].den = VALUE_ONE ;
            t[i].colonne[1].numero = ligne ;
            t[i].colonne[1].num = VALUE_MONE ;
            t[i].colonne[1].den = VALUE_ONE ;
            value_oppose(poidsM), 
	    value_oppose(valeur);
            value_addto(objectif[0].num,valeur) ;
            value_addto(objectif[1].num,poidsM) ;
        }
	/* Mise a jour des colonnes 0 et 1 */
        t[0].colonne[t[0].taille].numero = ligne ;
        t[0].colonne[t[0].taille].num = valeur ;
        t[0].colonne[t[0].taille].den = VALUE_ONE ;
        t[0].existe = 1 ;
        t[0].taille++ ;
	/* Element de poids M en 1ere colonne */
        t[1].colonne[t[1].taille].numero = ligne ;
        t[1].colonne[t[1].taille].num = poidsM ;
        t[1].colonne[t[1].taille].den = VALUE_ONE ;
        t[1].existe = 1 ;
        t[1].taille++ ;
	/* Creation d'une colonne cachee */
        CREVARCACHEE ;
	DEBUG1(dump_tableau("cre col cach 2", t, compteur);)
    }
    
    for(pc=sc->egalites ; pc!=0; pc=pc->succ, ligne++)
    {
	/* Added by bc: do nothing for nul equalities */
	if (pc->vecteur == NULL) continue;

        valeur = VALUE_ZERO ;
        poidsM = VALUE_ZERO ;
        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ)
            if(vect_coeff(pv->var,sc_base(sc)))
                value_substract(poidsM, pv->val) ;
            else 
		valeur = pv->val ; /* val terme const */

        for(pv=pc->vecteur ; pv !=0 ; pv=pv->succ) {
            if (vect_coeff(pv->var,sc_base(sc))) {
                h = hash((Variable) pv->var) ; trouve=0 ;
                while (VARIABLE_DEFINED_P(hashtable[h].nom))  {
                    if (hashtable[h].nom==pv->var) {
                        trouve=1 ;
                        break ;
                    }
                    else { h = (h+1) % MAX_VAR ; }
                }
                if(!trouve) {
                    hashtable[h].succ=premier_hash ;
                    premier_hash = h ;
                    hashtable[h].val = VALUE_NAN ;
                    hashtable[h].numero=compteur++ ;
                    hashtable[h].nom=pv->var ;
                    CREVARVISIBLE ;
                }
		assert((NUMERO) < (3 + NB_INEQ + NB_EQ + DIMENSION));
                if(value_neg_p(poidsM) || 
		   (value_zero_p(poidsM) && value_neg_p(valeur)))
                    {value_substract(t[NUMERO].colonne[0].num,pv->val),
		       t[NUMERO].colonne[0].den = VALUE_ONE ;}
                t[NUMERO].existe = 1 ;
                t[NUMERO].colonne[t[NUMERO].taille].numero=ligne ;
                if(value_neg_p(poidsM) || 
		   (value_zero_p(poidsM) && value_neg_p(valeur)))
                    tmpval = pv->val ; 
		else tmpval = value_uminus(pv->val) ;
                t[NUMERO].colonne[t[NUMERO].taille].num = tmpval ;
                t[NUMERO].colonne[t[NUMERO].taille].den = VALUE_ONE ;
                t[NUMERO].taille++ ;
            }
        }
	/* Creation de variable d'ecart ? */
        if(value_neg_p(poidsM) || 
	   (value_zero_p(poidsM) && value_neg_p(valeur))) {
            i=compteur++ ;
            CREVARVISIBLE ;
            t[i].existe = 1 ; t[i].taille = 2 ;
            t[i].colonne[0].num = VALUE_ONE ;
            t[i].colonne[0].den = VALUE_ONE ;
            t[i].colonne[1].numero = ligne ;
            t[i].colonne[1].num = VALUE_MONE ;
            t[i].colonne[1].den = VALUE_ONE ;
            value_oppose(poidsM), 
	    value_oppose(valeur);
            value_addto(objectif[0].num,valeur) ;
            value_addto(objectif[1].num,poidsM) ;
        }
	/* Mise a jour des colonnes 0 et 1 */
        t[0].colonne[t[0].taille].numero = ligne ;
        t[0].colonne[t[0].taille].num = valeur ;
        t[0].colonne[t[0].taille].den = VALUE_ONE ;
        t[0].existe = 1 ;
        t[0].taille++ ;
	/* Element de poids M en 1ere colonne */
        t[1].colonne[t[1].taille].numero = ligne ;
        t[1].colonne[t[1].taille].num = poidsM ;
        t[1].colonne[t[1].taille].den = VALUE_ONE ;
        t[1].existe = 1 ;
        t[1].taille++ ;
	/* Creation d'une colonne cachee */
        CREVARCACHEE ;
	DEBUG1(dump_tableau("cre col cach 3", t, compteur);)
    }
    
    /* FIN DE SOLUTION PROVISOIRE */
    
    /* Algorithme du simplexe - methode primale simple.
     * L'objectif est d'etudier la faisabilite' d'un systeme
     * de contraintes sans trouver l'optimum.
     *   Les contraintes ont la forme : a x <= b
     *      et  d x = e
     * La methode de resolution procede comme suit :
     *
     *  1) Creer un tableau
     *       a  b
     *       d  e
     *     Eliminer autant de variables que posible par
     *    Gauss-Jordan
     *
     *  2) Travailler sur les inegalites seulement.
     *      Poser  x = x' - M 1
     *    ou` 1 est le vecteur de chiffres 1.
     *     Les inequations prennent alors la forme :
     *      a1 x <= b1 + M c1
     *      a2 x >= b2 + M c2
     *    avec c1 et c2 positifs
     *     On introduit les variables d'ecart y (autant que 
     *    d'inequations du 2eme type) et on cherche
     *      max{1(a2 x - y) | x,y >= 0 ; a1 x <= b1 + M c1 ;
     *                                a2 x - y <= b2 + M c2}
     *     On cree donc le tableau :
     *        0  0  1 a2     1  0  0
     *        b1 c1  a1      0  I  0
     *        b2 c2  a2     -I  0  I
     *
     *     On applique ensuite l'algorithme du simplex en
     *    se souvenant que c1 et c2 sont a multiplier par un
     *    infiniment grand.
     *     Si l'optimum est egal a (1 b2 , 1 c2), il y a une
     *    solution.
     *
     * Structures de donnees : on travaille sur des tableaux
     * de fractions rationnelles.
     */
    nlle_colonne=(frac *) malloc((4 + 2*NB_EQ + NB_INEQ+1)*sizeof(frac)) ;
    frac_init(nlle_colonne,4 + 2*NB_EQ + NB_INEQ);
    while(1) {

        /*  Recherche d'un nombre negatif 1ere ligne  
	 */
        for(j=2, jj= -1 ;j<compteur;j++)
            if(t[j].existe && NEGATIF(t[j].colonne[0]))
            {
		jj=j ; break ;
	    }
        
	/*  Terminaison  */
        if(jj == -1) { 
	    bool cond;
            DEBUG1({
		printf ("solution :\n") ;
		dump_tableau("sol", t, compteur) ;
		printf("objectif : "); printfrac(objectif[0]) ; 
			printfrac(objectif[1]) ; printf("\n") ;
	    });
	    
	    if (ofl_ctrl == FWD_OFL_CTRL)
		cond = EGALOFL(objectif[0],t[0].colonne[0]) &&
		    EGALOFL(objectif[1],t[1].colonne[0]);
	    else
		cond = EGAL(objectif[0],t[0].colonne[0]) &&
		    EGAL(objectif[1],t[1].colonne[0]);
	    
	    if(cond)
	    {
                DEBUG1(printf("Systeme soluble (faisable) en rationnels\n");)
		SOLUBLE(1)
	    }
	    else 
	    {
		DEBUG1(printf("Systeme insoluble (infaisable)\n");)
		SOLUBLE(0)
	    }
	    DEBUG1(printf("fin\n");)
        }

	DEBUG1(printf("1 : jj= %ld\n",jj);
	      dump_tableau("avant ch pivot", t, compteur);)

	DEBUG1(min1.num = 32700; min1.den=1; min2=min1;)
	
        /*  Recherche de la ligne de pivot  
	 *  si ii==-1, pas encore trouve, min{1,2} non valides...
	 */
        for(i=1, i0=1, i1=1, ii=-1 ; i<t[jj].taille ; )
        {
	    bool cond;

	    DEBUG1(fprintf(stdout, "itering i{,0,1} = %ld %ld %ld\n", 
			  i, i0, i1);)

            if(((i0<t[0].taille && t[jj].colonne[i].numero <= 
		 t[0].colonne[i0].numero)  || i0>=t[0].taille)
	       && ((i1<t[1].taille && t[jj].colonne[i].numero <=
		    t[1].colonne[i1].numero) || i1>=t[1].taille)) {
		if( POSITIF(t[jj].colonne[i])) {
		    /* computing rapport{1,2} 
		     */
		    frac f1 = 
			(i0<t[0].taille &&
			 t[jj].colonne[i].numero==t[0].colonne[i0].numero)?
			     t[0].colonne[i0]:frac0;
		    frac f2 = t[jj].colonne[i];
		    frac f3 =
			(i1<t[1].taille && 
			 t[jj].colonne[i].numero==t[1].colonne[i1].numero)?
			     t[1].colonne[i1]:frac0;
			
		    frac_div(&rapport1,f1,f2,ofl_ctrl);
		    frac_div(&rapport2,f3,f2,ofl_ctrl);
	
    
		    DEBUG1(fprintf(stdout, "rapports:");
			  printfrac(rapport1);
			  printfrac(min1);
			  printfrac(rapport2);
			  printfrac(min2);
			  fprintf(stdout, "\nand cond: ");)

		    if (ii==-1) 
			cond = true; /* first assignment is forced */
		    else
			if (ofl_ctrl == FWD_OFL_CTRL)
			    cond = INFOFL(rapport2,min2) ||
				(EGALOFL(rapport2,min2) && 
				 INFOFL(rapport1,min1));
			else
			    cond = INF(rapport2,min2) || 
				(EGAL(rapport2,min2) && 
				 INF(rapport1,min1));
		    
		    DEBUG1(fprintf(stdout, "%d\n", cond);)

		    if (cond) {
			AFF(min1,rapport1) ;
			AFF(min2,rapport2) ;
			AFF(piv,t[jj].colonne[i]) ;
			frac_simplifie(&piv);
			ii=t[jj].colonne[i].numero ;
		    }
		} /* POSITIF(t[jj].colonne[i])) */
		i++ ;
	    }
	    else {
		if(i0<t[0].taille && /* it may skip over */
		   t[jj].colonne[i].numero> t[0].colonne[i0].numero) i0++ ;
		if(i1<t[1].taille && 
		   t[jj].colonne[i].numero > t[1].colonne[i1].numero) i1++ ;
	    }
	    
	    DEBUG1(printf("i=%ld i0=%ld i1=%ld   %d %d %d\n",
			 i,i0,i1,
			 i<t[jj].taille? t[jj].colonne[i].numero: -1,
			 i0<t[0].taille? t[0].colonne[i0].numero: -1,
			 i1<t[1].taille? t[1].colonne[i1].numero: -1);)
        }

        /* Cas d'impossibilite'  */
	if(ii==-1) {
	    DEBUG1(dump_tableau("sol infinie", t, compteur);
		   fprintf(stderr,"Solution infinie\n");)
	    SOLUBLE(1)
	}

	/* Modification des numeros des variables */

        j = variables[jj-2];
	k = variablescachees[ii-1];
        variables[jj-2] = k;
	variablescachees[ii-1] = j;

        DEBUG2({
	    printf("Visibles :");
	    for(j=0;j<compteur-2;j++)
		printf(" %d",variables[j]);
	    printf("\nCachees :");
	    for(j=0;j<nbvariables;j++)
		printf(" %d",variablescachees[j]);
	    printf("\n");
	});

        /*  Pivot autour de la ligne ii / colonne jj
         * Dans ce qui suit, j = colonne courante,
         *  k = numero element dans la nouvelle colonne
         *     qui remplacera la colonne j,
         *  cc = element (colonne j, ligne ii)
         */
	DEBUG(printf("Pivoter %ld %ld\n",ii,jj);)
        
	/* Remplir la derniere colonne temporaire de t
	 *   qui contient un 1 en position ligne ii
	 */
        t[compteur].taille = 2 ;
	t[compteur].colonne[0].numero = 0 ;
        t[compteur].colonne[0].num = VALUE_ZERO ;
	t[compteur].colonne[0].den = VALUE_ONE ;
        t[compteur].colonne[1].numero = ii;
        t[compteur].colonne[1].num = VALUE_ONE ;
        t[compteur].colonne[1].den = VALUE_ONE ;
        t[compteur].existe = 1 ;

        for(j=0 ; j<=compteur ; j=(j==(jj-1))?(j+2):(j+1)) {
	    if(t[j].existe)
	    {
		k=0 ;
		cc.num= VALUE_ZERO ; 
		cc.den= VALUE_ONE ;
		for(i=1;i<t[j].taille;i++)
		    if(t[j].colonne[i].numero==ii)
                    { AFF(cc,t[j].colonne[i]); break ; }
		    else if(t[j].colonne[i].numero>ii)
                    {cc.num= VALUE_ZERO ; cc.den=VALUE_ONE ; break ; }
		for(i=0,i1=0;i<t[j].taille || i1<t[jj].taille ;) {

		    DEBUG2(printf("k=%ld, j=%ld, i=%ld i1=%ld\n",k,j,i,i1);
			   printf("fractions: ");
			   printfrac(t[j].colonne[i]) ;
			   printfrac(t[jj].colonne[i1]) ;
			   if (value_zero_p(t[jj].colonne[i1].den))
			   printf("ATTENTION fraction 0/0 ");/*DN*/
			   printfrac(cc);
			   printfrac(piv);)
		    
		    if(i<t[j].taille &&  
		       i1<t[jj].taille && 
		       t[j].colonne[i].numero == t[jj].colonne[i1].numero) 
		    {   
			if(t[j].colonne[i].numero == ii) {
			    AFF(nlle_colonne[k],t[j].colonne[i]);
			} else {
			    frac *n = &nlle_colonne[k],
                                 *a = &t[j].colonne[i],
                                 *b = &t[jj].colonne[i1];
			     pivot(n, (*a), (*b), cc, piv,ofl_ctrl);
			}
			
			if(i==0||nlle_colonne[k].num!=0) {
			    nlle_colonne[k].numero = t[j].colonne[i].numero ;
			    k++ ;
			}

			i++ ; i1++ ;
		    }
		    else
			if(i>=t[j].taille || 
			   (i1<t[jj].taille && 
			    t[j].colonne[i].numero > t[jj].colonne[i1].numero))
			{  
			    DEBUG1(
				if (i<t[j].taille) 
				{
				    printf("t[j].colonne[i].numero > "
					   "t[jj].colonne[i1].numero , "
					   "k=%ld, j=%ld, i=%ld i1=%ld\n",
					   k,j,i,i1);
				    printf("j = %ld  t[j].taille=%d , "
					   "t[jj].taille=%d\n",
					   j,t[j].taille,t[jj].taille);
				    printf("t[j].colonne[i].numero=%d , "
					   "t[jj].colonne[i1].numero=%d\n",
					   t[j].colonne[i].numero,
					   t[jj].colonne[i1].numero);
				});

                        /* 0 en colonne j  ligne t[jj].colonne[i1].numero */
			    if(t[jj].colonne[i1].numero == ii) {
				AFF(nlle_colonne[k],frac0)
			    } else {
				frac *n = &(nlle_colonne[k]),
				     *b = &(t[jj].colonne[i1]);
				pivot(n,frac0,(*b),cc,piv,ofl_ctrl) ;
			    }

			    if(i==0||nlle_colonne[k].num!=0)
			    {
				nlle_colonne[k].numero = 
				    t[jj].colonne[i1].numero ;
				k++ ;
			    }
			    if(i1<t[jj].taille) i1++ ; else i++ ;
			}
			else if(i1>=t[jj].taille || 
				t[j].colonne[i].numero < 
				t[jj].colonne[i1].numero)
			{
			    /* 0 en col jj, ligne t[j].colonne[i].numero */
			    DEBUG2(printf("t[j].colonne[i].numero < "
					  "t[jj].colonne[i1].numero , "
					  "k=%ld, j=%ld, i=%ld i1=%ld\n",
					  k,j,i,i1);
				   printf("j = %ld  t[j].taille=%d , "
					  "t[jj].taille=%d\n",
					  j,t[j].taille,t[jj].taille););
			    AFF(nlle_colonne[k],t[j].colonne[i]) ;
			    if(i==0||nlle_colonne[k].num!=0) {
				nlle_colonne[k].numero = 
				    t[j].colonne[i].numero ;
				k++ ;
			    }
			    if(i<t[j].taille) i++ ; else i1++ ;
			}
			else
			    DEBUG2(printf(" ??? ");)

		    DEBUG2(printf(" -> ");
			   printfrac(nlle_colonne[k-1]);
			   printf(" [ligne numero %d]\n", 
				  nlle_colonne[k-1].numero);)

		}
		if(j==compteur) w = jj ; else w = j ;
		colo = t[w].colonne ;
		t[w].colonne=nlle_colonne ;
		nlle_colonne = colo ;
		t[w].taille=k ;
		DEBUG1(printf("w = %ld  t[w].taille=%d \n",w,t[w].taille);
		      dump_tableau("last", t, compteur););
	    }
        }
    }

    /* Restauration des entrees vides de la table hashee  */
    FINSIMPLEX :
    DEBUG1(dump_tableau("fin simplexe", t, compteur);)
    DEBUG(fprintf(stderr, "soluble = %d\n", soluble);)

    DEBUG(fprintf(stderr,"END SIMPLEX: %d th\n",simplex_sc_counter);)

    for(i = premier_hash ; i != PTR_NIL; i = hashtable[i].succ){
      hashtable[i].nom = VARIABLE_UNDEFINED;
      hashtable[i].numero = 0 ;
      hashtable[i].hash = 0 ;
      hashtable[i].val = (Value)0 ;
    }

    
    for(i=0;i<(3 + NB_INEQ + NB_EQ + DIMENSION); i++) 
      free(t[i].colonne);
    free(t);
    free(nlle_colonne);

#ifdef CONTROLING
    alarm(0); /*clear the alarm*/
#endif
    UNCATCH(simplex_arithmetic_error|timeout_error|overflow_error);
    
    /* restore initial base */
    vect_rm(sc_base(sc));
    sc_base(sc) = saved_base;
    sc_dimension(sc) = saved_dimension;
    
    return soluble;
}     /* main */

/* (that is all, folks!:-)
 */
