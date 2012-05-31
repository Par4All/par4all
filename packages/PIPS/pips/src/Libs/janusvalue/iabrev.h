/* ========================================================================= */
/*                       SIMPLEXE for integer variables                      */
/*                            ALL-INTEGER METHODS                            */
/*                             Jean Claude SOGNO                             */
/*                     Projet CHLOE -- INRIA ROCQUENCOURT                    */
/*                                Juin 1994                                  */
/* ========================================================================= */

/* ========================================================================= */
/*                             Duong NGUYEN QUE                              */
/*                 Adaption to abstract computation: janusvalue              */
/*                               CRI-ENSMP                                   */
/* ========================================================================= */

#define FTRACE XX->ftrace 
/* ................ variables resultats ....................... */
#define VRESULT XX->result
#define NITER XX->niter
/* ............... variables ou constantes de depart .............. */
#define MC XX->mcontr
#define NV XX->nvar
#define NP XX->nvpos
#define PMETH XX->meth
#define PMET2 XX->met2
#define PMET3 XX->met3
#define PDICO XX->met4
#define MSR XX->met5
#define PRECR XX->met6
#define PCOMP XX->met7
#define REDON XX->met8
#define DYN XX->dyn
#define VISU XX->ntrace
#define VIS2 XX->ntrac2
#define VIS3 XX->ntrac3
#define TMORE 20
#define VARC XX->varc
#define FORCER XX->forcer
#define PFOURIER XX->fourier
#define CRITERMAX XX->critermax
#define CHOIXPIV XX->choixpiv
#define CHOIXPRIM XX->choixprim
#define VW1 (XX->ntrac2>=1)
#define VW2 (XX->ntrac2>=2)
#define VW3 (XX->ntrac2>=3)
#define VW4 (XX->ntrac2>=4)
#define VW5 (XX->ntrac2>=5)
#define VW6 (XX->ntrac2>=6)
#define VW7 (XX->ntrac2>=7)
#define VW8 (XX->ntrac2>=8)
/* ............... macros pour tableaux passes .............. */
/*#define a(A,B) *(*(XX->pta+A)+B)*/
/*#define d(A) *(XX->ptd+A) * remplace int d[MAXLIGNES+1]; */
/*#define e(A) *(XX->pte+A) * remplace int e[MAXLIGNES+1]; */
/*#define A(A,B) XX->a[A][B]*/
#define A(A,B) II->a[A][B]
#define D(A) II->d[A] /* remplace int d[MAXLIGNES+1]; */
#define E(A) II->e[A] /* remplace int e[MAXLIGNES+1]; */
/* ............... macros for local arrays .............. */
#define AK(A,B) XX->ak[A][B]
#define DK(A) XX->dk[A]
#define B(A) XX->b[A] /*#define bb(A) *(&XX->b[0]+A) ok */
#define G(A) XX->g[A]
/* ............... visualization run-level .................*/
#define ZVB 1 /* bad problem parameters */
#define ZVO 1 /* overflow */
#define ZVS 1 /* step signale */
#define ZVP1 3 /* pivoting, iteration is indicated */
#define ZVP2 6 /* after pivoting, global set *//*R*/
#define ZVTS 3 /* tableau en fin de step */
/*       initial problem           */
#define ZVA1 2 /* nature and number of constraints */
#define ZVA4 4 /* initial global set */
/*       Variables status           */
#define ZVVF 4 /* constrained feature is specified by an inequality */
/*       Extended Euclid Algorithm           */
/*#define ZVX 5 */ /* Extended Euclid Algorithm is described */
/*     Unimodular change of variables        */
#define ZVU1 2  /* unimodular change is indicated */
/*#define ZVU2 4 */ /* unimodular change is described */
/*#define ZVU4 4 */ /* global set before changing */
/*#define ZVU9 7 */ /* global set after changing *//*R*/
/*    Equalities elimination       */
/*#define ZVEK 4 */ /* comments concerning elimination */
#define ZVEV 3  /* empty equation is removed */
/*#define ZVE4 4*/  /* global set before pivoting */
/*            GCD test             */
/*#define ZVG1 2*/ /* after GCD computation, GCD test is indicated */
#define ZVG2 1 /* polyedron proved empty by GCD test */
/*#define ZVG3 3 *//* global set when polyedron proved empty by GCD test */
/*     inequalities division       */
#define ZVI1 2 /* inequality division is indicated */
#define ZVI3 3 /* inequality division is shown */
/*           fourier               */
#define ZVF1 2 /* fourier elimination is indicated */
#define ZVF2 5 /* after fourier 2-2, before removing col, global set */
#define ZVF3 4 /* after fourier elimination global set */
#define ZVFEC 2 /* empty column */
#define ZVFW 2  /* warning, a column is empty and functions are not */
/*     Non-negative variables        */
#define ZVNPC 6  /* global set after possible columns permutation */ /*R*/
#define ZVN1 4  /* chosen inequality is indicated */
#define ZVN2 2  /* rhs of chosen inequality is positive */
#define ZVNW 2  /* warning, all columns of free variables are empty */
#define ZVNG 5  /* gcd of coefficients of free variables */
/*             cut                 */
#define ZVC1 3 /* cutting operation */
/*#define ZVC2 4 *//* cutting operation: resulting c. array */
/*#define ZVC3 5 *//* detailed cutting operation */
/*#define ZVC4 6 *//* more detailed cutting operation */
#define ZVCP1 4 /* global set before pivoting */
#define ZVCP2 7 /* global set after pivoting, before removing cut *//*R*/
#define ZVCP3 6 /* global set after pivoting */ /* redondant avec ZVP2 *//*R*/
/*             redundant inequalities                 */
#define ZVR1 2 /* redundant inequality is indicated */
#define ZVR3 4 /* final global set if redundancies *//*R*/
#define ZVR4 4 /* global set before redundant inequality is removed */
/*             dual                 */
#define ZVDS 3 /* surrogate constraint is indicated */
#define ZVDEND 3 /* global set after dual */
/*             constraint satisfaction                 */
#define ZVSAT2 2 /* one more satisfaction is specified*/
#define ZVSAT3 3 /* global set after one more satisfaction or contradiction */
#define ZVSAT4 4 /* details concerning satisfaction */
/*           primal                 */
#define ZVPRI 2 /* when improved cost function */
/*             last                 */
#define ZVL 1 /* solver result */
#define ZVL2 3 /* global set */
/*             branch                 */
#define ZVBR1 1 /*  */
#define ZVBR2 2 /*  */
#define ZVBR3 3 /* details  */
#define ZVBR4 4 /* details  */
/* ............... visualization variables .................*/
#define NSTEP XX->nstep
#define MAJSTEP XX->majstep
/* ********* local variables ******* */
#define NEGAL XX->negal
#define ICOUT XX->icout
#define MX XX->mx
#define NX XX->nx
#define NREDUN XX->nredun
#define VREDUN(A) XX->vredun[A]
#define TMAX XX->tmax
#define NUMERO XX->numero
#define NUMAX XX->numax
#define NUB XX->nub
#define LASTFREE XX->lastfree
#define IC1 XX->ic1
#define NTP XX->ntp
#define VDUM XX->vdum      /* variables for trace */
/* ******** in case of possible bug or array overflow ***************/
#define XBUG(A) return xbug(XX,A)
#define XDEB(A) return xdeb(XX,A)
