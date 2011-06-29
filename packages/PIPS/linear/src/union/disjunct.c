
#line 62 "disjunct.w"


#line 335 "UNION.w"

/* Package  :  C3/union
 * Author   :  Arnauld LESERVOT (leservot(a)limeil.cea.fr)
 * Date     :  
 * Modified :   04 04 95
 * Documents:   UNION.tex : ``Extension de C3 aux unions de polyedres''
 * Comments :
 */
/* 
 *                  WARNING
 * 
 *   THOSE FUNCTIONS ARE AUTOMATICALLY DERIVED 
 *    
 *           FROM THE WEB SOURCES !
 */

/* Ansi includes        */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

/* Linear includes      */
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "polyedre.h"
#include "union.h" 


#line 63 "disjunct.w"


#line 71 "disjunct.w"

/* Pdisjunct dj_new()   AL 26/10/93
 * Allocate a new Pdisjunct
 */
Pdisjunct dj_new() { return( (Pdisjunct) sl_new() ); }
 

/* Pdisjunct dj_dup( (Pdisjunct) in_dj ) AL 15/11/93
 * Duplicates input disjunction.
 */
Pdisjunct dj_dup( in_dj )
Pdisjunct in_dj; 
{ 
  if (dj_full_p(in_dj)) return dj_full();
  return (Pdisjunct) sl_dup( (Psyslist) in_dj ); 
}


/* Pdisjunct dj_free( (Pdisjunct) in_dj ) AL 31/05/94
 * w - 1 depth free of input disjunction.
 */
Pdisjunct dj_free( in_dj )
Pdisjunct in_dj; { return (Pdisjunct) sl_free( (Psyslist) in_dj ); }


/* Pdisjunct dj_dup1( (Pdisjunct) in_dj ) AL 31/05/94
 * 1st depth duplication of input disjunction.
 */
Pdisjunct dj_dup1( in_dj )
Pdisjunct in_dj;
{ return( (Pdisjunct) sl_dup1( (Psyslist) in_dj ) );  }


/* Pdisjunct dj_free1( (Pdisjunct) in_dj ) AL 31/05/94
 * 1st depth free of input disjunction.
 */
Pdisjunct dj_free1( in_dj )
Pdisjunct in_dj; { return (Pdisjunct) sl_free1( (Psyslist) in_dj ); }

#line 116 "disjunct.w"

/* Pdisjunct dj_full()  AL 18/11/93
 * Return full space disjunction = dj_new()
 */
Pdisjunct dj_full(){ return( dj_new() ); }


/* dj_full_p( (Pdisjunct) in_dj )   AL 30/05/94
 * Returns True if in_dj = (NIL) ^ (NIL)
 */
bool dj_full_p( in_dj )
Pdisjunct in_dj;
{
  return( (in_dj != DJ_UNDEFINED) &&
    ( in_dj->succ == NULL ) &&
    ( in_dj->psys  == NULL ) );
}

#line 138 "disjunct.w"

/* Pdisjunct dj_empty()    AL 18/11/93
 * Returns a disjunction with sc_empty() element.
 */
Pdisjunct dj_empty()
{ return (Pdisjunct) sl_append_system(NULL, sc_empty((Pbase) NULL)); }


/* dj_empty_p( (Ppath) in_pa )   AL 30/05/94
 * Returns True if in_dj = (1*TCST = 0) ^ (NIL)
 */
bool dj_empty_p( in_dj )
Pdisjunct in_dj;
{
  return( ( in_dj != DJ_UNDEFINED     )    &&
     ( in_dj->succ == NULL       )    &&
     ( in_dj->psys != NULL       )    &&
     ( sc_empty_p( in_dj->psys ) )       );
}

#line 169 "disjunct.w"

/* Pdisjunct dj_intersection_ofl_ctrl( in_dj1, in_dj2, ofl_ctrl )
 * Computes intersection of two disjunctions.       AL,BC 23/03/95
 * Very costly function : -> sc_faisabilite_ofl_ctrl used.
 * No sharing 
 */
Pdisjunct dj_intersection_ofl_ctrl( in_dj1, in_dj2, ofl_ctrl )
Pdisjunct in_dj1, in_dj2;
int ofl_ctrl;
{
  Pdisjunct dj1, dj2, ret_dj;
  
  if (DJ_UNDEFINED_P(in_dj1)||DJ_UNDEFINED_P(in_dj2)) return DJ_UNDEFINED   ;
  if (dj_full_p(in_dj1) && dj_full_p(in_dj2))         return dj_full()      ;
  if (dj_full_p(in_dj1))                              return dj_dup(in_dj2) ;
  if (dj_full_p(in_dj2))                              return dj_dup(in_dj1) ;
  if (dj_empty_p(in_dj1)||dj_empty_p(in_dj2))         return dj_empty()     ;
  
  ret_dj = (Pdisjunct) NULL; 
  for(dj1 = in_dj1; dj1 != NULL; dj1 = dj1->succ) {
    for(dj2 = in_dj2; dj2 != NULL; dj2 = dj2->succ) {
      Psysteme ps = sc_append( sc_dup(dj1->psys), dj2->psys );
      if (!sc_rational_feasibility_ofl_ctrl( ps, ofl_ctrl, true )) 
   { ps = sc_free( ps ); continue; }
      ret_dj = (Pdisjunct) sl_append_system( ret_dj, ps );
    }
  }
  if (ret_dj == (Pdisjunct) NULL) return dj_empty(); /* empty intersection */
  return ret_dj;
}

#line 205 "disjunct.w"

/* Pdisjunct dj_intersect_system_ofl_ctrl( )  */
Pdisjunct  dj_intersect_system_ofl_ctrl( in_dj, in_ps, ofl_ctrl )
Pdisjunct  in_dj    ;
Psysteme   in_ps    ;
int        ofl_ctrl ;
{ return dj_intersection_ofl_ctrl( in_dj, sl_append_system(NULL, in_ps), ofl_ctrl ); }

#line 219 "disjunct.w"

/* Pdisjunct dj_intersect_djcomp_ofl_ctrl( ) 
 * No sharing. in_dj1 and in_dj2 stay as is.
 */
Pdisjunct dj_intersect_djcomp_ofl_ctrl( in_dj1, in_dj2, ofl_ctrl )
Pdisjunct in_dj1, in_dj2;
int       ofl_ctrl;
{
  Pdisjunct dj, ret_dj;  
  
  /* Special cases */
  if (DJ_UNDEFINED_P(in_dj1) || DJ_UNDEFINED_P(in_dj2)) return DJ_UNDEFINED;
  
  if (dj_empty_p( in_dj1 ) || dj_full_p(in_dj2)) return dj_empty();
  if (dj_full_p ( in_dj1 ))                      return dj_disjunct_complement( in_dj2 );
  if (dj_empty_p( in_dj2 ))                      return dj_dup( in_dj1 );
  
  /* debuging */
  C3_DEBUG("dj_intersect_djcomp_ofl_ctrl",{
    fprintf(stderr,"Inputs (in_dj1, then in_dj2):");
    dj_fprint_tab(stderr, in_dj1, union_variable_name, 1);
    dj_fprint_tab(stderr, in_dj2, union_variable_name, 1);
  });

  /* General cases */
  ret_dj = dj_empty();
  for(dj = in_dj1; dj != NULL; dj = dj->succ ){ 
    Ppath pa = pa_make( dj->psys, (Pcomplist) in_dj2  ); 
    ret_dj   = dj_union( ret_dj, pa_path_to_few_disjunct_ofl_ctrl( pa, ofl_ctrl ) ); 
    free(pa);
  }
  C3_RETURN(IS_DJ, ret_dj);
}

#line 257 "disjunct.w"

/* Pdisjunct dj_union( (Pdisjunct) in_dj1, (Pdisjunct) in_dj2 ) 
 * Give the union of the two disjunctions. AL 15/11/93
 * Memory: systems of the 2 unions are shared. 
 *    in_dj1 = dj_union(in_dj1,in_dj2); 
 *    (in_dj1 = dj_free(in_dj1);      to remove in_dj1 and in_dj2
 */
Pdisjunct dj_union( in_dj1, in_dj2 )
Pdisjunct in_dj1, in_dj2;
{
  Pdisjunct dj;
   
  if (DJ_UNDEFINED_P(in_dj1) || DJ_UNDEFINED_P(in_dj2)) return DJ_UNDEFINED;
  if (dj_empty_p( in_dj2 )) {dj_free(in_dj2); return in_dj1;}
  if (dj_full_p ( in_dj2 )) {dj_free(in_dj1); return in_dj2;}
  if (dj_empty_p( in_dj1 )) {dj_free(in_dj1); return in_dj2;}
  if (dj_full_p ( in_dj1 )) {dj_free(in_dj2); return in_dj1;}

  for( dj = in_dj1; dj->succ != NULL; dj = dj->succ) {};
  dj->succ = in_dj2;
  return in_dj1;
}

#line 287 "disjunct.w"

/* bool dj_feasibility_ofl_ctrl( (Pdisjunct) in_dj, (int) ofl_ctrl ) 
 * Returns true if in_dj is a feasible disjunction. AL,BC 23/02/95
 */
bool dj_feasibility_ofl_ctrl( in_dj, ofl_ctrl )
Pdisjunct in_dj;
int ofl_ctrl;
{
  bool   ret_bool = false;
  Pdisjunct dj;

  if ( in_dj == DJ_UNDEFINED ) return false;
  for( dj = in_dj; dj != NULL && !ret_bool; dj = dj->succ ) {
    if (dj->psys == SC_UNDEFINED) return false;
    ret_bool = ret_bool || 
      sc_rational_feasibility_ofl_ctrl( dj->psys, ofl_ctrl, true );
  }
  return ret_bool;
}

#line 315 "disjunct.w"

/* Pdisjunct dj_system_complement( (Psystem) in_ps )  AL 26/10/93
 * Input  : A Psysteme.
 * Output : A disjunction which is complement of in_ps.
 */
Pdisjunct dj_system_complement( in_ps )
Psysteme in_ps;
{
  Pdisjunct       ret_dj = NULL;
  Pvecteur        v1 = NULL, pv = NULL;
  Psysteme        ps = NULL;
  Pcontrainte     eq = NULL, ineq = NULL;

  if ( in_ps == SC_UNDEFINED ) return DJ_UNDEFINED;
  if (sc_empty_p(in_ps))       return dj_full();
  
  /* debugging */
  C3_DEBUG("dj_system_complement (in_ps)",
      {sc_fprint(stderr, in_ps, union_variable_name);});


  /* v1 = 1*TCST to build complement system ... */
  v1 = vect_new( TCST, VALUE_ONE);
  /* Look for equalities */
  for( eq = in_ps->egalites; eq != NULL; eq = eq->succ ) {
    ps     = sc_make( CONTRAINTE_UNDEFINED,
       contrainte_make( vect_add( v1, eq->vecteur ) ) );
    ret_dj =  (Pdisjunct) sl_append_system( ret_dj, ps );
    pv     = vect_dup( eq->vecteur ); vect_chg_sgn( pv );
    ps     = sc_make( CONTRAINTE_UNDEFINED, contrainte_make(vect_add( v1, pv )));
    vect_rm( pv ); pv = NULL;
    ret_dj =  (Pdisjunct) sl_append_system( ret_dj, ps );
    
  }
  /* Look for inequalities */
  for(ineq = in_ps->inegalites; ineq != NULL; ineq = ineq->succ) {
    pv = vect_dup(ineq->vecteur);
    vect_chg_sgn( pv );
    ps = sc_make( CONTRAINTE_UNDEFINED, contrainte_make(vect_add( v1, pv )));
    vect_rm( pv ); pv = NULL;
    ret_dj =  dj_append_system( ret_dj, ps );
  }
   
  vect_rm( v1 );
  C3_RETURN(IS_DJ, ret_dj);
}

#line 367 "disjunct.w"

/* Returns complement of in_dj. No sharing */
Pdisjunct dj_disjunct_complement( in_dj )
Pdisjunct in_dj;
{
  Pdisjunct ret_dj;
  if DJ_UNDEFINED_P(in_dj) return DJ_UNDEFINED;
  if (dj_empty_p(in_dj)||dj_full_p(in_dj)) return dj_dup(in_dj);
  
  /* debugging */
  C3_DEBUG("dj_disjunct_complement (in_ps)",
      {dj_fprint_tab(stderr, in_dj, union_variable_name, 1);});

  ret_dj = dj_full();
  for(; in_dj != NULL; in_dj = in_dj->succ) 
    { ret_dj = dj_intersection(ret_dj, dj_system_complement(in_dj->psys)); }
  C3_RETURN(IS_DJ, ret_dj);
}

#line 391 "disjunct.w"

/* Returns projection of in_dj along vars of in_pv. Sharing : in_dj is modified */
Pdisjunct dj_projection_along_variables_ofl_ctrl( in_dj, in_pv, ofl_ctrl )
Pdisjunct  in_dj;
Pvecteur   in_pv;
int        ofl_ctrl;
{
  Pdisjunct dj;
  if DJ_UNDEFINED_P(in_dj) return DJ_UNDEFINED;
  if (dj_empty_p(in_dj)||dj_full_p(in_dj)) return in_dj;
  
  for(dj = in_dj; dj != NULL; dj = dj->succ) 
    { sc_projection_along_variables_ofl_ctrl( &(dj->psys), in_pv, ofl_ctrl ); }
  return in_dj;
}

#line 414 "disjunct.w"

/* Pdisjunct dj_simple_inegs_to_eg( in_dj ) transforms two opposite
 * inequalities in a simple equality in each system of the input disjunction.
 * Input disjunction is not modified.
 */
Pdisjunct dj_simple_inegs_to_eg( in_dj )
Pdisjunct in_dj;
{
  Pdisjunct  dj;
  Pdisjunct  ret_dj = NULL;

  /* Special case */
  if (DJ_UNDEFINED_P(in_dj) || dj_empty_p(in_dj) || dj_full_p(in_dj)) 
    return dj_dup(in_dj);

  /* General case */
  for( dj = in_dj; dj != NULL; dj = dj->succ) {
    Psysteme     ps = dj->psys, new_ps;
    Pcontrainte  ineq;
    
    assert(!SC_UNDEFINED_P(ps)&&!sc_empty_p(ps)&&!sc_full_p(ps));
    
    if (ps->nb_ineq <= 1) { 
      ret_dj = sl_append_system( ret_dj, sc_dup( ps )); 
      continue; 
    }

    /* Compare with inequalities */
    new_ps = sc_make( contraintes_dup(ps->egalites), CONTRAINTE_UNDEFINED );
    for (ineq = ps->inegalites; ineq != NULL; ineq = ineq->succ) {
      Pcontrainte  co, ineq2;
      Pvecteur     pv = vect_dup(ineq->vecteur); 
      vect_chg_sgn        ( pv );
      co    = contrainte_make( pv );
      ineq2 = contrainte_dup(ineq);

      /* Do we have ineq <= 0 and - ineq <= 0 ? */ 
      if (contrainte_in_liste(co, ps->inegalites)) {
   if (   !contrainte_in_liste(ineq, new_ps->egalites)
       && !contrainte_in_liste(co,   new_ps->egalites) )
       sc_add_egalite( new_ps, ineq2 );
      }
      else { sc_add_inegalite( new_ps, ineq2 ); }
      co = contrainte_free( co );
    }

    new_ps->base = NULL;
    sc_creer_base( new_ps );
    ret_dj = (Pdisjunct) sl_append_system( ret_dj, new_ps );
  }

  return ret_dj;
}

#line 477 "disjunct.w"

/* bool dj_is_system_p( (Pdisjunct) in_dj )  AL 16/11/93
 * Returns True if disjunction in_dj has only one Psysteme in it.
 */
bool dj_is_system_p( in_dj )
Pdisjunct in_dj;
{ return( sl_is_system_p( (Psyslist) in_dj ) ); }


/* Pdisjunct dj_append_system( (Pdisjunct) in_dj, (Psysteme) in_ps )
 * Input  : A disjunct in_dj to wich in_ps will be added. AL 10/11/93
 * Output : Disjunct in_dj with in_ps. => ! Sharing.
 * Comment: Nothing is checked on result in_dj.
 */
Pdisjunct dj_append_system( in_dj, in_ps )
Pdisjunct in_dj;
Psysteme  in_ps;
{ 
  Pdisjunct ret_dj;
  
  if (dj_full_p(in_dj)) { ret_dj = dj_new(); ret_dj->psys = in_ps; }
  else {ret_dj = (Pdisjunct) sl_append_system((Psyslist) in_dj, in_ps);}
  return ret_dj;
}

#line 507 "disjunct.w"

/* dj_variable_rename replaces in_vold with in_vnew : in_dj is modified */
Pdisjunct dj_variable_rename( in_dj, in_vold, in_vnew )
Pdisjunct in_dj;
Variable  in_vold;
Variable  in_vnew;
{ 
  Pdisjunct dj;
  if DJ_UNDEFINED_P(in_dj) return DJ_UNDEFINED;
  if (dj_empty_p(in_dj)||dj_full_p(in_dj)) return in_dj;
  
  for(dj = in_dj; dj != NULL; dj = dj->succ) 
    { sc_variable_rename( dj->psys, in_vold, in_vnew ); }
  return in_dj;
}

#line 532 "disjunct.w"

/* Pdisjunct  dj_variable_substitution_with_eqs_ofl_ctrl() AL160595
 * Substitutes to all systems of disjunction in_dj definitions
 * of variables in in_pv that are implied by in_pc.
 * in_pc are equality constraints that define uniquely each 
 * variable of in_pc. 
 * Function based on sc_variable_substitution_with_eq_ofl_ctrl().
 */
Pdisjunct dj_variable_substitution_with_eqs_ofl_ctrl( in_dj, in_pc, in_pv, ofl_ctrl )
Pdisjunct    in_dj    ;
Pcontrainte  in_pc    ;
Pvecteur     in_pv    ;
int          ofl_ctrl ;
{
  Pvecteur       vec1;

  /* Special cases */
  if (dj_full_p(in_dj)||dj_empty_p(in_dj)||DJ_UNDEFINED_P(in_dj)) return in_dj;


  C3_DEBUG( "dj_variable_substitution_with_eqs_ofl_ctrl", {
    dj_fprint        ( stderr, in_dj,        union_variable_name );
    contrainte_fprint( stderr, in_pc, false, union_variable_name );
    vect_fprint      ( stderr, in_pv,        union_variable_name );
  });


  /* For each variable in in_pv, 
   * we should have one and only one constraint in in_pc that defines it.
   * In each constraint, we should have only one variable in in_pv.
   */
  for(vec1 = in_pv; vec1 != NULL; vec1 = vec1->succ) {
    Variable      var     =  vec1->var;
    int           found   =  0;
    Pcontrainte   pr, def = NULL;
    Pvecteur      vec2    ;
    Pdisjunct     dj      ;

    /* Find constraint def that defines var (only one !) */
    for(pr = in_pc; pr != NULL; pr = pr->succ)
      { if(vect_coeff(var, pr->vecteur) != 0) {found++; def = pr;} }
    assert( found == 1);
    
    /* Assert that there is no other variable of in_pv that belongs to def */
    for(vec2 = in_pv; vec2 != NULL; vec2 = vec2->succ) {
      if (vec2->var == var) continue;
      assert( vect_coeff( vec2->var, def->vecteur ) == 0 );
    }
    
    /* Assert that there is no other variable of in_pv that belongs to def */
    for(dj = in_dj; dj != NULL; dj = dj->succ) 
      { dj->psys = sc_variable_substitution_with_eq_ofl_ctrl(dj->psys, def, var, ofl_ctrl); }

  }
  return( in_dj );
}

#line 597 "sl_io.w"

/* void dj_fprint_tab(FILE*, Pdisjunct, function, int) prints a Pdisjunct */
void    dj_fprint_tab( in_fi, in_dj, in_fu, in_tab )
FILE*       in_fi;
Pdisjunct   in_dj;
char        *(*in_fu)();
int         in_tab;
{
  char*  tabs = sl_get_tab_string( in_tab );

  if (dj_full_p(in_dj))    { fprintf(in_fi, "%sDJ_FULL\n",      tabs); return; }
  if DJ_UNDEFINED_P(in_dj) { fprintf(in_fi, "%sDJ_UNDEFINED\n", tabs); return; }

  fprintf      ( in_fi, "\n%s# -----DJ BEGIN-----\n", tabs   );
  sl_fprint_tab( in_fi, (Psyslist) in_dj, in_fu,      in_tab ); 
  fprintf      ( in_fi, "\n%s# -----DJ END-----\n",   tabs   );
}


/* void dj_read(FILE*) reads a Pdisjunct */
Pdisjunct dj_read( nomfic )
char* nomfic;
{ return ( (Pdisjunct) sl_read(nomfic) ); }

#line 64 "disjunct.w"

