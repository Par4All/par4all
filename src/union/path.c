
#line 55 "path.w"


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


#line 56 "path.w"


#line 97 "path.w"

/* Ppath pa_make(in_ps, in_pcomp)    AL 16/11/93
 * Allocates a Ppath and initialize it with in_ps and in_pcomp
 * SHARING.
 */
Ppath pa_make( in_ps, in_pcomp )
Psysteme in_ps;
Pcomplist in_pcomp;
{
  Ppath ret_pa = (Ppath) malloc( sizeof( Spath ) );
  if (ret_pa == NULL) {
    (void) fprintf(stderr,"pa_new: Out of memory space\n");
    exit(-1);
  }
  ret_pa->psys = in_ps; ret_pa->pcomp = in_pcomp;
  return ret_pa;
}


/* void pa_dup(Ppath pa)       AL 30/05/94 */
Ppath pa_dup(in_pa)
Ppath in_pa;
{
  if (in_pa == PA_UNDEFINED ) return PA_UNDEFINED;
  return pa_make( sc_dup(in_pa->psys), sl_dup(in_pa->pcomp) );
}

/* Ppath pa_free(Ppath pa)      BA, AL 30/05/94 */
Ppath pa_free(in_pa)
Ppath in_pa;
{
  if (in_pa != PA_UNDEFINED) {
    in_pa->psys  = sc_free(in_pa->psys);
    in_pa->pcomp = sl_free((Psyslist) in_pa->pcomp);
    free( in_pa ); in_pa = PA_UNDEFINED;
  }
  return((Ppath) PA_UNDEFINED);
}


/* void pa_dup1(Ppath pa)           AL 30/05/94
 * 1 depth duplication: system and complements are shared. 
 */
Ppath pa_dup1(in_pa)
Ppath in_pa;
{
  if (in_pa == PA_UNDEFINED) return PA_UNDEFINED;
  return pa_make( in_pa->psys, sl_dup1(in_pa->pcomp) );
}


/* Ppath pa_free1(Ppath pa)        BA, AL 30/05/94
 * 1 depth free. System and complement are not freed.
 */
Ppath pa_free1(in_pa)
Ppath in_pa;
{
  if (in_pa != PA_UNDEFINED) {
    sl_free1((Psyslist) in_pa->pcomp);
    free( in_pa ); in_pa = PA_UNDEFINED;
  }  
  return((Ppath) PA_UNDEFINED);
}

#line 171 "path.w"

/* Ppath pa_full()         AL 18/11/93
 * Returns full space path : pa_full = pa_new()
 */
Ppath pa_full() { return pa_new(); }


/* pa_full_p( (Ppath) in_pa )   AL 18/11/93
 * Returns True if in_pa = (NIL) ^ (NIL)
 */
bool pa_full_p( in_pa )
Ppath in_pa;
{
  return( (in_pa != PA_UNDEFINED) &&
    ( in_pa->pcomp == NULL ) &&
    ( in_pa->psys  == NULL ) );
}


/* Ppath pa_empty()           AL 18/11/93
 * Returns empty path : pa_empty = sc_empty(NULL) ^ (NIL)
 */
Ppath pa_empty() { return pa_make(sc_empty(NULL), NULL); }


/* pa_empty_p( (Ppath) in_pa )   AL 18/11/93
 * Returns True if in_pa = (1*TCST = 0) ^ (NIL)
 */
bool pa_empty_p( in_pa )
Ppath in_pa;
{
  return( (in_pa != PA_UNDEFINED) &&
    ( in_pa->pcomp == NULL ) &&
    ( in_pa->psys != NULL ) &&
    ( sc_empty_p( in_pa->psys ) ) );
}

#line 216 "path.w"

/* int pa_max_constraints_nb( (Ppath) in_pa )
 * Give the maximum constraints nb among systems of in_pa. 
 */
int pa_max_constraints_nb( in_pa )
Ppath    in_pa;
{
  Psysteme   ps;
  int        loc, ret_int;

  if (PA_UNDEFINED_P(in_pa)||pa_full_p(in_pa)) return 0;
  if ( pa_empty_p(in_pa) )                     return 1;

  ps      = in_pa->psys; 
  ret_int = 2*(ps->nb_eq) + ps->nb_ineq;
  loc     = sl_max_constraints_nb( (Psyslist) in_pa->pcomp );
  
  if (loc > ret_int) ret_int = loc;
  return ret_int;
}

#line 249 "path.w"

/* Ppath pa_intersect_system( (Ppath) in_pa, (Psysteme) in_ps )
 * Computes the intersection between in_pa and in_ps. AL 25/04/95
 * No sharing 
 */
Ppath pa_intersect_system( in_pa, in_ps )
Ppath    in_pa;
Psysteme in_ps;
{
  Psysteme ps;

  if (PA_UNDEFINED_P(in_pa)||SC_UNDEFINED_P(in_ps)) 
                            return PA_UNDEFINED;
  if ( pa_empty_p(in_pa) )  return pa_empty();
  if ( pa_full_p(in_pa) )   return pa_make(sc_dup(in_ps),NULL);
  
  ps = sc_normalize(sc_append( sc_dup(in_pa->psys), in_ps ));
  if (ps == NULL){ ps = sc_free(ps); return pa_empty(); }
  return pa_make(ps, sl_dup(in_pa->pcomp));
}

#line 274 "path.w"

/* Ppath pa_intersect_complement( (Ppath) in_pa, (Pcomplement) in_pc )
 * Computes the intersection between in_pa and in_ps.  AL 17/11/93
 * No sharing 
 */
Ppath pa_intersect_complement( in_pa, in_pc )
Ppath       in_pa;
Pcomplement in_pc;
{
  Pcomplist  pc;
  Psysteme   ps;

  if (PA_UNDEFINED_P(in_pa)||SC_UNDEFINED_P(in_pc)) return PA_UNDEFINED;
  if (pa_empty_p(in_pa))                            return pa_empty();

  if (pa_full_p(in_pa)) ps = sc_full(); else  ps = sc_dup(in_pa->psys);
  pc = sl_append_system( sl_dup(in_pa->pcomp), sc_dup(in_pc) );
  return pa_make(ps, pc) ;
}

#line 300 "path.w"

/* Ppath pa_reduce_simple_complement( (Ppath) in_pa )     AL 16/11/93
 * Scan all the complement. If one complement is a simple inequality,
 * its complement is computed and intersected with psys part of in_pa.
 * in_pa is modified. (Sharing with in_pa).
 */
Ppath pa_reduce_simple_complement( in_pa )
Ppath in_pa;
{
  Psysteme      pss;
  Pcomplist pco, pco2 = NULL, tofree = NULL;
  Ppath         ret_pa;
  bool       at_least_one = false ; /* Do we have a simple complement ? */
  
  if( pa_full_p(in_pa) || pa_empty_p(in_pa) || (in_pa == PA_UNDEFINED) ) 
                return (in_pa);
  
  C3_DEBUG("pa_reduce_simple_complement", {
    fprintf(stderr, "Input path:\n");  
    pa_fprint_tab( stderr, in_pa, union_variable_name, 1 );
  });

  pss = in_pa->psys;
  for( pco = in_pa->pcomp, pco2 = NULL; pco != NULL; pco = pco->succ ) {
    Psysteme ps = pco->psys;
    
    if (ps == SC_UNDEFINED) { 
   pco2  = sl_free1(pco2); 
   in_pa = pa_free1(in_pa); 
   return PA_UNDEFINED ; 
    }
    else if (sc_empty_p(ps)) continue;
    else if ((ps->nb_ineq == 1) && (ps->nb_eq == 0)) {
      Pdisjunct dj = dj_system_complement( ps );
      pss          = sc_safe_append( pss, dj->psys );
      tofree       = sl_append_system( tofree, ps );
      dj           = dj_free( dj ); 
      at_least_one = true;
    }
    else { pco2 = (Pcomplist) sl_append_system( pco2, ps ); }
  }

  if(!at_least_one) {
    pco2   = sl_free1(pco2);  
    ret_pa = in_pa;
  }
  else if(!sc_faisabilite_ofl(pss)) {
    pco2   = sl_free1( pco2   ); 
    tofree = sl_free1( tofree );
    in_pa  = pa_free ( in_pa  ); /* also frees pss */
    ret_pa = pa_empty(); 
  } 
  else {
    in_pa  = pa_free1( in_pa  ); 
    tofree = sl_free ( tofree );
    ret_pa = pa_make ( pss, pco2 );
  }

  C3_RETURN( IS_PA, ret_pa );
}

#line 366 "path.w"

/* Ppath pa_transform_eg_in_ineg( in_pa )
 * Transforms all equalities of all systems composing in_pa in
 * inequalities and returns in_pa.
 * in_pa is modified. (Sharing with in_pa).
 */
Ppath pa_transform_eg_in_ineg( in_pa )
Ppath in_pa;
{
  Pcomplist pco;

  if( pa_full_p(in_pa) || pa_empty_p(in_pa) || (in_pa == PA_UNDEFINED) ) 
                return (in_pa);
  
  sc_transform_eg_in_ineg( in_pa->psys );
  for( pco = in_pa->pcomp; pco != NULL; pco = pco->succ ) 
    { sc_transform_eg_in_ineg( pco->psys ); }

  return in_pa;
}

#line 395 "path.w"

/* bool pa_feasibility_ofl_ctrl( (Ppath) in_pa, int ofl_ctrl)   
 * Returns true if the input path is possible and false if it 
 * is not possible or undefined.                 
 */
#ifdef TRACE_Linear/C3 Library_PATH
extern char* entity_local_name() ;
#endif 

bool pa_feasibility_ofl_ctrl( in_pa, ofl_ctrl )
Ppath in_pa;
int ofl_ctrl;
{
  Pdisjunct  dj;
  Ppath      pa;
  bool    ret_bo = false;
#ifdef TRACE_Linear/C3 Library_PATH
  FILE*      report_file;
#endif 

  if ( PA_UNDEFINED_P( in_pa )) return false;
  if ( pa_empty_p    ( in_pa )) return false;
  if ( pa_full_p     ( in_pa )) return true;
  
#ifdef TRACE_Linear/C3 Library_PATH
  /* Just to keep trace of input paths if wanted */
  if (getenv("KEEP_PATH") != (char*) NULL) {
    struct timeval  *tp = (struct timeval*)  malloc(sizeof(struct timeval));
    struct timezone *tz = (struct timezone*) malloc(sizeof(struct timezone));
    int seconds;
    gettimeofday( tp, tz ); seconds = tp->tv_sec;
    report_file = fopen("mail_those_paths_to_arnauld","a");
    pa_fprint( report_file, in_pa, union_variable_name );
    fprintf( report_file, "# %s", ctime( &(seconds) ));
    fprintf( report_file, 
       "# Module:                            \t%s\n", db_get_current_module_name());
    fprintf( report_file, 
       "# Input number of complement:        \t%d\n", sl_length(in_pa->pcomp)     );
    fprintf( report_file, 
       "# Input max constrainst:             \t%d\n", pa_max_constraints_nb(in_pa));
    fflush ( report_file ); free( tp ); free( tz );
  }
#endif  

  pa = pa_supress_same_constraints( in_pa );
  dj = pa_path_to_few_disjunct_ofl_ctrl( pa, ofl_ctrl );
  if( dj_empty_p(dj) || (dj == NULL) ) ret_bo = false;
  else                                 ret_bo = true;


#ifdef TRACE_Linear/C3 Library_PATH
  /* keep trace of paths */
  if (getenv("KEEP_PATH") != (char*) NULL) {
    fprintf( report_file, 
       "# Output number of disjunctions:     \t%d\n", sl_length(dj)            );
    fprintf( report_file, 
       "# Output max constrainst:            \t%d\n", sl_max_constraints_nb(dj));
    fprintf( report_file, 
       "# Feasible:                          \t%s\n", (ret_bo) ? "YES":"NO"    );
    fclose ( report_file );
  }
#endif 

  pa = pa_free( pa );  dj = dj_free( dj );
  return ret_bo;
}

#line 470 "path.w"

/* Pdisjunct pa_path_to_disjunct_ofl_ctrl
 *       ( (Ppath) in_pa, (int) ofl_ctrl)    
 * Produces a Pdisjunct corresponding to the path Ppath.
 * No sharing.
 */
Pdisjunct pa_path_to_disjunct_ofl_ctrl( in_pa, ofl_ctrl )
Ppath   in_pa;
int     ofl_ctrl;
{
  Pdisjunct       ret_dj;
  Pcomplist       comp;
  int             meth1 = 0, meth2 = 1; /* comparison between 2 methods */

  if ( in_pa == PA_UNDEFINED )   return DJ_UNDEFINED;
  if (pa_full_p(in_pa))          return dj_full();
  if (pa_empty_p(in_pa))         return dj_empty();
  if ((in_pa->psys != NULL) && 
   sc_empty_p(in_pa->psys)) return dj_empty();
  
  ret_dj = (Pdisjunct) sl_append_system(NULL, sc_dup(in_pa->psys)); 
  for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ) {
    Pdisjunct dj1 = dj_system_complement( comp->psys ); 
    Pdisjunct dj2 = ret_dj;
    int       lg1 = sl_length( dj1 );
    int       lg2 = sl_length( dj2 );

    meth1 = meth1 + lg2*lg1 ; meth2 = meth2 * lg1;

    ret_dj        = dj_intersection_ofl_ctrl( ret_dj, dj1, ofl_ctrl);
    dj1           = dj_free( dj1 ); dj2 = dj_free( dj2 );
  }
   
  C3_DEBUG("pa_path_to_disjunct_ofl_ctrl", {
    fprintf(stderr, "Feasibility calls with method 1 and 2 : %d\t%d\n", 
      meth1, meth2);  
  });

  return( ret_dj );
}

#line 631 "sl_io.w"

/* void pa_fprint_tab(FILE*, Pdisjunct, function, tab) prints a Ppath */
void pa_fprint_tab( in_fi, in_pa, in_fu, in_tab )
FILE*   in_fi;
Ppath   in_pa;
char    *(*in_fu)();
int     in_tab;
{
  Psyslist    sl;
  char*       tabs = sl_get_tab_string( in_tab );
  
  if (pa_full_p(in_pa))    { 
    fprintf(in_fi, "%sPA_FULL\n", tabs); 
    free(tabs); return;
  }
  if PA_UNDEFINED_P(in_pa) { 
    fprintf(in_fi, "%sPA_UNDEFINED\n", tabs); 
    free(tabs); return;
  }

  sl = sl_new(); sl->succ = in_pa->pcomp; sl->psys = in_pa->psys;
  fprintf      ( in_fi, "\n%s# --------PA BEGIN------\n", tabs);
  sl_fprint_tab( in_fi, sl, in_fu, in_tab );
  fprintf      ( in_fi, "\n%s# --------PA END--------\n", tabs);
  free( sl ); free( tabs ); return;
}

/* void pa_read(FILE*) reads a Ppath */
Ppath pa_read( nomfic )
char* nomfic;
{
  Ppath       ret_pa;
  Psyslist    sl;

  sl = sl_read(nomfic);
  if (sl == SL_NULL) return PA_UNDEFINED;
  ret_pa = pa_make(sl->psys, (Pcomplist) sl->succ);
  free( sl );
  return ret_pa;
}

#line 57 "path.w"

