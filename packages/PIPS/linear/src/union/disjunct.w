%%
%% $Id$
%%
%% Copyright 1989-2014 MINES ParisTech
%%
%% This file is part of Linear/C3 Library.
%%
%% Linear/C3 Library is free software: you can redistribute it and/or modify it
%% under the terms of the GNU Lesser General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% any later version.
%%
%% Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
%% WARRANTY; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE.
%%
%% See the GNU Lesser General Public License for more details.
%%
%% You should have received a copy of the GNU Lesser General Public License
%% along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
%%


\section{Structure de donnée}
\paragraph{Une disjonction} $\cal D$, est la plus simple représentation d'une union de
polyèdres : c'est une liste de polyèdres, donc de Psystemes.
\[
{\cal D} = \bigvee_{i=1}^{n} {\cal P}_{i}
\]
La structure est identique à celle de Psyslist :
@d Type @{
typedef Ssyslist *Pdisjunct,Sdisjunct;

#define DJ_UNDEFINED    (Pdisjunct) NULL
@}

Géométriquement, cette structure correspond à
la vision présenté à la figure~\ref{disjunctfig}.

\begin{figure} \alepsf{disjunct.eps}{2/3}
\caption{Une union vue comme disjonction} \label{disjunctfig}
\end{figure}

Remarquons que cette structure est efficace pour manipuler
l'union de disjonctions, puisqu'il s'agit en fait
d'une simple concaténation de listes. Cette représentation n'est
par contre pas efficace pour les intersections d'unions.

Les disjonctions apparaissent ici comme des "sommes de produits" en
terme de logique ; c'est à dire des disjonctions de systèmes 
représentés par des conjonctions d'hyperplans. 
Les techniques classiques de réduction de formule booléenne
décrites dans \cite{Ger87} : table de Karnaugh et procédure de
Quine-McCluskey, ne sont pas ici applicables puisque nous n'avons pas
affaire à des variables booléenes libres.

\paragraph{Passage d'une disjonction à un chemin.}
La structure de chemin est décrite dans le chapitre
\ref{chemin-chap}. Ce dernier chapitre s'attache à produire une
disjonction équivalente à un chemin. Par symétrie, nous
présentons ici la formule de passage d'une disjonction à un
chemin. Il ne s'agit ici que d'informer le lecteur, puisque cette
formule ne sera pas utilisée.

\begin{eqnarray*}
\bigvee_{i=1}^{n} {\cal P}_{i} &=& \neg (\neg  \bigvee_{i=1}^{n} {\cal P}_{i} )  \\
 &=& \neg ( \bigwedge_{i=1}^{n} \neg {\cal P}_{i} ) \\
 &=& \neg ( \bigwedge_{i=1}^{n} \bigvee_{j_i = 1}^{p_i} \neg h_{ij_i} ) \\
 &=& \neg ( \bigvee_{j_1, \ldots, j_n} \bigwedge_{i=1}^{n} \neg h_{ij_i} )\\ 
 &=& \bigwedge_{j_1, \ldots, j_n} \neg {\cal P'}_{j_1,\ldots,j_n} \\
 && \mbox{ avec } {\cal P'}_{j_1,\ldots,j_n} = \bigwedge_{i=1}^{n} \neg h_{ij_i}
\end{eqnarray*}

Remarquons ici l'absence de polyèdre englobant, similaire à ${\cal P}_0$ dans
la description des chemins. Ce rôle peut être joué par l'enveloppe 
convexe ${\cal H}$ des ${\cal P}_i$ : 
$\bigvee_{i=1}^{n} {\cal P}_{i} = 
	{\cal H} \bigwedge_{j_1, \ldots, j_n} \neg {\cal P'}_{j_1,\ldots,j_n}$.



\section{Fonctions}
Les fonctions sont dans le fichier :
@O disjunct.c -d @{
@< includes @>
@< fonctions Pdisjunct @>
@}

\subsection{Allocation et désallocation mémoire}
Les fonctions d'allocation et de désallocation de la mémoire sont
similaires aux fonctions communes puisqu'un Pdisjunct est similaire
à un Psyslist.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_new()	AL 26/10/93
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
@| dj_new sj_dup dj_free dj_dup1 dj_free1 @}


\paragraph{dj\_full} est une union représentant l'espace entier.
C'est par convention un Pdisjunct dont les champs sont positionés
à NULL. \\
{\bf dj\_full\_p} teste si une disjonction représente l'espace entier.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_full()	AL 18/11/93
 * Return full space disjunction = dj_new()
 */
Pdisjunct dj_full(){ return( dj_new() ); }


/* dj_full_p( (Pdisjunct) in_dj )	AL 30/05/94
 * Returns True if in_dj = (NIL) ^ (NIL)
 */
boolean dj_full_p( in_dj )
Pdisjunct in_dj;
{
  return( (in_dj != DJ_UNDEFINED) &&
	 ( in_dj->succ == NULL ) &&
	 ( in_dj->psys  == NULL ) );
}
@| dj_full dj_full_p @}

\paragraph{dj\_empty} est la disjonction vide contenant un seul
Psysteme infaisable : sc\_empty. \\
{\bf dj\_empty\_p} détecte les disjonctions vides.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_empty()		AL 18/11/93
 * Returns a disjunction with sc_empty() element.
 */
Pdisjunct dj_empty()
{ return (Pdisjunct) sl_append_system(NULL, sc_empty((Pbase) NULL)); }


/* dj_empty_p( (Ppath) in_pa )	AL 30/05/94
 * Returns True if in_dj = (1*TCST = 0) ^ (NIL)
 */
boolean dj_empty_p( in_dj )
Pdisjunct in_dj;
{
  return( ( in_dj != DJ_UNDEFINED     )    &&
	  ( in_dj->succ == NULL       )    &&
	  ( in_dj->psys != NULL       )    &&
	  ( sc_empty_p( in_dj->psys ) )       );
}
@| dj_empty dj_empty_p @}





\subsection{Opérations sur les disjonctions}
\paragraph{dj\_intersection\_ofl\_ctrl} calcule l'intersection de
deux disjonctions. Seuls les systèmes faisables sont pris en compte.
Si dj1 a $n$ systèmes et dj2 $p$, la complexité est au moins de $n * p$ :
le test de faisabilité étant effectué, la complexité peut
être exponentielle. Il n'y a pas ici de partage.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_intersection_ofl_ctrl( in_dj1, in_dj2, ofl_ctrl )
 * Computes intersection of two disjunctions. 		 AL,BC 23/03/95
 * Very costly function : -> sc_faisabilite_ofl_ctrl used.
 * No sharing 
 */
Pdisjunct dj_intersection_ofl_ctrl( in_dj1, in_dj2, ofl_ctrl )
Pdisjunct in_dj1, in_dj2;
int ofl_ctrl;
{
  Pdisjunct	dj1, dj2, ret_dj;
  
  if (DJ_UNDEFINED_P(in_dj1)||DJ_UNDEFINED_P(in_dj2)) return DJ_UNDEFINED   ;
  if (dj_full_p(in_dj1) && dj_full_p(in_dj2))         return dj_full()      ;
  if (dj_full_p(in_dj1))                              return dj_dup(in_dj2) ;
  if (dj_full_p(in_dj2))                              return dj_dup(in_dj1) ;
  if (dj_empty_p(in_dj1)||dj_empty_p(in_dj2))         return dj_empty()     ;
  
  ret_dj = (Pdisjunct) NULL; 
  for(dj1 = in_dj1; dj1 != NULL; dj1 = dj1->succ) {
    for(dj2 = in_dj2; dj2 != NULL; dj2 = dj2->succ) {
      Psysteme ps = sc_append( sc_dup(dj1->psys), dj2->psys );
      if (!sc_rational_feasibility_ofl_ctrl( ps, ofl_ctrl, TRUE )) 
	{ ps = sc_free( ps ); continue; }
      ret_dj = (Pdisjunct) sl_append_system( ret_dj, ps );
    }
  }
  if (ret_dj == (Pdisjunct) NULL) return dj_empty(); /* empty intersection */
  return ret_dj;
}
@| dj_intersection_ofl_ctrl @}



\paragraph{dj\_intersect\_system\_ofl\_ctrl} calcule l'intersection
d'une disjonction et d'un système, sans partage sur les données entrées.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_intersect_system_ofl_ctrl( )  */
Pdisjunct  dj_intersect_system_ofl_ctrl( in_dj, in_ps, ofl_ctrl )
Pdisjunct  in_dj    ;
Psysteme   in_ps    ;
int        ofl_ctrl ;
{ return dj_intersection_ofl_ctrl( in_dj, sl_append_system(NULL, in_ps), ofl_ctrl ); }
@| dj_intersect_system_ofl_ctrl @}



\paragraph{dj\_intersection\_djcomp\_ofl\_ctrl} est l'intersection de
la première disjonction avec le complémentaire de la seconde. 
Les Psystemes sous-jacent ne sont pas partagés.
@D fonctions Pdisjunct @{
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
@| dj_intersect_djcomp_ofl_ctrl @}


\paragraph{dj\_union} est l'union de deux disjonctions. C'est une simple
concaténation de listes. Les Psystemes sous-jacent sont partagés.
@D fonctions Pdisjunct @{
/* Pdisjunct dj_union( (Pdisjunct) in_dj1, (Pdisjunct) in_dj2 ) 
 * Give the union of the two disjunctions. AL 15/11/93
 * Memory: systems of the 2 unions are shared. 
 * 	in_dj1 = dj_union(in_dj1,in_dj2); 
 * 	(in_dj1 = dj_free(in_dj1);      to remove in_dj1 and in_dj2
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
@| dj_union @}

\paragraph{dj\_feasibility\_ofl\_ctrl} renvoie Vrai si la disjonction
est faisable : un appel à la faisabilité est effectué pour
chaque polyèdre de la disjonction. On s'arrète dès que l'on
en trouve un faisable. La complexité théorique est donc
exponentielle. 

@D fonctions Pdisjunct @{
/* boolean dj_feasibility_ofl_ctrl( (Pdisjunct) in_dj, (int) ofl_ctrl )	
 * Returns true if in_dj is a feasible disjunction. AL,BC 23/02/95
 */
boolean dj_feasibility_ofl_ctrl( in_dj, ofl_ctrl )
Pdisjunct in_dj;
int ofl_ctrl;
{
  boolean 	ret_bool = FALSE;
  Pdisjunct	dj;

  if ( in_dj == DJ_UNDEFINED ) return FALSE;
  for( dj = in_dj; dj != NULL && !ret_bool; dj = dj->succ ) {
    if (dj->psys == SC_UNDEFINED) return FALSE;
    ret_bool = ret_bool || 
      sc_rational_feasibility_ofl_ctrl( dj->psys, ofl_ctrl, TRUE );
  }
  return ret_bool;
}
@| dj_feasibility_ofl_ctrl @}


\paragraph{dj\_system\_complement} calcule la disjonction
représentant le complémentaire d'un polyèdre $PS$, c'est à
dire $\neg PS$. Si $PS$ est composé de $p$ inégalités,
la complexité est de $p$ : on constitue juste une liste
de $p$ inégalités opposées.

@D fonctions Pdisjunct @{
/* Pdisjunct dj_system_complement( (Psystem) in_ps )	AL 26/10/93
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
@| dj_system_complement @}



\paragraph{dj\_disjunct\_complement} calcule la disjonction
représentant le complémentaire d'une disjonction {\tt in\_dj}.
@D fonctions Pdisjunct @{
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
@| dj_disjunct_complement @}


\paragraph{dj\_projection\_along\_variables\_ofl\_ctrl} projette chaque
système de la disjonction {\tt in\_dj} le long des varibles
contenues dans le vecteur {\tt in\_pv}.
@D fonctions Pdisjunct @{
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
@| dj_projection_along_variables_ofl_ctrl @}


\paragraph{dj\_simple\_inegs\_to\_eg} revoie une dijonction
équivalente à {\tt in\_dj} où deux inégalités
opposées $a \leq 0 \wedge -a \leq 0$ sont transformées en une
égalité simple pour chacun des systèmes de {\tt in\_dj}. Le
système d'entré n'est pas modifié.
@D fonctions Pdisjunct @{
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
@| dj_simple_inegs_to_eg  @}




\subsection{Fonctions de manipulation}
\paragraph{dj\_is\_system\_p} indique si la disjonction est limitée
à un seul système. \\ 
{\bf dj\_append\_system} ajoute un Psysteme ps à une disjonction
dj, sans test de faisabilité sur ps.
@D fonctions Pdisjunct @{
/* boolean dj_is_system_p( (Pdisjunct) in_dj )	AL 16/11/93
 * Returns True if disjunction in_dj has only one Psysteme in it.
 */
boolean dj_is_system_p( in_dj )
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
@| dj_is_system_p dj_append_system @}


\paragraph{dj\_variable\_rename} remplace la variable {\tt in\_vold}
par la variable {\tt in\_vnew}. La disjonction {\tt in\_dj},
modifiée, est renvoyée.  
@D fonctions Pdisjunct @{
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
@| dj_variable_rename @}


\paragraph{dj\_variable\_substitution\_with\_eqs\_ofl\_ctrl} substitue,
dans la disjonction entrée {\tt in\_dj}, aux variables contenues
dans le vecteur {\tt in\_pv} les vecteurs associés à ces
variables par les équations dans la contrainte {\tt in\_pc}. A
chaque variable à renommer dans {\tt in\_pv} doit correspondre une
unique équation dans {\tt in\_pc} contenant cette variable avec un
coefficient 1. La disjonction entrée est modifiée, puis renvoyée.
@D fonctions Pdisjunct @{
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
    contrainte_fprint( stderr, in_pc, FALSE, union_variable_name );
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
@| dj_variable_substitution_with_eqs_ofl_ctrl @}

