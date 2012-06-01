%%
%% $Id$
%%
%% Copyright 1989-2012 MINES ParisTech
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


\section{Contraintes redondantes}
\subsection{Opération de coupure}
\paragraph{}
Il est important de pouvoir réduire au fur et à mesure la taille
des systèmes générés. La première raison est la
rapidité des calculs : plus le système est de taille faible,
plus les tests de faisabilités (Fourrier-Motzkin), constamment
appelés, sont rapides. La deuxième raison est la possibilité
de mener ces tests : un système contenant un grand nombre de
contraintes peut exploser par Fourrier Motzkin ou mener à des
erreurs d'overflow pour le simplexe. 

Cette partie rappelle la notion de polyèdre sans contraintes
redondantes et introduit la notion de non redondance relative.

\paragraph{Ensemble de travail}
Nous nous placerons dans l'ensemble $\Bbb N^l$, un sytème de
contraintes représentant implicitement les points entiers dans le
polyèdre convexe délimité par ce système. Nous parlerons
donc de faisabilité et de complémentaire dans $\Bbb N^l$. Ainsi,
pour \( h(x) = \{ x\,|\, a.x + b \leq 0\}\) on a :
\( \complement h = \neg h = \{x\,|\, -a.x - b + 1 \leq 0\} \).

\paragraph{Représentation des Polyèdres}
Le polyèdre $P$ est l'intersection de $p$ hyperplans :
\[
P = \bigwedge_{1 \leq j \leq p} h_{j} \\
\mbox{ avec } h_{j} = a_{j}.x + b_{j} \leq 0
\]
Nous noterons pour $p > 1$ :
\[
P_{\neg j} = \bigwedge_{1 \leq k \leq p \wedge k \neq j} h_{k} \\
\]

Nous rajoutons à l'ensemble des polyèdres convexes les
éléments $Empty$, polyèdre vide, infaisable car ne contenant
aucun entier, et le polyèdre $Full$ l'ensemble $\Bbb N^l$.



\paragraph{Définition de l'opérateur de coupure $/$}
Un demi-espace $H$ coupe un polyèdre convexe $P$ si 
$P \cap \complement H = \varnothing$. Ceci nous permet de définir
l'opérateur de coupure :

\begin{define}
\begin{equation}
H/P = 
  \begin{cases}
	H     & \text{si \(\complement H \cap P\) est faisable.}, \\
	Full  & \text{sinon}.
  \end{cases}
\end{equation}
\end{define} 

Nous étendons cette définition à $Empty$ et $Full$. Ainsi,
$Full/P = Full$, $Empty/P$ est égal à $Full$ si $P$ est faisable
et à $Empty$ sinon.

Cette définition s'étend aux unions :
\begin{eqnarray}
P'/ P &=& \bigwedge_{1 \leq j \leq p} (h'_{j}/P) \\
(\bigvee_{1 \leq i' \leq n'} P'_{i'}) / P
	 &=& \bigvee_{1 \leq i' \leq n'} (P'_{i'}/P) \\
 (\bigvee_{1 \leq i' \leq n'} P_{i'}) / (\bigvee_{1\leq i\leq n} P_i)
	 &=& \bigvee_{1 \leq i \leq n \wedge 1 \leq i' \leq n'} (P'_{i'} / P_i)
\end{eqnarray}


\paragraph{}   
\begin{define}
Un polyèdre $P$ est non-redondant s'il est faisable et si :
\[ \forall j \in [1,p], h_j/P_{\neg j} = h_j \]
\end{define}


\subsection{Calculs}
Comment calculer $H/P$ ? Nous pouvons conclure sur la faisabilité
de $P \cap \complement H$ en ajoutant juste la contrainte $-a.x - b + 1 \leq 0$
à celles qui définissent $P$ puis tester la faisabilité de ce
nouveau polyèdre.

\paragraph{Le treillis des hyperplans parallèles}
Pour normaliser un polyèdre, il faut en théorie faire $p$ tests
de faisabilités, ce qui est couteux. Il peut être interessant de
détecter les contraintes parallèles. 

Dans C3 un demi-espace est
représenté par un vecteur $ax+b \leq 0$, ainsi qu'un hyperplan
$ax+b = 0$. Nous introduisons la fonction {\tt vect\_parallel} qui
prend en entré deux vecteurs et renvoit leur position relative dans
le treillis des signes, représenté à la figure~\ref{sign-fig}.

Le treillis des signes est représentable par la liste des éléments
qu'il contient {\tt hspara\_elem}, ainsi que par une matrice {\tt hspara\_jm} 
donnant dans sa partie basse le résultat de l'opérateur de jointure ($\vee$ join), 
et dans sa partie haute le résulatat de l'opérateur de rencontre 
($\wedge$ meet)~\cite{Gra71}.

\begin{figure} \alepsf{hspara_cases.eps}{1/2}
\caption{Différentes positions relatives d'hyperplans parallèles}
\label{sign-fig} \end{figure}

\begin{figure} \alepsf{hspara_lattice.eps}{1/2}
\caption{Le treillis des hyperplans parallèles}
\label{sign-fig} \end{figure}



@D Type @{
/* Implementation of the finite parallel half space lattice hspara
 *                
 *                      ________ full 
 *                     /          |
 *                    /          empty ___
 *                   /            |       \
 *                  /           keep       \
 *                 /           /    \       \
 *              ssplus        /    opplus    \
 *                 |      ssminus    |     opminus
 *               sszero        \   opzero    /
 *                  \           \   /       /
 *                   \________ unpara _____/
 */
enum hspara_elem
{                      /* compare   {h1: a1 X + b1 <= 0} with {hj: aj X + bj <= 0} */      
  unpara        = 0,   /*  unparallel ->   h1/hj = h1    */
  /**/                 /*  a1 == aj for same sign (ss)  part lattice */ 
    sszero      = 1,   /*  b1 == bj   ->   h1/hj = full  */ 
    ssplus      = 2,   /*  bj >  b1   ->   h1/hj = full  */
  /**/     
    /**/               /* keep part                      */
      /**/     
        ssminus = 3,   /*  bj <  b1   ->   h1/hj = h1    */ 
      /**/             /* -a1 == aj for opposite sign (op)  part lattice */
        opzero  = 4,   /*  b1 == bj   ->   h1/hj = h1    */ 
        opplus  = 5,   /*  bj >  b1   ->   h1/hj = h1    */
    keep        = 6,
    /**/               /* empty part                     */
      opminus   = 7,   /*  b1 <  bj   ->   h1/hj = empty */  
    empty       = 8,   
  full          = 9     
};
@}


@D global reduc @{
static char* hspara_string[10] __attribute__ ((unused)) = 
{  
  "unpara",
  /**/                
    "sszero",   
    "ssplus",   
  /**/     
    /**/     
      /**/     
        "ssminus",    
      /**/            
        "opzero",        
        "opplus",    
    "keep",
    /**/          
      "opminus", 
    "empty",
  "full"
};

static enum  hspara_elem       
             hspara_jm[10][10] = { /* Lower left is join, upper right is meet */

 /*join\meet   unpara sszero ssplus ssminus opzero opplus keep opminus  empty full  */
 /* unpara  */ {  0,    0,     0,      0,     0,    0,      0,     0,     0,    0    },
 /* sszero  */ {  1,    1,     1,      0,     0,    0,      0,     0,     0,    1    },
 /* ssplus  */ {  2,    2,     2,      0,     0,    0,      0,     0,     0,    2    },
 /* ssminus */ {  3,    9,     9,      3,     0,    0,      3,     0,     3,    3    },
 /* opzero  */ {  4,    9,     9,      6,     4,    4,      4,     0,     4,    4    },
 /* opplus  */ {  5,    9,     9,      6,     5,    5,      5,     0,     5,    5    },
 /* keep    */ {  6,    9,     9,      6,     6,    6,      6,     0,     6,    6    },
 /* opminus */ {  7,    9,     9,      8,     8,    8,      8,     7,     7,    7    },
 /* empty   */ {  8,    9,     9,      8,     8,    8,      8,     8,     8,    8    },
 /* full    */ {  9,    9,     9,      9,     9,    9,      9,     9,     9,    9    }};


#define hspara_join(se1, se2)   (((se1) >= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_meet(se1, se2)   (((se1) <= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_to_string(se)    (char*) hspara_string[(int) (se)]   
@| parallel_half_space_lattice hspara @}



\paragraph{vect\_parallel} 
Prenons un exemple trivial auquel peuvent ce ramener les autres pour
expliquer l'utilité de cette fonction. Supposons que {\tt in\_v1}
représente le demi-espace $h_1: x \geq 0$ ($-x \leq 0$)  et {\tt in\_v2}
le demi-espace $h_2: x \geq b$ ($-x +b \leq 0$). Ici, $b = \mbox{\tt
vect\_parallel($h_1$,$h_2$)}$. Nous avons alors :

\begin{itemize}
\item Si $b \geq 0$, alors $h_1 \wedge h_2 = h_2$, et nous pourrons
supprimer $h_1$ d'un système qui intersecterait un système
contenant $h_2$.
\item Si $b \leq 0$, alors $h_1 \wedge \neg h_2 = \varnothing$, et
nous pourrons supprimer $h_2$ des complémentaires si $h_1$ est une
contrainte du système positif, dans le cadre des chemins.
\end{itemize}

@D fonctions reduc @{
/* enum hspara_elem vect_parallel(Pvecteur in_v1, Pvecteur in_v2) AL950711
 * input:      2 Pvecteur in_v1 and in_v2 
 * output:     hspara_elem (element of the parallel half space lattice)
 * memory:     Inspector (nothing is shared, nor modified, output allocated).
 * complexity: length(in_v1) * length(in_v2)
 * comment:    in_v1 = a1 X + b1 represents a1 X+b1 <= 0 and in_v2 a2 X + b2 <=0.
 *             if      (a1!=a2) || (a1!=-a2), returns unpara 
 *             else if (a1==a2),  return sign(b2-b1)  in ss part of hspara
 *             else if (a1==-a2), return sign(-b2-b1) in op part of hspara.
 */
enum hspara_elem vect_parallel( in_v1, in_v2 )
Pvecteur in_v1, in_v2;
{
  Pvecteur            v1, v2;
  enum hspara_elem    ret_sle = unpara;
  boolean             first     = TRUE;
  boolean             same_sign = FALSE;
  Value                 gcd1, gcd2;   /* gcd of each vector                 */
  int                 l1, l2;   /* length of each vector without TCST */
  Value               b1, b2, diff; /* value of TCST and their diff       */

  if (!in_v1 || !in_v2) return unpara;

  /* debuging */
  /*
  C3_DEBUG("vect_parallel", {
    fprintf(stderr, "Input vectors, in_v1, then in_v2:\n");  
    vect_fprint( stderr, in_v1, union_variable_name );
    vect_fprint( stderr, in_v2, union_variable_name );
  });
  */


  /* get gcd of each vector and constant linked to TCST */

  l1 = 0; b1 = 0; gcd1 = value_abs(val_of(in_v1));
  for (v1 = in_v1; v1 != NULL; v1 = v1->succ) {
    gcd1 = pgcd( gcd1, value_abs(val_of(v1)) );
    if(var_of(v1)==TCST) b1 = val_of(v1); 
    else l1++;
  }

  l2 = 0; b2 = 0; gcd2 = value_abs(val_of(in_v2));
  for (v2 = in_v2; v2 != NULL; v2 = v2->succ) {
    gcd2 = pgcd( gcd2, value_abs(val_of(v2)) );
    if(var_of(v2)==TCST) b2 = val_of(v2);
    else l2++;
  }

  if (l1 != l2)    return unpara ;

  
  /* Determine what kind of parallel hyperplane we are in */
  for (v2 = in_v2; v2 != NULL; v2 = v2->succ) {
    Variable  var2  = var_of(v2);
    Value     val2  = val_of(v2);
    boolean   found = FALSE;

    if (var2 == TCST) continue;

    for (v1 = in_v1; v1 != NULL; v1 = v1->succ) {
      if (var_of(v1) == var2) {
	Value      i1 = value_mult(gcd2,val_of(v1));
	Value     i2 = value_mult(gcd1,val2);
	boolean  ss = value_eq(i1,i2);
	boolean  op = value_eq(i1,value_uminus(i2));
	
	if (!ss && !op) return unpara;
	if (first) {first = FALSE; same_sign = (ss)?ss:op ;}
	if ((same_sign && op)||(!same_sign && ss)) return unpara; 
	found = TRUE;
      }
    }

    /* coefficient value was 0 and was not represented */
    if(!found) return unpara;  
  }
	

  /* compute return value */
  {
      Value p1 = value_mult(gcd1,b2),
            p2 = value_uminus(value_mult(gcd2,b1));
      diff = (same_sign)? value_plus(p1,p2): value_minus(p2,p1);
  }
  if      (value_zero_p(diff)) ret_sle = (same_sign) ? sszero  : opzero  ;
  else if (value_pos_p(diff)) ret_sle = (same_sign) ? ssplus  : opplus  ;
  else if (value_neg_p(diff)) ret_sle = (same_sign) ? ssminus : opminus ;
  else ret_sle = unpara;

  /* debuging */
  /*
  C3_DEBUG("vect_parallel", 
     { fprintf(stderr, "Output hspara: %s\n", hspara_to_string(ret_sle));  });
  */

  return ret_sle;
}
@| vect_parallel @}




\paragraph{contrainte\_parallel\_in\_liste} étend la fonction 
{\tt contrainte\_in\_liste} de C3. La contrainte {\tt in\_co} est
considérée comme une inégalité et est comparée à la
liste des inégalités {\tt in\_lc}. La valeur renvoyée,
élément du treillis des hyperplans parallèles, est la jointure de 
la contrainte entrée avec tous les éléments de la liste.


@D fonctions reduc @{
/* enum enum hspara_elem contrainte_parallel_in_liste( in_co, in_lc ) AL950711
 * input:      1 constraint in_co and a list of constraints in_lc 
 * output:     hspara_elem (element of the parallel half space lattice)
 * memory:     Inspector (nothing is shared, nor modified, output allocated).
 * complexity: length(in_lc) * comp(vect_parallel())
 * comment:    in_co represents a1 X+b1 <= 0 and in_lc aj X + bj <=0.
 *             Returns in_co/in_lc = join_j( vect_parallel( in_co, in_lc_j ) ) 
 *             between keep, empty and full. 
 */
enum hspara_elem contrainte_parallel_in_liste( in_co, in_lc )
Pcontrainte in_co, in_lc;
{
  Pcontrainte         c;
  Pvecteur            vpos;
  enum hspara_elem    ret_sle = keep;	

  assert(!CONTRAINTE_UNDEFINED_P(in_co));
  if (CONTRAINTE_NULLE_P(in_co)) return keep;
  
  /* debuging */
  C3_DEBUG("contrainte_parallel_in_list", {
    fprintf(stderr, "Input in_co:");  
    inegalite_fprint( stderr, in_co, union_variable_name ); 
    fprintf(stderr, "Input in_lc:\n"); 
    inegalites_fprint( stderr, in_lc, union_variable_name );
  });

  vpos = in_co->vecteur;
  
  for (c = in_lc; !CONTRAINTE_UNDEFINED_P(c) && (ret_sle != full); c=c->succ) {
    Pvecteur         cv   = c->vecteur;
    enum hspara_elem hs   = vect_parallel(vpos, cv);

    C3_DEBUG("contrainte_parallel_in_list", {
      fprintf(stderr, "ret_sle: %s ,  hs: %s\n", 
	      hspara_to_string(ret_sle),  
	      hspara_to_string( hs  )  ); 
    });
    
    ret_sle = hspara_join( ret_sle, hs);
  }


  /* debuging */
  C3_DEBUG("contrainte_parallel_in_list", 
    { fprintf(stderr, "Output hspara: %s\n", hspara_to_string(ret_sle)); });
  
  return ret_sle;
}
@| contrainte_parallel_in_list @}






\section{Suppression de contraintes redondantes} 

\paragraph{sc\_supress\_parallel\_constraints} supprime dans le second système
les contraintes parallèles-positives (la différence des deux vecteurs représentatifs
est positive) à celles contenues dans le premier. 

Les égalités de {\tt in\_ps2} sont transformées en
inégalités. Puis, pour chaque demi-espace $h1: x\geq0\;(-x\leq 0)$
de {\tt in\_ps2}, nous calculons $res1 = \mbox{\tt contrainte\_parallel\_in\_list}()$.

@D fonctions reduc @{
/* Psysteme sc_supress_parallel_redund_constraints( in_ps1, in_ps2 )
 * input:    2 Psystemes in_ps1 and in_ps2
 * output:   in_ps1 / in_ps2   (cut operation on polyhedrons)
 * memory:   Inspector (nothing is shared, nor modified, output allocated).
 * comment:  Supress in dup(in_ps2) parallel constraints that are redundant 
 *           relatively to in_ps1.
 *           Returned Psysteme have only inequalities. 
 */
Psysteme sc_supress_parallel_redund_constraints( in_ps1, in_ps2 )
Psysteme in_ps1, in_ps2;
{
  Psysteme        ps1, ps2,  ret_ps = NULL;
  Pcontrainte     ineq1, ineqs2;
  boolean         stop = FALSE, dup1 = FALSE, dup2 = FALSE;
  
  if ( in_ps1 == SC_RN ) return sc_dup(in_ps2);
  
  /* debuging */
  C3_DEBUG("sc_supress_parallel_constraints", {
    fprintf(stderr, "Input systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });
  

  /* Transforms equalities in inequalities if necessary */
  if (in_ps1->nb_eq != 0) 
    { ps1 = sc_dup( in_ps1 ); sc_transform_eg_in_ineg( ps1 ); dup1 = TRUE; }
  else ps1 = in_ps1;
 
  if (in_ps2->nb_eq != 0) 
    { ps2 = sc_dup( in_ps2 ); sc_transform_eg_in_ineg( ps2 ); dup2 = TRUE; }
  else ps2 = in_ps2;


  /* Compare with inequalities */
  ineqs2 = ps2->inegalites;

  for (ineq1 = ps1->inegalites; ineq1 != NULL && !stop; ineq1 = ineq1->succ) {
    enum hspara_elem  sk = contrainte_parallel_in_liste( ineq1, ineqs2 );
    switch (sk) 
      {
      case keep:
	if (ret_ps != NULL){ sc_add_inegalite( ret_ps, contrainte_dup(ineq1) ); }
	else ret_ps = sc_make( NULL, contrainte_dup(ineq1) );
	break;
      case empty:
	ret_ps = sc_free(ret_ps);
	ret_ps = sc_empty(NULL);
	stop = TRUE; 
	break;
      case full: continue; break;
      default:  
	{
	  fprintf(stderr, "%s supress_kind == %d should not appear !",
		  "[sc_supress_parallel_redund_constraints]", (int) sk ); 
	  abort();
	} 
      }
      
  }

  /* update base and normalize */  
  if ((ret_ps != NULL) && !sc_empty_p(ret_ps))  { 
    vect_rm(ret_ps->base); 
    ret_ps->base = NULL; sc_creer_base( ret_ps );
    ret_ps = sc_normalize( ret_ps );
  }

  /* Manage memory and return */
  ps1 = (dup1)? sc_free(ps1) : ps1;
  ps2 = (dup2)? sc_free(ps2) : ps2;
  C3_RETURN( IS_SC, ret_ps );
}
@| sc_supress_parallel_redund_constraints @}



\paragraph{sc\_supress\_same\_constraints} supprime dans le second
système {\tt in\_ps2} les contraintes contenues dans le premier
{\tt in\_ps1}. Applique donc la fonction {\tt in\_ps2} = 
{\tt in\_ps2 } $/$ {\tt in\_ps1}.  
@D fonctions reduc @{
/* Psysteme sc_supress_same_constraints( in_ps1, in_ps2 ) supress in 
 * in_ps2 constraints that are in in_ps1. Nothing is shared, nor modified.
 * Returned Psysteme have only inequalities.
 * This function should be superseded by sc_supress_parallel_redund_contraints
 */
Psysteme sc_supress_same_constraints( in_ps1, in_ps2 )
Psysteme in_ps1, in_ps2;
{
  Psysteme        ret_ps = NULL;
  Pcontrainte     eq, ineq;
  
  if ( in_ps1 == SC_RN ) return sc_dup(in_ps2);
  
  C3_DEBUG("sc_supress_same_constraints", {
    fprintf(stderr, "\nInput systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });

  /* Compare with equalities a == 0   <=>   a <= 0 and -a <= 0 */
  for (eq = in_ps2->egalites; eq != NULL; eq = eq->succ) {
    Pcontrainte  co, eq2;
    Pvecteur     pv;
    boolean      eq_in_ineq, co_in_ineq;

    if (contrainte_in_liste(eq, in_ps1->egalites)) continue;
    
    pv = vect_dup(eq->vecteur); 
    vect_chg_sgn        ( pv );
    co = contrainte_make( pv );
    if (contrainte_in_liste(co, in_ps1->egalites ))
      { co = contrainte_free( co ); continue; }


    eq_in_ineq = contrainte_in_liste(eq, in_ps1->inegalites);
    co_in_ineq = contrainte_in_liste(co, in_ps1->inegalites);
    
    if (eq_in_ineq && co_in_ineq) { 
      co = contrainte_free( co ); 
    }
    else if (eq_in_ineq) { /* add co to returned inegs */
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, co ); }
      else ret_ps = sc_make( NULL, co );
    }
    else if (co_in_ineq) { /* add eq to returned inegs */
      eq2 = contrainte_dup(eq);
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, eq2 ); }
      else ret_ps = sc_make( NULL, eq2 );
      co = contrainte_free( co ); 
    }
    else { /* add co and eq to returned inegs */
      eq2 = contrainte_dup(eq);
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, eq2 ); }
      else ret_ps = sc_make( NULL, eq2 );
      sc_add_inegalite( ret_ps, co );
    }
  }

  /* Compare with inequalities */
  for (ineq = in_ps2->inegalites; ineq != NULL; ineq = ineq->succ) {
    Pcontrainte io;
    if (contrainte_in_liste(ineq, in_ps1->inegalites)) continue;
    if (contrainte_in_liste(ineq, in_ps1->egalites))   continue;
    io = contrainte_dup( ineq ); contrainte_chg_sgn( io );
    if (contrainte_in_liste(io, in_ps1->egalites)) {
      io = contrainte_free(io);  
      continue;
    }
    
    if (ret_ps != NULL){ sc_add_inegalite( ret_ps, contrainte_dup(ineq) ); }
    else ret_ps = sc_make( NULL, contrainte_dup(ineq) );
    io = contrainte_free(io);  
  }
  
  if (ret_ps != NULL) 
    { vect_rm(ret_ps->base); ret_ps->base = NULL; sc_creer_base( ret_ps );}
  
  ret_ps = sc_normalize( ret_ps );
  C3_RETURN( IS_SC, ret_ps );
}
@| sc_supress_same_constraints @}



\paragraph{sc\_elim\_redund\_with\_first} prend {\tt in\_ps1} et {\tt
in\_ps2} en entré et renvoie le Psysteme {\tt in\_ps2} modifié :
il ne garde que les contraintes qui coupent {\tt in\_ps1}. Pour cela,
on construit le Psysteme {\tt ps1} constitué d'une copie de {\tt
in\_ps1} et des contraintes de {\tt in\_ps2}, qui sont donc partagées.
On procède ensuite de façon classique pour éliminer les contraintes
redondantes dans ce nouveau système : inversion des contraintes de
{\tt in\_ps2} et test de faisabilité. 

Cette fonction ne traitant pas des égalités, plus de contraintes
redondantes pourraient être détectées si les égalités
des deux systèmes d'entré sont transformées en
inégalités. Cependant, le grand nombre de contraintes
générées peut entrainer un allongement des calculs de faisabilité. 

@D fonctions reduc @{
/* Psysteme sc_elim_redund_with_first_ofl_ctrl( in_ps1, in_ps2, ofl_ctrl )	
 * Returns constraints of in_ps2 which cut in_ps1. AL 06 04 95
 * It is assumed that in_ps1 and in_ps2 are feasible !
 * in_ps1 is not modified, in_ps2 is modified.
 */
Psysteme sc_elim_redund_with_first_ofl_ctrl(in_ps1, in_ps2, ofl_ctrl)
Psysteme in_ps1, in_ps2;
int      ofl_ctrl;
{
  Psysteme    ps1; 
  Pcontrainte prev_eq = NULL, eq, tail = NULL;
  Pbase       pb;

  /* Return on special cases */
  if ( sc_full_p(in_ps1) )    return in_ps2;
  if ( in_ps1->nb_ineq == 0 ) return in_ps2;

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "\nInput systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });


  /* build in_ps1.and.in_ps2 with sharing on in_ps2
   * This also works if in_ps1 is full space */
  if ( in_ps2->nb_eq != 0 ) sc_transform_eg_in_ineg( in_ps2 );
  ps1 = sc_dup( in_ps1 );
  for (eq = ps1->inegalites; eq != NULL; tail = eq, eq = eq->succ) {}
  tail->succ = in_ps2->inegalites;

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "ps1 old: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
	    ps1->nb_eq, ps1->nb_ineq, ps1->dimension);  
    vect_fprint(stderr, ps1->base, union_variable_name);
    fprintf(stderr, "in_ps2: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
	    in_ps2->nb_eq, in_ps2->nb_ineq, in_ps2->dimension);  
    vect_fprint(stderr, in_ps2->base, union_variable_name);
  });

  /* update information on ps1 */
  ps1->nb_eq     = ps1->nb_eq   + in_ps2->nb_eq;
  ps1->nb_ineq   = ps1->nb_ineq + in_ps2->nb_ineq;
  pb             = ps1->base;
  ps1->base      = base_union( ps1->base, in_ps2->base );
  ps1->dimension = vect_size ( ps1->base );
  vect_rm( pb );

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "ps1: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
	    ps1->nb_eq, ps1->nb_ineq, ps1->dimension);
    vect_fprint(stderr, ps1->base, union_variable_name);
  });

  /* Normalize 2 inputs systems */
  for (eq = ps1->inegalites; eq != NULL; eq=eq->succ)
  {
      vect_normalize(eq->vecteur);
  }
  /* returns if there is no intersection */
  if (!sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, TRUE)) { 
    tail->succ = NULL;  ps1 = sc_free(ps1); 
    in_ps2 = sc_free(in_ps2); in_ps2 = sc_empty(NULL);
    C3_RETURN( IS_SC, in_ps2 ); 
  }
    

  /* We run over in_ps2 constraints (shared by ps1) 
   * and detect redundance */
  assert(sc_weak_consistent_p(in_ps2));
  assert(sc_weak_consistent_p(ps1));
  for (eq = tail->succ, prev_eq = tail; eq != NULL; eq = eq->succ)
  {
      contrainte_reverse(eq);	
      assert(sc_weak_consistent_p(ps1));
      C3_DEBUG("sc_elim_redund_with_first", {
	  fprintf(stderr, "\nps1:\n");  
	  fprintf(stderr, "nb_eq= %d, nb_ineq= %d, dimension= %d\n", 
		  ps1->nb_eq, ps1->nb_ineq, ps1->dimension);
	  sc_fprint( stderr, ps1, union_variable_name );
      });

      if (sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, TRUE))
      {
	  contrainte_reverse(eq);
	  prev_eq = prev_eq->succ;
      }
      else{  
	  /* eliminate the constraint from in_ps2, and thus from ps1 */	
	  eq_set_vect_nul(eq);	
	  if (in_ps2->inegalites == eq)
	      in_ps2->inegalites = eq->succ;
	  prev_eq->succ = eq->succ;
	  eq->succ = CONTRAINTE_UNDEFINED;
	  eq = contrainte_free(eq);
	  eq = prev_eq;
	  in_ps2->nb_ineq--;
	  ps1->nb_ineq--;
	  assert(sc_weak_consistent_p(ps1));
	  assert(sc_weak_consistent_p(in_ps2));
      }
  }


  if ( in_ps2->inegalites == NULL ) 
    { in_ps2 = sc_free(in_ps2);  in_ps2 = sc_full(); }

  tail->succ = NULL; ps1 = sc_free( ps1 ); 
  C3_RETURN( IS_SC, in_ps2 );
}
@| sc_elim_redund_with_first_ofl_ctrl @}



\section{Réduction des disjonctions engendrées par un chemin}

\subsection{Motivation}
Un chemin représente de façon efficace l'information qui gouverne
le n{\oe}ud d'un arbre. Cette information doit parfois être
transformée sous forme de disjonction. C'est le cas pour le Data
Flow Graph (DFG) de Feautrier.
La transformation d'un chemin sous forme de disjonction
peut, si elle est faite de façon naïve, produire un très grand nombre
de disjonctions, notamment si le nombre de compléments dans le chemin
est grand. Ce problème se pose, entre autres, dans le calcul
des effets résumés exacts d'une procédure sur des tableaux.

Prenons l'exemple de lczos.f, dont le code est présenté à
l'annexe~\ref{lczos-annexe}, et cherchons quels sont
les éléments du tableau GAMMA qui sont écrit par le programme.
On cherche donc la source d'une instruction finale fictive
qui lirait tout les éléments du tableau GAMMA.
Les éléments qui ne sont pas écrit dans le programme
sont écrit par un n{\oe}ud d'entrée fictif.

L'arbre figure \ref{gammatree} représente la provenance
de l'élément GAMMA( i1, i2 ). Si cet élément
n'est pas écrit par le programme, il provient du n{\oe}ud
d'entré "Entry Node". Celui ci est gouverné par 4 compléments
de systèmes. La génération de la disjonction gouvernant
ce n{\oe}ud par des méthodes naïves produit
$2 \times 3 \times 3 \times 4 = 72$
disjonctions, dont on montre qu'elles peuvent
se réduire à 2 disjonctions indépendantes grâce aux
quelques règles de logique ci-dessous exposées.
Notons que ces règles portent sur des hyperplans communs au
système principal ou aux complémentaires, qui sont nombreux dans
notre exemple, d'où le fort taux de réduction. Dans des cas plus
généraux, comme par exemple pour la différence de deux
polyèdres, nous appliquerons une autre façon de générer
une disjonction équivalente.

\begin{figure}
\[
\Ifthenelse{\left\{      \begin{array}{c}
                        n \geq 1  \\
                        i1 \leq n \\
                        2 \leq i2 \leq m-1  
                \end{array} \right.}
        {operation 1}
	{\Ifthenelse{\left\{ \begin{array}{c}
                        i1 \leq n \\
                        2 \leq i2 \leq m-1 
                \end{array} \right.}
                {operation 2}
                {\Ifthenelse{\left\{ \begin{array}{c}
                        n \geq  1 \\
                        i1 \leq n \\
                        i2 \leq 1
                           \end{array} \right.}
                        {operation 3}
                        {\Ifthenelse{\left\{ \begin{array}{c}
                                i1 \leq n\\
                                i2 \leq 1
                                \end{array} \right.}
                                {operation 4}
                                {Entry Node}
                        }
                }
        }
\]
\caption{Arbre donnant la fonction source pour GAMMA}
\label{gammatree}
\end{figure}

\subsection{Règles de réduction}
Voici quelques règles de logique, démontrées à
l'annexe~\ref{demo-annexe}, qui permettent
de réduire fortement le nombre des disjonctions si le chemin a un
grand nombre d'hyperplans en commun.

Soit \( {C} = P_{0} \bigwedge_{i=1}^{n} \neg P_{i} \)
un chemin.

\paragraph{Règle 1 :}
\[ \left\{ \begin{array}{c}
        {P}_{0} = {P'}_{0} \wedge h \\
        \exists i_0 \in [1,n] {P}_{i_0} = {P'}_{i_0} \wedge h
           \end{array}
    \right.
        \Rightarrow
                {C} = {P}_{0} \wedge\neg {P'}_{i_0}
                           \bigwedge_{i \neq i_0} \neg {P}_{i}
\]
On peut donc enlever dans les compléments tout les hyperplans
qui délimitent ${P_0}$.
Remarque~: s'il existe un complémentaire composé uniquement
de l'hyperplan $h$, alors ${C}$ est vide.


\paragraph{Règle 2 :}
\[ \exists h, \forall i \in [1,n] {P}_{i} = {P'}_{i} \wedge h
        \Rightarrow
                {C} =({P}_{0}  \bigwedge_{i=1}^{n}\neg {P'}_{i})
                           \vee ({P}_{0} \wedge \neg h)
\]
On peut donc enlever dans les compléments tout les hyperplans communs.
Remarque~: s'il existe un complémentaire composé uniquement
de l'hyperplan commun $h$, alors ${C} = {P}_0 \wedge \neg h$.

\paragraph{Règle 3 :}
\begin{eqnarray*}
\exists h, \exists n' < n,
        \left\{ \begin{array}{c}
                \forall i \in [1,n'] \; {P}_{i} = {P'}_{i} \wedge h \\
                \forall i \in [n'+1,n] \; {P}_{i} ={P'}_{i}\wedge\neg h\\
                \end{array}
        \right. && \\
\Rightarrow {C} =
   [({P}_{0}\wedge h) \bigwedge_{i=1}^{n'} \neg {P'}_{i}] &
   \vee\; [({P}_{0}\wedge\neg h) \bigwedge_{i=n'+1}^{n} \neg{P'}_{i}]&
\end{eqnarray*}
Remarque~: s'il existe un complémentaire composé uniquement
de l'hyperplan $h$, alors
${C} = ({P}_{0} \wedge \neg h) \bigwedge_{n'+1}^{n}\neg{P'}_i$.

\paragraph{Règle 4 :}
\[ {C} = {P}_{0} \bigwedge_{i=1}^{n} \neg ({P}_0 \wedge {P}_{i}) \]
Cette règle sera utilisée pour éviter de générer des
disjonction redondantes. La figure~\ref{path2-fig} nous donne un
exemple o\`u les règles 1, 2 et 3 ne peuvent pas s'appliquer et o\`u une
génération naïve de la disjonction équivalentes produit 5
disjonctions, dont 3 sont redondantes (celles introduites par les
hyperplans H3, H4 et H5).

\begin{figure}
\centerline{\epsf{path2.eps}[xscale=2/3,yscale=2/3]}
\caption{Application de la règle 4}
\label{path2-fig}
\end{figure}



\subsection{Algorithme appliquant la règle 1} 
On parcourt chaque complémentaire {\tt comp} du chemin entré
pour ne garder de ses contraintes que celles qui ne sont pas contenues
dans $P_0$.

@D fonctions reduc @{
/* Ppath pa_supress_same_constraints( (Ppath) in_pa )   
 * Supress from complements of in_pa same constraints than those in
 * positif Psystem in_pa->psys. Returned path have no more equalities. AL050795
 * No sharing, no modification of inputs.
 */
Ppath pa_supress_same_constraints( in_pa )
Ppath in_pa;
{
  Ppath        ret_pa = PA_UNDEFINED;
  Pcomplist    comp;
  Psysteme     positif;
  Psyslist     psl = NULL;

  /* Special cases */
  if ( PA_UNDEFINED_P( in_pa )) return PA_UNDEFINED;
  if ( pa_empty_p    ( in_pa )) return pa_empty();
  if ( pa_full_p     ( in_pa )) return pa_full ();

  /* debuging */
  C3_DEBUG( "pa_supress_same_constraints", {
    fprintf(stderr, "Input path:\n");
    pa_fprint_tab(stderr, in_pa, union_variable_name, 1);
  });

  /* General case */
  positif = in_pa->psys;
  if (!sc_faisabilite_ofl(positif)) return pa_empty();
  
  for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ) {
 /*   Psysteme ps = sc_supress_same_constraints( positif, comp->psys ); */
    Psysteme ps = sc_supress_parallel_redund_constraints( comp->psys, positif );
    if (ps == NULL) 
      {psl = sl_free(psl); ret_pa = pa_empty(); C3_RETURN(IS_PA, ret_pa);}
    else psl = sl_append_system( psl, ps );
  }

  positif = sc_dup(positif); sc_transform_eg_in_ineg( positif );
  ret_pa  = pa_make( positif, (Pcomplist) psl );
  C3_RETURN(IS_PA, ret_pa);
}
@| pa_supress_same_constraints @}


\subsection{Algorithme appliquant la règle 4} 
On parcourt chaque complémentaire {\tt comp} du chemin entré
pour ne garder de ses contraintes que celles qui coupent $P_0$.
L'application de cette règle rend caduque l'application de la
règle 1 puisque les hyperplans des complémentaires que l'on
retrouve dans $P_0$ sont redondants dans $P_0 \wedge P_i$. 
Un nouveau chemin avec la liste {\tt lcomp} de complémentaires 
modifiés est générée.
@D fonctions reduc @{
/* Pdisjunct pa_path_to_disjunct_rule4_ofl_ctrl( (Ppath) in_pa, int ofl_ctrl)   
 * Returns the corresponding disjunction according rule 4. AL 05/16/95
 * No sharing.
 */
Pdisjunct pa_path_to_disjunct_rule4_ofl_ctrl( in_pa, ofl_ctrl )
Ppath in_pa;
int   ofl_ctrl;
{
  Pcomplist   comp, lcomp = NULL;
  Pdisjunct   ret_dj  ; 
  Psysteme    systeme ; 
  Ppath       pa      ; 
  int         pa_clength1, pa_clength2; 

  if (in_pa == PA_UNDEFINED) return DJ_UNDEFINED;
  if (pa_empty_p(in_pa))     return dj_empty();

  C3_DEBUG( "pa_path_to_disjunct_rule4_ofl_ctrl", {
    fprintf(stderr, "\n\n Input path:\n\n");
    pa_fprint(stderr, in_pa, union_variable_name );
  });


  if (pa_max_constraints_nb(in_pa) > PATH_MAX_CONSTRAINTS) 
    C3_RETURN(IS_DJ, pa_path_to_disjunct_ofl_ctrl( in_pa, ofl_ctrl));
    
  systeme = in_pa->psys;
  if (in_pa->pcomp == NULL) 
    C3_RETURN(IS_DJ, sl_append_system(NULL,sc_dup(systeme)));

  for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ ) {
    Psysteme ps;
    if (comp->psys == SC_UNDEFINED) 
      { sl_free(lcomp); C3_RETURN( IS_DJ, DJ_UNDEFINED ); }

    ps = sc_dup(comp->psys);

    ps = sc_elim_redund_with_first_ofl_ctrl( systeme, ps, ofl_ctrl );

    if (sc_empty_p( ps )) { ps = sc_free(ps); continue; }
    if (sc_full_p ( ps ))  
      { ps = sc_free(ps); C3_RETURN( IS_DJ, dj_empty() ); }

    lcomp = sl_append_system( lcomp, ps );
  }
 
  pa          = pa_make(sc_dup(in_pa->psys), lcomp); 
  pa_clength1 = sl_length( pa->pcomp );
  pa          = pa_reduce_simple_complement( pa );
  pa_clength2 = sl_length( pa->pcomp );
  systeme     = pa->psys;


  /* Returns according to different cases */
  if (pa_empty_p(pa)) 
       { ret_dj = dj_empty(); } 
  else if (pa_clength2 == 0) 
       { ret_dj = dj_append_system(NULL,sc_dup(systeme)); } 
  else if (pa_clength1 != pa_clength2)  /* we've modified P0 systeme */
       { ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl); } 
  else { ret_dj = pa_path_to_disjunct_ofl_ctrl( pa, ofl_ctrl); }

  pa = pa_free( pa );

  C3_RETURN( IS_DJ, ret_dj );
}
@| pa_path_to_disjunct_rule4_ofl_ctrl @}



\subsection{Algorithme complet}
\paragraph{}
Nous utilisons ces règles dans l'algorithme~\ref{reduc-algo} de réduction des
disjonctions d'un chemin.

\begin{figure}
\begin{algorithm}{Réduction des disjonctions d'un chemin} \label{reduc-algo}
1- Soit ${D}$ la disjonction équivalente à ${C}$.
Mettre dans ${D}$ la disjonction vide. \\
2- R1 : enlever des complémentaires les hyperplans communs à $P_0$.\\
3- R3 : Parcourir les complémentaires restant et en extraire les hyperplans
dont $h$ est commun à une partie des complémentaires, $\neg h$ étant
commun au reste des complémentaires. Consituer deux chemins et 
répéter l'algorithme à l'étape 3 avec ces deux chemins.\\
4- R4 : Enlever des complémentaires les hyperplans qui ne coupent pas ${P}_{0}$.\\
5- Générer de façon naïve les disjonctions restantes.
\end{algorithm}
\label{reduc-fig}
\caption{Algorithme de réduction des disjonctions d'un chemin}
\end{figure}


\paragraph{Détail de la fonction}
Les variables sont d'abord initialisées, et l'on retourne
si le chemin entré est infaisable ou s'il n'a pas de compléments.
@D variables @{ Psysteme  systeme;  Pdisjunct ret_dj = DJ_UNDEFINED; @}
@D initialisation @{
C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
  fprintf(stderr, "\n\n Input path:\n\n");
  pa_fprint(stderr, in_pa, union_variable_name );
});

if (PA_UNDEFINED_P( in_pa ))   C3_RETURN(IS_DJ, DJ_UNDEFINED);
if (pa_full_p     ( in_pa ))   C3_RETURN(IS_DJ, dj_full());
if (pa_empty_p    ( in_pa ))   C3_RETURN(IS_DJ, dj_empty());
  
/* If it's an empty path or if it has no complements : return */ 
systeme = in_pa->psys ; 
if (!sc_faisabilite_ofl( systeme )) C3_RETURN(IS_DJ,dj_empty());
if (in_pa->pcomp == NULL) C3_RETURN(IS_DJ,(Pdisjunct) sl_append_system(NULL,sc_dup(systeme)));
@}

\paragraph{Traitement des complémentaires simples.}
Première chose à faire : réduire le chemin s'il existe des
complémentaires n'ayant qu'une seule contrainte. Le chemin réduit sera
{\tt pa} et la nouvelle liste de complémentaires {\tt lcomp}.
@D variables @{	Ppath pa; Pcomplist lcomp;@}
@D reduction des complementaires simples @{
pa      = pa_make(sc_dup(systeme), sl_dup(in_pa->pcomp)); 
pa      = pa_reduce_simple_complement( pa );

if (pa_empty_p(pa)) {pa = pa_free(pa); C3_RETURN(IS_DJ,dj_empty());}

pa      = pa_transform_eg_in_ineg    ( pa ); 
lcomp   = pa->pcomp ; 
systeme = pa->psys  ;

C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
  fprintf(stderr, "pa:\n");
  pa_fprint_tab(stderr, pa, union_variable_name, 1 );
});

if ( pa->pcomp == NULL ) { 
  pa     = pa_free1(pa); 
  ret_dj = (Pdisjunct) sl_append_system(NULL, systeme);
  
  C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
    fprintf(stderr, "No complement, returning:\n");
    dj_fprint_tab(stderr, ret_dj, union_variable_name, 1 );
  });
  
  return ret_dj;
}
@}

\paragraph{Recherche de complémentaires communs.}
Toute contrainte dans {\tt common\_cons} se retrouve dans 
tous les complémentaires ; soit telle quelle, soit opposée.
Cette variable est initialisée par les contraintes
du premier complémentaire. 

On cherche le premier élément de cette liste de contraintes qui soit
commun à tous les complémentaires. Cette contrainte commune sera
{\tt common\_cons} et son vecteur associé {\tt cons\_pv}.

@D variables @{	
Pcontrainte common_cons = NULL, cons, cons_oppose = NULL; 
Pvecteur    vect_1, cons_pv = NULL;
Pcomplist   comp;
@}
@D contrainte commune @{
/* We are looking for a common hyperplan */
vect_1 = vect_new(TCST, VALUE_ONE); common_cons = NULL;

for(cons = (lcomp->psys)->inegalites;
	(cons != NULL)&&(lcomp->succ != NULL);cons = cons->succ){
  boolean is_common = TRUE;
  cons_pv           = vect_dup( cons->vecteur ); vect_chg_sgn( cons_pv );
  cons_oppose       = contrainte_make(vect_add( cons_pv, vect_1 )); 

  for(comp = lcomp->succ;(comp != NULL) && is_common; comp = comp->succ){
    Pcontrainte ineg = (comp->psys)->inegalites;
    boolean     is_common1, is_common2;

    is_common1 = contrainte_in_liste( cons,        ineg );
    is_common2 = contrainte_in_liste( cons_oppose, ineg );
    is_common  = is_common1 || is_common2;
  }
  if (!is_common) { 
    /* removes cons_pv and vect_dup(vect_1) */
    cons_oppose = contrainte_free(cons_oppose);
    vect_rm( cons_pv ); cons_pv = (Pvecteur) NULL;
    continue; 
  }
  common_cons = cons;
  vect_chg_sgn( cons_pv );
  break;
} 

C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
  fprintf(stderr, "cons_pv: ");
  if (common_cons == NULL) fprintf(stderr, "NULL\n"); 
  else vect_fprint(stderr, cons_pv, union_variable_name); 
});
@}


\paragraph{Application des la règle 3 et 4.}
S'il existe un hyperplan commun, on applique récursivement la
règle 3. Sinon, il ne peut pas plus avoir d'hyperplan commun et
c'est la règle 4 qui est appelée récursivement. 
@D regle 3 et 4 @{
if( common_cons != NULL ) {
  @< regle 3 construction des chemins @>
  @< regle 3 appel recursif @>
}
else { 
  @< regle 4 @>
}

/* Manage memory */
pa = pa_free(pa); vect_rm(vect_1); vect_1 = NULL;

C3_RETURN(IS_DJ, ret_dj);
@}

\paragraph{Constructions des chemins de la règle 3.}
On prend la première contrainte commune {\tt common\_cons}
et l'on génère, suivant la règle 3, deux nouveaux chemins :
{\tt pa1} et {\tt pa2}. Pour prendre en compte la remarque de 
la règle 3, les chemins dont l'un des complémentaires 
est égal à l'hyperplan commun est mis à vide.

{\tt common\_cons\_oppose} est la contrainte opposée à {\tt
common\_cons} et {\tt common\_ps\_oppose} le Psysteme constitué de
cette contrainte opposée.
@D variables @{	
Pcontrainte	common_cons_oppose;
Psysteme 	common_ps, common_ps_oppose;
Ppath		pa1, pa2; 
boolean		pa1_empty  = FALSE, pa2_empty  = FALSE;
boolean		pa1_filled = FALSE, pa2_filled = FALSE;
@}
@D regle 3 construction des chemins @{
common_ps          = sc_make( CONTRAINTE_UNDEFINED, contrainte_make(cons_pv) );
cons_pv            = vect_dup( common_cons->vecteur ); vect_chg_sgn( cons_pv );
common_cons_oppose = contrainte_make(vect_add(cons_pv,vect_1));
common_ps_oppose   = sc_make( CONTRAINTE_UNDEFINED, common_cons_oppose );
pa1 = pa_new(); pa2= pa_new();

for(comp = lcomp; comp != NULL; comp = comp->succ){
  Psysteme     local_ps;
  Pcontrainte  co = comp->psys->inegalites;
  
  if (!pa1_empty && contrainte_in_liste(common_cons, co)) {
    local_ps = sc_supress_same_constraints( common_ps, comp->psys );
    if (local_ps == SC_EMPTY) { pa1 = pa_empty(); pa1_empty = TRUE; continue;}
    pa1->pcomp = sl_append_system( pa1->pcomp, local_ps ); pa1_filled = TRUE;
  }
  else if(!pa2_empty &&  contrainte_in_liste(common_cons_oppose, co)) {
    local_ps = sc_supress_same_constraints( common_ps_oppose, comp->psys );
    if (local_ps == SC_EMPTY) {pa2 = pa_empty(); pa2_empty = TRUE; continue;}
    pa2->pcomp = sl_append_system( pa2->pcomp, local_ps ); pa2_filled = TRUE;
  }
}
@}

On  rappelle la fonction pour réduire à nouveaux les deux chemins et l'on
construit l'union des disjonctions {\tt dj1} et {\tt dj2} résultantes.
@d variables @{	Pdisjunct  dj1 = dj_empty(), dj2 = dj_empty(); @}
@D regle 3 appel recursif @{
if (pa1_filled) {
  /* take care of rule 2 */
  if (pa_full_p( pa2 )) pa1->psys = sc_dup( systeme );
  else pa1->psys = sc_safe_append( sc_dup(common_ps), systeme );

  C3_DEBUG("pa_path_to_few_disjunct", {
    fprintf(stderr, "pa1:\n");  
    pa_fprint_tab( stderr, pa1, union_variable_name, 1 );
  });

  if (pa_full_p(pa2)||sc_faisabilite_ofl(pa1->psys)) 
	{dj_free(dj1);dj1 = pa_path_to_few_disjunct_ofl_ctrl(pa1, ofl_ctrl);}

}

if (pa2_filled) {
  /* take care of rule 2 */
  if (pa_full_p( pa1 )) pa2->psys = sc_dup( systeme );
  else pa2->psys = sc_safe_append( sc_dup(common_ps_oppose), systeme );

  C3_DEBUG("pa_path_to_few_disjunct", {
    fprintf(stderr, "pa2:\n");  
    pa_fprint_tab( stderr, pa2, union_variable_name, 1 );
  });
  if (pa_full_p(pa1)||sc_faisabilite_ofl(pa2->psys)) 
	{dj_free(dj2);dj2 = pa_path_to_few_disjunct_ofl_ctrl(pa2, ofl_ctrl);}


}

ret_dj = dj_union( dj1, dj2 ); 

/* Manage memory, free:
 * cons_oppose, common_ps, common_ps_oppose, 
 * cons_pv, vect_1, pa1, pa2
 */
cons_oppose      = contrainte_free( cons_oppose );
common_ps        = sc_free( common_ps );
common_ps_oppose = sc_free( common_ps_oppose );
vect_rm(cons_pv); cons_pv = NULL;
pa1 = pa_free(pa1);   pa2 = pa_free(pa2); 
@}

\paragraph{Règle 4.} Il s'agit juste d'appliquer la fonction qui
implante cette règle.
@D regle 4 @{ ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl ); 
@}


\paragraph{Commentaire et génération de la fonction.}
On rajoute le commentaire pour les programmeurs :
@D commentaire reduction chemin @{
Pdisjunct    pa_path_to_few_disjunct_ofl_ctrl( (Ppath) in_pa, (int) ofl_ctrl )  
Produces a Pdisjunct corresponding to the path Ppath and
reduces the number of disjunctions.
See "Extension de C3 aux Unions de Polyedres" Version 2,
for a complete explanation about this function.
in_pa is modified.		AL 23/03/95
@}

Et l'on a pour finir la fonction elle même :
@D fonctions reduc @{
/* @< commentaire reduction chemin @> 
*/
Pdisjunct pa_path_to_few_disjunct_ofl_ctrl( in_pa, ofl_ctrl )
Ppath   in_pa;
int     ofl_ctrl;
{
  @< variables @>

  @< initialisation @>
  @< reduction des complementaires simples @>
  @< contrainte commune @>
  @< regle 3 et 4 @>
}
@| pa_path_to_few_disjunct_ofl_ctrl @}





\section{Égalité, différence et inclusion de polyèdres}
La notion de chemin peut être utilisée dans un autre contexte que
celui des arbres de décisions.
Une bonne partie de ces fonctions ont été écrites par Béatrice
Apvrille. Qu'elle en soit ici remerciée.

\paragraph{Inclusion et égalité.} 
Montrons qu'un polyèdre ${P}_{1}$ est inclus dans
le polyèdre ${P}_{2}$. Nous avons, en confondant
les notions d'ensemble et de prédicat qui les définissent :
\begin{eqnarray*}
{P}_{1} \subset {P}_{2} & \Leftrightarrow &
                {P}_{1} \Rightarrow {P}_{2} \\
        & \Leftrightarrow & \neg {P}_{1} \vee {P}_{2} \\
        & \Leftrightarrow & \neg ({P}_{1} \wedge \neg {P}_{2}) \\
\end{eqnarray*}
Donc ${P}_{1} \subset {P}_{2}$ si et seulement si
le chemin ${C} = {P}_{1} \wedge \neg {P}_{2}$ est
infaisable.
La fonction pa\_faisabilite est donc utilisée dans ce cas.

{\bf pa\_inclusion\_p} teste si un système est inclus dans un autre.
@D fonctions reduc @{
/* boolean pa_inclusion_p(Psysteme ps1, Psysteme ps2)	BA, AL 31/05/94
 * returns TRUE if ps1 represents a subset of ps2, false otherwise
 * Inspector (no sharing on memory).
 */
boolean pa_inclusion_p_ofl_ctrl(ps1, ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
  boolean   result;
  Ppath     chemin = pa_make(ps1, sl_append_system(NULL, ps2));
  
  CATCH(overflow_error) {
    result = FALSE; 
  }
  TRY {
    result = ! (pa_feasibility_ofl_ctrl(chemin, ofl_ctrl));
    UNCATCH(overflow_error);
  }
  chemin = pa_free1(chemin); 
  return(result);
}
@| pa_inclusion_p_ofl_ctrl @}

{\bf pa\_system\_equal\_p\_ofl\_ctrl} déclare égaux deux systèmes si
chacun est inclus dans l'autre.
@D fonctions reduc @{
/* boolean pa_system_equal_p(Psysteme ps1, Psysteme ps2) BA
 */
boolean pa_system_equal_p_ofl_ctrl(ps1,ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
    return (  pa_inclusion_p_ofl_ctrl(ps1,ps2, ofl_ctrl) && 
	      pa_inclusion_p_ofl_ctrl(ps2,ps1, ofl_ctrl) );
}
@| pa_system_equal_p_ofl_ctrl @}

\paragraph{Différence de deux systèmes.} C'est une disjonction
équivalente à l'intersection du premier système avec le
complémentaire du deuxième.
@D fonctions reduc  @{
/* Pdisjunct pa_system_difference_ofl_ctrl(ps1, ps2)
 * input    : two Psystemes
 * output   : a disjunction representing ps1 - ps2
 * modifies : nothing
 * comment  : algorihtm : 
 * 	chemin = ps1 inter complement of (ps2)
 * 	ret_dj = dj_simple_inegs_to_eg( pa_path_to_few_disjunct(chemin) )
 */
Pdisjunct pa_system_difference_ofl_ctrl(ps1, ps2, ofl_ctrl)
Psysteme ps1,ps2;
int      ofl_ctrl;
{
  Ppath     chemin;
  Pdisjunct dj, ret_dj;
  
  if ((ps1 == SC_UNDEFINED)||(ps2 == SC_UNDEFINED)) return DJ_UNDEFINED; 
  if (sc_empty_p(ps2)) return sl_append_system(NULL,sc_dup(ps1));
  if (sc_empty_p(ps1)) return dj_empty();
  
  chemin  =  pa_make(ps1, sl_append_system(NULL,ps2));
  dj      =  pa_path_to_few_disjunct_ofl_ctrl(chemin, ofl_ctrl);
  chemin  =  pa_free1( chemin );
  ret_dj  =  dj_simple_inegs_to_eg( dj );
  dj      =  dj_free( dj );
  return ret_dj;
}
@| pa_system_difference_ofl_ctrl @}

\paragraph{Égalité d'une enveloppe convexe et de l'union de deux polyèdres.}
pa\_convex\_hull\_equals\_union\_p\_ofl\_ctrl est vrai si l'enveloppe
convexe des deux systèmes d'entrée est égale à l'union de ses
mêmes deux systèmes.
@D fonctions reduc @{
/* boolean pa_convex_hull_equals_union_p(conv_hull, ps1, ps2)
 * input    : two Psystems and their convex hull	AL,BC 23/03/95
 * output   : TRUE if ps1 U ps2 = convex_hull, FALSE otherwise
 * modifies : nothing
 * comment  : complexity = nb_constraints(ps1) * nb_constraints(ps2)	
 *            if ofl_ctrl = OFL_CTRL, conservatively returns ofl_ctrl 
 *            when an overflow error occurs
 */
boolean pa_convex_hull_equals_union_p_ofl_ctrl
            (conv_hull, ps1, ps2, ofl_ctrl, ofl_res)
Psysteme  conv_hull, ps1, ps2;
int       ofl_ctrl;
boolean   ofl_res;
{
  Ppath    chemin;
  boolean  result;
  int      local_ofl_ctrl = (ofl_ctrl == OFL_CTRL)?FWD_OFL_CTRL:ofl_ctrl;
  
  chemin = pa_make(conv_hull,sl_append_system(sl_append_system(NULL,ps1),ps2));
  
  if (ofl_ctrl==OFL_CTRL) {
	CATCH(overflow_error) {
            result = ofl_res;
        }
        TRY {
            result = !(pa_feasibility_ofl_ctrl(chemin, local_ofl_ctrl));
            UNCATCH(overflow_error);
        }
  }
  else
      result = !(pa_feasibility_ofl_ctrl(chemin, local_ofl_ctrl));

  chemin = pa_free1(chemin);
  return(result);
}
@| pa_convex_hull_equals_union_p_ofl_ctrl @}





Ces fonctions sont dans le fichier :
@O reduc.c  -d @{
@< includes @>
@< global reduc    @>
@< fonctions reduc @>
@}

