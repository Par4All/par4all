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


%\paragraph{Règle 1.} On enlève des complémentaires les
%contraintes similaires à celles du système $\cal P_0$ : {\tt systeme}.
%Une nouvelle liste de complémentaires est alors créée : {\tt lcomp}, 
%qui sera désormais la liste des complémentaires du chemin {\tt in\_pa}.
%Bien sûr, on retourne si l'on a qu'un seul complément, sinon on
%continue d'examiner les autres compléments.
@D titi @{	 Pcomplist	comp, lcomp; @}
@D tutu @{
lcomp = NULL;
for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ ) {
  Psysteme ps;	
  ps = sc_supress_same_constraints( comp->psys, systeme );
  if ( ps == SC_UNDEFINED ) return( dj_empty() );
  lcomp = sl_append_system( lcomp, ps );
}
in_pa->pcomp = lcomp;

/* If there is only one complement, returns a naive way */
if (lcomp->succ == NULL)  return( pa_path_to_disjunct_ofl_ctrl(in_pa,ofl_ctrl) );
@}


\paragraph{Le treillis des signes}
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

\begin{figure}
\centerline{\epsf{sign_lattice.eps}[xscale=2/3,yscale=2/3]}
\caption{Le treillis des signes}
\label{sign-fig}
\end{figure}


@D global reduc @{
/* Implementation of the finite sign lattice siglat
 *                
 *                sgtop
 *                /   \
 *             minus  plus
 *                \   /
 *                zero
 *                  |
 *                sgbot
 */
enum         siglat_elem         {sgbot=0, zero=1, minus=2, plus=3, sgtop=4} ;
static char* siglat_string[5]  = {"sgbot", "zero", "minus", "plus", "sgtop"};
static enum  siglat_elem       
             siglat_jm[5][5]   = { /* Lower left is join, upper right is meet */

   /*join\meet   sgbot zero minus plus sgtop */
   /* sgbot  */ {  0,    0,   0,    0,   0   },
   /* zero   */ {  1,    1,   1,    1,   1   },
   /* minus  */ {  2,    2,   2,    1,   2   },
   /* plus   */ {  3,    3,   4,    3,   3   },
   /* sgtop  */ {  4,    4,   4,    4,   4   }};

#define siglat_join(se1, se2)  (((se1) >= (se2))?siglat_jm[(se1)][(se2)]:siglat_jm[(se2)][(se1)])
#define siglat_meet(se1, se2)  (((se1) <= (se2))?siglat_jm[(se1)][(se2)]:siglat_jm[(se2)][(se1)])
#define siglat_positif(se)     (((se) == zero)||((se) == plus )) 
#define siglat_negatif(se)     (((se) == zero)||((se) == minus)) 
#define siglat_top_positif(se) (((se) == zero)||((se) == plus )||((se) == sgtop))  
#define siglat_top_negatif(se) (((se) == zero)||((se) == minus)||((se) == sgtop))   
#define siglat_to_string(se)   (char*) siglat_string[(int) (se)]     
@| sign_lattice siglat @} 



@D toto @{
/* Implementation of the finite parallel half space lattice hspara
 *                
 *                   _________ hstop ________
 *                  /            |            \
 *                sstop          |           optop
 *                /   \          |           /   \
 *           ssminus ssplus   hszero    opminus opplus
 *                \   /          |           \   /
 *               sszero ________/ \________ opzero
 *                  \                         /
 *                   \________ hsbot ________/
 */
enum hspara_elem
{                     /* compare   {h1: a1 X + b1 <= 0} with {hj: aj X + bj <= 0} */      
hsbot          = 0,   /*  unparallel ->   h1/hj = h1    */
   /**/               /*  a1 == aj for same sign (ss)  part lattice */ 
     sszero    = 1,   /*  b1 == bj   ->   h1/hj = full  */ 
       ssminus = 2,   /*  b1 <  bj   ->   h1/hj = full  */ 
       ssplus  = 3,   /*  b1 >  bj   ->   h1/hj = h1    */ 
     sstop     = 4,   /*  b1  ? bj   ->   h1/hj = full  */ 
   /**/               /* -a1 == aj for opposite sign (op)  part lattice */
     opzero    = 5,   /* -b1 == bj   ->   h1/hj = h1    */ 
       opminus = 6,   /* -b1 <  bj   ->   h1/hj = empty */ 
       opplus  = 7,   /* -b1 >  bj   ->   h1/hj = h1    */ 
     optop     = 8,   /* -b1 ?  bj   ->   h1/hj = empty */ 
    hszero     = 9,   /* -b1 ?  bj   ->   h1/hj = full  */ 
hstop          = 10   /* a1?aj^b1?bj ->      ?          */ 
};

/* Result of h1/hj according to hspara_elem */
enum supress_kind {hjempty, empty, keep, full}; 
static enum supress_kind  hspara_supress[11] =
{
keep,             
   /**/        
     full,       
       full,      
       keep,  
     full,
   /**/ 
     keep,
       empty, 
       keep,
     empty,
    full,
keep
}; 
#define hspara_to_supress(hs)   hspara_supress[(int)(hs)]


static char* hspara_string[11] = 
{
"hsbot",
   /**/
     "sszero", 
       "ssminus", 
       "ssplus",  
     "sstop",
   /**/
     "opzero",
       "opminus", 
       "opplus",
     "optop",
    "hszero",
"hstop" 
};

static enum  hspara_elem       
             hspara_jm[11][11] = { /* Lower left is join, upper right is meet */

 /*join\meet    hsbot sszero ssminus ssplus sstop opzero opminus opplus optop hszero hstop */
 /* hsbot   */ {  0,    0,     0,      0,     0,    0,      0,     0,     0,    0,     0   },
 /* sszero  */ {  1,    1,     1,      1,     1,    0,      0,     0,     0,    1,     1   },
 /* ssminus */ {  2,    2,     2,      1,     2,    0,      0,     0,     0,    1,     2   },
 /* ssplus  */ {  3,    3,     4,      3,     3,    0,      0,     0,     0,    1,     3   },
 /* sstop   */ {  4,    4,     4,      4,     4,    0,      0,     0,     0,    1,     4   },
 /* opzero  */ {  5,    9,    10,     10,    10,    5,      5,     5,     5,    5,     5   },
 /* opminus */ {  6,   10,    10,     10,    10,    5,      6,     5,     6,    5,     6   },
 /* opplus  */ {  7,   10,    10,     10,    10,    5,      5,     7,     7,    5,     7   },
 /* optop   */ {  8,   10,    10,     10,    10,    5,      6,     7,     8,    5,     8   },
 /* hszero  */ {  9,    9,    10,     10,    10,    9,     10,    10,    10,    9,     9   },
 /* hstop   */ { 10,   10,    10,     10,    10,   10,     10,    10,    10,   10,    10   }};


#define hspara_join(se1, se2)   (((se1) >= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_meet(se1, se2)   (((se1) <= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_to_string(se)    (char*) hspara_string[(int) (se)]   
#define siglat_to_ss_hspara(se) (se)      
#define siglat_to_op_hspara(se) (((se)==sgbot)?(hsbot):((se)+4))   

@}

\paragraph{contrainte\_parallel\_in\_liste} étend la fonction 
{\tt contrainte\_in\_liste} de C3. La contrainte {\tt in\_co} est
considérée comme une inégalité et est comparée à la
liste des inégalités {\tt in\_lc}. La valeur renvoyée,
élément du treillis des signes, est la jointure de la contrainte
entrée avec tous les éléments de la liste.

Reprenons l'exemple trivial utilisé pour {\tt vect\_parallel}.
Ici, {\tt in\_co} représente le demi-espace $h_1: x \geq 0$ ($-x \leq 0$)  
et {\tt in\_lc} les demi-espaces $h_j: x \geq b_j$ ($-x +b_j \leq 0$).
Ici, $b_j = \mbox{\tt vect\_parallel($h_1$,$h_j$)}$, et le résultat 
$res = \bigvee_{j\geq 2} b_j$. Nous avons alors, en appelant $S$ le système
représenté par la conjonction des $h_j$, plusieurs
possibilités suivant les valeurs de $res$ :

\begin{enumerate}

\item[$bottom$]  $h_1$ n'est parallèle à aucun des
demi-espaces de $h_j$. Donc $h_1/S = h_1$.  

\item[$zero$ ou $minus$] $h_1$ est redondant dans $S \wedge h_1$ : il
ne sépare pas S. Donc $h_1/S = Full$. 

\item[$plus$]  $\exists\, j$ tel que $h_j$ est redondant dans
$S \wedge h_1$. $h_1$ peut donc séparer $S$. Donc $h_1/S = h_1$. 

\item[$top$]  $\exists\, j_1, j_2$ tels que 
$b_{j_1} \leq 0 \wedge b_{j_2} \geq 0 \wedge b_{j_1} \neq b_{j_2}$,
donc $h_{j_1}$ est redondant dans $S$. $h_1$ et $h_{j_2}$ sont redondants dans
$S\wedge h_1$. $h_1$ ne sépare donc pas $S$, et $h_1/S = Full$.

\end{enumerate}
