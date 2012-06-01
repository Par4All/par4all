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

% title 	: Extension de C3 aux Unions de Polyèdres.
% author      	: Arnauld LESERVOT
% begin date  	: 10/93
% file name 	: UNION.w 
% called file 	: path.eps, disjunct.eps






% ---------- Dimensions et style
\documentstyle[a4,11pt,leservot-fre]{report}

% Il semble que ceci ne soit pas interprete de la meme facon par
% le latex sur les sun (colle la marge droite) et sur babar (correct).
\addtolength{\textwidth}{2cm}
\addtolength{\leftmargin}{-1cm}

%\title{Extension de C3 aux Unions de Polyèdres}
%\author{Arnauld LESERVOT}
%\date{\today}



% ---------- Debut document
\begin{document}
%\maketitle
\input{union-CEA-title}
\tableofcontents
\newpage

% ---------- Introduction
\chapter{Introduction} 

\paragraph{Nécessité de l'union de polyèdres.}
Les polyèdres convexes sont intensément 
utilisés dans le paralléliseur automatique Linear/C3 Library~\cite{IJT91},
notamment pour le calcul des dépendances de données.

La méthode de Feautrier pour
calculer un graphe de dépendance exact, l'Array Data Flow 
Graph~\cite{Fea91b}, fournit une fonction source pour chaque
opérande du programme séquentiel à traiter.

Cette fonction source est présentée sous forme 
de {\em quast} (QUasi Affine Sélection Tree), 
c'est à dire sous forme d'un arbre de décision : 
arbre binaire dont les noeuds sont des systèmes 
d'équations et d'inéquations linéaires. 
La figure~\ref{decision-fig} représente un tel arbre.

\begin{figure}
\[ \Ifthenelse{n - 3 \leq i \leq 2n - 2}
	{\mbox{opération 1}}
	{ \Ifthenelse{i \leq 4n + 5}
		{\mbox{opération 2}}
		{\mbox{opération 3}}
	}
\]
\caption{Un arbre de décision}
\label{decision-fig}
\end{figure}

Chaque feuille de l'arbre est acceptable si le chemin
qui la gouverne est vrai : on obtient ainsi, pour chaque opérande
du programme, un ensemble de solutions conditionnées par un prédicat.

Si nous reprenons l'exemple de la figure~\ref{decision-fig},
l'opération 2 est gouvernée par $i \leq 4n+5 \wedge \neg ( n - 3
\leq i \leq 2n - 2)$. Ce prédicat est un {\em chemin} dans l'arbre de
décision. Il peut être mis sous forme d'une {\em disjonction}
équivalente : $i \leq n - 4  \vee 2n-1 \leq i \leq 4n+5$.
Ces disjonctions proviennent de la prise, dans l'arbre de décision,
d'une branche fausse : la négation d'un système ayant plus
d'une inéquation est une union de polyèdres.

L'objet de ce rapport est de présenter les structures de
bases définissant ces deux types de données, ainsi que les
fonctions les manipulant. Nous nous attacherons plus
particulièrement à réduire le nombre des disjonctions
équivalentes à un chemin. 


\paragraph{C3 et son extension aux unions.}
La bibliothèque d'Algèbre linéaire C3, écrite en C, regroupe les
structures de vecteur, de matrice, 
de polynôme, de système d'équations ou
d'inéquations en nombre entiers et de polyèdre,
ainsi que les fonctions de manipulation de ces structures.
Le lecteur est renvoyé à \cite{Anc91} et \cite{Lam93} pour
plus d'information sur cette bibliothèque.

{\em Polyèdre} désigne, ici comme dans C3,
un polyèdre convexe dans un espace de dimension finie : il 
est donc représentable par un système d'équations ou 
d'inéquations linéaires ou par un système générateur
de points, droites et rayons. Nous renvoyons le lecteur
à~\cite{Hal79} pour plus de détails. 
C'est la représentation sous forme de système qui sera 
utilisée ici. Les polyèdres seront notés dans la suite $P_{i}$. 

Les structures de données et les fonctions ont été
implanté en C et conçu pour s'intégrer à la
bibliothèque d'algèbre linéaire C3.

\paragraph{Travaux similaires.}
D. Wilde décrit dans \cite{Wil93} une
extension de C3-IRISA pour contenir les unions de polyèdres. La structure
de chemin n'y est cependant pas représentée et la fonction de
différence, n'étant pas basées sur les chemins, n'intègre
pas les réductions introduites dans ce rapport.

Citons aussi les travaux de Pugh~\cite{Pug91} à l'université de
Maryland dont le calculateur symbolique Omega réalise des
opérations similaires à celles de C3 augmenté des unions. 
La structure de chemin n'y est par contre pas particulièrement étudiée. 


\paragraph{Plan du rapport.}
Le chapitre ~\ref{strucom-chap} présente une structure de donnée
réutilisée par la suite pour décrire les disjonctions et les
chemins : la liste de systèmes. Le chapitre~\ref{disjonction-chap}
décrit les disjonctions et le chapitre~\ref{chemin-chap} les
chemins. Les fonctions de lecture et d'écritures des disjonctions et
des chemins sont présentées au chapitre~\ref{lectec-chap}.


\paragraph{Remerciements.}
Je remercie B. Apvrille pour ces remarques ainsi que pour sa contribution 
à certaines des fonctions présentées ici.



% ---------- Structures Communes
\chapter{Structures Communes}
\label{strucom-chap}
@i sc_list.w


% ---------- Disjonctions
\chapter{Les Disjonctions}
\label{disjonction-chap}
@i disjunct.w


% ---------- Chemins
\chapter{Les Chemins}
\label{chemin-chap}
@i path.w

% ---------- Reduction
\chapter{Réduction des hyperplans redondants}
\label{reduc-chap}
@i reduc.w


% ---------- Entree-Sortie
\chapter{Lecture-Écriture}
\label{lectec-chap}
@i sl_io.w


% ---------- Bibliographie
\bibliographystyle{alpha}
\bibliography{/gandhi/home1/s8/leservot/Biblio/biblio,/gandhi/local/pips/Pcp/Biblio/pcp}
\addcontentsline{toc}{chapter}{Bibliographie}


% ---------- Annexes
\appendix
\chapter{Index général}

\paragraph{Fichiers générés}
@f

\paragraph{Macros définies}
@m

\paragraph{Fonctions et termes indexés}
@u


\chapter{LCZOS.F}
\label{lczos-annexe}
\begin{verbatim}
       program lczos
       real alpha(400), a(400,400), beta(400), real y(400,400), x(400)
       real Norm1, Norm2(400), gamma(400,400), yp(400,400)
       do 3 i = 1, n
          alpha(i) = 0.0
3         beta(i) = 0.0
       do 5 i = 1, n
5         y(i, 1) = 0.0
       Norm1 = 0.0
       do 10 i = 1, n
10        Norm1 = Norm1 + x(i)*x(i)
       Norm1 = 2*Norm1
       do 20 i = 1, n
20        y(i,1) = x(i)/Norm1
       do 30 i = 1, n
          gamma(i,1) = 0.0
          do 30  k = 1, n
30           gamma(i,1) = gamma(i,1) + a(i,k)*y(k,1)
       do 40 i = 1, n
40        alpha(1) = alpha(1)+ gamma(i,1)*y(i,1)
       do 50 i = 1, n
50        yp(i,2) = gamma(i,1) - alpha(1)*y(i,1)
       Norm2(1) = 0.0
       do 60 i = 1, n
60        Norm2(1) = Norm2(1) + yp(i,2)*yp(i,2)
       beta(1) = sqrt(Norm2(1))
       do 70 i = 1, n
70        y(i,2) = yp(i,2)/ beta(1)
       do 800 j = 2, m-1
          do 300 i = 1, n
             gamma(i,j) = 0.0
             do 300  k = 1,n
300             gamma(i,j) = gamma(i,j) + a(i,k)*y(k,j)
          do 400 i = 1, n
400          alpha(j) = alpha(j)+ gamma(i,j)*y(i,j)
          do 500 i = 1,n
500          yp(i, j+1) = gamma(i,j)-alpha(j)*y(i,j)- beta(j-1)*y(i,j-1)
          Norm2(j) = 0.0
          do 600 i = 1, n
600           Norm2(j) = Norm2(j) + yp(i,j+1)*yp(i,j+1)
              beta(j) = sqrt(Norm2(j))
          do 700 i = 1, n
700           y(i,j+1) = yp(i,j+1)/ beta(j)
800    continue
       end
\end{verbatim}


\chapter{Démonstration des règles 1, 2, 3.}
\label{demo-annexe}
\input{/gandhi/home1/s8/leservot/These/avancement-regles}


\chapter{Types et entête}
\label{entete-annexe}
\section{Fichier des types.} Ce fichier comprend la définition des types et des
redéfinitions de fonctions pour des questions de compatibilité avec
les versions antérieures.
@o union-local.h @{
@< Type @> 
/* FOR BACKWARD COMPATIBILITY */
#define my_sc_full()         sc_full()
#define my_sc_empty()        sc_empty((Pbase) NULL)
#define is_sc_my_empty_p(ps) sc_empty_p((ps))
#define is_dj_full_p(dj)     dj_full_p((dj))
#define is_dj_empty_p(dj)    dj_empty_p((dj))
#define is_pa_full_p(pa)     pa_full_p((pa))
#define is_pa_empty_p(pa)    pa_empty_p((pa))


/* FOR BACKWARD COMPATIBILITY */
#define sc_difference(ps1, ps2)      pa_system_difference_ofl_ctrl((ps1),(ps2),FWD_OFL_CTRL)
#define sc_inclusion_p(ps1, ps2)     pa_inclusion_p_ofl_ctrl((ps1), (ps2), NO_OFL_CTRL)
#define sc_inclusion_p_ofl(ps1, ps2) pa_inclusion_p_ofl_ctrl((ps1), (ps2), FWD_OFL_CTRL)
#define sc_inclusion_p_ofl_ctrl(ps1, ps2, ofl) pa_inclusion_p_ofl_ctrl((ps1), (ps2), (ofl))
#define sc_equal_p(ps1,ps2)          pa_system_equal_p_ofl_ctrl((ps1), (ps2), NO_OFL_CTRL)
#define sc_equal_p_ofl(ps1,ps2)      pa_system_equal_p_ofl_ctrl((ps1), (ps2), FWD_OFL_CTRL)
#define sc_equal_p_ofl_ctrl(ps1, ps2, ofl) pa_system_equal_p_ofl_ctrl((ps1), (ps2), (ofl))
#define sc_convex_hull_equals_union_p(conv_hull, ps1, ps2) \
  pa_convex_hull_equals_union_p_ofl_ctrl((conv_hull), (ps1), (ps2),NO_OFL_CTRL, FALSE)
#define sc_convex_hull_equals_union_p_ofl(conv_hull, ps1, ps2) \
  pa_convex_hull_equals_union_p_ofl_ctrl((conv_hull), (ps1), (ps2), OFL_CTRL, FALSE)
#define sc_convex_hull_equals_union_p_ofl_ctrl(conv_hull, ps1, ps2, ofl, bo) \
  pa_convex_hull_equals_union_p_ofl_ctrl((conv_hull), (ps1), (ps2), (ofl), (bo))

/* OTHERS */
#define sc_elim_redund_with_first(ps1, ps2) sc_elim_redund_with_first_ofl_ctrl((ps1), (ps2), NO_OFL_CTRL)

#define dj_fprint(fi,dj,fu)           dj_fprint_tab((fi), (dj), (fu), 0)
#define DJ_UNDEFINED_P(dj)            ((dj) == DJ_UNDEFINED)
#define dj_faisabilite(dj)            dj_feasibility_ofl_ctrl((dj), NO_OFL_CTRL)
#define dj_feasibility(dj)            dj_feasibility_ofl_ctrl((dj), NO_OFL_CTRL)
#define dj_faisabilite_ofl(dj)        dj_feasibility_ofl_ctrl((dj), FWD_OFL_CTRL)
#define dj_intersection(dj1, dj2)     dj_intersection_ofl_ctrl((dj1), (dj2), NO_OFL_CTRL)
#define dj_intersect_system(dj,ps)    dj_intersect_system_ofl_ctrl((dj), (ps), NO_OFL_CTRL ) 
#define dj_intersect_djcomp(dj1,dj2)  dj_intersect_djcomp_ofl_ctrl( (dj1), (dj2), NO_OFL_CTRL )
#define dj_projection_along_variables(dj,pv) \
  dj_projection_along_variables_ofl_ctrl((dj),(pv),NO_OFL_CTRL)
#define dj_variable_substitution_with_eqs(dj,co,pv) \
  dj_variable_substitution_with_eqs_ofl_ctrl( (dj), (co), (pv), NO_OFL_CTRL )

#define pa_fprint(fi,pa,fu)           pa_fprint_tab((fi), (pa), (fu), 0)
#define PA_UNDEFINED_P(pa)            ((pa) == PA_UNDEFINED)
#define pa_new()                      pa_make(NULL, NULL)
#define pa_faisabilite(pa)            pa_feasibility_ofl_ctrl((pa), NO_OFL_CTRL)
#define pa_feasibility(pa)            pa_feasibility_ofl_ctrl((pa), NO_OFL_CTRL)
#define pa_faisabilite_ofl(pa)        pa_feasibility_ofl_ctrl((pa), FWD_OFL_CTRL)
#define pa_path_to_disjunct(pa)       pa_path_to_disjunct_ofl_ctrl((pa), NO_OFL_CTRL )
#define pa_path_dup_to_disjunct(pa)   pa_path_to_disjunct_ofl_ctrl((pa), NO_OFL_CTRL )
#define pa_system_difference(ps1,ps2) pa_system_difference_ofl_ctrl((ps1),(ps2),NO_OFL_CTRL)
#define pa_system_equal_p(ps1,ps2)    pa_system_equal_p_ofl_ctrl((ps1),(ps2),NO_OFL_CTRL)
#define pa_inclusion_p(ps1,ps2)       pa_inclusion_p_ofl_ctrl((ps1),(ps2),NO_OFL_CTRL)
#define pa_path_to_disjunct_ofl(pa)   pa_path_to_disjunct_ofl_ctrl((pa), FWD_OFL_CTRL )
#define pa_path_to_disjunct_rule4(pa) pa_path_to_disjunct_rule4_ofl_ctrl((pa), FWD_OFL_CTRL )
#define pa_path_to_few_disjunct(pa)   pa_path_to_few_disjunct_ofl_ctrl((pa), NO_OFL_CTRL)
#define pa_system_difference(ps1,ps2) pa_system_difference_ofl_ctrl((ps1),(ps2),NO_OFL_CTRL)
#define pa_convex_hull_equals_union_p(conv_hull, ps1, ps2) \
  pa_convex_hull_equals_union_p_ofl_ctrl((conv_hull), (ps1), (ps2), NO_OFL_CTRL, FALSE)

#define un_fprint(fi,un,fu,ty)        un_fprint_tab((fi), (un), (fu), (ty), 0)


/* Misceleanous (debuging...) */
#define PATH_MAX_CONSTRAINTS          12

#define IS_SC                         1
#define IS_SL                         2
#define IS_DJ                         3
#define IS_PA                         4

extern char* (*union_variable_name)(Variable);

#if(defined(DEBUG_UNION_C3) || defined(DEBUG_UNION_Linear/C3 Library))
#define C3_DEBUG( fun, code )         \
  {if(getenv("DEBUG_UNION")){fprintf(stderr,"[%s]\n", fun); {code}}}
#define C3_RETURN( type, val )      \
  {if(getenv("DEBUG_UNION")){ \
     char* val1 = (char*) val; \
     fprintf(stderr,"Returning:\n"); \
     un_fprint_tab(stderr,(char*)val1,union_variable_name,type,1); return val1;} \
   else{ return val; }}
#else 
#define C3_DEBUG( fun, code )
#define C3_RETURN( type, val )        {return val;}
#endif

/* For the parsers: */
extern void sl_init_lex();
extern int slx_parse ();
@}

\section{Entête de fichiers C.}
@d includes @{
/* Package  :	C3/union
 * Author   : 	Arnauld LESERVOT (leservot(a)limeil.cea.fr)
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

@}



\end{document}

