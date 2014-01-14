/*

  $Id: sc_janus_feasibility.c 1539 2012-05-31 07:27:03Z maisonneuve $

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

/* This file provides functions to convert a system 
 * of constraints into format of Janus.
 */ 
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "linear.h"
#include "janusvalue.h"


/*sc must be consistant*/
static bool 
sc_to_iproblem(Psysteme sc, Pinitpb I, Pproblem Z, FILE *fdebug)
{
  Pcontrainte peq;
  Pvecteur pv;
  Value temp;
  bool special_constraint_p = false;

  /*Janus 
    int p6,p5,p4,p3,p2,p1;*/
  int vi0,targ,pr,nblevel,mtlevel,tas;
  int i,j,choix,vision,methode;    
  int par1,par2,par3,par4,par5,par6,par7;
  /* int tfois = 1; for time measurement only. other initiation removed.*/

  /* prefixed parameter for Janus - to be changed in futur */
  tas=0; pr=0;
  par1 = 0; par2 = 0; par3 = 0;
  par4 = 0; /* no debug mode*/
  par5 = 0; par6 = 1;  par7 = 1;

  /*have to remove this debug in the futur for exact time measument*/
  if (par4) {
    fdebug=fopen("jtrace.tex","w") ; /* file for trace */
  }else {
    fdebug = NULL;
  }
  Z->ftrace=fdebug;
  
  /**************** BEGIN parameters into structure ************************/
  Z->turbo=0; Z->remove=1;
  mtlevel = par1; /* technique for hierarchical computation - 0 if simplex */
  nblevel = par2; /* number of levels - 0 if simplex */
  methode = par3;            Z->met8 = methode/10000000;
  methode -= 10000000*Z->met8;Z->met7 = methode/1000000;
  methode -= 1000000*Z->met7; Z->met6 = methode/100000;
  methode -= 100000*Z->met6;  Z->met5 = methode/10000;
  methode -= 10000*Z->met5;   Z->met4 = methode/1000;
  methode -= 1000*Z->met4;    Z->met3 = methode/100;
  methode -= 100*Z->met3;     Z->met2 = methode/10;
  methode -= 10*Z->met2;      Z->meth = methode;
  vision = par4;    Z->dyn = vision/1000;   /* dynamic visualization*/
  vision -= 1000*Z->dyn;      Z->ntrac3 = vision/100; /* amount of information */
  vision -= 100*Z->ntrac3;    Z->ntrac2 = vision/10;  /* amount of information */
  Z->ntrace = vision-10*Z->ntrac2;                    /* amount of information */
  Z->fourier = par5;       /* Fourier_Motzkin */
  targ =  par6; Z->forcer= targ/10;
  targ -= 10*Z->forcer; Z->varc= targ; /*How to introduce constrained variables*/
  choix = par7; Z->critermax = choix/100; /*Criterion  max iterations*/
  choix -= 100*Z->critermax; Z->choixprim = choix/10; /* Choice pivot primal */
  Z->choixpiv = choix-10*Z->choixprim; /* Choice of the pivot for dual simplex */  
  /**************** END parameters into structure ************************/
  /**************** BEGIN print debug into structure ************************/
  if (Z->dyn) dynam(Z,0);
  if (Z->ntrace||Z->ntrac2)
  {fprintf(Z->ftrace,"\\documentstyle[fullpage,epic,eepic,fancybox]{article} \n");
   fprintf(Z->ftrace,"\\begin{document}\n");
   tableau_janus(I,Z);
  }
  /**************** END print debug into structure ************************/

/**************** BEGIN reset some parameters in the structures ************************/
  Z->negal=0;Z->icout=0;Z->minimum=0;Z->mx=0;Z->nx=0;Z->ic1=0;Z->tmax=0;Z->niter=0;Z->itdirect=0;Z->nredun=0;Z->numero=0;Z->numax=0; Z->lastfree=0;Z->nub= 0;Z->ntp = 0;Z->vdum = 0;Z->nturb = 0;
/*
  for ( i=1;i<=MAXLIGNES+1; i++) { 
  for (j=1; j<=MAXCOLONNES+1; j++)     
  value_assign(I->a[i][j],VALUE_ZERO);
  }
  for ( i=1 ; i <= MAXLIGNES +1 ; i++)  {      
  value_assign(I->d[i],VALUE_ZERO);
  }
  for ( i=1 ; i <= MAXLIGNES +1 ; i++)  {
  I->e[i]=0;
  }
  for ( i=1;i<=AKLIGNES; i++) { 
  for (j=1; j<=AKCOLONNES; j++)     
  value_assign(Z->ak[i][j],VALUE_ZERO);
  }
  for ( i=1 ; i <= AKLIGNES ; i++)  { 
  value_assign(Z->dk[i],VALUE_ZERO);
  }
*/
  /*TODO: If these parameters are already initialized before used in the code (which is not easy to see now), 
    then it's not necessary to reinitialize them here. If not, then we might have a bug.*/

  /**************** END reset some parameters in the structures ************************/

  /**************** BEGIN to insert DIMENSION with tests into structure *************/
  /* removed reading from file. replace by direct assign*/
  /* we can chosse probleme janus ou simplexe entier classique or other in this section*/
 
  /*fscanf(fh,"%d",&Z->nvar);    nombre total de variables */
  Z->nvar = sc->dimension; /* to be verified */
  if (Z->nvar < 0) { /*sc given is not correct (normally tested before)*/
    ifscdebug(5) {
      printf("\n[sc_to_iproblem]: Janus has a number of variables < 0 ");
    }
    return false;/* Janus cannot handle this system of constraints*/
  }
  if (Z->nvar> MAXCOLONNES) {
    ifscdebug(5) {
      printf("\n[sc_to_iproblem]: Too many variables %3d > max=%3d\n",Z->nvar,MAXCOLONNES);
    }
    return false;/* Janus cannot handle this system of constraints*/
  }

  /*fscanf(fh,"%d",&Z->mcontr);  nombre de contraintes */
  Z->mcontr = sc->nb_eq + sc->nb_ineq; /* sc's parameters must be true*/
  if (Z->mcontr> MAXLIGNES) {
    ifscdebug(5) {
      printf("\n[sc_to_iproblem]: Too many constraints %3d > max = %3d \n",Z->mcontr,MAXLIGNES);
    }
    return false;/* Janus cannot handle this system of constraints*/
  }

  /*fscanf(fh,"%d",&Z->nvpos); nombre de variables >=0 donc: Z->nvpos<= Z->nvar */
  Z->nvpos = 0;  /* to be verified */
  if (Z->nvpos > Z->nvar) { 
    ifscdebug(5) {
      printf("\n[sc_to_iproblem]: %3d variables positives, greater than %3d\n",Z->nvpos,Z->nvar);
    }
    return false;/* ?? can we put nvpos = 0 here */
  }

  /*fscanf(fh,"%d",&vi0);  unused parameter */
  vi0 = 0;      
  /**************** END to insert DIMENSION into structure ********************/

  /**************** BEGIN to insert MATRIX into structure*********************/
  /* need exact DIMENSION, similar to sc_to_matrix*/
  /* for ( i=1 ; i <= Z->mcontr ; i++)
    { fscanf(fh,"%d",&ventier) ; I->e[i]=ventier ;
      for ( j=1 ; j <= Z->nvar ; j++)
	{ fscanf(fh,"%d",&ventier) ; I->a[i][j]=ventier ;
	}
      fscanf(fh,"%d",&ventier) ; I->d[i]=ventier ;
    }
  */
  /*ATTENTION, first elements in array in struct initpb are not used. */
  /*this array is not initiated. In use : rows 1 -> Z->mcontr, columns 1-> Z->nvar*/

  /*DN: need to check the special inequality here: 0 < 1, mean a vector of one element.*/

  for ( i=1 ; i <= Z->mcontr ; i++) /* differentiation eq and ineq, running by constraints*/
    { 
      /*if egalite then 0, inegalite then -1. Put in egalites first*/
      if (i<=sc->nb_eq) I->e[i]= 0;
      else I->e[i]= -1;
    }

  /*included constant (or rhs) here, which is the last element in the vector, donot change the sign of constant
    DN 6/11/02. use Marcos of Value. Needed Value in struct problem.*/

  /*egalites coeffs, i for columns, j for rows*/
  for (peq = sc->egalites, i = 1; !CONTRAINTE_UNDEFINED_P(peq); peq=peq->succ, i++) 
    {      
      for (pv=sc->base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
	{     
	  value_assign(I->a[i][j],value_uminus(vect_coeff(vecteur_var(pv),peq->vecteur)));
	}
      value_assign(I->d[i],vect_coeff(TCST,peq->vecteur));
    }
	
  /*inegalites*/
  for (peq = sc->inegalites; !CONTRAINTE_UNDEFINED_P(peq); peq=peq->succ, i++) 
    {       
      value_assign(temp,VALUE_ZERO); 
      for (pv=sc->base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
	{     
	  value_assign( I->a[i][j],value_uminus(vect_coeff(vecteur_var(pv),peq->vecteur)));
	  value_addto(temp,value_abs(I->a[i][j]));
	}
      if value_zero_p(temp) {
	special_constraint_p = true;
      }
     value_assign(I->d[i],vect_coeff(TCST,peq->vecteur));
    } 
  if (special_constraint_p) {
    /*sc_default_dump(sc); 
      TODO: what to do with this special constraint? if 0<1 ok, if 1 < 0 then not ok, return false */
  }
  
  /**************** END to insert MATRIX into structure *********************/ 
  return true;
}

bool 
sc_janus_feasibility(Psysteme sc)
{ 
  
  FILE *fopen(), *fdebug;
  initpb I; 
  problem Z;

  bool ok = false;
  int r = 0;

  /**************** change format into Janus, using global struct iproblem */
  ok = sc_to_iproblem(sc,&I,&Z,fdebug); /*Attention with dimension=0*/
  
  /**************** BEGIN execution simplexe entier classique */
  if (ok) r = isolve(&I,&Z,0);
  else r = 9;/*DN janus should not be used here.*/

  /* the third parameter means test only one time
     only I is initialized. Z is still empty */
  /*
  if (r==VRFIN) printf("solution finie\n") ;
  else if (r==VRVID) printf("polyedre vide\n") ;
  else if (r==VRINF) printf("solution infinie\n") ;
  else if (r==VRINS) printf("nombre insuffisant\n") ; 
  else if (r==VRDEB) printf("debordement tableaux\n") ;
  else if (r==VRCAL) printf("some wrong parameter\n") ;
  else if (r==VRBUG) printf("bug de programmation\n") ;
  else if (r==VROVF) printf("data overflow\n") ;
  else if (r==VREPS) printf("pivot anormalement petit\n") ;
  */
  /* #define VRFIN 0	 solution finie */
  /* #define VRVID 1	 polyedre vide */
  /* #define VRINF 2	 solution infinie */
  /* #define VRINS 3	 nombre de pivotages insuffisant, bouclage possible */
  /* #define VRDEB 4	 debordement de tableaux */
  /* #define VRCAL 5	 appel errone */
  /* #define VRBUG 6	 bug */
  /* #define VROVF 7	 overflow */

  
  /***************** END execution simplexe entier classique */

  if (Z.ntrace||Z.ntrac2)
    { fprintf(Z.ftrace,"\\end{document}\n");
    }
  
  if (fdebug) fclose(fdebug);/* DN: In Solaris, we can fclose(NULL), but in LINUX, we cannot.*/

  if (r==VRFIN) {ok = true; return ok;}
  else if ((r==VRVID)||(r==VRINF)) {ok = false;return ok;}
  else return r; /* in case of failure then return r >=3
		    donnot want to use exception here, nor one more parameter, 
		    while wanting to keep return bool for compatibility. To be changed.*/
}

