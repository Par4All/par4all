/* This file provides functions to convert a system 
 * of constraints into format of Janus.
 */ 
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
//#include <assert.h>
//#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sc-private.h"

/* Janus */
#include <sys/types.h>
//#include "Janus/sjanus.h"
//#include "Janus/stask.h"
//#include "Janus/rproblem.h"
#include "Janus/iproblem.h"
/* Janus */

FILE *fopen(), *fdebug;
struct initpb I; 
struct problem Z;

#define FTRACE fdebug
//sc must be consistant
//return TRUE if janus can handle, FALSE if not
static boolean 
sc_to_iproblem(Psysteme sc)
{
  //sc
  // int nbrows, nbcolumns; // common temporary counter i,j 
  Pcontrainte peq;
  Pvecteur pv;
  Value v;

  //Janus 
  // int p6,p5,p4,p3,p2,p1;
  int vi0,targ,pr,nblevel,mtlevel,tas;
  int i,j,choix,vision,methode;    
  int par1,par2,par3,par4,par5,par6,par7;
  // int tfois = 1;// for time measurement only. other initiation removed.

  /* prefixed parameter for Janus - to be changed in futur */
  tas=0; pr=0;
  par1 = 0; par2 = 0; par3 = 0;
  par4 = 0; // no debug mode
  par5 = 0; par6 = 1;  par7 = 1;

  //have to remove this debug in the futur for exact time measument
  fdebug=fopen("jtrace.tex","w") ; /* file for trace */
  Z.ftrace=fdebug;
  
  /**************** BEGIN parameters into structure ************************/
  Z.turbo=0; Z.remove=1;
  mtlevel = par1; /* technique for hierarchical computation - 0 if simplex */
  nblevel = par2; /* number of levels - 0 if simplex */
  methode = par3;            Z.met8 = methode/10000000;
  methode -= 10000000*Z.met8;Z.met7 = methode/1000000;
  methode -= 1000000*Z.met7; Z.met6 = methode/100000;
  methode -= 100000*Z.met6;  Z.met5 = methode/10000;
  methode -= 10000*Z.met5;   Z.met4 = methode/1000;
  methode -= 1000*Z.met4;    Z.met3 = methode/100;
  methode -= 100*Z.met3;     Z.met2 = methode/10;
  methode -= 10*Z.met2;      Z.meth = methode;
  vision = par4;    Z.dyn = vision/1000;   /* dynamic visualization*/
  vision -= 1000*Z.dyn;      Z.ntrac3 = vision/100; /* amount of information */
  vision -= 100*Z.ntrac3;    Z.ntrac2 = vision/10;  /* amount of information */
  Z.ntrace = vision-10*Z.ntrac2;                    /* amount of information */
  Z.fourier = par5;       /* Fourier_Motzkin */
  targ =  par6; Z.forcer= targ/10;
  targ -= 10*Z.forcer; Z.varc= targ; /*How to introduce constrained variables*/
  choix = par7; Z.critermax = choix/100; /*Criterion  max iterations*/
  choix -= 100*Z.critermax; Z.choixprim = choix/10; /* Choice pivot primal */
  Z.choixpiv = choix-10*Z.choixprim; /* Choice of the pivot for dual simplex */  
  /**************** END parameters into structure ************************/
  /**************** BEGIN print debug into structure ************************/
  if (Z.dyn) dynam(&Z,0);
  if (Z.ntrace||Z.ntrac2)
  {fprintf(FTRACE,"\\documentstyle[fullpage,epic,eepic,fancybox]{article} \n");
   fprintf(FTRACE,"\\begin{document}\n");
   tableau_janus(&I,&Z);
  }
  /**************** END print debug into structure ************************/
  /**************** BEGIN to insert DIMENSION with tests into structure *************/
  // removed reading from file. replace by direct assign
  // we can chosse probleme janus ou simplexe entier classique or other in this section
 
  //fscanf(fh,"%d",&Z.nvar);   /* nombre total de variables */
  Z.nvar = sc->dimension; /* to be verified */
  if (Z.nvar <= 0)   // let's forget this case for the moment. haven't seen the use yet.  
    {
      printf("number of variables <= 0");
      return FALSE;
      // put exception here saying Janus cannot handle this system of constraints
    }
  if (Z.nvar> MAXCOLONNES)
    {
      printf("Too many variables %3d > max=%3d\n",Z.nvar,MAXCOLONNES);
      return FALSE;// put exception here saying Janus cannot handle this system of constraints
    }

  //fscanf(fh,"%d",&Z.mcontr); /* nombre de contraintes */
  Z.mcontr = sc->nb_eq + sc->nb_ineq; /* to be verified*/
  if (Z.mcontr> MAXLIGNES)
    { 
      printf("Too many constraints %3d > max = %3d \n",Z.mcontr,MAXLIGNES);
      return FALSE;// put exception here saying Janus cannot handle this system of constraints
    }

  //fscanf(fh,"%d",&Z.nvpos);/* nombre de variables >=0 donc: Z.nvpos<= Z.nvar */
  Z.nvpos = 0;  /* to be verified */
  if (Z.nvpos > Z.nvar)
    { 
      printf(" %3d variables positives, depasse %3d\n",Z.nvpos,Z.nvar);
      return FALSE;// ?? can we put nvpos = 0 here ??
    }

  //fscanf(fh,"%d",&vi0); /* unused parameter */
  vi0 = 0;
      
  /**************** END to insert DIMENSION into structure ********************/

  /**************** BEGIN to insert MATRIX into structure*********************/
  // need exact DIMENSION, similar to sc_to_atrix
  /* for ( i=1 ; i <= Z.mcontr ; i++)
    { fscanf(fh,"%d",&ventier) ; I.e[i]=ventier ;
      for ( j=1 ; j <= Z.nvar ; j++)
	{ fscanf(fh,"%d",&ventier) ; I.a[i][j]=ventier ;
	}
      fscanf(fh,"%d",&ventier) ; I.d[i]=ventier ;
    }
  */
  //ATTENTION, first elements in array in struct initpb are not used. 
  //this array is not initiated. In use : rows 1 -> Z.mcontr, columns 1-> Z.nvar

  for ( i=1 ; i <= Z.mcontr ; i++) // differentiation eq and ineq, running by constraints
    { 
      //if egalite then 0, inegalite then -1. Put in egalites first
      if (i<=sc->nb_eq) I.e[i]= 0;
      else I.e[i]= -1;
    }

  //included constant (or rhs) here, which is the last element in the vector, donot change the sign of constant
  //

  //egalites coeffs, i for columns, j for rows
  for (peq = sc->egalites, i = 1; !CONTRAINTE_UNDEFINED_P(peq); peq=peq->succ, i++) 
    {      
      for (pv=sc->base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
	{     
	  value_assign(v,value_uminus(vect_coeff(vecteur_var(pv),peq->vecteur)));
	  I.a[i][j] = VALUE_TO_INT(v);
	  if (value_ne(INT_TO_VALUE(I.a[i][j]),v)) {return FALSE;} //coeff exceeds the limit of type int
	}
       value_assign(v,vect_coeff(TCST,peq->vecteur));
       I.d[i]= VALUE_TO_INT(v);
       if (value_ne(INT_TO_VALUE(I.d[i]),v)) {return FALSE;} //coeff exceeds the limit of type int
    }
	
  //inegalites
  for (peq = sc->inegalites; !CONTRAINTE_UNDEFINED_P(peq); peq=peq->succ, i++) 
    {      
      for (pv=sc->base,j=1; !VECTEUR_NUL_P(pv); pv=pv->succ,j++)
	{     
	  value_assign(v,value_uminus(vect_coeff(vecteur_var(pv),peq->vecteur)));
	  I.a[i][j] = VALUE_TO_INT(v);
	  if (value_ne(INT_TO_VALUE(I.a[i][j]),v)) {return FALSE;} //coeff exceeds the limit of type int
	}
       value_assign(v,vect_coeff(TCST,peq->vecteur));
       I.d[i]= VALUE_TO_INT(v);
       if (value_ne(INT_TO_VALUE(I.d[i]),v)) {return FALSE;} //coeff exceeds the limit of type int
    }   
     
  /**************** END to insert MATRIX into structure *********************/ 
  return TRUE;
}

boolean 
sc_janus_feasibility(Psysteme sc)
{ 
  boolean ok = FALSE;
  int r = 0;

  /**************** change format into Janus, using global struct iproblem */
  ok = sc_to_iproblem(sc);
  
  /**************** BEGIN execution simplexe entier classique */
  if (ok) r = isolve(&I,&Z,0);
  else return(9); //means Janus is not ready for this sc
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
    { fprintf(FTRACE,"\\end{document}\n");
    }
      
  fclose(fdebug);

  if (r==VRFIN) {ok = TRUE; return ok;}
  else if ((r==VRVID)||(r==VRINF)) {ok = FALSE;return ok;}
  else return r; // in case of failure then return r >=3
  //donnot want to use exception here, nor one more parameter, 
  //while wanting to keep return boolean for compatibility. To be changed.
}
