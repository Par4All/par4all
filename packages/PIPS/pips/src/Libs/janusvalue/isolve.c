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
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include "arithmetique.h"
#include <stdio.h>
#include "assert.h"
#include "iproblem.h"
#include "iabrev.h"
#include "rproblem.h"

/* ============================================================== */
/*        PROCEDURES for printing computation information         */
/* ============================================================== */

static int xbug(Pproblem XX,int nu)
{ printf("DIRECT xBug=%d Pmeth=%d Pmet2=%d Pmet3=%d\\\\\n",
	 nu,PMETH,PMET2,PMET3);
 if (VISU||VIS2) fprintf(FTRACE,"xBug=%d\\\\\n",nu); return(VRBUG);
}
static int xdeb(Pproblem XX,int nu)
{ if (VISU||VIS2) fprintf(FTRACE,"xDeb=%d\\\\\n",nu); return(VRDEB);
}
static void vzphase(Pproblem XX,int n) /*****************************/
{ 
  if (VISU>=ZVS)
    {
      fprintf(FTRACE,"{\\bf{*} PHASE: ");
      if (n==1) fprintf(FTRACE,"PRE-PROCESSING PHASE");
      else if (n==2)fprintf(FTRACE,"DUAL SIMPLEX");
      else if (n==3)fprintf(FTRACE,"PRIMAL SIMPLEX");
      fprintf(FTRACE,"}\\\\\n");
    }
}
static void vzstep(Pproblem XX,int n) /*****************************/
{ 
  if (VISU>=ZVS)
    {
      fprintf(FTRACE,"{\\bf STEP %d: ",n);
      if (n==1) fprintf(FTRACE,"EQUALITIES ELIMINATION");
      else if (n==2)fprintf(FTRACE,"INEQUALITIES DIVISION");
      else if (n==3)fprintf(FTRACE,"FOURIER-MOTZKIN");
      else if (n==4)fprintf(FTRACE,"NON-NEGATIVE VARIABLES");
      else if (n==5)fprintf(FTRACE,"FOURIER-MOTZKIN on CONSTRAINED VARIABLES");
      else if (n==6) fprintf(FTRACE,"REDUNDANT INEQUALITIES");
      fprintf(FTRACE,"}\\\\\n");
    }
}
static void vznowstep(Pproblem XX) /*****************************/
{ if (MAJSTEP) return;
  vzstep(XX,NSTEP); MAJSTEP=1;
}
static void vznewstep(Pproblem XX,int n) /*****************************/
{ NSTEP=n; MAJSTEP=0;
}
void tableau_janus(Pinitpb II, Pproblem XX) /**************************/
{ int i,j,i00; i00=1 ;
  if (NV>100) return;
  fprintf(FTRACE,"%%%d %d %d %d\n",NV,MC,NP,i00);
  /*****fprintf(FTRACE,"%%%d %d %d %d\n",XX->nvar,XX->MCONTR,XX->NVPOS,i00);**/
  for (i=i00 ; i<=MC ; i++ )
    {
      fprintf(FTRACE,"%%%4d",E(i));
      for (j=1 ; j <= NV ; j++) fprint_Value(FTRACE,A(i,j));
      fprint_Value(FTRACE,D(i)); fprintf(FTRACE," \n");
      /*fprintf(FTRACE,"%%%4d",E(i));
      for (j=1 ; j <= NV ; j++) fprintf(FTRACE," %2d",A(i,j));
      fprintf(FTRACE," %3d\n",D(i));DN*/
    }
  /*fprintf(FTRACE," \\\\\n");*/
}
/*DN note: if use janus.c then not static DN*/
/* static void tableauk(Pproblem XX,int j1,int j2)  */
/* { int i,j,i00; i00=1 ; */
/*   if (j1>NX) {fprintf(FTRACE," --- pas de colonne %d ----\n",j1); return; }; */
/*   if (j2>NX) {fprintf(FTRACE," --- pas de colonne %d ----\n",j2); return; }; */
/*   fprintf(FTRACE,"%d %d %d %d\n",NX,MX,NX,i00); */
/*   for (i=i00 ; i<=MX ; i++ ) */
/*     { fprintf(FTRACE,"1 "); */
/*       if (j2>0) */
/* 	{ for (j=1 ; j <= NX ; j++) */
/* 	  { if (j==j1) fprint_Value(FTRACE,AK(i,j2)); */
/* 	    else if (j==j2) fprint_Value(FTRACE,AK(i,j1)); */
/* 	    else fprint_Value(FTRACE,AK(i,j)); */
/* 	  /\*if (j==j1) fprintf(FTRACE," %2d",AK(i,j2)); */
/* 	    else if (j==j2) fprintf(FTRACE," %2d",AK(i,j1)); */
/* 	    else fprintf(FTRACE," %2d",AK(i,j));DN*\/ */
/* 	  } */
/* 	} */
/*       else */
/* 	{ for (j=1 ; j <= NX ; j++) if (j!=j1) fprint_Value(FTRACE,AK(i,j)); */
/* 	  if (j1>0 && (j1 <= NX)) fprint_Value(FTRACE,AK(i,j1)); */
/* 	  /\*for (j=1 ; j <= NX ; j++) if (j!=j1) fprintf(FTRACE," %2d",AK(i,j)); */
/* 	    if (j1>0 && (j1 <= NX)) fprintf(FTRACE," %2d",AK(i,j1));DN*\/ */
/* 	} */
/*       fprint_Value(FTRACE,DK(i)); fprintf(FTRACE," \n"); */
/*       /\*fprintf(FTRACE," %3d\n",DK(i));DN*\/ */
/*     } */
/* } */
static void symbol(Pproblem XX,int v) /*****************************/
{ if (v>NUMAX) fprintf(FTRACE,"w") ;
  else if (v<0) fprintf(FTRACE,"v") ;
  else if (freevariable(XX,v)) fprintf(FTRACE,"x") ;
  else fprintf(FTRACE,"y") ;
  fprintf(FTRACE,"_{%d}",abs(v));
}
static void symbold(Pproblem XX,int v) /****************************/
{ fprintf(FTRACE,"$");symbol(XX,v);fprintf(FTRACE,"$");
}
/* static void symboldn(Pproblem XX,int v) /\***************************\/ */
/* { symbold(XX,v);fprintf(FTRACE,"\\\\\n"); */
/* } */
static void vzcut(Pproblem XX,int i,int divisor,int i2)
{
  int line,j ;
  if (NX>16) fprintf(FTRACE,"{\\scriptsize\n");
  else if (NX>13) fprintf(FTRACE,"{\\footnotesize\n");
  fprintf(FTRACE,"\\begin{center}\n");
  fprintf(FTRACE,"\\fbox{$\n");
  fprintf(FTRACE," \\begin{array}{rrrrrrrrrrrrrrrrrrrrrrrrr}\n");
  for (line=1; line<=2; line++)
    {
      int s,nk;//unused ab
      Value k;//DN
      int fleche,carre,carre2; fleche=carre=carre2=0;
      s=0; nk=0;
	  /*if (line==1) symbol(XX,G(i)); else fprintf(FTRACE," cut") ;*/
      if (line==1) symbol(XX,G(i)); else symbol(XX,G(i2));
      fprintf(FTRACE,":") ;
      for ( j=1 ; j <= NX ; j++) /*j0*/
	{ int boite; boite=0;
	  if (value_assign(k,AK(i,j))) ++nk;
	  if (nk==17)
	    { nk=0; fprintf(FTRACE,"	\\\\ & \n") ;
	    }
	  if (( NX <= 24) || value_notzero_p(k) ) fprintf(FTRACE,"	&") ;
	  if value_notzero_p(k)
	    { if ((value_neg_p(k)) && (line==1)) fprintf(FTRACE,"-"); else if (s) fprintf(FTRACE,"+");
	    if (line==1)
	      {	if (value_notone_p(value_abs(k))) fprint_Value(FTRACE,k) ;
	      }
	    else
	      {// fprintf(FTRACE,"\\lfloor%d/%d\\rfloor ",k,divisor) ;
		fprintf(FTRACE,"\\lfloor");fprint_Value(FTRACE,k);fprintf(FTRACE,"/%d\\rfloor ",divisor) ;//DN_? divisor??
	      }
	      symbol(XX,B(j));
	      s=1;
	    }
	  if (boite)
	    { fprintf(FTRACE,"$}"); boite=0;
	    }
	}
      if ((i<IC1)||janus_egalite(XX,i)) fprintf(FTRACE,"=") ;
                 else fprintf(FTRACE,"\\leq") ;
	    //DN      if (line==1) fprintf(FTRACE,"	& %3d \\\\\n",DK(i)) ;
	    // else fprintf(FTRACE,"	&\\lfloor%d/%d\\rfloor ",DK(i),divisor);
	    if (line==1) {fprintf(FTRACE,"	& \\\\\n");fprint_Value(FTRACE,DK(i));}
	    else {fprintf(FTRACE,"	&\\lfloor");fprint_Value(FTRACE,DK(i));fprintf(FTRACE,"/%d\\rfloor ",divisor);}
    }
  fprintf(FTRACE," \\end{array}  \n") ;
  fprintf(FTRACE,"$}\n");
  fprintf(FTRACE,"\\end{center}\n");
  if (NX>13) fprintf(FTRACE,"}\n");
}
static void igvoir3(Pproblem XX,int ia,int ib,int carre, int ip, int jp,
		   int carre2, int ip2, int jp2, int fleche, int ifl)
{ int i,j ;  /*print latex arrays * ak et dk between lines ia and ib */
  if (NX>16) fprintf(FTRACE,"{\\scriptsize\n");
  else if (NX>13)
    fprintf(FTRACE,"{\\footnotesize\n");
  fprintf(FTRACE,"\\begin{center}\n");
  fprintf(FTRACE,"\\fbox{$\n");
  fprintf(FTRACE," \\begin{array}{rrrrrrrrrrrrrrrrrrrrrrrrr}\n");
  for ( i=ia ; i <= ib ; i++)
    { 
      int s,nk;  Value k;//DN, unused ab
      s=0; nk=0;     
      if (fleche && i==ifl) fprintf(FTRACE,"\\Rightarrow ") ;
      if (i<IC1)
	{ 
	  if (i==ICOUT) fprintf(FTRACE,"\\bullet ") ;

	      if ((i==ICOUT) || (G(i)==0)) fprintf(FTRACE,"-d_{%d}",IC1-i);
	      else symbol(XX,G(i));
	      s=1;/*distances*/
	}
      else 
	{ if (janus_egalite(XX,i)) fprintf(FTRACE,"e_{%d}",G(i));
	  else symbol(XX,G(i));
	  fprintf(FTRACE,":") ;
	}
      for ( j=1 ; j <= NX ; j++) /*j0*/
	{ 
	  int boite; boite=0;
	  if (value_assign(k,AK(i,j))) ++nk;
	  if (nk==17)  { nk=0; fprintf(FTRACE,"	\\\\ & \n") ;	}
	  if (( NX <= 24) || value_notzero_p(k) ) fprintf(FTRACE,"	&") ;
	  if ((((fleche && i==ifl) || (carre && i==ip)) && j==jp)
	      || (carre2 && i==ip2 && j==jp2))
	    { 
	      fprintf(FTRACE,"\\fbox{$"); boite=1;
	    }
	  if value_notzero_p(k)
	    { 
	      if value_neg_p(k) fprintf(FTRACE,"-"); else if (s) fprintf(FTRACE,"+");
	      if value_notone_p(value_abs(k)) fprint_Value(FTRACE,value_abs(k)); 
	      symbol(XX,B(j));
	      s=1;
	    }
	  if (boite) { fprintf(FTRACE,"$}"); boite=0;    }
	}
      fprintf(FTRACE,"	& ") ;
      if ((i<IC1)||janus_egalite(XX,i)) fprintf(FTRACE,"=") ;
      else fprintf(FTRACE,"\\leq") ;
      {fprintf(FTRACE,"	& ");fprint_Value(FTRACE,DK(i));fprintf(FTRACE," \\\\\n") ;}
      if (i==IC1-1)	{ 
	if (NX>24) fprintf(FTRACE,"	.....") ;
	else 
	  for (j=1; j <= NX; j++) fprintf(FTRACE,"	&.....") ; /*j0*/
	fprintf(FTRACE,"	\\\\\n") ;
      }
    } ;
  fprintf(FTRACE," \\end{array}  \n") ;
  fprintf(FTRACE,"$}\n");
  fprintf(FTRACE,"\\end{center}\n");
  if (NX>14) fprintf(FTRACE,"}\n");
} /* fin igvoir3 */
static void igvoir(Pproblem XX,int ia,int ib) /*print latex arrays */
{ igvoir3(XX,ia,ib,0,0,0,0,0,0,0,0);
} /* fin igvoir */

static void w1voir(Pproblem XX,int ix)
{ igvoir(XX,ix,ix);   /******* print line ak (ix) + dk (ix) */
}
/*static void wvoir(Pproblem XX)*/ /**** display arrays ak and dk */
static void wvoir(Pproblem XX) /**** acces pour mise au point janus */
{
  if (NV>100) { 
    int i,j;
    Value cab,cmax;//DN coeef max
    cmax = VALUE_ZERO;
    for ( i=1 ; i <= MX ; i++)
      for ( j=1 ; j <= NX ; j++)
	if (value_assign(cab,value_abs(AK(i,j))))
	  if value_gt(cab,cmax) value_assign(cmax,cab);
    {fprintf(FTRACE,"COEFFICIENT MAXIMUM=");fprint_Value(FTRACE,cmax);fprintf(FTRACE," \n");}
    return;
  }
  igvoir(XX,1,MX); /*i0*/
}
/* static void stepvoir(Pproblem XX) /\**** acces pour mise au point janus *\/ */
/* {  */
/*  if (NV>100) {  */
/*     int i,j; */
/*     Value cab,cmax;//DN coeef max */
/*     cmax = VALUE_ZERO; */
/*     for ( i=1 ; i <= MX ; i++) */
/*       for ( j=1 ; j <= NX ; j++) */
/* 	if (value_assign(cab,value_abs(AK(i,j)))) */
/* 	  if value_gt(cab,cmax) value_assign(cmax,cab); */
/*     {fprintf(FTRACE,"COEFFICIENT MAXIMUM=");fprint_Value(FTRACE,cmax);fprintf(FTRACE," \n");} */
/*     return; */
/*   }    */
/*   wvoir(XX); /\*i0*\/ */
/* } */
static void vzemptycol(Pproblem XX,int j)
{ fprintf(FTRACE,"empty column, variable $"); symbol(XX,B(j));
  fprintf(FTRACE,"$\\\\\n");
}
static void vzredundant(Pproblem XX,int i,int i2)
{ if (VISU>=ZVS) vznowstep(XX);
  if(VISU>=ZVR1)
    {
  if(i2<0) fprintf(FTRACE," empty inequality $y_{%d}$\\\\",G(i));
  else if(i2==0) fprintf(FTRACE," useless inequality $y_{%d}$\\\\",G(i));
  else fprintf(FTRACE,
     " redundant inequality $y_{%d}$ compared with $y_{%d}$\\\\\n",G(i),G(i2));
  if(VISU>=ZVR4) igvoir3(XX,1,MX,0,0,0,0,0,0,1,i);
    }
}
static void vzgcd(Pproblem XX,int i, int gcd, int viz)//DN int gcd or Value gcd???
{ if (VISU>=viz) fprintf(FTRACE,
    "* Polyedron proved empty by GCD test on ligne%3d gcd=%2d\\\\\n",i,gcd);
  if (VISU>=viz+2) wvoir(XX);
}
static void vzextgcd(Pproblem XX,int a1,int a2,int u1,int u2,int u3,
		     int zvx)
{ if (VISU<zvx) return;
  fprintf(FTRACE,"Let us compute vector ($\\alpha_1, \\alpha_2, \\alpha_3$)\
 such that $%d \\alpha_1+ %d \\alpha_2=\\alpha_3=gcd(%d,%d)$\\\\\n",
 a1,a2,a1,a2);
 fprintf(FTRACE,"The Extended Euclid Algorithm gives:\
 $\\alpha_1=%d$, $\\alpha_2=%d$, $\\alpha_3=%d$\\\\\n",u1,u2,u3);
 fprintf(FTRACE,"Using values $\\alpha_1$,\
 $\\alpha_2$, $%d/\\alpha_3$, $%d/\\alpha_3$, \n",a2,a1);
}
//DNstatic void term(Pproblem XX,int s,int k,int x) /*******************/
//{ fprintf(FTRACE,"	&") ; if (!k) return;
//  if (k<0) fprintf(FTRACE,"-"); else if (s) fprintf(FTRACE,"+");
//  if (abs(k)!=1) fprintf(FTRACE,"%d",abs(k)) ; symbol(XX,x);
//}
static void term(Pproblem XX,int s,Value k,int x) /*******************/
{ fprintf(FTRACE,"	&") ; if value_zero_p(k) return;
  if value_neg_p(k) fprintf(FTRACE,"-"); else if (s) fprintf(FTRACE,"+");
  if value_notone_p(value_abs(k)) fprint_Value(FTRACE,value_abs(k)) ; symbol(XX,x);
}
//DNstatic void ucformula(Pproblem XX,int u1,int u2,int v,int new1,int new2)
//{ symbol(XX,v); fprintf(FTRACE,"=");
//  term(XX,0,u1,new1); term(XX,1,u2,new2); fprintf(FTRACE,"\\\\\n");
//}
static void ucformula(Pproblem XX,Value u1,Value u2,int v,int new1,int new2)
{ symbol(XX,v); fprintf(FTRACE,"=");
  term(XX,0,u1,new1); term(XX,1,u2,new2); fprintf(FTRACE,"\\\\\n");
}
//DN static void vzunimod(Pproblem XX, int u1,int u2,int u3,
//		     int za,int zb, int old1,int old2,int new1,int new2)
static void vzunimod(Pproblem XX, Value u1,Value u2,int u3,
		     int za,int zb, int old1,int old2,int new1,int new2)
{ if (VISU>=ZVU1+2)
    { if (VISU<ZVU1+3)
        { if (VDUM++) fprintf(FTRACE,"As in a previous step,\n");
          else fprintf(FTRACE,"After applying the Extended Euclid Algorithm \
to the coefficients of both variables,\n");
	}
      fprintf(FTRACE," we can perform a unimodular change:\n");
      fprintf(FTRACE,"\\[ \\left\\{  \\begin{array}{rrrr}  \n");
      ucformula(XX,u1,-zb/u3,old1,new1,new2);
      ucformula(XX,u2,za/u3,old2,new1,new2);
      fprintf(FTRACE,"\\end{array}  \\right.   \\]\n");
    }
  else if (VISU>=ZVU1)
    { fprintf(FTRACE,"unimodular change $");
      symbol(XX,old1); fprintf(FTRACE,", "); symbol(XX,old2);
      fprintf(FTRACE," \\Rightarrow ");
      symbol(XX,new1); fprintf(FTRACE,", ");
      symbol(XX,new2); fprintf(FTRACE,"$\\\\\n");
    }
  if (VISU>=ZVU1+5) wvoir(XX);
}
/* ======================================================================= */
/*                        FOR REAL SIMPLEX                                 */
/* ======================================================================= */
static int integrite(float vf) /* ***************************** */
{ int vi ; float vfp, epss; epss=0.00001; /*epss=0.000001;*/
  if (vf<0) vfp= -vf ; else vfp= vf ;
  vi=vfp+epss ;
  if (vfp-vi<epss) return(1) ; return(0) ;
}
static Value dessus(float vf) /* ***************************** */
{ Value borne; float vf2;
  if (integrite(vf)) vf2=vf-0.5; else vf2=vf;
  if (vf2<0) borne=vf2; else borne=vf2+1;
  return(borne);
}
static Value dessous(float vf) /* ***************************** */
{ return(-dessus(-vf));
}
/************************ in case of test
			  void testdessusdessous(Pproblem XX,float vf)*/
/*{ fprintf(FTRACE,"variable flottante=%14.7f dessus=%d dessous=%d\\\\\n",
	  vf,dessus(vf),dessous(vf));
  if (vf>0) testdessusdessous(XX,-vf);
}
void testsdessusdessous(Pproblem XX)
{ testdessusdessous(XX,0); testdessusdessous(XX,7.6543);
  testdessusdessous(XX,4.000002); testdessusdessous(XX,4.000001);
  testdessusdessous(XX,1.999995); testdessusdessous(XX,1.999999);
}
              in case of test - end *******************/
/************************* for real simplex ********************************/
static void integerrealline(Pproblem XX,struct rproblem *RR, int ii, int ir)
{ int j;
  RR->d[ir]=DK(ii); if (ir==0) RR->e[0]=XX->minimum ; else RR->e[ir]=1 ;
  for (j=1 ; j<=NX ; j++) RR->a[ir][j]=AK(ii,j);
  if (MSR>=7) RR->iname[NX+ir]=G(ii);
}
static void integertoreal(Pproblem XX,struct rproblem *RR,int mini)
{ int i,j;
  RR->nvar=NX; RR->mcontr=MX-IC1+1; RR->nvpos=RR->nvar; RR->copie=0;
  RR->ntrace=XX->ntrac3; RR->meth=0; if (PRECR) RR->testp=0; else RR->testp=3;
  RR->ftrace=FTRACE; RR->base=0; RR->cost=0; RR->rhs=0;
  for (i=IC1 ; i<=MX ; i++) integerrealline(XX,RR,i,i+1-IC1);
  if (ICOUT) integerrealline(XX,RR,ICOUT,0);
  else
    { RR->d[0]=0; if (mini==1) RR->e[0]=1 ; else 
	{RR->e[0]= -1 ;fprintf(FTRACE,"creation real mini %d\\\\\n",mini);
	}
      for (j=1 ; j<=NX ; j++) RR->a[0][j]=0;
    }
  if (mini<0) RR->e[0]= -1 ;
  RR->state=0; if (MSR>=7) for (j=1 ; j<=NX ; j++) RR->iname[j]=B(j);
}
static void messrsimplex(Pproblem XX,struct rproblem *RR,int r, int ph)
  { if (!r) {fprintf(FTRACE," REEL-PHASE %d:%13.6f \n",ph,RR->x[0]); return; }
  if(r==VRVID)fprintf(FTRACE,"(EMPTY)");
  else if(r==2)fprintf(FTRACE,"(INFINI)");
  else if(r==3)fprintf(FTRACE,"INSUF");
  else if(r==8)fprintf(FTRACE,"(EPSILON)");
 fprintf(FTRACE,"ANOMALIE PHASE %d REEL %d it=%d\\\\\n",ph,r,RR->iter);
}
static float localnewrsimplex(Pproblem XX,struct rproblem *RR,int min)
{ int rsr,rd=0; float rrd =0,rrp;
  if (MSR==0)
    { integertoreal(XX,RR,min); RR->meth=1; rd=realsimplex(RR); rrd=RR->x[0];
	  /*fprintf(FTRACE,"Real dual simplex x[0]=%13.6f\\\\\n",vpara.x[0]);*/
    }
  integertoreal(XX,RR,min); rsr=realsimplex(RR);
  if (rsr) messrsimplex(XX,RR,rsr,1); rrp=RR->x[0];
      /*if (VW6) fprintf(FTRACE,"REEL-PRIMAL:%13.6f \n",rrp);*/
  if (MSR==0)
    { if (rd!=rsr) fprintf(FTRACE,"DIVERGENCE DUAL:%d PRIMAL:%d\n",rd,rsr);
      else if (VW6) fprintf(FTRACE,
              "REEL DUAL:%13.6f PRIMAL:%13.6f ECART:%13.6f\n",rrd,rrp,rrd-rrp);
    }
  else if (VW6) fprintf(FTRACE,"REEL PRIMAL:%13.6f \n",rrp);
      /*XX->itemp=rsr; XX->ftemp=RR->x[0]; * revoir */
  return(RR->x[0]); 
}
static float localrsimplex(Pproblem XX,int min)
{ struct rproblem ARR; localnewrsimplex(XX,&ARR,min); return(ARR.x[0]); 
}
static void copybound(Pproblem XX,Pproblem YY)
{ int i; for (i=1; i<= XX->numax; i++) value_assign(YY->ibound[i],XX->ibound[i]);
         for (i=1; i<= XX->numax; i++) value_assign(YY->ilbound[i],XX->ilbound[i]);
}
static void vid(Pproblem XX)
{ XX->state= -1;
}
static void vidr(struct rproblem *RR)
{ RR->state= -1;
}
/*static int copystructure(Pproblem XX,Pproblem YY)*/
/*int copystructure(Pproblem XX,Pproblem YY)*/
static int copystructure(Pproblem XX,Pproblem YY)
{ int i,j;
  if (XX==YY) return(1); if (XX->state<0) return(2); YY->state=XX->state;
  YY->nvar=XX->nvar;    YY->mcontr=XX->mcontr;    YY->nvpos=XX->nvpos;
  YY->ftrace=XX->ftrace;   YY->dyn=XX->dyn;
  YY->ntrace=XX->ntrace;   YY->ntrac2=XX->ntrac2; YY->ntrac3=XX->ntrac3;
  YY->fourier=XX->fourier; YY->varc=XX->varc; YY->forcer=XX->forcer;
  YY->meth=XX->meth; YY->met2=XX->met2; YY->met3=XX->met3; YY->met4=XX->met4;
  YY->met5=XX->met5; YY->met6=XX->met6; YY->met7=XX->met7; YY->met8=XX->met8;
  YY->critermax=XX->critermax; YY->remove=XX->remove; YY->turbo=XX->turbo;
  YY->choixpiv=XX->choixpiv;  YY->choixprim=XX->choixprim;
  YY->negal=XX->negal;        YY->icout=XX->icout;
  YY->minimum=XX->minimum;
  YY->mx=XX->mx;              YY->nx=XX->nx;
  YY->tmax=XX->tmax;
  YY->numero=XX->numero;      YY->numax=XX->numax;   YY->lastfree=XX->lastfree;
  YY->niter=XX->niter;
  /*YY->result=XX->result;*/ /*YY->jinfini=XX->jinfini;*/ /*YY->nub=XX->nub;*/
  YY->ic1=XX->ic1;            YY->ntp=XX->ntp;
  YY->vdum=XX->vdum;
  for (i=1; i<= XX->mx; i++) { 
    for (j=1; j<= XX->nx; j++)
      value_assign(YY->ak[i][j],XX->ak[i][j]);
    value_assign(YY->dk[i],XX->dk[i]);
    YY->g[i]=XX->g[i];
  }
  for (j=1; j<= XX->nx; j++)
    value_assign(YY->b[j],XX->b[j]);
  if (XX->state) copybound(XX,YY); return(0);
}
/* ========================================================================= */
/*           PROCEDURES FOR INFORMATION ABOUT VARIABLES AND SYSTEM           */
/* ========================================================================= */
static int janus_egalite(Pproblem XX,int i) /*********** deplacer *****************/
{ return(i>MX-NEGAL);
}
/* static int costvariable(Pproblem XX) /\**************************\/ */
/* { return(NV+1); */
/* } */
static int freevariable(Pproblem XX,int v) /**************************/
{ return((v>NP && v<=NV) || v<0);
}
//DNstatic int freevariable(Pproblem XX,Value v) /**************************/
//{ return((value_gt(v,int_to_value(NP)) && value_le(v,int_to_value(NV)) || value_neg_p(v));
//}
static int cutvariable(Pproblem XX, int v) /**************************/
{ return(v>NUMAX);
}
static int fluidvariable(Pproblem XX, int v) /******* free variable */
{ return( freevariable(XX,v) || cutvariable(XX,v) ); /*    or cut variable */
}
static int cutline(Pproblem XX,int i) /*******************************/
{ return(cutvariable(XX,G(i))); /* is the constraint a cut inequality ? */
}
static int fluidline(Pproblem XX,int i) /*****************************/
{ return(fluidvariable(XX,G(i)));
}
/* static int cutcolumn(Pproblem XX,int j) /\*****************************\/ */
/* { return(cutvariable(XX,B(j))); */
/* } */
/* static int fluidcolumn(Pproblem XX, int j) /\**************************\/ */
/* { return(fluidvariable(XX,B(j))); */
/* } */
static int freecolumn(Pproblem XX,int j) /****************************/
{ return(freevariable(XX,B(j)));
}
//DNstatic int emptylhs(Pproblem XX,int i) /*****************************/
//{ int j; for (j=1; j<=NX; j++) if (AK(i,j)) return(0);
//  return(1);
//}
static int emptylhs(Pproblem XX,int i) /*****************************/
{ int j; for (j=1; j<=NX; j++) if value_notzero_p(AK(i,j)) return(0);//DN
  return(1);
}
static int useless(Pproblem XX,int i) /*****************************/
{ 
  int j,w;
  Value a;
  w=1;
  if value_neg_p(DK(i)) return(0);//DN
  for (j=1; j<=NX; j++)
    if (value_assign(a,AK(i,j)))
      {	if value_pos_p(a>0) return(0); if (cutvariable(XX,B(j))) w=2;
      }
  return(w);
}
static int presence(Pproblem XX,int iv)
{ int i,j; XX->cu=0; XX->lu=0;
  for (i=IC1 ; i<=MX ; i++) if (G(i)==iv) {XX->lu=i; return(-i);}
  for (j=1 ; j<=NX ; j++) if (B(j)==iv) return(XX->cu=j);
  return(0);
}
static int newfreevariable(Pproblem XX) /**************************/
{ return(--LASTFREE);
}
/* ======================================================================== */
/*                      PROCEDURES for data overflow test                   */
/* ======================================================================== */
//DNstatic int correctm(Pproblem XX,int k1,int k2)
//{ int k ; k=k1*k2 ; /*examen si multiplication termes non nuls sans overflow*/
//  if (k1>0 && k2>0 || k1<0 && k2<0)
//   { if (k>0) return(k); }
//  else if (k<0) return(k);
//  if (VISU>=ZVO)
//    { fprintf(FTRACE,"t:%4d k1=%12d,k2=%12d,k=%12d mult deb*\n",NITER,k1,k2,k);
//    }
//  return(0);
//}
//static int corrects(Pproblem XX,int k1,int k2) /*provisoire*/
//{ int k; k=k1+k2; /* examen si somme termes dont un non nul sans overflow */
//  if (k1>=0 && k2>=0 && k<=0 || k1<=0 && k2<=0 && k>=0)
//   { if(VISU>=ZVO)fprintf(FTRACE,"t:%4d k1=%12d,k2=%12d,k=%12d somm deb***\n",
//			 NITER,k1,k2,k); return(0);
//    }
//  return(1);
//}
static Value dn_multiply(Value v1, Value v2)
{
  Value v;  
  if (value_zero_p(v1) || value_zero_p(v2)) return(VALUE_ZERO);
  v = value_direct_multiply(v1,v2);
  if value_eq(v1,value_div(v,v2)) return(v);
  else {
    fprintf(stderr,"\nDNDNDN JANUS WARNING, multiplication overflow");
    assert(false);
    return VALUE_NAN;
  }
}
static Value correctm(Pproblem XX,Value k1,Value k2)
{ 
  Value k;
  if (value_notzero_p(k1) && value_notzero_p(k2)) {// must not be zero
    k = value_direct_multiply(k1,k2);
    if value_eq(k1,value_div(k,k2)) return(k);
  }
  if (VISU>=ZVO)
    {       
      fprintf(FTRACE,"\nDNDNDN JANUS WARNING: correctm overflow");
      assert(false);
    }
  return(0);//should check parameter notzero before use, so return zero means overflow
}
static int corrects(Pproblem XX,Value k1,Value k2)
{ 
  Value k;
  k=value_plus(k1,k2);
  if value_eq(k1,value_minus(k,k2)) return(1);
  if (VISU>=ZVO)
    { 
      assert(false);
      fprintf(FTRACE,"\nDNDNDN JANUS WARNING: corrects overflow");
    }
  return(0);//means overflow
}
/* ======================================================================== */
/*                            PIVOTING PROCEDURE                            */
/* ======================================================================== */
static int ipivotage2(Pproblem XX,int i1,int j1,int ipiv1,int ipiv2,
		      int vtu)
     /*	La matrice ak et les seconds membres dk sont entiers
	Le pivot est egal a 1 ou -1.
	le pivotage de la matrice s'effectue a partir de la ligne i1 et de
	la colonne j1. Les numeros de la variable d'ecart (basique) de la
	ligne i1, et de la variable independante (hors-base) de la colonne j1,
	contenus respectivement dans les tableaux g et b, sont permutes. Dans
	le cas de l'elimination d'une egalite, l'operation equivaut a chasser
	de la base une variable d'ecart identiquement nulle */
{ 
  int i,j; /* variables travail */
  Value pv,p2,p3,pd,pp,dn_tmp;
i = 0;j=0; 
value_assign(pv,VALUE_ZERO);value_assign(p2,VALUE_ZERO);value_assign(p3,VALUE_ZERO);
value_assign(pd,VALUE_ZERO);value_assign(pp,VALUE_ZERO);value_assign(dn_tmp,VALUE_ZERO);
/*fprintf(stderr,"%d %d %d %d %d \n",i1,j1,ipiv1,ipiv2,vtu);*///DNDB

  if (NITER >=TMAX) return(VRESULT=VRINS);
  if (DYN) dynam(XX,0);
  NITER +=1; VRESULT=VRFIN;
  if (vtu) value_assign(pv,XX->tturb[i1][j1]);
  else { value_assign(pv,AK(i1,j1)); j=B(j1); B(j1)=G(i1); G(i1)=j ; }//DN
  
/*fprintf(stderr,"%d %d ",i,j);fprint_Value(stderr,pv);fprintf(stderr," ");
fprint_Value(stderr,p2);fprintf(stderr," ");fprint_Value(stderr,p3);fprintf(stderr," ");
fprint_Value(stderr,pd);fprintf(stderr," ");fprint_Value(stderr,pp);fprintf(stderr," ");
fprint_Value(stderr,dn_tmp);fprintf(stderr," \n");*///DNDB

  if (value_notone_p(value_abs(pv)))
    { /*if(VISU)*/ 
      fprintf(FTRACE,"*** bug,pivot different +-1:");
      fprint_Value(FTRACE,pv);fprintf(FTRACE,"\n");
      return(VRESULT=VRBUG);
    }
  if (XX->turbo)
    { if (vtu)
	{ for (i=ipiv1; i<=ipiv2; i++)
	  if (value_assign(p2,AK(i,j1))) { //p2 <> 0
	      if value_neg_p(pv) value_oppose(p2);

	      //DK(i)-=XX->dturb[i1]*p2;
	      if value_notzero_p(XX->dturb[i1])
		 value_assign(dn_tmp,dn_multiply(XX->dturb[i1],p2));
	      else value_assign(dn_tmp,VALUE_ZERO);
	      value_substract(DK(i),dn_tmp);//DN

	      for (j=1; j<=NX; j++)
		if (j==j1) value_assign(AK(i,j),value_uminus(p2));
		else if (value_assign(p3,XX->tturb[i1][j])) {
		  //AK(i,j)-=p2*p3;
		  if (value_assign(dn_tmp,dn_multiply(p2,p3))) value_substract(AK(i,j),dn_tmp);//DN
		}
	  }
	return(VRESULT=0);
	}
      for (i=ipiv1; i<=ipiv2; i++)
	if (i != i1 && (value_assign(p2,AK(i,j1)))) { 
	  if (value_neg_p(pv)) value_oppose(p2); 
	  //DK(i)-=DK(i1)*p2 ;
	  if (value_assign(dn_tmp,dn_multiply(DK(i1),p2))) value_substract(DK(i1),dn_tmp);
	  for (j=1; j<=NX; j++)
	    if (j==j1) value_assign(AK(i,j),value_uminus(p2));
	    else if (value_assign(p3,AK(i1,j))) {
	      //AK(i,j)-=p2*p3;
	      if (value_assign(dn_tmp,dn_multiply(p2,p3))) value_substract(AK(i,j),dn_tmp);//DN
	    }
	}
      if (value_mone_p(pv))
	{ value_oppose(DK(i1));
	  for (j=1; j<=NX; j++)
	    if (j!=j1) value_oppose(AK(i1,j));
	}
    }
  else
    { for (i=ipiv1; i<=ipiv2; i++)
	if ((value_assign(p2,AK(i,j1))) && i != i1)
	  { if (value_neg_p(pv)) value_oppose(p2);
	  if (value_assign(pd,DK(i1))) {
	      if (value_assign(pp,dn_multiply(pd,p2)))
		{ if (corrects(XX,DK(i),pp)==0) return(VRESULT=VROVF); value_substract(DK(i),pp);
		}
	      else return(VRESULT=VROVF);
	  }
	    for (j=1 ; j<=NX ; j++) /*j0*/
	      if (j==j1) value_assign(AK(i,j),value_uminus(p2));
	      else if (value_notzero_p(AK(i1,j)))
		{ if (value_zero_p(value_assign(p3,correctm(XX,p2,AK(i1,j))))) return(VRESULT=VROVF);
		  if (corrects(XX,AK(i,j),value_uminus(p3))==0) return(VRESULT=VROVF);
		  value_substract(AK(i,j),p3);
		}
	  }
      if (value_mone_p(pv))
      { value_oppose(DK(i1));
	for (j=1 ; j<=NX ; j++) /*j0*/
	  if (j!=j1) value_oppose(AK(i1,j)) ;
      }
    }
  if (VISU>=ZVP1)
    { fprintf(FTRACE,"{\\bf iteration} %d: ",NITER);
      fprintf(FTRACE,"pivot(%3d,%3d), ",i1,j1);
      fprintf(FTRACE," variables $");
      if (janus_egalite(XX,i1)) fprintf(FTRACE,"e_{%d}",B(j1));
      else symbol(XX,B(j1));
      fprintf(FTRACE,"$ - $"); symbol(XX,G(i1)); fprintf(FTRACE,"$\\\\\n");
      if (VISU>=ZVP2) wvoir(XX) ;
    }
/*fprintf(stderr,"ketthucipivotage2 :i %dj %d pv",i,j);fprint_Value(stderr,pv);fprintf(stderr," p2");
fprint_Value(stderr,p2);fprintf(stderr," p3");fprint_Value(stderr,p3);fprintf(stderr," pd");
fprint_Value(stderr,pd);fprintf(stderr," pp");fprint_Value(stderr,pp);fprintf(stderr," dn_tmp");
fprint_Value(stderr,dn_tmp);fprintf(stderr," \n");*///DNDB
  return(VRESULT) ;
}
static int ipivotage(Pproblem XX,int i1,int j1)
{ return ipivotage2(XX,i1,j1,1,MX,0);
}
static int redundant(Pproblem XX,int i1,int i2) /**********************/
{ int j;
 Value a,s;
 value_assign(s,value_minus(DK(i1),DK(i2)));
  for (j=1; j<=NX; j++)
    if (value_assign(a,value_minus(AK(i2,j),AK(i1,j))))
      { if (cutvariable(XX,B(j))) return(0);
      if (value_notzero_p(s))
	  { if (value_pos_p(s)) { if (value_neg_p(a)) return(0); }
	    else if (value_pos_p(a)) return(0);
	  }
	else value_assign(s,a);
    }
  if (value_pos_p(s)) return(i1); else return(i2);
}
static int satisfait(Pproblem XX) /****** for a system including only */
{ int i;         /* inequalities, does a basic solution is obvious? */
  for (i=MX; i>=IC1; i--) if (value_neg_p(DK(i))) return(0);
  return(1) ;
}
/* ========================================================================= */
/*     PROCEDURE FOR REPLACING A CONSTRAINED VARIABLE BY A FREE VARIABLE     */
/* ========================================================================= */
static int makecolumnfree(Pproblem XX,int jx) /************************/
{ if (freecolumn(XX,jx)) return(0);
  if (cutvariable(XX,B(jx))) B(jx)= newfreevariable(XX);
  else
    { int jj; if (++MX >MAXMX) return(VRDEB); G(MX)= B(jx) ;
      value_assign(DK(MX),VALUE_ZERO); 
      for (jj=1 ; jj<=NX ; jj++) value_assign(AK(MX,jj),VALUE_ZERO); 
      value_assign(AK(MX,jx),VALUE_MONE);
      B(jx) = newfreevariable(XX);
      if (VISU>=ZVVF)
	{ fprintf(FTRACE,
      "constrained feature of column %2d is specified by an inequality:\n",jx);
	  w1voir(XX,MX);
	}
    }
  return(0);
}
/* ========================================================================= */
/*                     PROCEDURES FOR HANDLING MATRIX                        */
/* ========================================================================= */
static void celimination(Pproblem XX,int j1) /*************************/
{ if (j1 != NX)       /* column j1 is eliminated (and replaced by column NX) */
    { int i;
      B(j1)=B(NX) ;
      for (i=1 ; i<=MX ; i++) value_assign(AK(i,j1),AK(i,NX)) ; /*i0*/
    } ;
  --NX ;
}
static void emptycolelim(Pproblem XX,int j1) /*************************/
{ if (j1 != NX)   /* empty column j1 is eliminated and replaced by column NX */
    { int i;
      B(j1)=B(NX) ;
      for (i=1 ; i<=MX ; i++) /*i0*/
	if (value_notzero_p(AK(i,NX))) value_assign(AK(i,j1),AK(i,NX));
    }
  --NX ;
}
static void permutec(Pproblem XX,int j1, int j2) /*********************/
{ if (j1 != j2)                           /* columns j1 and j2 are permuted */
    { int i,dn_tmp;
    Value vi;
      dn_tmp=B(j1); B(j1)=B(j2) ; B(j2)=dn_tmp; 
      for (i=1 ; i<=MX ; i++) /*i0*/
	{ value_assign(vi,AK(i,j1)); value_assign(AK(i,j1),AK(i,j2)); value_assign(AK(i,j2),vi);
	}
    }
}
static void permutel(Pproblem XX,int i1, int i2) /*********************/
{ if (i1 != i2)                           /* lines i1 and i2 are permuted */
    { int j,v;
    Value vi;
      v=G(i1) ; G(i1)=G(i2) ; G(i2)=v ;
      value_assign(vi,DK(i1)); value_assign(DK(i1),DK(i2)); value_assign(DK(i2),vi);
      for (j=1 ; j<=NX ; j++) /*j0*/
	{ value_assign(vi,AK(i1,j)); value_assign(AK(i1,j),AK(i2,j)); value_assign(AK(i2,j),vi);
	}
    }
}
static void oppositeline(Pproblem XX,int io) /************/
{ int j ;
  value_oppose(DK(io));
  for (j=1; j<=NX; j++) value_oppose(AK(io,j)) ; /*j0*/
}
static void retirer(Pproblem XX,int i1) /******************************/
{ if (i1 != MX)              /* line i1 is removed (and replaced by line MX) */
    { int j ;
      value_assign(DK(i1),DK(MX)); G(i1)=G(MX);
      for (j=1; j<=NX; j++) value_assign(AK(i1,j),AK(MX,j)); /*j0*/
    }
  --MX ;
}
/* static void badretirerineq(Pproblem XX,int i1) /\***********************\/ */
/* { if (NEGAL) permutel(XX,i1,MX-NEGAL); retirer(XX,MX-NEGAL); */
/* } */
static void retirerineq(Pproblem XX,int i1) /***********************/
{ permutel(XX,i1,MX-NEGAL); retirer(XX,MX-NEGAL);
}
static void razcut(Pproblem XX) /******************************/
{ int i;
  for (i=MX; i>=IC1; i--)
    if (cutline(XX,i)) retirerineq(XX,i);
}
/* ======================================================================= */
/*                     PROCEDURES FOR FOURIER-MOTZKIN                      */
/* ======================================================================= */
static int possible(Pproblem XX,int j)
{ int i; /** Fourier-Motzkin should not spoil functions ********/
  for (i=1 ; i<IC1 ; i++)
    if (value_notzero_p(AK(i,j))) return(0);
  return(1) ;
}
static void fourier0(Pproblem XX, int j) /**** trivial case ***********/
{ int i ; /* all non-zero coefficients of a column have the same sign */
  if (VISU>=ZVS) vznowstep(XX);
  if (VISU>=ZVF1)
    { fprintf(FTRACE,"fourier-Motzkin 0 : variable $x_{%d}$ - inequalities:",
	      B(j)); /*vvv*/
      for (i=MX ; i>=1 ; i--)
	if (value_notzero_p(AK(i,j))) fprintf(FTRACE,"  $y_{%d}$ ",G(i)) ;
      if (VISU<ZVF3) fprintf(FTRACE,"\\\\\n") ;
    }
  for (i=MX ; i>=IC1 ; i--)
    if (value_notzero_p(AK(i,j))) retirer(XX,i) ;
  emptycolelim(XX,j) ;
  if (VISU>=ZVF3) wvoir(XX);
} /* fin fourier0 */
static void ajouter(Pproblem XX,Value sk, int ip1, int i) /**************/
{ int jk;
 Value vt,dn_tmp;
  if (value_one_p(sk))
    { for (jk=NX ; jk>=1 ; jk--) /*j0*/
	if ( value_assign(vt,AK(ip1,jk))) value_addto(AK(i,jk),vt);
      value_addto(DK(i),DK(ip1));
    }
  else
    { for (jk=NX ; jk>=1 ; jk--)	/* if (jk !=j) si j transmis */ /*j0*/
	if ( value_assign(vt,AK(ip1,jk))) {
	  if (value_notzero_p(sk)) value_assign(dn_tmp,dn_multiply(vt,sk));
	  else value_assign(dn_tmp,VALUE_ZERO);
	  value_addto(AK(i,jk), dn_tmp);
	}
    if (value_notzero_p(sk)&&value_notzero_p(DK(ip1))) value_assign(dn_tmp,dn_multiply(DK(ip1),sk));
    else value_assign(dn_tmp,VALUE_ZERO);
    value_addto(DK(i),dn_tmp);
    //DK(i)+= DK(ip1) *sk ;
    }
}
static void fourier1(Pproblem XX, int pn, int j, int ip1) /************/
     /* integer fourier elimination criterion and: */
     /* a single positive coefficient or a single negative coefficient */
{ int i,bj; 
 Value sk,vt;
 bj=B(j);
 if (VISU>=ZVS) vznowstep(XX);
  /*if (pn==1)*/
 if (value_one_p(value_assign(vt,AK(ip1,j))))
    { for (i=MX ; i>=IC1 ; i--)
	if (value_neg_p(value_assign(sk,AK(i,j)))) ajouter(XX,value_uminus(sk),ip1,i);/* ligne ip1 sur ligne i */
    }
  else
  for (i=MX ; i>=IC1 ; i--)
    if (value_pos_p(value_assign(sk,AK(i,j)))) ajouter(XX,sk,ip1,i);
  retirer(XX,ip1) ; emptycolelim(XX,j) ;
  if (VISU) {  
    if (VISU>=ZVF1) {
      fprintf(FTRACE,"fourier-Motzkin ");fprint_Value(FTRACE,vt);fprintf(FTRACE," variable $x_{%d}$\\\\\n",bj);}
    if (VISU>=ZVF3) wvoir(XX);
  }
} /* fin fourier1 */
static void fourier22(Pproblem XX, int j) /****************************/
     /* integer fourier elimination criterion and: */
     /* 2 positive coefficients and 2 negative coefficients */
{ int ip1=0,ip2=0,im1=0,im2=0,i,jk;
 Value sk,rp1,rp2=0,rm1,rm2=0,dp1,dp2,dm1,dm2,ap1,ap2,am1,am2;

  if (VISU>=ZVS) vznowstep(XX);
  rp1=0 ; rm1=0;
  for (i=MX ; i>=IC1 ; i--)
    if (value_assign(sk,AK(i,j))){
      if (value_pos_p(sk))
	{ if (value_zero_p(rp1))
	    { value_assign(rp1,sk) ; ip1=i;
	    }
	  else
	    { value_assign(rp2,sk) ; ip2=i;
	    }
	}
      else {
	if (value_zero_p(rm1))
	     { value_assign(rm1,sk) ; im1=i;
	     }
           else
	     { value_assign(rm2,sk) ; im2=i;
	     }
      }}
  if (VISU>=ZVF1) fprintf(FTRACE, /*vvv*/  
  "Fourier-Motzkin 2-2 variable $x_{%d}$ inequalities +: $y_{%d}$ $y_{%d}$ inequalities -: $y_{%d}$ $y_{%d}$\\\\\n",B(j),G(ip1),G(ip2),G(im1),G(im2));

  value_assign(dp1,DK(ip1)); value_assign(dp2,DK(ip2));
  value_assign(dm1,DK(im1)); value_assign(dm2,DK(im2));

  value_assign(DK(im1),value_minus(dn_multiply(rp1,dm1),dn_multiply(rm1,dp1)));
  value_assign(DK(im2),value_minus(dn_multiply(rp2,dm1),dn_multiply(rm1,dp2)));
  value_assign(DK(ip1),value_minus(dn_multiply(rp1,dm2),dn_multiply(rm2,dp1))); 
  value_assign(DK(ip2),value_minus(dn_multiply(rp2,dm2),dn_multiply(rm2,dp2)));
  //DK(im1) = rp1*dm1 - rm1*dp1; DK(im2) = rp2*dm1 - rm1*dp2;
  //DK(ip1) = rp1*dm2 - rm2*dp1; DK(ip2) = rp2*dm2 - rm2*dp2;

  for (jk=NX ; jk>=1 ; jk--) /*j0*/ { 
    value_assign(ap1,AK(ip1,jk)); value_assign(ap2,AK(ip2,jk));
    value_assign(am1,AK(im1,jk)); value_assign(am2,AK(im2,jk));

   value_assign(AK(im1,jk),value_minus(dn_multiply(rp1,am1),dn_multiply(rm1,ap1))); 
   value_assign(AK(im2,jk),value_minus(dn_multiply(rp2,am1),dn_multiply(rm1,ap2)));
   value_assign(AK(ip1,jk),value_minus(dn_multiply(rp1,am2),dn_multiply(rm2,ap1))); 
   value_assign(AK(ip2,jk),value_minus(dn_multiply(rp2,am2),dn_multiply(rm2,ap2)));
   //AK(im1,jk),rp1*am1 - rm1*ap1; AK(im2,jk) = rp2*am1 - rm1*ap2;
   //      AK(ip1,jk) = rp1*am2 - rm2*ap1; AK(ip2,jk) = rp2*am2 - rm2*ap2;
  }
  if (VISU>=ZVF2) wvoir(XX);
  emptycolelim(XX,j);
  if (VISU>=ZVF3) wvoir(XX);
} /* fin fourier22 */
/* ======================================================================= */
/*              PROCEDURES FOR UNIMODULAR CHANGES OF VARIABLES             */
/* ======================================================================= */
//DNstatic int pgcd2(int z1, int z2) /********** GCD of two numbers *************/
//{ int p,v,r; p=abs(z1); v=abs(z2);
//  while (v!=0)
//    { r=p-(p/v)*v; p=v; v=r;
//    }    /*fpprintf(FTRACE,"pgcd2(%3d,%3d)=%3d:\n",z1,z2,p);*/
//  return(p); /* retour 0 si z1==z2==0 */
//}
static Value pgcd2(Value z1, Value z2) /********** GCD of two numbers *************/
{ 
  Value p,v,r; 
  value_assign(p,value_abs(z1)); value_assign(v,value_abs(z2));
  while (value_notzero_p(v)) { 
    value_assign(r,value_minus(p,dn_multiply(value_div(p,v),v))); 
    value_assign(p,v); value_assign(v,r);
  } 
  return(p); /* retour 0 si z1==z2==0 */
}
/*static int extgcd(Pproblem XX,int a1,int a2,int *pu1,int *pu2,int *pu3,
		  int zvx)
{ int v1,v2,v3,t1,t2,t3,q;
  if (a1==0 || a2==0) return(1);
  *pu1=1 ; *pu2=0 ; *pu3=abs(a1) ;
  v1=0 ; v2=1 ; v3=abs(a2) ;
  while (v3!=0)
    { q=*pu3/v3 ;
      t1= *pu1 -v1*q ; t2= *pu2-v2*q ; t3=*pu3-v3*q ;
      *pu1=v1 ; *pu2=v2 ; *pu3=v3 ;
      v1=t1 ; v2=t2 ; v3=t3 ;
    }
  if (a1<0) *pu1= - *pu1 ; if (a2<0) *pu2= - *pu2 ;
  if(VISU>=zvx) vzextgcd(XX, a1, a2, *pu1, *pu2, *pu3,zvx);
  return(0);
}DN*/
static Value extgcd(Pproblem XX, Value a1, Value a2, Value *pu1,Value *pu2, Value *pu3, int zvx)
{ Value v1,v2,v3,t1,t2,t3,q;
  if (value_zero_p(a1) || value_zero_p(a2)) return(VALUE_ONE);
  value_assign(*pu1,VALUE_ONE); value_assign(*pu2,VALUE_ZERO); value_assign(*pu3,value_abs(a1));
  value_assign(v1,VALUE_ZERO); value_assign(v2,VALUE_ONE); value_assign(v3,value_abs(a2)) ;
  while (value_notzero_p(v3))
    { value_assign(q,value_div(*pu3,v3)) ;
      value_assign(t1,value_minus(*pu1,dn_multiply(v1,q))); 
      value_assign(t2,value_minus(*pu2,dn_multiply(v2,q))); 
      value_assign(t3,value_minus(*pu3,dn_multiply(v3,q)));
      value_assign(*pu1,v1); value_assign(*pu2,v2); value_assign(*pu3,v3);
      value_assign(v1,t1); value_assign(v2,t2); value_assign(v3,t3) ;
    }
  if (value_neg_p(a1)) value_oppose(*pu1) ; if (value_neg_p(a2)) value_oppose(*pu2) ;
  if(VISU>=zvx) vzextgcd(XX, a1, a2, *pu1, *pu2, *pu3,zvx);
  return(VALUE_ZERO);
}
static int jnsegment(Pproblem XX,int ik,int j1,int j2)
{ /*indique indice jk sur segment ligne ik tel que abs(coefficient)non nul, sinon renvoie -1 */
  int j ;    
  if (j1<=j2) { 
    for (j=j1 ; j<=j2 ; j++) if (value_abs(AK(ik,j))) return(j);
  }else {
    for (j=j1 ; j>=j2 ; j--) if (value_abs(AK(ik,j))) return(j);}
  return(-1);
}
static int jsegment(Pproblem XX,int ik,int j1,int j2,Value p)
{ /* indique indice jk sur la ligne ik tel que abs(coefficient)==p , sinon renvoie -1 */
  int j ;                 
  for (j=j1 ; j<=j2 ; j++) if (value_eq(value_abs(AK(ik,j)),p)) return(j);
  return(-1);
}
static Value pgcdsegment(Pproblem XX,int ik, int j1, int j2) /**********/
{ int j;
 Value aa,p = VALUE_ZERO;
  for (j=j1 ; j<=j2 ; j++)
    if (value_assign(aa,AK(ik,j)))
      if (value_one_p(value_assign(p,pgcd2(p,aa)))) break;
  return(p); /* return: 0 if all coefficients null - otherwise gcd */
}
static int changing2(Pproblem XX, int i1, int ja, int jb) /***********/
{ 
  int iz,old1,old2,new1,new2; 
  Value za,zb, u1,u2,u3;
  value_assign(za,AK(i1,ja)) ; value_assign(zb,AK(i1,jb));
  if (ja==jb) XBUG(7);
  old1=B(ja); old2=B(jb); new1=newfreevariable(XX); new2=newfreevariable(XX);

  if (VISU>=ZVU1+2) igvoir3(XX,1,MX,1,i1,ja,1,i1,jb,1,i1);
  else if (VISU>=ZVU1+1) igvoir3(XX,i1,i1,1,i1,ja,1,i1,jb,1,i1);

  if (extgcd(XX,za,zb,&u1,&u2,&u3,ZVU1+3)) XBUG(10); /*u3!=1 t97*/
//  fprintf(stderr,"\n(in changing2: ");//DNDB
  for (iz=MX ; iz>=1 ; iz--) { 
    Value aka,akb; value_assign(aka,AK(iz,ja)); value_assign(akb,AK(iz,jb));
    // AK(iz,ja)= u1*aka + u2*akb ;
    //AK(iz,jb)= (za*akb - zb*aka)/u3 ;
   
    value_assign(AK(iz,ja),value_plus(dn_multiply(u1,aka),dn_multiply(u2,akb))) ;
    value_assign(AK(iz,jb),value_div(value_minus(dn_multiply(za,akb),dn_multiply(zb,aka)),u3)) ;
  //  fprint_Value(stderr,AK(iz,ja));fprintf(stderr,"AKizja");//DNDB
  //  fprint_Value(stderr,AK(iz,jb));fprintf(stderr,"AKizjb");//DNDB
  }
//fprintf(stderr,"in changing2)\n");//DNDB
  B(ja)=new1; B(jb)=new2;
  if (VISU) vzunimod(XX,u1,u2,u3,za,zb, old1,old2,new1,new2);
  return(0); /* si pas d'incident */
}
static int build0(Pproblem XX, int i1,int jj,int *pj1) /***************/
{ int jn1=0,jn2=0 ;       /* A unitary coefficient is built on equation i1 */
  if (VISU>=ZVEV+1) fprintf(FTRACE,
     "Equation $e_{%d}$ does not include a unitary coefficient.\\\\\n",G(i1));
  if ((jn1=jnsegment(XX,i1,NX,jj))<0) XBUG(5);     /* jj,NX */
  while (value_notone_p(AK(i1,jn1))) { 
    jn2 = jnsegment(XX,i1,jn1-1,jj); 
    if (jn2<0) XBUG(6); /*jn1+1,NX*/
    if (makecolumnfree(XX,jn1)) XDEB(1); 
    if (makecolumnfree(XX,jn2)) XDEB(2);
    if ((VRESULT=changing2(XX,i1,jn1,jn2))) return(VRESULT);
  }
  if (VISU>=ZVEV+1) fprintf(FTRACE,"Now, a unitary coefficient is available for a gaussian elimination:\n");
  *pj1=jn1; return(0);
}
static int build1(Pproblem XX,int i1, int jj) /***********************/
{ int vj=0,jn1=0,jn2=0;/*unitary or single coefficient built on inequality i1*/
 Value delta= VALUE_ZERO;
  jn1=jnsegment(XX,i1,jj,NX);if (jn1<0) return(jn1);/*no non-zero coefficient*/
  permutec(XX,jj,jn1);
  value_assign(delta,pgcdsegment(XX,i1,jj,NX));
  if (VISU>=ZVNG) {fprintf(FTRACE,"Delta%3d=",i1);fprint_Value(FTRACE,delta);fprintf(FTRACE,"\\\\\n");}
  vj = jsegment(XX,i1,jj,NX,delta); /* recherche coefficient=delta */
  if (vj>=0) /*on sait que valeur absolue d'un coefficient=pgcd*/
    permutec(XX,jj,vj);
  else /* t98 2eme exemple RR1828 7.3.2 */
    while (value_ne(value_abs(AK(i1,jj)),delta))
      { jn2=jnsegment(XX,i1,jj+1,NX);
        if ((VRESULT=changing2(XX,i1,jj,jn2))) return(VRESULT);
      }
  return(0);
}
/* ======================================================================= */
/*                      PROCEDURES FOR CUT OPERATIONS                      */
/* ======================================================================= */
//DNstatic int cfloor(int v1, int v2) /********** v2 doit etre positif *********/
//{ if (v1==0) return(0); if (v1>0) return(v1/v2); return(((v1+1)/v2)-1);
//} /* division reelle v1/v2=cfloor+reste * cfloor:entier * 0<=reste<1 */
static Value cfloor(Value v1, Value v2) /********** v2 doit etre positif *********/
{ Value v;
//fprintf(stderr,"(v1=");fprint_Value(stderr,v1);fprintf(stderr," ");//DNDB
//fprintf(stderr,"v2=");fprint_Value(stderr,v2);fprintf(stderr," ");//DNDB
 if (value_zero_p(v1)) {/*fprintf(stderr,"cfloor VALUE_ZERO)");*/return(VALUE_ZERO); }//DNDB
 if (value_pos_p(v1)) {/*
fprintf(stderr,"cfloor_pos ");fprint_Value(stderr,value_div(v1,v2));fprintf(stderr,")");*///DNDB
return(value_div(v1,v2));}
 value_assign(v,value_div(value_plus(v1,VALUE_ONE),v2));
 value_decrement(v);
	/*fprintf(stderr,"cfloor_neg ");fprint_Value(stderr,v);fprintf(stderr,")");*///DNDB
 return(v);
}

static int newcut(Pproblem XX) /*********/
{ if (++MX >MAXMX) return(VRDEB); G(MX)= ++NUMERO ; return(0);
}
static void coupe(Pproblem XX, int i1, int i2, Value abpivot) /**********/
{int j2;
/* fprintf(stderr,"\n(in coupe: abpivot = ");fprint_Value(stderr,abpivot);fprintf(stderr,"");*///DNDB
 value_assign(DK(i2),cfloor(DK(i1),abpivot));
/* fprintf(stderr," DKi2 =");fprint_Value(stderr,DK(i2));fprintf(stderr,"\n");*///DNDB
 for (j2=1 ; j2<=NX ; j2++) /*j0*/ {
   value_assign(AK(i2,j2),cfloor(AK(i1,j2),abpivot));
/*   fprintf(stderr," AKi2j2 = ");fprint_Value(stderr,AK(i2,j2));fprintf(stderr,"\n");*///DNDB
}
// fprintf(stderr,"\nin coupe)\n");//DNDB
 if (VISU>=ZVC1) /*3*/
   { if (VISU>=ZVC1+1) fprintf(FTRACE,"Let's build at "); /*zvc2 4*/
      fprintf(FTRACE,"cut $"); symbol(XX,G(i2));
      if(i1==i2) {fprintf(FTRACE,"$ (by "); fprint_Value(FTRACE,abpivot);
       fprintf(FTRACE,") from surrogate inequality");}
      else {fprintf(FTRACE,"$ (by "); fprint_Value(FTRACE,abpivot);
      fprintf(FTRACE,") from source row $y_{%d}$",G(i1));}
      if (VISU>=ZVC1+3) w1voir(XX,i2); /*zvc4 6*/
      if (VISU>=ZVC1+2)  /*zvc3 5*/
	{ fprintf(FTRACE," as follows:"); vzcut(XX,i1,abpivot,i2);
	}
      else fprintf(FTRACE,"\\\\\n");  /*suppression annulee 13juin 00 */
    }
}
static int cutiteration(Pproblem XX, int i1, int j1, int turb)
     /******** coupe eventuelle + pivotage + retrait ************/
{ int r=0,ix=0;
 Value abpivot =VALUE_ZERO;
 if (value_notone_p(value_assign(abpivot,value_abs(AK(i1,j1))))) {
   if (value_zero_p(abpivot)) { NUB=13; return(VRBUG); }
      if (cutline(XX,i1)) /* surrogate cut*/
	{ coupe(XX,i1,i1,abpivot); ix=i1;
	}
      else
	{ if ((r=newcut(XX))) return(r); coupe(XX,i1,MX,abpivot); ix=MX;
	  if (PMETH==9) /* PPP9 */
	    {localrsimplex(XX,1);fprintf(FTRACE,"(apres cut,avant piv)\\\\\n");
	    }
	}
      if (VISU>=ZVCP1) igvoir3(XX,1,MX,1,i1,j1,1,ix,j1,1,i1);
    }
  else
    { ix=i1; if (VISU>=ZVCP1) igvoir3(XX,1,MX,1,i1,j1,0,0,0,0,0);
    }
  if (turb)
    { int tu,j; 
    tu= ++XX->nturb; XX->jturb[tu]=j1; value_assign(XX->dturb[tu],DK(ix));
      if (XX->nturb>29)
	{fprintf(FTRACE,"turbo insuffisant %d\\\\\n",XX->nturb);return(VROVF);}
      for (j=1; j<=NX; j++) value_assign(XX->tturb[tu][j],AK(ix,j));
      if ((r=ipivotage2(XX,ix,j1,IC1,MX,0))) return(r);
    }
  else if ((r=ipivotage(XX,ix,j1))) return(r);
  if (fluidline(XX,ix))
      { if (VISU>=ZVCP2) wvoir(XX); retirer(XX,ix);
      }
  if (VISU>=ZVCP3) wvoir(XX);
  return(0);
}
/* ======================================================================= */
/*                            INEQUALITY DIVISION                          */
/* ======================================================================= */
//DNstatic int pgcd2in(int z1, int z2) /********** GCD of two numbers ************/
//{ int p,v,r; p=abs(z1); v=abs(z2);
//  while (v!=0)
//    { r=p-(p/v)*v; p=v; v=r;
//    }    /*fpprintf(FTRACE,"pgcd2(%3d,%3d)=%3d:\n",z1,z2,p);*/
//  return(p); /* retour 0 si z1==z2==0 */
//}
static Value pgcd2in(Value z1, Value z2) /********** GCD of two numbers ************/
{ Value p,v,r; value_assign(p,value_abs(z1)); value_assign(v,value_abs(z2));
 while (value_notzero_p(v))
    { value_assign(r,value_minus(p,dn_multiply(value_div(p,v),v))); 
    value_assign(p,v);value_assign(v,r);
    }
  return(p); /* retour 0 si z1==z2==0 */
}
static Value pgcdseg(Pproblem XX,int ik, int j1, int j2) /**********/
{ int j; 
 Value aa, p;
 value_assign(p,VALUE_ZERO);
  for (j=j1 ; j<=j2 ; j++)
    if (value_assign(aa,AK(ik,j)))
      if (value_one_p(value_assign(p,pgcd2in(p,aa)))) break;
  return(p); /* return: 0 if all coefficients null - otherwise gcd */
}
static Value inequaldivision(Pproblem XX,int i)
{ int j;
 Value gcd;        /* return 0: all coefficients are null * otherwise gcd */
 if (value_zero_p(value_assign(gcd,pgcdseg(XX,i,1,NX)))) return(VALUE_ZERO);	/* is LHS empty? */
 else if (value_gt(gcd,VALUE_ONE))
   { Value dp ;
   if (VISU>=ZVS) vznowstep(XX);
   if (VISU>=ZVI1)
     { if (VISU>=4) wvoir(XX); /* 2000 */
     fprintf(FTRACE,"  line %d: inequality $",i); symbol(XX,G(i));
     {fprintf(FTRACE," $ divided by gcd=");fprint_Value(FTRACE,gcd);fprintf(FTRACE,"\n");}
     if (VISU<ZVI3) fprintf(FTRACE,"\\\\\n");
     }
   if (value_assign(dp,DK(i))) value_assign(DK(i),cfloor(dp,gcd));
   for (j=NX ; j>=1 ; j--) value_division(AK(i,j),gcd);         /* division by GCD */
   if (VISU>=ZVI3) w1voir(XX,i);
   }
 return(gcd);
}
/* ======================================================================= */
/*                                  GCD TEST                               */
/* ======================================================================= */
static int gcdtest(Pproblem XX,int i, int *pj1, Value *ppivot, int viz)
{ 
  int j;
  Value gcd; /* GCD test + division by GCD ** 0: fails * -1: empty polyedron */
  if (value_zero_p(value_assign(gcd,pgcdsegment(XX,i,1,NX))))/* is LHS empty? */ { 
    if (value_notzero_p(DK(i))) { 
      if (VISU) vzgcd(XX,i,0,viz); return(-1); /*1*/
    }
  } else if (value_gt(gcd,VALUE_ONE)) { 
    Value dp ;
    if (VISU>=viz+1) { fprintf(FTRACE,"* ligne%3d gcd=",i);fprint_Value(FTRACE,gcd);fprintf(FTRACE,"\\\\\n");} /*2*/    
    if (value_notzero_p(DK(i)))			/* GCD test */ { 
      value_assign(dp,value_div(DK(i),gcd));
      if (value_ne(dn_multiply(dp,gcd),DK(i))) { 
	if (VISU) vzgcd(XX,i,gcd,viz); return(-1);}
      value_assign(DK(i),dp);
    }
    for (j=NX; j>=1; j--) value_division(AK(i,j),gcd);         /* division by GCD */
    for (j=NX; j>=1; j--) if (value_one_p(value_abs(AK(i,j)))) { 
      *pj1=j ; value_assign(*ppivot,VALUE_ONE); break;	}
    if (VISU>=viz+1){
      fprintf(FTRACE,"Line %d was divided by gcd = ",i);fprint_Value(FTRACE,gcd);fprintf(FTRACE,"\\\\\n");}
  }
  return(0);
}
/* ======================================================================= */
/*                   BASIC ALL-INTEGER SIMPLEX PROCEDURES                  */
/* ======================================================================= */
static int dualentier(Pproblem XX) /********************************/
{ int i,j,i1=0,j1,boul2,nn,nvn;
 Value teta,tet,pivot=VALUE_ZERO;
  if (VISU>=ZVS)
    { fprintf(FTRACE,"{*} STEP DUAL ALL.I. ,NX=%3d,MX=%3d,t=%4d,CHOIXPIV=%d ",
	      NX,MX,NITER,CHOIXPIV);
      if (PMETH==MTDI2) fprintf(FTRACE," surrogate\\\\\n");
      else fprintf(FTRACE," simple\\\\\n");
    }
  XX->nturb=0; if (satisfait(XX)) return(VRFIN) ;
  for (;;)
    { int vg,r; vg=10000; value_assign(tet,VALUE_ZERO); nn=0 ;
      for (i=IC1 ; i<=MX ; i++)
	if (value_neg_p(value_assign(teta,DK(i))))
	  { boul2=1; ++nn ;
	    for (j=1 ; j<=NX ; j++)
	      if (value_neg_p(AK(i,j))) { boul2=0; break; }
	    if (boul2) { if (VISU>=ZVDEND) wvoir(XX); return(VRVID); }
	    if (CHOIXPIV==4) { if (vg>G(i)) { i1=i; value_assign(tet,teta); vg=G(i);} }
	    else if (value_lt(teta,tet)) { i1=i; value_assign(tet,teta); }
	  }
      if (value_zero_p(tet))
	{ if (XX->turbo>=3) if (XX->nturb)
	    { int tu;
	      for (tu=1 ; tu<=XX->nturb ; tu++)
	      if ((r=ipivotage2(XX,tu,XX->jturb[tu],1,IC1-1,1))) return(r);
	      XX->nturb=0;
	    }
	  if (VISU>=ZVDEND) wvoir(XX); return(VRFIN) ; /* available solution */
	}
      nvn=0;
      for (j=1 ; j<=NX ; j++) if (value_neg_p(AK(i1,j)))
	  { if (++nvn >1) break; j1=j ; value_assign(pivot, AK(i1,j)) ;
	  }
      if (nvn >1) /* more than one possible pivot */
	{ if (CHOIXPIV==2||CHOIXPIV==4)
	    { int comptj,compp; compp=0;
	      for (j=1 ; j<=NX ; j++) if (AK(i1,j)<0)
		  { comptj=0;
		    for (i=IC1; i<=MX; i++) if ( DK(i) < 0)
			if (AK(i,j)<0) ++comptj;
		    if (comptj>compp) { compp=comptj; j1=j; }
		  }
	    }
	  else
	  if (CHOIXPIV==3)
	    { int comptj,compp; compp=0;
	    for (j=1 ; j<=NX ; j++) if (value_neg_p(AK(i1,j)))
		  { comptj=0;
		    for (i=IC1 ; i<=MX ; i++)
		      if ((value_neg_p(DK(i))) && (value_neg_p(AK(i,j)))) ++comptj; /*> > */
		    if (comptj>compp) { compp=comptj; j1=j; }
		  }
	    }
	  else
	  if (CHOIXPIV)
	    { for (j=1 ; j<=NX ; j++) if (value_neg_p(AK(i1,j)))
		  if (value_lt(pivot,AK(i1,j))) value_assign(pivot, AK(i1,j1=j));
	    }
	  else for (j=1; j<=NX; j++) if (value_gt(pivot,AK(i1,j))) value_assign(pivot,AK(i1,j1=j));
	}
      if (nn>1 && PMETH==MTDI2) /************ construction surrogate */
	{ int mx1; if (VISU>=ZVDS) fprintf(FTRACE,
			"surrogate inequality from %d inequalities\n",nn);
	  mx1=MX ; if ((r=newcut(XX))) return(r); /*i1=MX;*/ 
	  value_assign(DK(MX),VALUE_ZERO); 
	  for (j=1 ; j<=NX ; j++) value_assign(AK(MX,j),VALUE_ZERO);
	  for (i=IC1 ; i<=mx1 ; i++)
	    if (value_neg_p(DK(i)))
	      { value_addto(DK(MX),DK(i)) ;
		for (j=1 ; j<=NX ; j++) value_addto(AK(MX,j),AK(i,j));
	      }
	  if (value_posz_p(DK(MX))) return(VROVF);
	  if (VISU>=ZVDS) igvoir(XX,MX,MX);
	  boul2=1; value_assign(pivot,VALUE_ZERO);  /*** choix pivot surrogate ***/
	  if (CHOIXPIV)
	    { for (j=1; j<=NX; j++) if (value_neg_p(AK(MX,j)))
	      if (boul2 || value_lt(pivot,AK(MX,j)))
		{ boul2=0 ; j1=j ; value_assign(pivot,AK(MX,j)) ;
		}
	    }
	  else for (j=1 ; j<=NX ; j++) if (value_neg_p(AK(MX,j)))
	    if (boul2 || value_gt(pivot,AK(MX,j)))
	      { boul2=0 ; j1=j ; value_assign(pivot, AK(MX,j)) ;
	      }
	  if (boul2) return(VRVID); i1=MX;
	}
      r=0;
      if (DYN)
	{ if (VISU) { fprintf(FTRACE,"Dual, constraints:"); wvoir(XX);}
	  r=dynam(XX,1);
	  if (VISU)
	    { if (r)fprintf(FTRACE,"sortie dynam Dual constraints: OUI\\\\\n");
	      else fprintf(FTRACE,"sortie dynam Dual constraints: non\\\\\n");
	    }
	}
      /*if (VISU) fprintf(FTRACE,"Etat sortie dyn=%d r=%d\\\\\n",DYN,r);*/
      if (DYN && r)
	{ if (VISU) fprintf(FTRACE,"unimodular change in DUAL\\\\\n");
	  if ((r=build0(XX,i1,1,&j1))) return(r);
	}
      if (XX->turbo>=2)
	{ if ((r=cutiteration(XX,i1,j1,1))) return(r);
	  if (XX->turbo==2)
	    { int tu; tu=XX->nturb;
	      if (tu!=1) {fprintf(FTRACE,"ARRET D URGENCE\\\\\n"); XBUG(37);}
	      if ((r=ipivotage2(XX,tu,XX->jturb[tu],1,IC1-1,1))) return(r);
	      XX->nturb=0;
	    }
	}
      else if ((r=cutiteration(XX,i1,j1,0))) return(r);
    }
} /* fin dualentier */ ;
/* ======================================================================= */
static void razfreq(Pproblem XX) /*****/
{ int i; for (i=1 ; i<=NUMAX ; i++) XX->frequency[i]=0;
}
/* static void voirfreq(Pproblem XX) /\*****\/ */
/* { int i; fprintf(FTRACE,"FREQUENCY:\\\\\n"); */
/*   for (i=1 ; i<=NUMAX ; i++) if (XX->frequency[i]) */
/*       fprintf(FTRACE,"frequence %d: %d\\\\\n",i,XX->frequency[i]); */
/* } */
static int iprimal(Pproblem XX,int minimum,int i2,int stopcout)
     /************ RUDIMENTARY PRIMAL ALL-INTEGER * cost line i2 ***********/
{ int r,i,j,i1=0,j1=0,i3=0,bool1,lastit;
  float c1,c2,alpha=0,zeta ; 
  Value pivot,lastdk,tet1,tet2;

  if (CRITERMAX) TMAX=(NX * CRITERMAX)+NITER;
  if (i2>=IC1 || !satisfait(XX)) { NUB=12; return(VRBUG);}
  if (VW6) fprintf(FTRACE,"Choix primal: %d (dual %d) iter %d $->$ %d",
	      CHOIXPRIM,CHOIXPIV,NITER,TMAX);
  if (VISU)
    { vzphase(XX,3);
      if (minimum) fprintf(FTRACE," MIN"); else fprintf(FTRACE," MAX");
      {fprintf(FTRACE,"IMIZATION function line %d (current value: ",i2);
      fprint_Value(FTRACE,value_uminus(DK(i2)));fprintf(FTRACE,")\\\\\n");}
      if (VISU>=ZVBR3) wvoir(XX) ;            /* encore provisoire */
    }
  if (NTP<NX) /* very particular case, some variables are unconstrained */
    { for (j=NX ; j>NTP ; j--)
      if (value_notzero_p(AK(i2,j)))
	  { for (i=IC1 ; i<= MX ; i++)
	      if (value_notzero_p(AK(i,j))) { NUB=14; return(VRBUG);}
	    XX->jinfini=j; return(VRINF);
	  }
    }
  value_assign(lastdk,DK(i2)); lastit=NITER; razfreq(XX);
  for (;;)
    { bool1=1; c1=0 ; value_assign(tet2,VALUE_ZERO);
      for (j=NX ; j>=1 ; j--)
	{ if (value_neg_p(value_assign(tet1,(minimum) ? AK(i2,j) : value_uminus(AK(i2,j)))))
	    { bool1=1;
	      for (i=IC1 ; i<= MX ; i++)
		if (value_pos_p(AK(i,j)))
		  { int selection;
		    zeta = VALUE_TO_FLOAT(value_div(DK(i), AK(i,j))) ;
		    if (CHOIXPRIM<=1) selection=(bool1||zeta<alpha);
		    else selection=(bool1 || ((zeta>=1.0)&&(zeta<alpha))
				  || ((zeta<1.0)&&(zeta>alpha))
				    || ((zeta<1.0)&&(alpha>=1.0)) );
		    if (selection)
		      { bool1=0; alpha=zeta ; i3=i ; value_assign(pivot, AK(i,j));
		      }
		  }
	      if (bool1) { XX->jinfini=j; return(VRINF);} /*solution infinie*/
	      if (CHOIXPRIM==1) if (alpha<1.0) alpha=0; /* 10/2/98 */
	      if (((c2=alpha*(VALUE_TO_FLOAT(tet1)))<c1) || (c2==c1) && (value_lt(tet1,tet2))) //float multiplied by Value
		{ c1=c2 ; j1=j ; i1=i3 ; value_assign(tet2,tet1) ;
		}
	    }
	}
      if (bool1) return(VRFIN);
      ++XX->frequency[G(i1)];
      if ((r=cutiteration(XX,i1,j1,0))) return(r); /* PPP9 */
      if (stopcout==9) return(stopcout);
      if (PMETH==9){localrsimplex(XX,1);fprintf(FTRACE,"(apres tou)\\\\\n");}
      if (value_ne(DK(i2),lastdk))	{ 
	if (VISU>=ZVPRI) {fprintf(FTRACE,"{\\bf iteration} %d improved cost value: ",NITER);
	fprint_Value(FTRACE,value_uminus(DK(i2)));fprintf(FTRACE,"\\\\\n");}
	value_assign(lastdk,DK(i2)); lastit=NITER; razfreq(XX);
      }
      if (stopcout && stopcout!=9)
	{ if (value_posz_p(DK(i2))) return(10);
	  for (i=i2+1; i< IC1 ; i++)
	    if (value_posz_p(DK(i)))
	      { razfreq(XX); return(11);
	      }
	}
    }
} /* fin iprimal ================================================ */
static void anonymouscopyline(Pproblem XX,int is, int id) /************/
{ int j ; if (is==id) return;   /* line is (source) is copied on line id */
  value_assign(DK(id), DK(is)) ;
  for (j=1 ; j<=NX ; j++) value_assign(AK(id,j), AK(is,j)); /*j0*/
}
static void copyline(Pproblem XX,int is, int id) /********************/
{ if (is==id) return;   /* line is (source) is copied on line id) */
  anonymouscopyline(XX,is,id) ; G(id)=G(is) ;
}
/* ========================= PROCEDURES FOR FAST =========================== */
static void inhibit(Pproblem XX, int in) /*********/
{ permutel(XX,in,IC1++);
}
static void remettre(Pproblem XX, int in) /*********/
{ permutel(XX,in,IC1-1); IC1--;
}
static int remise(Pproblem XX, int icf) /*********/
{ IC1=icf; retirer(XX,IC1);
}
static int fast(Pproblem XX,Pproblem ZZ) /*******************/
{ int icf,iopt,r,i,j,exicou;
  exicou=ICOUT; icf=IC1 ; newcut(XX) ; inhibit(XX,MX); G(icf)=0;
  for (i=IC1; i<=MX; i++) if (value_neg_p(DK(i))) inhibit(XX,i);
  while (icf<IC1-1)
    { iopt=0;
      if (PMETH<=3) /* PPP3 */
	{ value_assign(DK(icf),VALUE_ZERO);
	  for (j=1; j<=NX; j++) value_assign(AK(icf,j),VALUE_ZERO);
	  for (i=icf+1; i<IC1; i++)
	    { value_addto(DK(icf), DK(i));
	      for (j=1; j<=NX; j++) value_addto(AK(icf,j),AK(i,j));
	    }
	}
      else
	{ Value dkopt; value_assign(dkopt,VALUE_ZERO);
	  for (i=icf+1; i<IC1; i++) if (value_lt(DK(i),dkopt)) value_assign(dkopt,DK(iopt=i));
	  copyline(XX,iopt,icf);
	}
      ICOUT=icf;
      if(VW4){fprintf(FTRACE,"passage fast(IC1=%d), ",IC1);symbold(XX,G(icf));}
      if (PMET3) { if (copystructure(XX,ZZ)) XBUG(41); remise(ZZ,icf);}
      r=iprimal(XX,1,icf,1);
      if (VW4)
	{ if (r==VRINS) fprintf(FTRACE," (too many)");
	  fprintf(FTRACE," %d iterations\\\\\n",NITER);
	}
      if (r==VRINS) { remise(XX,icf); return(r);}
      if (r==VRINF)
	{ int ik;
	  fprintf(FTRACE,"ENTIER INFINI %d\\\\\n",XX->jinfini); wvoir(XX);
	  if (value_mone_p(AK(icf,XX->jinfini)))
	    { if (!iopt) XBUG(24);
	      remettre(XX,iopt); ik=IC1;
	    }
	  else ik=icf;
	  if ((r=cutiteration(XX,ik,XX->jinfini,0))) return(r); r=10;
	}
      ICOUT=exicou;
      if (value_neg_p(DK(icf)) && r!=11) /* no solution */
	{ remise(XX,icf);
	  if (VISU>=ZVSAT3) { fprintf(FTRACE," fast vide\n"); wvoir(XX); }
	  return(VRVID);
	}
      for (i=IC1-1 ; i>icf ; i--) if (value_posz_p(DK(i))) remettre(XX,i);
      if (VISU>=ZVSAT4) {fprintf(FTRACE,"STEP FAST IC1==%d\n",IC1); wvoir(XX);}
    }
  remise(XX,icf);
  if (VW2 || VISU>=ZVSAT4)
    { fprintf(FTRACE,"Feasibility proof (primal), cost = ");
    fprint_Value(FTRACE,value_uminus(DK(1)));fprintf(FTRACE,"\n");
      vzlast(XX); wvoir(XX);
    }
  if (!satisfait(XX)) { NUB=27; return(VRBUG); } return(VRFIN);
} /* ========================== fin fast =================================== */

static void anonymctol(Pproblem XX,int js, int id) /**********/
{ int j ;
  for (j=1 ; j<=NX ; j++) value_assign(AK(id,j),VALUE_ZERO) ; /*j0*/
  value_assign(AK(id,js),VALUE_MONE);value_assign(DK(id),VALUE_ZERO);
}
static void columntoline(Pproblem XX,int js, int id) /*****************/
{ anonymctol(XX,js,id) ; G(id)=B(js) ;
}
//DN static void boundline(Pproblem XX,int io,int bou) /************/
//{ oppositeline(XX,io); DK(io) += bou; if (io>MX-NEGAL) permutel(XX,io,MX-NEGAL);
//}
static void boundline(Pproblem XX,int io,Value bou) /************/
{ oppositeline(XX,io); value_addto(DK(io),bou); if (io>MX-NEGAL) permutel(XX,io,MX-NEGAL);
}
//DNstatic int addnonbasicbound(Pproblem XX,int jnb,int nbb,int nonfixe)
static int addnonbasicbound(Pproblem XX,int jnb,Value nbb,int nonfixe)
{ int r;
  if (value_zero_p(nbb)) /*if (nonfixe)*/
    { celimination (XX,jnb); return(0);
    }
  if ((r=newcut(XX))) return(r); anonymctol(XX,jnb,MX);boundline(XX,MX,nbb);
  if (!nonfixe) ++NEGAL; return(0);
}
//DNstatic int addedbound(Pproblem XX,int iv,int borne)//DN??? void or int???
static void addedbound(Pproblem XX,int iv,Value borne)//DN??? void or int???
{ int iy;
  for (iy=NX ; iy>=1 ; iy--) if (B(iy)==iv) addnonbasicbound(XX,iy,borne,1);
  for (iy=IC1 ; iy<=MX-NEGAL ; iy++) if (G(iy)==iv)
      { newcut(XX); anonymouscopyline(XX,iy,MX); boundline(XX,MX,borne);
      }
}
//DN static int modifyconstr(Pproblem XX,int jfixe,int ifixe,int k,int ineq)
static int modifyconstr(Pproblem XX,int jfixe,int ifixe,Value k,int ineq)
{ if (jfixe)
    { if (ineq)
	{ int v; TMAX +=1;
	  newcut(XX);v=G(MX);columntoline(XX,jfixe,MX);B(jfixe)=v;value_substract(DK(MX),k);
	  if (ipivotage(XX,MX,jfixe)) XBUG(81);
	  retirer(XX,MX); --NUMERO; if (VW5) wvoir(XX);
	} 
      else addnonbasicbound(XX,jfixe,k,0); /*k==0: celimination(XX,jfixe);*/
    }
  else
    { value_substract(DK(ifixe),k);
      if (ifixe<IC1) { permutel(XX,ifixe,--IC1); permutel(XX,IC1,MX-NEGAL);}
      else permutel(XX,ifixe,MX-NEGAL);
      if (ineq==0) ++NEGAL;
    }
  return(0);
}
static int findandmod(Pproblem XX,int v,Value k,int ineq)
{ 
  int c,l,r; c=l=0;
  if (!(r=presence(XX,v)))XBUG(82); if (r>0) c=r; else l= -r;
  if (VW4&&ineq) {fprintf(FTRACE,"increment y(%d)= ",v);fprint_Value(FTRACE,k);}
  if (modifyconstr(XX,c,l,k,ineq)) XBUG(90);
  if (VW5){if(c)fprintf(FTRACE,"(column)");fprintf(FTRACE," - variable %d",v);}
  return(0);
} 
/* ================================================================== */
static int hb(struct rproblem *RR, int iv)
{ int j; for (j=1; j<=RR->nb; j++) if (RR->iname[RR->b[j]]==iv) return(j);
  return(0);
}
static void copyrline(struct rproblem *RR,int is, int id) /************/
{ if (is != id)    /* line is (source) is copied on line id) */
    { int j ; RR->d[id]= RR->d[is] ;
      for (j=1 ; j<=RR->n ; j++) RR->a[id][j] = RR->a[is][j] ;
    }
}
static void rinverse(struct rproblem *RR,int ir) /************/
{ int j ;
  RR->d[ir]= - RR->d[ir] ;
  for (j=1 ; j<=RR->n ; j++) RR->a[ir][j] = - RR->a[ir][j] ;
}
static int recherche(struct rproblem *RR, int varia)
{ int i;
  for (i=1 ; i<=RR->m ; i++) if (RR->g[i]==varia) return(-i); /* line */
  for (i=1 ; i<=RR->n ; i++) if (RR->b[i]==varia) return(i); /* column */
  return(0);
}
static void razcoutr(struct rproblem *RR, int mini)
{ int j;
  RR->d[0]=0; if (mini==1) RR->e[0]=1 ; else RR->e[0]= -1 ;
  for (j=1 ; j<=RR->nvar ; j++) RR->a[0][j]=0;
}
static int preparephase2(Pproblem XX,struct rproblem *RR,int rvar,
			 int mini,int kost)
{ int ir;
  if (kost== 0) { if (rvar<=RR->n) ir= rvar; else ir=RR->n-rvar; }
  else ir=recherche(RR,rvar);
  if (ir<0) copyrline(RR,-ir,0);
  else { razcoutr(RR,mini); RR->a[0][ir] = -1 ; };
  if (mini==1) RR->e[0]=1 ;
  else { RR->e[0]= -1 ; if (kost== -1) rinverse(RR,0); }
  if (kost== -1) {
    if (ir<0) {
      RR->inf[0]=RR->inf[-ir]; *(RR->pconor)=*(RR->pconor+RR->g[-ir]);
    } else { 
      RR->inf[0]=0; *(RR->pconor)=*(RR->pconor+RR->b[ir]);
    }}
  RR->cost= kost; return(ir);
}
static int phase1(Pproblem XX,struct rproblem *RR)
{ int r; integertoreal(XX,RR,1); razcoutr(RR,1);
  if (MSR>=5) RR->meth=1; r=realsimplex(RR); RR->meth=0; return(r);/*?*/
}
static int phase2(Pproblem XX,struct rproblem *RR,int rvar,int mini)
{ int rsr;
  if (MSR<=2)
    { integertoreal(XX,RR,1); razcoutr(RR,1); if (MSR==1) RR->meth=1;
      preparephase2(XX,RR,rvar,mini,0);
    }
  else
    { if (MSR<=3) rsr=phase1(XX,RR); /* repetee pour verifications */
      if (preparephase2(XX,RR,rvar,mini,-1)==0) XBUG(51);/*introuvable*/
    }
  if ((rsr=realsimplex(RR))) if (VIS2) messrsimplex(XX,RR,rsr,2); return(rsr);
}
static int dealtvariable(Pproblem XX,int v) /*********************/
{ return((v<=NV && v>0));
}
static int inevariable(Pproblem XX,int v) /*********************/
{ return(v>NV+1 && v<NV+MC);
}
/* static void etatfonctions(Pproblem XX) */
/* { int iv; if (!REDON) return; fprintf(FTRACE,"FONCTIONS "); /\*sans info*\/ */
/*   for (iv=1; iv<=NUMAX ; iv++) if (inevariable(XX,iv)) */
/*     if (presence(XX,iv)) if (value_notzero_p(XX->ilbound[iv])) */
/*       {symbold(XX,iv);fprintf(FTRACE,"(");fprint_Value(FTRACE,XX->ilbound[iv]);fprintf(FTRACE,") "); */
/*       } */
/*   fprintf(FTRACE,"\\\\\n"); */
/* } */
static int majlowbounds(Pproblem XX,struct rproblem *RR,int ope)
{ int i,j,iv; if (MSR<6) return(0);
  if (ope==0) { 
    if(REDON)for(iv=1;iv<=NUMAX;iv++){
      value_assign(XX->ilbound[iv],VALUE_MONE);value_assign(XX->ibound[iv],100000);XX->utilb[iv]=-1;}
    else for (iv=1;iv<=NUMAX;iv++) if(dealtvariable(XX,iv))
      { value_assign(XX->ilbound[iv],VALUE_MONE); XX->utilb[iv]=-1; }  /*6mai1999*/
  }
  else if (MSR<7)
    { for (i=IC1 ; i<=MX ; i++) if (dealtvariable(XX,iv=G(i)))
	  if (recherche(RR,RR->nvar+i+1-IC1)>0) value_assign(XX->ilbound[iv],VALUE_ZERO);
      for (j=1 ; j<=NX ; j++) if (dealtvariable(XX,iv=B(j)))
	  if (recherche(RR,j)>0) value_assign(XX->ilbound[iv],VALUE_ZERO);
    }
  else { 
    for (j=1 ; j<=RR->nvar ; j++)
      if (dealtvariable(XX,iv=RR->iname[RR->b[j]])||REDON) { 
	if(inevariable(XX,iv)) {
	  if (value_pos_p(XX->ilbound[iv])) XBUG(85);
	  else if(value_neg_p(XX->ilbound[iv])&&VW5) fprintf(FTRACE,"%d:HB ",iv);
	}
	value_assign(XX->ilbound[iv],VALUE_ZERO);
      } /*etatfonctions(XX);*/
    if (MSR>=9&&XX->state>0)  /*6mai1999*/
      for (i=1; i<=RR->mcontr; i++)
	if (dealtvariable(XX,iv=RR->iname[RR->g[i]]))
	  if (XX->utilb[iv])
	    if (RR->x[RR->g[i]]>=XX->ibound[iv]) //comparison of float and Value. DN
	      { XX->utilb[iv]=0;
	        if(VW3) {fprintf(FTRACE,"y%d ancien ",iv);fprint_Value(FTRACE,XX->ibound[iv]);
		fprintf(FTRACE," actuel %13.6f\\\\\n",RR->x[RR->g[i]]);}
	      }
    }
  return(0);
}
/* static int cutbounds(Pproblem XX,struct rproblem *RR,int iv, int rvar,float epss) */
/* {  */
/*   float rrv; int r; */
/*   if (VW6) {fprintf(FTRACE,"bornes cut "); symbold(XX,iv);} */
/*   if ((r=phase2(XX,RR,rvar,1))) return(r); */
/*   rrv=RR->x[0]+epss; fprintf(FTRACE,"var [%d] max=%13.6f ",iv,rrv); */
/*   if (rrv<1) fprintf(FTRACE," {*}CTILT {*}\\\\\n"); */
/*   if ((r=phase2(XX,RR,rvar,-1))) return(r); */
/*   rrv=RR->x[0]-epss; fprintf(FTRACE," min=%13.6f\\\\\n",rrv); return(0); */
/* } */
static int upperbound(Pproblem XX,struct rproblem *RR,int iv,int rvar, float epss)
{ 
  float rrv; int r; Value dn_tmp;
  if (VW6) symbold(XX,iv);
  if(MSR>=7) if ((r=majlowbounds(XX,RR,1))) return(r);
  if (MSR>=9&&XX->state>0) {
    if(!XX->utilb[iv])
      {if (VW3) fprintf(FTRACE,"upper inutile y%d\\\\\n",iv); return(0);}
    else 
      { XX->utilb[iv]=0;if (VW3) fprintf(FTRACE,"upper effectif y%d\\\\\n",iv);
      }}
  if ((r=phase2(XX,RR,rvar,1))) return(r);
  if (VW6) messrsimplex(XX,RR,r,2);
  rrv=RR->x[0]+epss; XX->rbound[iv]=rrv;//float to Value.DN
  //r=XX->ibound[iv]; XX->ibound[iv]=dessous(rrv); 
  // XX->decrement += (r-XX->ibound[iv]);
  //r is used just like a temporary variable, replaced by dn_tmp
  value_assign(dn_tmp,XX->ibound[iv]); value_assign(XX->ibound[iv],dessous(rrv));
  value_addto(XX->decrement,value_minus(dn_tmp,XX->ibound[iv]));
  if(VW6) {
    if (value_lt(XX->ibound[iv],VALUE_ONE)) fprintf(FTRACE," {*} TILT {*}\\\\\n");
    else {fprintf(FTRACE," bound = ");fprint_Value(FTRACE,XX->ibound[iv]);fprintf(FTRACE,"\\\\\n");}}
  return(0);
}
static int lowbound(Pproblem XX,struct rproblem *RR,int iv,int rvar, int *first,float epss)
{ 
  Value range,r=VALUE_ZERO; float rrv;
  if(VW2&&inevariable(XX,iv)) fprintf(FTRACE,"examen fonc y%d\\\\\n",iv);
  if (MSR<6 || XX->ilbound[iv])
    { if (MSR>=7) if ((r=majlowbounds(XX,RR,1))) return(r);; /* remplacer par 7 */
      if ((r=phase2(XX,RR,rvar,-1))) return(r);
      rrv=RR->x[0]-epss; XX->rlbound[iv]=rrv; value_assign(XX->ilbound[iv],dessus(rrv));
      value_addto(XX->decrement,XX->ilbound[iv]);
      if(VW2&&inevariable(XX,iv)) fprintf(FTRACE,"calcul fonc y%d\\\\\n",iv);
      if(VW6||(VW2&&inevariable(XX,iv))) if (MSR>=6 && (value_zero_p(XX->ilbound[iv])))
	{symbold(XX,iv);fprintf(FTRACE," preuve basse par calcul\\\\\n"); };
    }
  rrv=XX->rbound[iv]; value_assign(range,value_minus(XX->ibound[iv],XX->ilbound[iv]));
  if(VW6) if (value_notzero_p(XX->ilbound[iv])){ 
    symbold(XX,iv); messrsimplex(XX,RR,r,2); 
    fprintf(FTRACE," lbound = ");fprint_Value(FTRACE,XX->ilbound[iv]);fprintf(FTRACE," bound = ");
    fprint_Value(FTRACE,XX->ibound[iv]);fprintf(FTRACE," RANGE ");fprint_Value(FTRACE,range);
    if (value_zero_p(range)) {fprintf(FTRACE," {*} TILT {*} ");fprint_Value(FTRACE,XX->ibound[iv]);fprintf(FTRACE,"\\\\\n");}
    else if (value_neg_p(range)) fprintf(FTRACE," $->$ {*}{*} EMPTY {*}{*}\\\\\n");
    else fprintf(FTRACE,"\\\\\n");
  }
  if (value_neg_p(range)) return(VRVID); if (value_zero_p(range)) ++XX->tilt;
  if (dealtvariable(XX,iv))
    if (*first||rrv<XX->rrbest) { XX->rrbest=rrv; XX->vbest=iv; *first=0;}
  return(0);
}
static Value boundssum(Pproblem XX)
{ 
  int i,j,v;
  Value s = VALUE_ZERO;
  for (i=IC1 ; i<=MX ; i++) if (dealtvariable(XX,v=G(i)))
    { value_addto(s,XX->ibound[v]); value_substract(s,XX->ilbound[v]); }
  for (j=1 ; j<=NX ; j++) if (dealtvariable(XX,v=B(j)))
    { value_addto(s,XX->ibound[v]); value_substract(s,XX->ilbound[v]); }
  return(s);
}
static int computebounds(Pproblem XX,struct rproblem *RR,int up)
{ 
  int i0,i,j,r,first; float epss; Value s1=VALUE_ZERO,s2=VALUE_ZERO;
  epss=0.0001; /*epss=0.000001;*/
  if ((r=phase1(XX,RR))) { if(r==VRVID&&VW4) fprintf(FTRACE,"immediate "); return(r); }
  if (VW2) value_assign(s1,boundssum(XX));
  value_assign(XX->decrement,VALUE_ZERO);XX->tilt=0;first=1;i0=RR->nvar+1-IC1; majlowbounds(XX,RR,0);
  if (up) { 
    for (i=IC1 ; i<=MX ; i++) if (dealtvariable(XX,G(i)))
      if ((r=upperbound(XX,RR,G(i),i0+i,epss))) return(r);
    for (j=1 ; j<=NX ; j++) if (dealtvariable(XX,B(j)))
      if ((r=upperbound(XX,RR,B(j),j,epss))) return(r);
  }
  for (i=IC1 ; i<=MX ; i++) if (dealtvariable(XX,G(i)))
    if ((r=lowbound(XX,RR,G(i),i0+i,&first,epss))) return(r);
  for (j=1 ; j<=NX ; j++) if (dealtvariable(XX,B(j)))
    if ((r=lowbound(XX,RR,B(j),j,&first,epss))) return(r);
  if (REDON) {
    for (i=IC1;i<=MX;i++) if(inevariable(XX,G(i)))
      if ((r=lowbound(XX,RR,G(i),i0+i,&first,epss))) return(r);
  }
  if (!XX->tilt) { 
    value_assign(XX->bestlow,XX->ilbound[XX->vbest]); value_assign(XX->bestup,XX->ibound[XX->vbest]);}
  if (VW2) value_assign(s2,boundssum(XX));
  if (VW3) {fprintf(FTRACE,"etat=%d s1=",XX->state);fprint_Value(FTRACE,s1);fprintf(FTRACE," s2=");
  fprint_Value(FTRACE,s2);fprintf(FTRACE," diff=");fprint_Value(FTRACE,value_minus(s1,s2));
  fprintf(FTRACE," decrement=");fprint_Value(FTRACE,XX->decrement);fprintf(FTRACE,"\\\\\n");}
  if (PMET2>=66) if (value_zero_p(XX->decrement)) XBUG(93);
  XX->state=1; return(0); /*xs*/
}
static void voirbornes(Pproblem XX)
{ 
  int iv;
  for (iv=1; iv<=NUMAX ; iv++) 
    if (dealtvariable(XX,iv)) {
      if (presence(XX,iv)) {
	symbold(XX,iv); fprintf(FTRACE,"(");
	if (value_notzero_p(XX->ilbound[iv])) {fprint_Value(FTRACE,XX->ilbound[iv]);fprintf(FTRACE," - ");}
	fprint_Value(FTRACE,XX->ibound[iv]);fprintf(FTRACE,") ");
      } else fprintf(FTRACE," .... ");}
  if (REDON){ 
    fprintf(FTRACE,"redondances: ");
    for (iv=1; iv<=NUMAX ; iv++) if (inevariable(XX,iv)) if (presence(XX,iv)) 
      if (value_notzero_p(XX->ilbound[iv])) {
	symbold(XX,iv); fprintf(FTRACE,"(");fprint_Value(FTRACE,XX->ilbound[iv]);fprintf(FTRACE,") - ");
      }
  }
  fprintf(FTRACE,"\\\\\n"); /*retour*/
}
static void listehb(Pproblem XX,struct rproblem *RR)
{ 
  int iv,c;
  for (iv=1; iv<=NUMAX ; iv++) if (dealtvariable(XX,iv)) if ((c=hb(RR,iv)))
    if (value_le(XX->ibound[iv],VALUE_ONE)) {fprintf(FTRACE," (%d):",c); symbold(XX,iv);}
  fprintf(FTRACE," hors-base possibles\\\\\n");
}
static int choose(Pproblem XX,struct rproblem *RR)
{ 
  int i,j,iv,prem; Value bes=VALUE_ZERO;
  if (XX->state<0) XBUG(61);if (XX->state==0) XBUG(62);
  prem=1;XX->tilt=0;
  for (i=IC1 ; i<=MX ; i++) if (dealtvariable(XX,iv=G(i)))
    if (prem||(value_lt(XX->ibound[iv],bes))) {
      value_assign(bes,XX->ibound[iv]); XX->vbest=iv; prem=0;}//bes initialized by prem = true :-(
  for (j=1; j<=NX; j++) if (dealtvariable(XX,iv=B(j)))
    if (prem||(value_lt(XX->ibound[iv],bes))||((MSR>=8)&&(RR->state>=0)&&(value_eq(XX->ibound[iv],bes))&&(hb(RR,iv))))
      {value_assign(bes,XX->ibound[iv]);XX->vbest=iv;prem=0;}
  value_assign(XX->bestlow,XX->ilbound[XX->vbest]); value_assign(XX->bestup,XX->ibound[XX->vbest]);
  if (prem) XBUG(63);
  if (VW4&&(value_gt(XX->bestup,int_to_value(PCOMP))))  /*if (VIS2&&XX->bestup>=2)*/ { //DN. int_to_value
    symbold(XX,XX->vbest);fprintf(FTRACE,"DILATATION = ");fprint_Value(FTRACE,XX->bestup); voirbornes(XX);}
  return(0);
}
static int ajout(Pproblem XX)
{ 
  int iv,i; Value bi;
  if (REDON)
    for (i=MX;i>=IC1;i--) if (inevariable(XX,iv=G(i))) 
      if (XX->ilbound[iv]) {
	if(VW4) fprintf(FTRACE,"retrait inequa %d\\\\\n",iv);
	retirerineq(XX,i);
      }
  if (PMET2<7) return(0); razcut(XX);
  for (iv=1 ; iv<=NUMAX ; iv++) { if (dealtvariable(XX,iv)) {
    if (value_pos_p(value_assign(bi,int_to_value(XX->ibound[iv])))) addedbound(XX,iv,bi);
    else {
      if (value_neg_p(bi)) XBUG(72); 
      else {if(VW6) symbold(XX,iv);}
    }}
  }
  if(VW7) wvoir(XX); return(0);
}
static int majuu(Pproblem XX,Pproblem UU,struct rproblem *RR)
{ int iv,ns; Value bl; ns=0;
  for (iv=1; iv<=NUMAX; iv++) 
    if (dealtvariable(XX,iv)) { 
      if (value_pos_p(value_assign(bl,XX->ilbound[iv]))) { 
	if (VW4||(VW3&&MSR>=8)) {fprintf(FTRACE,"majuu, y%d=",iv);fprint_Value(FTRACE,bl);fprintf(FTRACE,"\\\\\n");}
	if (findandmod(UU,iv,bl,1)) XBUG(95); vidr(RR);
      }
      value_assign(UU->ibound[iv],value_minus(XX->ibound[iv],XX->ilbound[iv]));
      value_assign(UU->ilbound[iv],VALUE_ZERO);
      if (value_ge(UU->ibound[iv],2)) ++ns;
    }
  return(0);
}
static int majb(Pproblem XX,Pproblem UU,struct rproblem *RR)
{ 
  if (PMET2>=3) {/*2MMM3*/
    if (majuu(XX,UU,RR)) XBUG(97); UU->state=1; /*xs*/
    if(VW4) { fprintf(FTRACE,"apres majuu: "); voirbornes(UU);}
    if (satisfait(UU)) { if (VW4) wvoir(UU); return(99); } /*(VRFIN)*/
  } else { copybound(XX,UU); UU->state=1;} /*xs*/
  return(0);
}
static int failuresp(Pproblem XX,struct rproblem *RR,int *compb)
{ 
  int r,up; *compb=1; XX->bloquer=0; up=1;
  if (VW7) {fprintf(FTRACE," failure:\n"); wvoir(XX); }
  if (PMET2>=66) XX->bloquer=1; 
  else if (PCOMP&&XX->state>0) { 
    if ((r=choose(XX,RR))) return(r); 
    if (XX->tilt) XBUG(67);
    if ((PCOMP==9)||(value_le(XX->bestup,VALUE_ONE))) *compb=0; /*XX->bestup<=PCOMP*/
    else if (PCOMP==2) up=0;
  }
  if (!(*compb)) { 
    if (MSR>=8&&RR->state==0&&RR->vc){ 
      if (VW3) fprintf(FTRACE,"SIMULER y%d col%d\\\\\n",RR->namev,RR->vc);
      if (!RR->vc) XBUG(84); if (RR->vc!=hb(RR,RR->namev)) XBUG(87);
      elimination(RR,RR->vc);
    } else if ((r=phase1(XX,RR))) { 
      if (r==VRVID&&VW4) fprintf(FTRACE,"qimmediate "); return(r);}
    if ((r=choose(XX,RR))) return(r); 
    XX->state++; /*xs*/
    if (VW3&&MSR>=8) {
      if (hb(RR,XX->vbest)) fprintf(FTRACE,"HB:%d\\\\\n",XX->vbest);
      else fprintf(FTRACE,"echec HB:%d\\\\\n",XX->vbest);}
  }
  else { 
    if (VW3&&MSR>=8) fprintf(FTRACE,"COMPUTE TOTAL %d\\\\\n",XX->state);
    if ((r=computebounds(XX,RR,up))) return(r);
    if (VW4) {fprintf(FTRACE,"apres compute:"); voirbornes(XX);}
  }
  return(0);
}
static int reducedpb(Pproblem XX,Pproblem VV,struct rproblem *RR)
{ 
  ++VV->repere;
  if (VW2) { 
    int tete,i;  
    if (NV>10) tete=NV-NX; else tete=10-NX;
    for (i=1; i<=tete; i++) fprintf(FTRACE,"---");
    fprintf(FTRACE,"(%d Variables) REDUCED PROBLEM %d equa ",NX,NEGAL);
    for (i=MX+1-NEGAL; i<=MX; i++) symbold(XX,G(i));
    if (VW4) fprintf(FTRACE," repere %d\\\\\n",VV->repere);
    else fprintf(FTRACE,"\\\\\n"); if (VW5) wvoir(XX);
  }
  ICOUT=0; VRESULT=is2(XX,VV,RR,1) ;
  if (VIS2) if ((VRESULT!=VRFIN) && (VRESULT!=VRVID))
    fprintf(FTRACE,"DEAD reduced pb NX=%d NEGAL=%d vr=%d\\\\\n",NX,NEGAL,VRESULT);
  return(VRESULT) ;
}
static int splitpb(Pproblem XX,Pproblem VV,struct rproblem *RR, int jfixe,int ifixe, Value k,int ineq)
{ 
  int vhb; ++VV->repere;
  if (VW2) { 
    int tete,i,va;  
    if (NV>10) tete = NV - NX; else tete = 10 - NX;
    for (i=1; i<=tete; i++) fprintf(FTRACE,"---");
    fprintf(FTRACE,"{.} (%d Variables) SPLIT PROBLEM ",NX);
    if (VW4) fprintf(FTRACE," repere %d ",VV->repere);
    if (jfixe) va = B(jfixe) ; else va = G(ifixe); fprintf(FTRACE,"$");symbol(XX,va);
    if (ineq) {fprintf(FTRACE,", cost <= ");fprint_Value(FTRACE,value_uminus(k));fprintf(FTRACE,"$\\\\\n");}
    else fprintf(FTRACE," = ");fprint_Value(FTRACE,k);fprintf(FTRACE,"$\\\\\n");
  }
  if (ineq) { 
    copyline(XX,1,++MX); if (modifyconstr(XX,jfixe,MX,k,ineq)) XBUG(91); 
  } else if (modifyconstr(XX,jfixe,ifixe,k,ineq)) XBUG(92);
  if (VW4&&(ineq||VW6)) wvoir(XX);
  if (MSR>=8&&RR->state==0) { 
    if ((vhb=hb(RR,XX->vbest))) {
      if (RR->vc!=hb(RR,RR->namev)) XBUG(88);if (XX->vbest!=RR->namev) XBUG(89);
    } else if (VW3) fprintf(FTRACE,"NON, car basic:%d\\\\\n",XX->vbest);
  }
  ICOUT=0; VRESULT=is2(XX,VV,RR,1) ;
  if (VIS2) if (VRESULT!=VRFIN && VRESULT!=VRVID) {
    fprintf(FTRACE,"DEAD split pb NX=%d vr=%d k=",NX,VRESULT);fprint_Value(FTRACE,k);fprintf(FTRACE,"\\\\\n");}
  return(VRESULT) ;
}
static int recup(Pproblem XX,Pproblem VV) /*2MMM4*/
{ if (PMET2>=4) if (VV->marque==0)
     { if (copystructure(XX,VV)) XBUG(43); VV->marque=1;}
  return(0);
}
static int reduction(Pproblem XX,struct rproblem *RR)
{ 
  int iv; Value k; vidr(RR);
  for (iv=1; iv<= XX->numax; iv++) if (dealtvariable(XX,iv)) 
    if (value_eq((value_assign(k,XX->ibound[iv])),XX->ilbound[iv]))
      if (findandmod(XX,iv,k,0)) XBUG(98);
  return(ajout(XX));
}
static int eclater(Pproblem XX,Pproblem UU,Pproblem VV, struct rproblem *RR)
{ 
  int r,ivb; Value k; struct problem AYY; vid(&AYY);     /*ssss*/
  if ((r=recup(UU,VV))) return(r);
  if ((r=choose(UU,RR))) return(r); ivb=UU->vbest;
  if (value_notzero_p(UU->ilbound[ivb])) XBUG(54);
  for (value_assign(k,UU->ilbound[ivb]); value_le(k,UU->ibound[ivb]); value_increment(k)) { 
    if (copystructure(UU,&AYY)) XBUG(44); AYY.vbest=ivb;
    if ((r=ajout(&AYY))) return(r); if (value_notzero_p(k)) vidr(RR);
    RR->namev=ivb; RR->vc = hb(RR,ivb);
    if (!presence(&AYY,ivb)) XBUG(53); r=splitpb(&AYY,VV,RR,AYY.cu,AYY.lu,k,0);
    if (r==VRFIN) value_assign(XX->dk[1],AYY.dk[1]); if (r!=VRVID) return(r); }
  if (VW4) fprintf(FTRACE,"INDIRECT PROOF NX=%d\\\\\n",NX);
  return(VRVID);
}
static int failure(Pproblem XX,Pproblem UU,Pproblem VV, struct rproblem *RR)
{ 
  int r,compb;
  if ((r=failuresp(XX,RR,&compb))) { 
    if (r!=VRVID) XBUG(99); if (VW4) fprintf(FTRACE,"real proof\\\\\n");
    return(r);}
  if (compb) if ((r=majb(XX,UU,RR))) {
    if (r!=99) XBUG(96); 
    else { 
      value_assign(XX->dk[1],UU->dk[1]); 
      if ((r=recup(UU,VV))) XBUG(94); 
      if (VW4) fprintf(FTRACE,"integ sol\\\\\n"); 
      return(VRFIN);
    }}
  if (VW3&&MSR>=8) {fprintf(FTRACE,"U: "); listehb(UU,RR);}
  if (XX->tilt||XX->bloquer) { 
    if ((r=reduction(UU,RR))) return(r);
    if ((r=reducedpb(UU,VV,RR))==VRFIN) { 
      if (VW4&&XX->bloquer) fprintf(FTRACE,"solution bloquee\\\\\n");
      value_assign(XX->dk[1],UU->dk[1]);
    }
    return(r);
  } else return(eclater(XX,UU,VV,RR));
}
static int fastplus(Pproblem XX,Pproblem VV,struct rproblem *RR)
{ 
  int r; struct problem AUU; vid(&AUU);     /*ssss*/
  if (PMETH>=8) return(failure(XX,XX,VV,RR)); /*PPP8 mise au point*/
  if (copystructure(XX,&AUU)) XBUG(45); /*revoir utilite*/
  if (PMETH>=7) return(failure(XX,&AUU,VV,RR)); /*PPP7*/
  NITER=0; if (CRITERMAX) TMAX=(NX * CRITERMAX)+NITER;
  if (PMETH>=6) /*PPP6*/ { 
    if ((r=dualentier(XX))==VRINS) return(failure(XX,&AUU,VV,RR));
    if (VW2) if (r==VRFIN){ 
      fprintf(FTRACE,"Feasibility proof (dual), cost = ");
      fprint_Value(FTRACE,value_uminus(DK(1)));fprintf(FTRACE,"\n");
	vzlast(XX); if (VW4) wvoir(XX);
    }
  } else { 
    struct problem AZZ; vid(&AZZ);    /*ssss*/
    if ((r=fast(XX,&AZZ))==VRINS) { 
      if (PMET3==0) r=failure(XX,&AUU,VV,RR);
      else { r=failure(&AZZ,&AUU,VV,RR); value_assign(XX->dk[1],AZZ.dk[1]); }
      ICOUT=0; /******* revoir!!!!! necessaire 2 cas ********/
    }
  }
  return(r);
}
static int pseudocopie(Pproblem XX,Pproblem YY,int condition)
{ 
  Pproblem pt; 
  if (XX->state<0) XBUG(42);
  if (condition) {pt=XX;XX=YY;YY=pt;} else if (copystructure(XX,YY)) XBUG(47);
  return(0);
}
static int iprimalplus(Pproblem XX,struct rproblem *RR,int minimum,int i2,int stopcout)
{ 
  int r,step,dispo;float rsr;
  Value bornesr,bornepai,bornea,solution;
  struct problem AYY,AZZ,AVV,AWW,*pty,*ptv,*ptw;

  vid(&AYY); vid(&AZZ); vid(&AVV); vid(&AWW); dispo=0;     /*ssss*/ /*xs*/
  pty=&AYY;ptv=&AVV;ptw=&AWW; XX->state=0;if (copystructure(XX,&AZZ)) XBUG(46);
  if ((r=iprimal(XX,minimum,i2,stopcout))!=VRINS) return(r); value_assign(bornepai,DK(ICOUT));
  rsr=localrsimplex(XX,1); 
  if (XX->minimum) value_assign(bornesr,dessous(rsr)); else value_assign(bornesr,dessus(rsr));
  if (VW2) {fprintf(FTRACE,"PRIMAL FAILS (%d ITER), possible integer values %13.6f $<$ ",NITER,-rsr); 
  fprint_Value(FTRACE,value_uminus(bornesr));fprintf(FTRACE," $<=$ cost $<=$ ");
  fprint_Value(FTRACE,value_uminus(DK(ICOUT)));fprintf(FTRACE,"\\\\\n");}
  value_assign(solution,bornepai); value_assign(bornea,bornesr); VRESULT=VRFIN; step=0;
  while (value_lt(solution,bornea)) {
    Value possible,expossible=VALUE_ZERO,typspl; ++step; /*if (XX->minimum)*/

    if (value_eq(value_plus(solution,VALUE_ONE),bornea)) value_assign(possible,value_plus(solution,VALUE_ONE));
    else if (PDICO==1 || (step==1&&PDICO==2)) value_assign(possible,value_div(value_plus(bornea,solution),2));
    else if (PDICO==3) value_assign(possible,value_plus(solution,VALUE_ONE));  /* descendant */
    else if (PDICO==2 && VRESULT==VRFIN) value_assign(possible,value_plus(solution,VALUE_ONE));
    else value_assign(possible,bornea); /* ascendant. Par curiosite, si 5, ->inequation */
    
    if ((value_eq(possible,bornea)) && (PDICO!=5)) typspl=0; else typspl=1;
    if (VW2) {fprintf(FTRACE,"dichoto%d NX=%d poss=",step,NX);fprint_Value(FTRACE,possible);fprintf(FTRACE,"\\\\\n");}
    ptw->marque=0; ptw->repere=11;
    if (dispo) {  /*2MMM5*/ 
      if (pseudocopie(ptv,pty,(PMET2>=5 && (value_eq(possible,value_plus(solution,VALUE_ONE)))))) XBUG(47);
      findandmod(pty,G(1),(value_minus(possible,expossible)),1); /*provisoire*/
      if (VW4) wvoir(pty); vidr(RR); VRESULT=reducedpb(pty,ptw,RR);
    }
    else { 
      if (copystructure(&AZZ,pty)) XBUG(48);
      vidr(RR); VRESULT=splitpb(pty,ptw,RR,0,ICOUT,possible,typspl);
    }
    if (VRESULT!=VRFIN&&VRESULT!=VRVID) return(VRESULT);
    if (PMET2>=4) { if(PMET2<7)if (!ptw->marque) XBUG(55);if (VIS2&&PMET2<7) wvoir(ptw);}
    if (VRESULT==VRVID) value_assign(bornea,value_minus(possible,VALUE_ONE));
    else { 
      if (pseudocopie(ptw,ptv,(PMET2>=5 && dispo))) XBUG(49); /*2MMM5*/
      if (PMET2>=4) {dispo=1; value_assign(expossible,possible); } /*2MMM4*/
      if (typspl==0) value_assign(solution,possible);
      else 
	if (value_eq(value_assign(solution,pty->dk[1]),possible)) {
	  if (VIS2) {
	    fprintf(FTRACE," p = ");fprint_Value(FTRACE,value_uminus(possible));
	    fprintf(FTRACE," s = ");fprint_Value(FTRACE,value_uminus(solution));}
	  XBUG(56);}
	else 
	  if (VW1) {
	    fprintf(FTRACE," verified solution ");fprint_Value(FTRACE,value_uminus(possible));
	    fprintf(FTRACE," improved solution ");fprint_Value(FTRACE,value_uminus(solution));
	    fprintf(FTRACE," \\\\\n");
	  }
    } /*if(PMETH==8) XBUG(39);*//*mesure cout step 1*/
  }
  if (VW1) {fprintf(FTRACE,"Final solution:");fprint_Value(FTRACE,value_uminus(solution));fprintf(FTRACE,"\\\\\n");}
  return(VRESULT);
}
/* ======================================================================= */
/*                             DYNAM.c                            */
/* ======================================================================= */
int dynam(Pproblem XX,int nd) /****************************/
{ /*struct problem AVV; struct rproblem ARR; vid(&AVV); vidr(&ARR);*/
  /*fprintf(FTRACE,"fourier=%d varc=%d choixpiv=%d forcer=%d\\\\\n",
     PFOURIER,VARC,CHOIXPIV,FORCER);*/
  int ca,v,uni; ++XX->dyit; uni=0;
  if (DYN==1)
    { printf("dynamic visualization\n");
      XX->itdirect=0; DYN=2; return(0);
    }
  if (XX->niter < XX->itdirect) return(0);
  printf("passage dynamic iter no=%d ----------------\n",XX->dyit);
  if (nd==1) printf("dual dual dual dual dual dual iteration\n");
  if (DYN) printf("dynam, iteration %d ",NITER);
  printf("query: "); ca= getchar();
  fclose(XX->ftrace) ;
  while (ca!=10)
    { 
      if (ca==117) { uni=1;printf("unimodular change "); }
      else if (ca==118) { printf("visu= "); scanf("%2d",&v); VISU=v;}
      else if (ca==104) /*h*/
	{ printf("help\n");
	  printf("    visu=%d\n",VISU);
	  printf("    direct ");
	  if(DYN)printf("NO\n");else printf("YES\n");
	  printf("    itdirect=%d\n",XX->itdirect);
	}
      else if (ca==100) DYN=0; /*d*/
      else if (ca==115) /*s*/
	{ printf("System state will be vizualized\n");
	  XX->ftrace=fopen("/tmp/jtrace.tex","a") ;
	  wvoir(XX) ; fclose(XX->ftrace) ;
	}
      else if (ca==105) /*i*/
	{ printf("go to iteration: "); scanf("%2d",&v); XX->itdirect=v; DYN=2;}
      else if (ca!=10) printf("commande %d?\n",ca);
      /*else if (ca!=10) printf("commande ?\n");*/
      printf("another query: ");
      ca= getchar();
      ca= getchar();
    }
  XX->ftrace=fopen("/tmp/jtrace.tex","a") ;
  return(uni);
  /*************************  while (c!=10)
    { c= getchar(); printf("vidage caractere =%d\n",c); }************/
  /*printf("selon c c=%c\n",c); printf("selon d c=%d\n",c);*/
  /*v=isalpha(c); printf("produit par isalpha v=%d\n",v);*/
  /*v=isdigit(c); printf("produit par isdigit v=%d\n",v);*/
  /*printf("conversion \n"); v=atoi(&c); printf("produit par atoi v=%d\n",v);*/

  /*printf("action direct:0 suite:1 visu:2 dynam:3 ? ");scanf("%2d",&v);*/
}
/* ======================================================================= */
/*                             RESULT PROCEDURE                            */
/* ======================================================================= */
static int vzlast(Pproblem XX) /********************************/
{ 
  if (VISU>=ZVL2) wvoir(XX); if (VISU>=ZVL2) fprintf(FTRACE,"******** ");
  if (VRESULT==VRVID) fprintf(FTRACE,"empty polyhedron");
  else if (VRESULT==VRBUG) fprintf(FTRACE,"bug programme %3d",NUB);
  else if (VRESULT==VRINF) fprintf(FTRACE,"solution infinie");
  else if (VRESULT==VRFIN) fprintf(FTRACE,"integer solution");
  else 
    if (VRESULT==VRDEB) { 
      fprintf(FTRACE,"debordement tableaux ");
      if (MX >MAXMX) fprintf(FTRACE," line MX=%4d\n",MX);
      else fprintf(FTRACE," column NX=%4d\n",NX); 
    } else if (VRESULT==VRINS) fprintf(FTRACE,"trop de pivotages");
    else if (VRESULT==VROVF) fprintf(FTRACE,"overflow");
    else if (VRESULT==VRCAL) fprintf(FTRACE,"appel errone **");
    else fprintf(FTRACE,"bug message * result=%5d",VRESULT);
  fprintf(FTRACE," after %d iterations\\\\\n",NITER) ;
  return 0;
}
/* ======================================================================= */
/*                              MAIN PROCEDURE                             */
/* ======================================================================= */
static int is2(Pproblem XX,Pproblem VV,struct rproblem *RR)
{ 
  int nlibre;   /* nlibre: information donnee par step 4 au step 5 */
  int prevnx,stfou; /*int forcer; forcer=(PFOURIER>3);*/
  /* ICOUT: ligne ou se trouve la fonction cout a optimiser; information
     indiquee par le step 1 (initialisations); en absence de cout, =0 */
  /* 111111111111111111111111 VERIFICATIONS INITIALISATIONS 11111111111111 */
  /* 22222222222222222222 ELIMINATION EGALITES 22222222222222222222222222 */
  nlibre =0; prevnx = 0; stfou = 0;
  if (VISU) if (NEGAL) vzstep(XX,1);
  while (NEGAL) { 
    int r,ik=0,jk; Value pivot; { 
      int i; Value dm = VALUE_MONE;
      for (i=MX ; i>=1 ; i--) if (janus_egalite(XX,i)) { 
	int j; Value sk,de; value_assign(pivot,VALUE_ZERO);
	for (j=NX; j>=1; j--) { if (value_assign(sk,value_abs(AK(i,j)))) 
	  if (value_one_p(value_assign(pivot,sk))) { ik=i; jk=j; break;}}
	if (value_zero_p(pivot)) /* equation surabondante ou incompatible */ { 
	  if (value_notzero_p(DK(i))) return(VRVID);
	  if (VISU>=ZVEV) fprintf(FTRACE, "Equation $e_{%d}$ is empty.\\\\\n",G(i));
	  retirer(XX,i); NEGAL-- ; break ; /*t603 */
	}                                      /* next line : t602 */
	if (value_one_p(pivot)) break;
	if ((value_lt(value_assign(de,value_abs(DK(i))),dm))||(value_neg_p(dm))) { 
	  value_assign(dm,de); ik=i; /*choice of the equality with the smallest rhs*/
	}
      }
    }
    if (value_zero_p(pivot)) continue ;
    if (value_notone_p(pivot)) {   /* unimodular change of variables is necessary */
      int jojo;
      if (gcdtest(XX,ik,&jk,&pivot,ZVG2)) return(VRVID);
      value_assign(pivot,VALUE_ZERO);
      for (jojo=NX; jojo>=1; jojo--)
	if (value_one_p(value_abs(AK(ik,jojo)))) { jk=jojo; value_assign(pivot,VALUE_ONE); break;}
      if (value_notone_p(pivot)) if ((r=build0(XX,ik,1,&jk))) return(r);
    } else if (VISU>=ZVEV+1) fprintf(FTRACE,"Equation $e_{%d}$ includes a unitary coefficient.\\\\\n",G(ik));
    if (VISU>=ZVEV+1) igvoir3(XX,1,MX,1,ik,jk,0,0,0,0,0);
    if ((r=ipivotage(XX,ik,jk))) return(r);
    celimination(XX,jk);
    if (fluidline(XX,ik)) retirer(XX,ik); else permutel(XX,ik,MX+1-NEGAL);
    NEGAL-- ;
  }//of while
  if (NEGAL) XBUG(15); /* equalities still remain */
  /* 2323232323232323232323 INEQUALITIES DIVISION 23232323232323232323 */
  {
    int i; if (VISU) vznewstep(XX,2);
    for (i=IC1 ; i<=MX ; i++)
      { inequaldivision(XX,i); }
  }
  /* 3333333333333333333333 FOURIER-MOTZKIN trivial 33333333333333333333333 */
  if (PFOURIER) { 
    int i,j; if (VISU) vznewstep(XX,3);
    for (j=NX ; j>=1 ; j--)
      if (freecolumn(XX,j))
	if (possible(XX,j)) {
	  int tplus,tmoins; Value k; tplus= tmoins= 0 ;
	  for (i=MX ; i>=IC1 ; i--) 
	    if (value_assign(k,AK(i,j))) {if (value_pos_p(k)) tplus++ ; else tmoins++ ;}
	  if (tplus==0||tmoins==0) {
	    if (tplus+tmoins) { 
	      fourier0(XX,j); if (!FORCER) if (ICOUT==0 && satisfait(XX)) return (VRFIN);
	    } else { /* ********** colonne vide */
	      if(VISU>=ZVFEC) vzemptycol(XX,j); emptycolelim(XX,j); }
	  }
	}
  }
  /* 4444444444444444444  FOURIER-MOTZKIN - DUMMY ELIMINATIONS 4444444444444*/
  stfou=0;//	fprintf(stderr,"%d %d %d\n",nlibre,prevnx,stfou);
  while (stfou==0 || (NX<prevnx && nlibre)) { 
    int i=0,j=0;//DN
    prevnx = NX; nlibre=0 ; stfou++;
    {for (j=NX ; j>=1 ; j--)
      if (freecolumn(XX,j)) {
	int tplus,tmoins,ip1=0,im1=0,tp2,tm2,i1; Value sk,dkmin;
	tplus= tmoins= tp2= tm2=0 ; i1= -100000 ;
	value_assign(sk,VALUE_ZERO); value_assign(dkmin,int_to_value(1000000));
	for (i=MX ; i>=IC1 ; i--)
	  if (value_notzero_p(AK(i,j))) { 
	    value_assign(sk,AK(i,j));
	    if (value_pos_p(sk)) { 
	      tplus++ ;
	      {if (value_one_p(sk)) { 
		ip1= i ; if (value_lt(DK(i),dkmin)) { i1=i; value_assign(dkmin,DK(i)); }
	      } else tp2++;}
	    } else { 
	      tmoins++ ;
	      {if (value_mone_p(sk)) { 
		im1=i;if (value_lt(DK(i),dkmin)) {i1=i; value_assign(dkmin,DK(i)); }
	      } else tm2++ ;}}
	  }
	if (value_zero_p(sk)) /* ********** colonne vide */ { 
	  if (possible(XX,j)) { 
	    if (VISU>=ZVFEC) vzemptycol(XX,j); emptycolelim(XX,j);
	  } else if (VISU>=ZVFW) fprintf(FTRACE,"****colonne%3d vide bloquee\n",j);/*provisoire*/
	} else { 
	  if (i1 < 0)  { nlibre++ ; continue ; } /*  ***** pas de pivot unitaire possible */
	  if (possible(XX,j)) { 
	    if (!FORCER) if (ICOUT==0 && satisfait(XX)) return (VRFIN) ;
	    if (PFOURIER>0) 
	      if (tplus==0||tmoins==0) { fourier0(XX,j); continue; }
	    if (PFOURIER>1) {
	      if (tp2==0||tm2==0) /* criteres F-M 1 et 2 */ { 
		if (tplus==1&&tp2==0) {fourier1(XX,1,j,ip1); continue;}
		else if (tmoins==1 && tm2==0 ) { fourier1(XX, -1,j,im1); continue; }
		else if (PFOURIER>2 && tmoins==2 && tplus==2 ) { fourier22(XX,j) ; continue ; }}}
	  }
	  if (VARC) /* dummy elimination */ { 
	    if (VISU>2) fprintf(FTRACE,"Dummy elimination: ");
	    if (ipivotage(XX,i1,j)) return(VRESULT) ; retirer(XX,i1) ;
	    if (VISU>=ZVTS) wvoir(XX); /*mauvais*/
	  } else nlibre++ ;
	}}
    }      
  }
  /*                  -------------------------------                    */
  if (!FORCER) if (ICOUT==0 && satisfait(XX)) return (VRFIN) ;
  if (NX==0 && (!satisfait(XX))) return(VRVID) ;
  /* envisager traitement cas trivial NX==1 */
  /* 555555555555555 NON-NEGATIVE VARIABLES 55555555555555 */
//	fprintf(stderr,"%d %d %d\n",nlibre,prevnx,stfou);//DNDB
  if (!nlibre) NTP=NX;
  else {
    int i,j; 
    if (VISU) vzstep(XX,4);
    if (VARC<=2)  /******** strategie simplexe *******/ { 
      int jj=0,idk=0; Value dkn = VALUE_ZERO;
      NTP=0; /** regroupement des variables contraintes **/
      for (jj=1 ; jj<=NX ; jj++)
	if (!freecolumn(XX,jj)) permutec(XX,++NTP,jj);
      if (VISU>=ZVNPC) wvoir(XX);
      for (;;) { 
	if (NTP==NX) break;
	if (!FORCER) if (ICOUT==0 && satisfait(XX)) return (VRFIN) ;
	//	value_assign(dkn,VALUE_ZERO); 
	value_assign(dkn,int_to_value(1000000)); /*provisoire*/
//	fprintf(stderr,"dkn = int_to_value(1000000) = ");fprint_Value(stderr,dkn);fprintf(stderr,"\n");//DNDB
	for (i=MX ; i>=IC1 ; i--)
	  if (value_lt(DK(i),dkn)) { value_assign(dkn,DK(i)); idk=i; }
	jj=NTP+1;
	if (VISU>=ZVN1) {
	  fprintf(FTRACE,"rhs=");fprint_Value(FTRACE,dkn);fprintf(FTRACE,", let us select inequality $y_{%d}$.\\\\\n",G(idk));
	}	
	if (build1(XX,idk,jj) <0) { /* segment libre est vide! */
	  int absent ; absent=1;
	  if (VISU>=ZVN1) fprintf(FTRACE,"No non-zero coefficient!!\\\\\n");
	  for (i=MX ; i>=IC1 ; i--)
	    if ((absent||(value_lt(DK(i),dkn))) && jnsegment(XX,i,jj,NX)>=0) { 
	      value_assign(dkn,DK(i)); idk=i; absent=0;
	    }
	  if (absent) { 
	    if (VISU>=ZVNW) fprintf(FTRACE,"WARNING, all columns of free variables are empty\\\\\n");
	    if (ICOUT==0) NX=NTP; break; }
	  if (VISU>=ZVN1) fprintf(FTRACE,"Then, we select inequality $y_{%d}$.\\\\\n",G(idk));
	  if (value_pos_p(dkn)) if (VISU>=ZVN2) fprintf(FTRACE,"WARNING SIMPLEX STRATEGY rhs positif\\\\\n");
	  if (build1(XX,idk,jj) <0) XBUG(33);
	}
	++NTP;
//	fprintf(stderr,"%d %d %d %d %d %d %d",nlibre,prevnx,stfou, i,j,jj,idk);fprint_Value(stderr,dkn);fprintf(stderr,"\n");//DNDB
	if ((i=cutiteration(XX,idk,jj,0))) return(i);
      }//of for
    } else if (VARC==3) { /********* variable unique *************/
      if (VISU>1) fprintf(FTRACE,"SINGLE ADDED VARIABLE\n");
      if (++NX>MAXNX) return(VRDEB) ;
      for (i=MX ; i>=1 ; i--) { 
	Value temporaire ; value_assign(temporaire,VALUE_ZERO);
	for (j=1 ; j<NX ; j++) if (freecolumn(XX,j)) value_substract(temporaire, AK(i,j)) ;
	value_assign(AK(i,NX),temporaire);
      }
      B(NX)= NUMERO = ++NUMAX ; if (VISU>3) wvoir(XX) ;
    }   
    else if (VARC>=4) {/******* variables libres doublees *******/
      for (j=NX ; j>=1 ; j--)
	if (freecolumn(XX,j)) { 
	  if (++NX>MAXNX) return(VRDEB); B(NX)=B(j) ; /* tres provisoire */
	  for (i=MX; i>=1; i--) value_assign(AK(i,NX),value_uminus(AK(i,j)));
	}
      if (VISU) fprintf(FTRACE,"map: variables libres ont ete doublees \n");
    }
  }
  /* 5656565656565656 FOURIER ON CONSTRAINED VARIABLES 5656565656565656 */
  if (PFOURIER) { 
    int i,j; if (VISU) vznewstep(XX,5);
    for (j=NX ; j>=1 ; j--)
      if (possible(XX,j)) { /*if (freecolumn(XX,j))*/
        int tplus,tmoins; Value k;
	tplus= tmoins= 0 ;
	for (i=MX ; i>=IC1 ; i--)
	  if (value_assign(k,AK(i,j))) { if (value_pos_p(k)) tplus++ ; else tmoins++ ;}
	if (tplus==0) { 
	  if (VISU>=ZVS) vznowstep(XX);
	  if (tplus+tmoins) { fourier0(XX,j); if(!FORCER) if (ICOUT==0 && satisfait(XX)) return(VRFIN);}
	  else { if(VISU>=ZVFEC) vzemptycol(XX,j); emptycolelim(XX,j); } /* ********** colonne vide */
	}
      }
  }
  /* 66666666666666 VARIOUS REDUNDANT INEQUALITIES 666666666666666 */
  if (XX->remove) { 
    int i,u,i1,i2,exmx; exmx=MX;
    if (VISU>=ZVS) vznewstep(XX,6);
    for (i=MX ; i>=IC1 ; i--) /******** empty lhs (left-hand side */
      if (emptylhs(XX,i)) {	
	if (value_neg_p(DK(i))) return(VRVID); VREDUN(++NREDUN)=G(i);
	if(VISU>=ZVR1) vzredundant(XX,i,-1); retirer(XX,i);
      }
    for (i=MX ; i>=IC1 ; i--) /************ intrinsic redundancies */
      if (useless(XX,i)==1) {if (VISU>=ZVR1) vzredundant(XX,i,0); VREDUN(++NREDUN)=G(i);retirer(XX,i);}
    for (i1=MX ; i1>=IC1+1 ; i1--) /****** redundancies when comparison */
      for (i2=i1-1 ; i2>=IC1 ; i2--)
	if ((u=redundant(XX,i1,i2))) {
	  if (VISU>=ZVR1) { int v; if (u==i1)v=i2;else v=i1; vzredundant(XX,u,v);}
	  VREDUN(++NREDUN)=G(u); retirer(XX,u); break;
	}
    if ((VISU>=ZVR3) && MX<exmx) { 
      fprintf(FTRACE,"%d redundant inequalities were removed\\\\",exmx-MX);
      wvoir(XX);
    }
    if (XX->nredun!=exmx-MX) XBUG(69);
  }
  /* 66666666666666666 RESOLUTION SIMPLEXE ENTIERS phase 1 666666666666666666 */
  {
    int r;
    if (satisfait(XX)) { 
      r=VRFIN; if (VISU>=ZVS) fprintf(FTRACE,"{\\bf {*} PHASE: DUAL SIMPLEX} is unnecessary\\\\\n");
    } else { 
      if (PMETH==MTDAI||PMETH==MTDI2) r=dualentier(XX);
      else r=fastplus(XX,VV,RR);
    }
    if ((r!=VRFIN) || !ICOUT) return(r);
  }
  /* 777777777777777777 OPTIMISATION FONCTION COUT ENTIERE 777777777777777777 */
  if (VW2) {
    fprintf(FTRACE,"Primal %d variables precr=%d comp=%d msr=%d dico=%d met3=%d met2=%d meth=%d - varc=%d choixpiv=%d crit=%d m8=%d\\\\\n",NX,PRECR,PCOMP,MSR,PDICO,PMET3,PMET2,PMETH,VARC,CHOIXPIV,CRITERMAX,REDON);
    wvoir(XX);
  }
//	fprintf(stderr,"%d %d %d\n",nlibre,prevnx,stfou);//DNDB
  return(iprimalplus(XX,RR,XX->minimum,ICOUT,0)); /* ./shj p-6  ./shj p3-7 */
}
static int init_janus(Pinitpb II,Pproblem XX,int suit1)
{ /* ICOUT: ligne ou se trouve la fonction cout a optimiser; information
     indiquee par le step 1 (initialisations); en absence de cout, =0 */
  /* 111111111111111111111111 VERIFICATIONS INITIALISATIONS 11111111111111 */
  NITER=0; TMAX=2*MC+NV; VRESULT=0 ; XX->nredun=0;
  if (VISU>=ZVS) fprintf(FTRACE, "\\begin{center} {\\bf ENTER INTEGER SOLVER} \\end{center}\n");
  if (MC>MAXLIGNES || NV>MAXCOLONNES) {
    if (VISU) fprintf(FTRACE,"entree: mc=%d maxl=%d --- nv=%d maxcol=%d\n",MC,MAXLIGNES,NV,MAXCOLONNES);
    return(VRDEB);
  }
  if (NP>NV) return(VRCAL) ;
  if (!suit1) {/* equations a la fin, inequations m sens */
    int i,j,i1,i2,ifi,ni,ii2,mf,ncou,Ei;
    NUMAX=NUMERO=NV+MC ; NX=NV; MX=MC; mf=ni=0 ; ncou=0; ICOUT=0; LASTFREE=0;//ii2=0;
    for (j=1 ; j<=NV ; j++) B(j)=j ;
    for (i=1 ; i<=MC ; i++) {
	Ei = E(i); //if (ii2!=0) fprintf(stderr,"Ei = %d, E(i) = %d, I->e[i] = %d",Ei,II.e(i),II.e[i]);//DNDB
      if (( abs(Ei)==2)||(Ei==3)) { 
	++mf; //number of function "cout" DN
	if (( abs(Ei) ==2)&&(++ncou>1)) { 
	  if (VISU>=ZVB) fprintf(FTRACE,"nombre de couts>1\n");
	  return(VRCAL);
	}
      } 
      else 
	if (Ei!=0) { 
	  if (Ei!=1 && Ei!= -1) { 	    
	    if (VISU>=ZVB) fprintf(FTRACE,"! e[%3d]=%3d\n",i,Ei);
	    return(VRCAL);
	  }
//          if (ii2!=0) fprintf(stderr,"Ei danglefai+= %d",Ei);//DNDB
	  ni++ ;//number of inequation DN
	}
    }//of for
    ii2=0;ifi=0;i1=mf;i2=mf+ni; IC1=mf+1; ii2=IC1-1+ni;/*derniere inequation*/
    NEGAL=MX-ii2;
    for (i=1 ; i<=MC ; i++) { 
      int i3; i3=(abs(Ei=E(i))==2)? mf :(Ei==3)? ++ifi :(Ei) ? ++i1 :++i2;
      if (Ei== -1) {
	value_assign(DK(i3),value_uminus(D(i))); 
	for (j=1; j<=NV; j++) value_assign(AK(i3,j),value_uminus(A(i,j)));
      } else { 
	value_assign(DK(i3),value_uminus(D(i))); 
	for (j=1; j<=NV; j++) value_assign(AK(i3,j),value_uminus(A(i,j))); 
      }
      G(i3)=NV+i ;
      if (abs(Ei)==2) { 
	ICOUT=i3; XX->minimum=(Ei==2);
      }
    }
    if (VISU) { 
      VDUM=0;
      if (VISU>=ZVA1) fprintf(FTRACE," %d variables %d functions\
          %d inequalities %d equalities\\\\\n",NX,mf,ni,MC-ni-mf);
      if(VISU>=ZVA4) wvoir(XX); /*stepvoir(XX);*/
    }
//fprintf(stderr,"\n DN %d %d %d %d %d %d %d %d %d %d",i,j,i1,i2,ifi,ni,ii2,mf,ncou,Ei); //DNDB
  } //of if suit1 
  return(0);
}

/**********************************************************************/
int isolve(Pinitpb II, Pproblem XX,int suite) /***********/
{
  struct problem AVV;struct rproblem ARR;int suit1,suit2,su3,su,r,stopc;
  vid(&AVV);vidr(&ARR);
  /*fprintf(FTRACE,"fourier=%d varc=%d choixpiv=%d forcer=%d\\\\\n",
    PFOURIER,VARC,CHOIXPIV,FORCER);*/
  if (DYN) dynam(XX,0);
  su3=suite/100; su=suite-100*su3; suit2=su/10; suit1=suite-10*suit2;
  /*suit2=suite/10; suit1=suite-10*suit2;*/
  /*if (su3) fprintf(FTRACE,"DEBUT ISOLVE icout=%d\\\\\n",ICOUT);*/
  if ((r=init_janus(II,XX,suit1))) return(r); stopc=0; if (su3) stopc=9;
  if (suit2) VRESULT=iprimal(XX,XX->minimum,ICOUT,stopc); /*stopcout*/
  else VRESULT=is2(XX,&AVV,&ARR); /*ssss*/
  if (VISU>=ZVL) vzlast(XX) ; return(VRESULT) ;
}
