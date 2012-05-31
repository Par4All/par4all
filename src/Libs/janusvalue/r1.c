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

#include <stdio.h>
#include "rproblem.h"

#define FT RR->ftrace
#define TRACE RR->ntrace
#define A(i,j) RR->a[i][j]
#define D RR->d
#define E RR->e
#define M1 RR->m
#define M2 RR->m2
#define MB RR->mb
#define N1 RR->n
#define NB RR->nb
#define NP RR->np
#define J0 RR->j0
#define TESTP RR->testp
#define DUAL RR->meth
#define COPIE RR->copie
#define BASE RR->base
#define COST RR->cost
#define RHS RR->rhs
#define VRESULT RR->vresult
#define ITER RR->iter
#define ITEMAX RR->tmax
#define X RR->x
#define PI RR->pi
#define B RR->b
#define G RR->g
#define INF RR->inf
#define CONORM(A) *(RR->pconor+A)
#define INFCOL(A) *(RR->pinfcolonn+A)
#define VRNUL VREPS
#define VARIABLELIBRE(v) v>NP && v<=N1
static float absf(float vf) /* ************************************************* */
{	if (vf<0) return(-vf) ;
	return(vf) ;
}
static int fraction(struct rproblem *RR,float vf) /* ***************************** */
{	int vi ;
	float vfp ;
	if (vf<0) vfp= -vf ; else vfp= vf ;
	vi=vfp+RR->epss ;
	if (vfp-vi<RR->epss) return(0) ; return(1) ;
}
static int voir4(struct rproblem *RR, int v, int i11, int j11) /* ********************** */
{	int i,j ;
	if (TRACE<2) return(0) ; if (TRACE<3&&v!=30) return(0) ; /*99*/
	if (v==0) { NB=N1 ; M2=M1 ; } ;/* appel avant les initialisations */
	fprintf(FT,"m=%2d mb=%2d nb=%2d t=%2d ",M1,MB,NB,ITER) ;
	if (v==0) fprintf(FT,"tout debut") ;
	if (v==1)fprintf(FT,"situation initiale, contraintes <= , = , >=");
	if (v==2) fprintf(FT,"forme standard, contraintes <= , =") ;
	if (v==3) fprintf(FT,"lignes ont ete normalisees") ;
	if (v==4) fprintf(FT,"colonnes ont ete normalisees") ;
	if (v==5) fprintf(FT,"apres norm") ;
	if (v==6) fprintf(FT,"forme standard normalisee") ;
	if (v==7) fprintf(FT,"egalites eliminees, forme canonique <=") ;

	if (v==8) { fprintf(FT,"methode primale\n") ; return(0) ; } ;
	if (v==9) fprintf(FT,"phase 1 primal") ;
	if (v==10) fprintf(FT,"variable artificielle primal introduite") ;
 if (v==11) {fprintf(FT,"algorithme simplexe primal ph. 1\n"); return(0);};
	if (v==12) fprintf(FT,"phase 2 primal") ;
	if (v==13) { fprintf(FT,"methode duale\n") ; return(0) ; } ;
	if (v==14) fprintf(FT,"phase 1 dual") ;
    if (v==15) fprintf(FT,"variable artificielle phase 1 dual introduite");
   if (v==16) {fprintf(FT,"algorithme simplexe dual ph. 1\n"); return(0);};
	if (v==17) fprintf(FT,"ligne artificielle inutile deplacee") ;
	if (v==18) fprintf(FT,"retour necessaire de dual a primal") ;
	if (v==19) fprintf(FT,"phase 2 dual") ;
/*if(v==30){fprintf(FT,"pppivot\(%2d,%2d\):%6.3f",i11,j11,1/ A(i11,j11));*/
if (v==30) {fprintf(FT,"pivot(%2d,%2d):%6.3f",i11,j11,1/ A(i11,j11));
	fprintf(FT," variables %2d <-> %2d",B[j11],G[i11]) ; };
    /*fprintf(FT,"d=%6.3f a=%15.9f\\\\\n",D[i11],A(0,j11)) ;*provisoire*/
		fprintf(FT,"\\\\\n") ; /*99*/
if (v==1)
{	for ( i=0 ; i <= M1 ; i++)
	{	if (i==0)
		{	if (E[i] ==1) fprintf(FT,"min   ") ;
			else if (E[i] == -1) fprintf(FT,"max   ") ;
			else fprintf(FT,"\?\?    ") ;
		} else
			if (i+N1 < 10) fprintf(FT," x%1d : ",i+N1) ;
				else fprintf(FT,"x%2d : ",i+N1) ;
		for ( j=1 ; j <= N1 ; j++)
		{	if (A(i,j) == 0) continue ;
			if (A(i,j) > 0) fprintf(FT,"+") ;
			if (j < 10) fprintf(FT,"%6.3fx%1d ",A(i,j),j) ;
				else fprintf(FT,"%6.3fx%2d ",A(i,j),j) ;
		} ;
		if (i>0)
		{	if (E[i] ==1) fprintf(FT,"<= ") ;
			else if (E[i] == -1) fprintf(FT,">= ") ;
			else if (E[i] ==0) fprintf(FT,"= ") ;
			fprintf(FT,"%7.3f",D[i]) ;
		} 
		else
			if (D[i]!=0)
			{	if (D[i]>0) fprintf(FT,"+") ;
				fprintf(FT,"%7.3f",D[i]) ;
			}
		fprintf(FT,"\n") ;
	} ;
	if (NP==N1) fprintf(FT,"variables hors-base toutes >=0\n") ;
	else if (NP==0)
	   fprintf(FT,"variables hors-base toutes de signe quelconque\n") ;
	else fprintf
(FT,"%3d variables hors-base >=0 ,%3d variables libres \n",NP,N1-NP) ;
};
	if (TRACE<4) return(0) ;
	if (TRACE<5 && (v!=1 ) ) return(0) ;
	if (TRACE<6 && (v!=1 && v!=6 ) ) return(0) ;
	if (TRACE<7 && (v!=1 && v!=6 && v!=30) ) return(0) ;
		fprintf(FT,"b	") ;
		for ( j=0 ; j <= NB ; j++)
		{ fprintf(FT,"[%2d]=%2d	",j,B[j]) ;
		} ;
		fprintf(FT,"\n") ;
	for ( i=0 ; i <= M2 ; i++)
	{	fprintf(FT,"g[%2d]=%2d ",i,G[i]) ;
		for ( j=0 ; j <= NB ; j++)
		{	fprintf(FT,"%6.3f	",A(i,j)) ;
		} ;
		fprintf(FT,"e=%4d d=%7.3f",E[i],D[i]) ;
		fprintf(FT,"\n") ;
	} ;
	return(0);
} /* fin voir */
static int voir(struct rproblem *RR, int v) /* ********************** */
{//	int i,j ;
  if (TRACE<3) return(0) ; voir4(RR,v, 0,0); return(0);
}
/***************************************************************/
/***************************************************************/
int realsimplex(Prproblem RR)
{ int mee,nvt;
/*********************** initialisations **********************/
  M1=RR->mcontr ; N1=RR->nvar ; NP=RR->nvpos ;
  mee=M1; nvt=N1;
  if (M1>RMAXLIGNES) return(VRDEB) ;
  if (N1>RMAXCOLONNES) return(VRDEB) ;
  if (NP>nvt) return(VRDEB) ;
  VRESULT=0 ;
  RR->pconor= &RR->tconor[1] ;	/* pconor utilise par macro CONORM */
  RR->pinfcolonn= &RR->tinfcolonn[1] ;	/* utilise par macro INFCOL */
  RR->eps=0.0000001 ;  /* peut-etre a modifier 0.0000001  0.0000003 */
  NB=N1 ; M2=M1 ;
  if (TRACE>0)
    { fprintf(FT,"%3d contraintes%3d variables%3d variables non-negatives\n",
	      M1,N1,NP) ;
      if (DUAL)fprintf(FT,"  dual"); else fprintf(FT,"  primal");
      if (COPIE) fprintf(FT,", doublage");
	  /*else fprintf(FT,",memes colonnes" );*/
	  /*if (TESTP==0) fprintf(FT,", tests precision pousses\n" );
      if (TESTP==1) fprintf(FT,", tests precision moyens\n" );
      if (TESTP==2) fprintf(FT,", tests precision sommaires\n" );
      if (TESTP>2) fprintf(FT,",sommaire,entiers inconnus\n");*/
      if (TESTP==0) fprintf(FT,", tests precision pousses" );
      if (TESTP==1) fprintf(FT,", tests precision moyens" );
      if (TESTP==2) fprintf(FT,", tests precision sommaires" );
      if (TESTP>2) fprintf(FT,",sommaire,entiers inconnus");
      fprintf(FT,", niveau trace: %d\\\\\n",TRACE); /*99*/
    }
  if (COPIE)
    { int ii,jj,jk ;
      if(TRACE)fprintf(FT,"systeme original contraintes\n"); voir(RR,1) ;
      if (TRACE) fprintf(FT,"systeme modifie contraintes double:\n");
      if ((N1=(2*nvt)-NP) >RMAXCOLONNES) return(VRDEB) ;
      for (ii=0 ; ii<= M1 ; ii++)
	{ jk=N1 ;
	  for (jj=nvt ; jj>= 1 ; jj--)
	    { float s; s=A(ii,jj);
	      if (jj>NP) A(ii,jk--)= -s;
	      A(ii,jk--)= s;
	    };
	} ;
      NP=N1 ;			/* variables rendues toutes >=0 */
    }
      /*if (simbad(RR)) return(VRESULT);*/
  VRESULT=simbad(RR);
  if (TRACE)
    { fprintf(FT,"resultat simplexe apres %d iterations: ",ITER) ;
      if (VRESULT==VRFIN) fprintf(FT,"solution finie\n") ;
      if (VRESULT==VRVID) fprintf(FT,"polyedre vide\n") ;
      if (VRESULT==VRINF) fprintf(FT,"solution infinie\n") ;
      if (VRESULT==VRINS) fprintf(FT,"nombre insuffisant\n") ;
      if (VRESULT==VRDEB) fprintf(FT,"debordement tableaux\n") ;
      if (VRESULT==VRCAL) fprintf(FT,"appel errone\n") ;
      if (VRESULT==VREPS) fprintf(FT,"pivot anormalement petit\n") ;
      if (VRESULT==VRBUG) fprintf(FT,"bug de programmation\n") ;
      fprintf(FT,"\\\\\n") ; /*99*/
    }
  if (VRESULT) return(VRESULT);
/******************** cas ou une solution finie est obtenue ******************/
  if (TRACE>1)
    { int jj,jk ; float xf ;
      for (jj=0 ; jj<= N1+M1 ; jj++)
	{ if (X[jj] !=0)
	    { if (jj<= N1) fprintf(FT,"	" );
				  else fprintf(FT,"%2d:	",jj-N1 );
	      /*fprintf(FT,"x[%2d] =%9.3f\n",jj,X[jj] );*/
	      fprintf(FT,"x[%2d] =%13.6f\n",jj,X[jj] );
	    } ;
	} ;
      if (TESTP>2 && MB!=M1)
	{ fprintf(FT," sans signification pour variables libres:\n");
	  for (jj=MB+1 ; jj<= M1 ; jj++) fprintf(FT," %3d",G[jj]);
	  fprintf(FT,"\n");
	}
      if (COPIE && (nvt-NP)) /* if (NP!=nvp) */
	{ jk=0 ;
	  for (jj=1 ; jj<= nvt+mee ; jj++)
	    { jk++ ; xf=X[jk] ;
	      if (jj>NP && jj<=nvt)
		{ jk++ ; if (xf==0) xf= -X[jk] ;
		};
	      X[jj] = xf ;
	    };
	  fprintf(FT,"resultats probleme initial:\n") ;
	  for (jj=0 ; jj<= nvt+mee ; jj++)
	    if (X[jj] !=0)
	      { if (jj <= nvt) fprintf(FT,"	" );
		else fprintf(FT,"%2d:	",jj-nvt );
		fprintf(FT,"x[%2d] =%9.3f\n",jj,X[jj] );
	      }
	};
      fprintf(FT,"\\\\\n"); /*99*/
    }
  VRESULT=VRFIN ;
  if (RR->tfrac==0) return(VRESULT) ;
  /* examen si variables fractionnaires utilisation marginale */
  if (TESTP<3)
    /*	if (mx==M1)*/
    { int bfract, jj ; bfract=0 ;/*FAUX*/
      for (jj=0 ; jj<= N1 ; jj++)
	if (fraction(RR,X[jj]))
	  { if (TRACE>0)
	      {	if (!bfract) fprintf(FT," variables non entieres: ");
		fprintf(FT," %2d",jj);
	      } ;
	    bfract=1 ;/*VRAI*/
	  } ;
      if (TRACE>0){
	if (bfract) fprintf(FT,"\n");
	else fprintf(FT," certitude variables toutes entieres\n");
      }
      RR->rfrac=0 ; if (bfract) RR->rfrac=1 ; 
    } ;
      /*if (TRACE) fprintf(FT,"\\end{document}\n");*/
return(VRESULT) ;
}
/***************************************************************/
/***************************************************************/
static void norm(struct rproblem *RR)
/*	cette procedure normalise la fonction cout, calcule les valeurs des
	seconds membres resultant d'une normalisation du tableau a, ou
	effectue ces deux operations */
{
	float s; int i,j;
	if (COST || ! COST && ! RHS)
	{
		s=0 ;
		for (j=1 ; j<=N1 ; j++)
		{
			A(0,j)= A(0,j)* CONORM(j) ;
			if (s < absf(A(0,j))) s=absf(A(0,j)) ;
		} ;
		if (s==0) s=1 ;
		for (j=1 ; j<=N1 ; j++) A(0,j)= A(0,j)/s ;
		CONORM(0)=s ;
	} ;
	if (RHS || ! COST && ! RHS)
	{
		s=0 ;
		for (i=1 ; i<=M1 ; i++)
		{
			D[i]= D[i]/ CONORM(N1+i) ;
			if (s < absf(D[i])) s=absf(D[i]) ;
		} ;
		CONORM(-1)=s ;
	} ;
	D[0]= D[0]/ CONORM(0) ;
} /* fin norm() */ ;
/*static*/ void elimination(Prproblem RR,int j1) 
/* cette procedure permute les colonnes j1 (qui sera eliminee) et NB */
{ 
	if (j1 != NB)
	{
		int i ;
		float s ;
		s= INFCOL(j1) ; INFCOL(j1)= INFCOL(NB) ;
		INFCOL(NB)=s ;
		i=B[j1] ; B[j1]=B[NB] ; B[NB]=i ;
		for (i=0 ; i<=M1 ; i++)
		{
			s= A(i,j1) ; A(i,j1)= A(i,NB) ; A(i,NB)=s ;
		} ;
	} ;
	NB=NB-1 ;
} /* fin elimination */ ;

static void retrait(struct rproblem *RR, int i1) 
/* cette procedure permute les lignes i1 (qui sera eliminee) et mb */
{ 
	if (i1 != MB)
	{
		int j ; float s ;
		s=INF[i1] ; INF[i1]=INF[MB] ; INF[MB]=s ;
		s= D[i1] ; D[i1]= D[MB] ; D[MB]=s ;
		j=G[i1] ; G[i1]=G[MB] ; G[MB]=j ;
		for (j=0 ; j<=N1 ; j++)
		{
			s= A(i1,j) ; A(i1,j)= A(MB,j) ; A(MB,j)=s ;
		} ;
	} ;
	MB=MB-1 ;
} /* fin retrait */ ;

static void reduction(struct rproblem *RR)
/*      cette procedure groupe les vecteurs provenant de l'elimination des
	egalites dans les colonnes NB+1 a n */
{	int j;
	for (j=N1 ; j>=1 ; j--)
	if (B[j] > N1)
	{
		if (E[B[j]-N1] == 0) elimination(RR,j) ;
	} ;
	for (j=M1 ; j>=1 ; j--)
	  if (VARIABLELIBRE(G[j])) retrait(RR,j) ;
} /* fin reduction() */ ;

static int validite (struct rproblem *RR)
/*	cette procedure verifie, lorsque la base est donnee ou lorsqu'on
	utilise une matrice des contraintes ayant deja pivote, que le
	tableau g ou que les tableaux g et b fournis sont compatibles */
{	int i,j;
	for (i=1 ; i<=M1 ; i++)
	{
		if (G[i]<1 || G[i]>N1+M1)
			return(VRESULT=VRCAL) ;
		for (j=i+1 ; j<=M1 ; j++)
			if (G[j]==G[i]) return(VRESULT=VRCAL) ;
		if (COST || RHS)
		{
			for (j=1 ; j<=N1 ; j++)
			  if (B[j]==G[i]) return(VRESULT=VRCAL);
		}
		else
		if (G[i]>N1 && G[i]!=N1+i)
		{
			j=G[G[i]-N1] ; G[G[i]-N1]=G[i] ; G[i]=j ;
			i=i-1 ;
		} ;
	} ;
	if (COST || RHS)
	for (j=1 ; j<=N1 ; j++)
	{
		if (B[j]<1 || B[j]>N1+M1)
			return(VRESULT=VRCAL) ;
		for (i=j+1 ; i<=N1 ; i++)
			if (B[i]==B[j]) return(VRESULT=VRCAL) ;
	}
	return(0) ;
} /* fin validite */ ;

static void precision(struct rproblem *RR,int i,int nx) 
/*
	cette procedure evalue l'incertitude moyenne dont sont affectes
	les termes de la ligne i du tableau a */
{
	int j;
	float s ;
	if (i==0) s=1
	; else
		s=(i>M2 || G[i]>N1) ? 1 : 0 ;
	for (j=J0 ; j<=nx ; j++)
		if (B[j]>N1) s=s+absf(A(i,j)) ;
	s=s*RR->eps1 ;
	if (s>INF[i]) INF[i]=s ;
} /* fin precision */ ;

static void precisioncolonne(struct rproblem *RR, int j, int mx) 
/*
	cette procedure evalue l'incertitude moyenne dont sont affectes
	les termes de la colonne j du tableau a, ou ceux de la colonne
	des seconds membres (j=-1) */
{
	int i ;
	float s ;
	if (j != -1)
	{
		s=(B[j]>N1) ? 0 : 1 ;
		for (i=1 ; i<=mx ; i++)
		if (G[i] <= N1)
		{
			if (A(i,j) != 0) s=s+absf(A(i,j)) ;
		}
	}
	else
	{
		s= CONORM(-1) ;
		for (i=1 ; i<=mx ; i++)
		if (G[i] <= N1)
		{
			if (D[i] != 0) s=s+absf(D[i]) ;
		} ;
	} ;
	s=s*RR->eps1 ;
	if (s > INFCOL(j)) INFCOL(j)=s ;
} /* fin precisioncolonne */ ;
static int pivotage(struct rproblem *RR, int i1, int j1)
/*	le pivotage de la matrice a s'effectue a partir de la ligne i1 et de
	la colonne j1. Les numeros de la variable d'ecart (basique) de la
	ligne i1, et de la variable independante (hors-base) de la colonne j1,
	contenus respectivement dans les tableaux g et b, sont permutes. Dans
	le cas de l'elimination d'une egalite, l'operation equivaut a chasser
	de la base une variable d'ecart identiquement nulle */
{
	float p1,p2,s ;
	int i,j,mx,nx ;
	if (ITER==ITEMAX) return(VRESULT=VRINS);
	s= A(i1,j1); if (absf(s)<RR->eps2) return(VRESULT=VREPS);
	nx=N1 ; mx=M1 ; if (TESTP>0) nx=NB ; if (TESTP>2) mx=MB ;
	ITER=ITER+1 ; p1=1/s ;
	j=B[j1] ; B[j1]=G[i1] ; G[i1]=j ;
	for (i=0 ; i<=mx ; i++)
	if (i != i1 && A(i,j1) !=0)
	{
		p2= A(i,j1)*p1 ; D[i]= D[i]-D[i1]*p2 ;
		if (RR->sommaire) if (absf(D[i])<RR->epss) D[i]=0 ;
		for (j=J0 ; j<=nx ; j++)
			if (j==j1) A(i,j)= -p2
			; else
			if (A(i1,j) != 0)
			{
				A(i,j)= A(i,j)-p2* A(i1,j) ;
				if (RR->sommaire)
					if (absf(A(i,j))<RR->epss) A(i,j)=0 ;
			} ;
		if (! RR->sommaire) precision(RR,i,nx) ;
	} ;
	D[i1]= D[i1]*p1 ;
	for (j=J0 ; j<=nx ; j++)
		A(i1,j)=(j==j1) ? p1 : A(i1,j)*p1 ;
	if (RR->sommaire) goto finpivotage ;
	p1=absf(p1) ;
	INF[i1]=INF[i1]*p1 ;
	/* les quantites inferieures en valeurs absolue a la tolerance
		correspondante sont annulees */
	if (D[i1] != 0)
	{	precisioncolonne(RR,-1,mx) ;
		s= INFCOL(-1)* CONORM(-1) ;
		for (i=0 ; i<=mx ; i++)
		if (D[i] != 0)
		{
			if (D[i]* D[i] <= INF[i]*s) D[i]=0 ;
		} ;
	} ;
	for (j=J0 ; j<=nx ; j++)
	if (j==j1) INFCOL(j)= INFCOL(j)*p1
	; else
	if (A(i1,j) != 0)
	{	precisioncolonne(RR,j,mx) ;
		s= INFCOL(j) ;
		for (i=0 ; i<=mx ; i++)
		if (A(i,j) != 0)
		{
			if (A(i,j)* A(i,j) <= INF[i]*s) A(i,j)=0 ;
		} ;
	} ;
finpivotage:
	voir4(RR,30,i1,j1) ;
	return(0);
} /* fin pivotage */ ;
static int simplexe(struct rproblem *RR,int i2,int bphase2,int *pj1,int *pbool1)
/*		Cette procedure minimise le cout defini dans la ligne i2
	(s'il s'agit du cout artificiel, il est represente en valeur opposee)*/
{
	int i,j;
	int i1= 0,i3 =0;
	float gamma,teta,c1,c2,zeta,tet,pivot ;
	float alpha =0;
	/* en phase 1, bphase2 est faux */
choix:
	/*
	Le pivot est choisi de facon a assurer la diminution la plus grande du
	cout. Si plusieurs pivots realisent cette condition (ce qui se produit
	essentiellement dans le cas d'un sommet multiple, la plus grande
	diminution possible pouvant etre nulle), on prendra celui situe dans
	la colonne dont le facteur de cout relatif est le plus negatif, et
	dans une meme colonne, le plus grand pivot. Si au cours de
	l'exploration d'une colonne aucun coefficient positif n'est rencontre,
	on se renvoie a l'etiquette solution infinie */
	*pbool1=1 ; c1=0 ; tet=0 ;
	for (j=J0 ; j<=NB ; j++)
	{
		teta=(bphase2) ? A(i2,j) : -A(i2,j) ;
		if (teta < 0)
		{
			*pbool1=1 ; pivot=0 ;
			for (i=1 ; i<=MB ; i++)
			if (A(i,j) > 0)
			{
				zeta= D[i]/ A(i,j) ;
				if (*pbool1
				|| zeta<alpha
				|| zeta==alpha && pivot<A(i,j))
				{
					*pbool1=0;
					alpha=zeta ; i3=i ; pivot= A(i,j) ;
				}
			} /* fin i */ ;
			if (*pbool1) return(VRESULT=VRINF);
			c2=alpha*teta ;
			if (c2<c1 || c2==c1 && teta<tet)
			{
				c1=c2 ; *pj1=j ; i1=i3 ; tet=teta ;
			}
		}
	} /* fin j */ ;
	if (*pbool1) goto optimum ;
	if (pivotage(RR, i1,*pj1)) return(VRESULT);
	/* si la variable artificielle quitte la base, on sort de la
		procedure simplexe */
	if (i1==i2) goto sortie ;
	if (bphase2 || D[i2]>0) goto choix ;
optimum:
	/* le cout minimum est atteint */
	if (bphase2) goto sortie ;
	/* on chasse la variable artificielle de la base */
	gamma=0 ;
	for (j=J0 ; j<=NB ; j++)
	if (absf(A(i2,j))>gamma)
	{
		gamma=absf(A(i2,j)) ; *pj1=j ;
	} ;
	if (pivotage(RR, i2,*pj1)) return(VRESULT);
sortie:
	/* en phase 1 le booleen bool1 indique si le polyedre est vide */
} /* fin simplexe */
static int simplexedual(struct rproblem *RR, int j2, int bphase2, int *pi1, int *pbool1)
/*	cette procedure maximise dans le systeme dual soit le cout defini
	par la colonne des seconds membres (lorsque j2=0), soit le cout
	defini par la colonne j2 */
{
	int i,j;
	int j1 = 0 ,j3 = 0 ;
	float gamma,teta,c1,c2,zeta,tet,pivot ;
	float alpha=0;
	J0=1;
choix:
	*pbool1=1 ; c1=0 ; tet=0 ;
	for (i=1 ; i<=MB ; i++)
	{
		teta=(bphase2) ? D[i] : A(i,j2) ;
		if (teta < 0)
		{
			*pbool1=1; pivot=0 ;
			for (j=1 ; j<=NB ; j++)
			if (A(i,j)<0) {
			  zeta= A(0,j)/ A(i,j) ;
			  /*if (A(0,j)<0.0001) zeta=0; provisoire*/
			  /*fprintf(FT,"zetafoir d=%6.3f a=%15.9f\\\\\n",D[i],A(0,j)) ;*/
			  if (*pbool1
			      || zeta>alpha
			      || zeta==alpha && pivot>A(i,j))
			    {
			      *pbool1=0 ;
			      alpha=zeta ; j3=j ; pivot= A(i,j) ;
			    }
			} /* fin j */ ;
			if (*pbool1) return(VRESULT=VRVID);
			c2=alpha*teta ;
			if (c2>c1 || c2==c1 && teta<tet)
			{
				c1=c2 ; *pi1=i ; j1=j3 ; tet=teta ;
			} ;
		}
	} /* fin i */ ;
	if (*pbool1) goto optimum ;
	if (pivotage(RR,*pi1,j1)) return(VRESULT);
	if (j1==j2) goto sortie ;
	if (bphase2 || A(0,j2)>0) goto choix ;
optimum:
	if (bphase2) goto sortie ;
	/* nulle ou non, la variable artificielle est chassee de la base
			duale */
	gamma=0 ;
	for (i=1 ; i<=MB ; i++)
	if (gamma < absf(A(i,j2)))
	{
		gamma=absf(A(i,j2)) ; *pi1=i ;
	} ;
	if (pivotage(RR,*pi1,j2)) return(VRESULT);
sortie:
} /* fin simplexedual */ ;
static int simbad(struct rproblem *RR)
{ 
int i,j,i1,j1,bphase2,bool1,risqueinfini; float s,gamma,pivot;
        RR->sommaire=(TESTP>1) ? 1 : 0 ;
	if (NP<0)
	  { if (TRACE) fprintf(FT,"Negative value for nvpos:%d\n",NP);
	    return(VRCAL);
	  }
	if (N1<NP)
	  { if (TRACE)fprintf(FT,"nvar (%d) < nvpos (%d)\n",N1,NP);
	    return(VRCAL);
	  }
	if (M1<0)
	  { if (TRACE) fprintf(FT,"Negative value for mcontr:%d\n",M1);
	    return(VRCAL);
	  }
	ITER=0 ; ITEMAX=2*M1+N1 ; RR->eps1=RR->eps*10 ;
	RR->eps2=RR->eps*100 ;RR->epss=RR->eps*10 ;
	J0=1 ; bphase2=1 ; NB=N1 ; M2=M1 ;
	MB=M1 ; VRESULT=0;
	risqueinfini=0 ;
	if (E[0]>0) E[0]=1;
	if (E[0]<0) E[0]= -1;
	if (COST<0) /* apres modification directe de la matrice */
	  {
	  }
	else
	if (! COST && ! RHS)
	{
		for (j=1 ; j<=N1 ; j++) B[j]=j ;
		voir(RR,1) ;
		for (i=0 ; i<=M1 ; i++)
		if (E[i]== -1)
		{
			/*	on donne aux inequations le meme sens et on
				se ramene eventuellement a un probleme de
				minimisation */
			D[i]= -D[i] ;
			for (j=1 ; j<=N1 ; j++) A(i,j)= -A(i,j) ;
		} ;
		voir(RR,2) ;
	    if (! RR->sommaire)
	    {
		for (i=1 ; i<=M1 ; i++)
		{
			/* normalisation des lignes */
			s=0 ;
			for (j=1 ; j<=N1 ; j++)
				if (s < absf(A(i,j))) s=absf(A(i,j)) ;
			if (s==0) s=1 ;
			for (j=1 ; j<=N1 ; j++)
				A(i,j)= A(i,j)/s ;
			CONORM(N1+i)=s ;
		} ;
	voir(RR,3) ;
		for (j=1 ; j<=N1 ; j++)
		{
			/* normalisation des colonnes */
			s=0 ;
			for (i=1 ; i<=M1 ; i++)
				if (s < absf(A(i,j))) s=absf(A(i,j)) ;
			if (s==0) s=1 ;
			if (s<1)
			{
				for (i=1 ; i<=M1 ; i++)
					A(i,j)= A(i,j)/s ;
				CONORM(j)=1/s ;
			}
			else CONORM(j)=1 ;
		} ;
	voir(RR,4) ;
		norm(RR) ;
	voir(RR,5) ;
		for (i=0 ; i<=M1+1 ; i++) INF[i]=0 ;
		for (j= -1 ; j<=N1 ; j++) INFCOL(j)=0 ;
	    } ;
/*#include "base.c"*/
		if (BASE)
		{
			if (validite(RR)) return(VRESULT) ;
			for (i=1 ; i<=M1 ; i++)
			if (G[i] <= N1)
			{
				s=0 ;
				for (i1=i ; i1<=M1 ; i1++)
				if (G[i1] <= N1)
				{
					if (s <= absf(A(i,G[i1])))
					{
						s=absf(A(i,G[i1])) ; j=i1 ;
					} ;
				} ;
				j1=G[j] ; G[j]=G[i] ; G[i]=N1+i ;
				if (s==0)
				{
					for (i1=i+1 ; i1<=M1 ; i1++)
						G[i1]=N1+i1 ;
					return(VRESULT=VRNUL) ;
				} ;
				M2=i ;
				if (pivotage(RR,i,j1)) return(VRESULT);
			} ;
			reduction(RR) ;
		}
		else
/* reprendre ................................................ */
			for (i=1 ; i<=M1 ; i++) G[i]=N1+i ;
/*#include "corh.c"*/
	}
	else
	{
		if (validite(RR)) return(VRESULT) ;
		reduction(RR) ;
		norm(RR) ;
		if (E[0]== -1) D[0]= -D[0] ;
		if (COST)
		{
			/* calcul de la ligne de la nouvelle fonction
			cout a l'aide de l'inverse de la base du systeme dual.
			pi est utilise comme tableau auxiliaire */
			for (j=1 ; j<=N1 ; j++)
				PI[j]=(E[0]== -1) ? -A(0,j) : A(0,j) ;
			for (j=1 ; j<=N1 ; j++)
			{
				A(0,j)=(B[j]>N1) ? 0 : PI[B[j]] ;
				for (i=1 ; i<=M1 ; i++)
				if (G[i] <= N1)
					A(0,j)= A(0,j)-A(i,j)*PI[G[i]];
			} ;
			INF[0]=0 ; precision(RR,0,N1) ;
			s=INF[0] ;
			for (j=1 ; j<=N1 ; j++)
				if (A(0,j)* A(0,j) <= s* INFCOL(j)) A(0,j)=0 ;
		} ;
		if (RHS)
		{
			/* calcul de la colonne des nouveaux seconds
			membres a l'aide de l'inverse de la base (systeme
			primal). x est utilise comme tableau auxiliaire */
			for (i=1 ; i<=M1 ; i++)
				X[i]=(E[i]== -1) ? -D[i] : D[i] ;
			for (i=0 ; i<=M1 ; i++)
			{
				if (i != 0)
				D[i]=(G[i]>N1) ? X[G[i]-N1] : 0 ;
				for (j=1 ; j<=N1 ; j++)
					if (B[j]>N1)
						D[i]= D[i]+A(i,j)*X[B[j]-N1] ;
			} ;
			INFCOL(-1)=0 ; precisioncolonne(RR,-1,M1) ;
			s= INFCOL(-1)* CONORM(-1) ;
			for (i=1 ; i<=M1 ; i++)
				if (D[i]* D[i] <= s*INF[i]) D[i]=0 ;
		}
		else
		for (i=1 ; i<=M1 ; i++)
			if (G[i] <= N1)
				D[0]= D[0]-D[i]*PI[G[i]] ;
/* reprendre ................................................ */
	} ;
	voir(RR,6) ;
	/*	elimination des egalites */
	for (i=M1 ; i>=1 ; i--)
	if (E[i]==0 && G[i]==N1+i)
	{
		pivot=0 ;
		for (j=NB ; j>=1 ; j--)
		if (pivot>=0)
		{
			s=absf(A(i,j)) ;
			if (s>pivot)
			{
				pivot=s ; j1=j ;
				/* la premiere variable libre possible est
					choisie d'autorite,signalee par pivot*/
				if (VARIABLELIBRE(B[j])) pivot= -1 ;
			}
		} ;
		if (pivot==0)
		{
			/*	l'equation est surabondante ou incompatible */
			if (D[i] != 0) return(VRESULT=VRVID);
			/*	l'equation surabondante est ignoree */
		}
		else
		{
			/*	l'equation est changee en inequation dont la
				variable d'ecart provient du vecteur j1, qui
				entre dans la base */
			if (pivotage(RR,i,j1)) return(VRESULT);
			elimination(RR,j1) ;
			if (pivot<0) retrait(RR,i) ;
		} ;
	} ;
	voir(RR,7) ;
	/*	le systeme auquel on s'est ramene ne comporte que des
		inequations, dont le nombre de variables est egal a NB. dans
		les colonnes NB+1 a n ont ete groupes les vecteurs provenant
		des pivotages effectues au cours des eliminations des egalites
		ces vecteurs ne sont pas detruits, car ils appartiennent a
		l'inverse de la base, utilisee dans les tests de precision, et
		eventuellement dans le calcul de la colonne des nouveaux
		seconds membres */
	/*	traitement variables libres */
	for (j=NB ; j>=1 ; j--)
	if (VARIABLELIBRE(B[j]))
	{
		pivot=0 ;
		for (i=MB ; i>=1 ; i--)
		{
			s=absf(A(i,j)) ;
			if (s>pivot)
			{
				pivot=s ; i1=i ;
			} ;
		} ;
		if (pivot==0)
		{
			/*	la variable n'intervient pas dans le domaine
				mais si le cout relatif n'est pas nul, il ne
				pourra exister de solution finie */
			if (A(0,j) != 0) risqueinfini=1 ;
			elimination(RR,j) ;
		}
		else
		{
			/*	on fait entrer dans la base la variable libre
				la variable d'ecart resultante sera desormais
				ignoree */
			if (pivotage(RR,i1,j)) return(VRESULT);
			retrait(RR,i1) ;
		} ;
	} ;
	if (DUAL) goto methodeduale ;
methodeprimale:
	voir(RR,8) ;
	gamma=0 ;
	for (i=1 ; i<=MB ; i++)
	if (D[i]<gamma)
	{
		bphase2=0 ; gamma= D[i] ; i1=i ;
	} ;
	if (bphase2) goto phase2 ;
	/*	une premiere phase est necessaire. une (seule) variable
		artificielle est introduite dans le systeme au moyen de la
		colonne 0 du tableau a */
	voir(RR,9) ;
	J0=0 ;
	B[0]=0 ; A(0,0)=0 ; INFCOL(0)=0 ;
	for (i=1 ; i<=M1 ; i++)
		A(i,0)=(D[i]<0 && i<=MB) ? -1 : 0 ;
	voir(RR,10) ;
	if (pivotage(RR,i1,0)) return(VRESULT);
	voir(RR,11) ;
        (simplexe(RR,i1,bphase2,&j1,&bool1)) ; if (VRESULT) return(VRESULT);
	B[j1]=B[0] ; INFCOL(j1)= INFCOL(0) ;
	for (i=0 ; i<=M1 ; i++) A(i,j1)= A(i,0) ;
	if (bool1) return(VRESULT=VRVID);
	J0=1 ; bphase2=1 ;
	/* fin de la premiere phase avec obtention d'une solution
		realisable */
phase2:
	voir(RR,12) ;
        (simplexe(RR,0,bphase2,&j1,&bool1)) ; if (VRESULT) return(VRESULT);
	goto resultats ;
methodeduale:
	voir(RR,13) ;
	gamma=0 ;
	for (j=1 ; j<=NB ; j++)
		if (A(0,j)<gamma)
		{
			bphase2=0 ; gamma= A(0,j) ; j1=j ;
		} ;
	if (bphase2) goto phase2dual ;
	/*	une (seule) variable artificielle est ajoutee au systeme dual
		au moyen de la ligne mb+1 du tableau a */
	voir(RR,14) ;
	M2=M1=M1+1 ;
	MB=MB+1 ;
	/* ligne conservee uniquement pour tests de precision */
	if (MB!=M1)
	{
		G[M1]=G[MB] ; D[M1]= D[MB] ; INF[M1]=INF[MB] ;
		for (j=1 ; j<=N1 ; j++) A(M1,j)= A(MB,j) ;
	} ;
	G[MB]=M1+N1 ; D[MB]=0 ; INF[MB]=0 ;
	for (j=1 ; j<=NB ; j++)
		A(MB,j)=(A(0,j)<0) ? 1 : 0 ;
	for (j=NB+1 ; j<=N1 ; j++) A(MB,j)=0 ;
	voir(RR,15) ;
	if (pivotage(RR,MB,j1)) return(VRESULT);
	voir(RR,16) ;
	simplexedual(RR,j1,bphase2,&i1,&bool1); if (VRESULT) return(VRESULT);
	if (i1 != MB)
	{
		D[i1]= D[MB] ; G[i1]=G[MB] ; INF[i1]=INF[MB] ;
		for (j=1 ; j<=N1 ; j++) A(i1,j)= A(MB,j) ;
	} ;
	if (MB != M1)
	{
		D[MB]= D[M1] ; G[MB]=G[M1] ; INF[MB]=INF[M1] ;
		for (j=1 ; j<=N1 ; j++) A(MB,j)= A(M1,j) ;
	} ;
	M2=M1=M1-1 ;
	MB=MB-1 ;
	voir(RR,17) ;
	bphase2=1 ;
	if (bool1)
	{
		/*	il n'existe pas de solution realisable dans le domaine
			dual, mais on ignore si cela correspond a un polyedre
			vide ou a une solution infinie pour le probleme
			primal*/
		voir(RR,18) ;
		ITEMAX=ITEMAX+ITER ; goto methodeprimale ;
	} ;
phase2dual:
	voir(RR,19) ;
	simplexedual(RR,0,bphase2,&i1,&bool1); if (VRESULT) return(VRESULT);
resultats:
	if (risqueinfini) return(VRESULT=VRINF);
	for (i=1 ; i<=M1+N1 ; i++) X[i]=0 ;
	if (RR->sommaire)
	{
		for (i=1 ; i<=M1 ; i++) X[G[i]]= D[i] ;
		X[0]= D[0] ;
	}
	else
	{
	for (i=1 ; i<=M1 ; i++)
		X[G[i]]= D[i]* CONORM(G[i]) ;
	X[0]= D[0]* CONORM(0) ;
	} ;
	if (E[0]== -1) X[0]= -X[0] ;
	for (j=1 ; j<=N1+M1 ; j++) PI[j]=0 ;
	if (RR->sommaire)
	  for (j=1 ; j<=N1 ; j++)
		PI[B[j]]= A(0,j) ;
	else
	  for (j=1 ; j<=N1 ; j++)
		PI[B[j]]= A(0,j)* CONORM(0)/ CONORM(B[j]) ;
	return(VRESULT=VRFIN) ;
} /* fin simbad */ ;
