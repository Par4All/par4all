/*

  $Id$

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include "genC.h"    
#include "database.h"  
#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "transformations.h"
#include "locality.h"

#define DEFAULT_INT_PREFIX 	"I_"
#define DEFAULT_FLOAT_PREFIX 	"F_"
#define DEFAULT_LOGICAL_PREFIX 	"L_"
#define DEFAULT_COMPLEX_PREFIX	"C_"
#define DEFAULT_STRING_PREFIX	"S_" 
#define  depth_max  40    /* La profondeur des differents  nids */
#define  st_max 40       /* Le nombre maximum de stencils q'un nid peut contenir */
#define nid_nbr 6    /* nombre de nid de la sequence */


/* La structure d'une boucle : son index et ces deux bornes */

typedef struct INFO_LOOP {    
  Variable index;
  Value upper, lower;  
} info_loop;


/* La structure d'un nid */

typedef struct NID { 
  entity tableau;       /*  le tableau en ecriture */
  Pmatrix  *st;        /* les stencils */
  info_loop *nd;       /* les boucles */ 
  int nbr_stencil;    /* nombre de stencils */
  Pmatrix delai;      /* le delai qu'on doit ajouter a ce nid */ 
  statement s;        /* le corps de ce nid */
  Value surface;      /* le volume memoire du tableau en ecriture de ce nid */
  Pmatrix coef;       /* le coefficient des fonctions d'acces  */
  Pvecteur pv_acces;     /*  */ 
  entity ref;           /*   */
  Pmatrix n;
} nid;

static Pvecteur *tiling_indice;
static int k1=0, k2=0;  /* k1 represente le nombre de nid et k2 leur profondeur */

static nid  sequen[nid_nbr]; /* la sequence de nids */

static int depth;     /* la pronfondeur des nids */

static info_loop   *merged_nest, *tiled_nest;  /* le nid fusionne */

loop loop1;       /* une copie de l'un des nids de la sequence */

entity first_array;  /* le tableau en entree */

/* J'ai ameliore la fonction make_new_scalar_variable */
/* afin de l'etendre  a des tableau   */
static entity internal_make_new_array_variable(int i, int j, entity module,
                         basic b,list lis)
{
  char *buffer;
  if(j==-1)
  asprintf(&buffer,"%s%d", "B",i);
  else 
    asprintf(&buffer,"%s%d_%d", "B",i,j);
  entity e = make_new_array_variable_with_prefix(buffer, module, b,lis);
  free(buffer);
  return e;
}

/* Compteurs des suffixes de nouvelles references */
/*
static int unique_integer_number = 0,
  unique_float_number = 0,
  unique_logical_number = 0,
  unique_complex_number = 0,
  unique_string_number = 0;
*/
/* first_turn permet de detecter le premier retour de la fonction loop_rwt()  */
/* overflow permet de detecter un debordemebnt dans l'un des nids */

static bool first_turn=false, overflow=false;


/* pour marquer si la sequence repond a nos hypotheses */

typedef enum {is_a_stencil, is_a_continue, is_a_no_stencil } contenu_t;


static bool loop_flt(loop l )
{  
  range range1;
  expression lower, upper;
  normalized norm1, norm2;
  Variable  index;
  Pvecteur pv1,pv2;
  if( ! first_turn )        /* la premiere boucle d'un nid donne */
    {
      loop1=copy_loop(l);
      first_turn = true; 
      /* allouer de la memoire pour les differentes boucles de ce nid */
      sequen[k1].nd= (struct INFO_LOOP  *)malloc(depth_max *sizeof(struct INFO_LOOP));
      /* allouer de la memoire pour les differents stencils  de ce nid */
      sequen[k1].st= ( Pmatrix *)malloc(st_max *sizeof(Pmatrix));
    }; 
  index =(Variable) loop_index(l);
  range1=loop_range(l);    
  lower=range_lower(range1);  /* la borne inferieure */  
  upper=range_upper(range1);  /* la borne superiere */
  normalize_all_expressions_of(lower);
  normalize_all_expressions_of(upper);
  norm1=expression_normalized(lower);
  norm2=expression_normalized(upper);
  pips_assert("normalized are linear", normalized_linear_p(norm1)&& normalized_linear_p(norm2));
  pv1= normalized_linear(norm1);
  pv2= normalized_linear(norm2); 
  sequen[k1].nd[k2].index=index;    /* stocker l'index de la boucle l */
  sequen[k1].nd[k2].lower=vect_coeff(TCST,pv1);  /* stocker sa borne inferieure */
  sequen[k1].nd[k2].upper=vect_coeff(TCST,pv2);   /* stocker sa borne superieure  */
  k2++;    /* incrementer le compteur des profondeur des nids */
  return true;
} 

static void loop_rwt(loop l, context_p context  ) 
{  
  contenu_t contenu;
  intptr_t depth;
  statement s = loop_body(l);    /* recuperer le coprs de le boucle l */
  contenu = (contenu_t) hash_get(context->contenu, s);
  depth= (intptr_t) hash_get(context->depth, s);   
  depth++;
  hash_put(context->depth,stack_head(context->statement_stack), ( void *) depth); 
  hash_put(context->contenu,stack_head(context->statement_stack), (void *) contenu  );
  if (first_turn)
    {
      first_turn=false;
      k2=0;
      k1++;
    }
}

static bool stmt_flt(statement s,context_p context )
{  
  stack_push(s, context->statement_stack); 
  return true;
}


static void stmt_rwt( statement s, context_p context)
{  
  stack_pop(context->statement_stack);   
}

static bool seq_flt(   )
{
  return true;
}

static void seq_rwt(sequence sq, context_p context)
{
  contenu_t contenu;
  intptr_t depth1=0, depth2=0;
  intptr_t max=0;
  int i=0;
  list l= sequence_statements(sq);
  contenu=is_a_stencil;
  hash_put(context->contenu, stack_head(context->statement_stack), (void *) contenu);
  MAP(STATEMENT, s,
  {
    contenu = (contenu_t) hash_get(context->contenu, s);
    if (i==0) depth1 = (intptr_t ) hash_get(context->depth, s);
    if (contenu ==is_a_stencil)  
      {  
	depth2= (intptr_t ) hash_get(context->depth, s);
	if (depth1!=depth2)
	  {  
	    contenu=is_a_no_stencil;
	    hash_put(context->contenu, stack_head(context->statement_stack),(void *) contenu);
	  };
	depth1 =  depth2;
      }
    else 
      {
	if ( contenu !=is_a_continue)
	  { 
	    contenu=is_a_no_stencil;
	    hash_put(context->contenu, stack_head(context->statement_stack),(void *) contenu);
	  };
      };
    if (depth2 > max) max=depth2;
    i++;
  };, l); 
  hash_put(context->depth,stack_head(context->statement_stack), (void *) max   );
} 

static bool uns_flt(   )
{
  return true;
}

static void uns_rwt( context_p context)
{  contenu_t contenu;
 
 contenu=is_a_no_stencil; 
 hash_put(context->contenu,stack_head(context->statement_stack),(void *)contenu );
 hash_put(context->depth,stack_head(context->statement_stack), ( void *) 0);  
}

static bool test_flt( )
{
  return true;
}

static void test_rwt( context_p context)
{
  contenu_t contenu;
  contenu=is_a_no_stencil; 
  hash_put(context->contenu, stack_head(context->statement_stack), (void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), ( void *) 0);   
} 
static bool call_flt( )
{
  return true ;
}

static void call_rwt(call  ca, context_p context)
{ 
  contenu_t contenu=is_a_stencil;
  if ((strcmp(entity_name(call_function(ca)),"TOP-LEVEL:CONTINUE" )==0)||
      (strcmp(entity_name(call_function(ca)),"TOP-LEVEL:RETURN" )  ==0))
    {
      contenu=is_a_continue; 
    }
  else
    {
      if (strcmp(entity_name(call_function(ca)),"TOP-LEVEL:=")==0)
	{
	  list lis;
	  expression gauche=NULL, droite=NULL;
	  syntax stg,std;
          normalized norm;
	  Pvecteur pv;       
	  int i=0,j=0;
	  lis =call_arguments(ca);
	  MAP(EXPRESSION,exp,{  
	    if (i==0)  gauche=exp;
	    if (i==1)  droite=exp;
	    i++;
	  },lis) ; 
	  stg= expression_syntax(gauche);
	  std= expression_syntax(droite);
	  if ((syntax_reference_p(stg))&&(syntax_call_p(std))&&(i==2))
	    { 
	      reference ref;
              list lis,lis2;
	      Variable vt=NULL;
	      Pvecteur pvt;
	      Value v1;
	      ref =syntax_reference(stg);
	      lis  =reference_indices(ref);
	      i=1;
	      sequen[k1].tableau=reference_variable(ref);
	      MAP(EXPRESSION,exp,{ 
                if (k2-i >=0) {
		  vt=sequen[k1].nd[k2-i].index;
		}
                else{
		  contenu=is_a_no_stencil;
		}
		normalize_all_expressions_of(exp);
		norm=expression_normalized(exp);
		pv= normalized_linear(norm);
                pvt = vect_make(VECTEUR_NUL, vt, VALUE_ONE,
				TCST,VALUE_ZERO);
		if (!vect_equal(pvt, pv))
		  {
		    contenu=is_a_no_stencil;
		  } ;
		i++;
	      },lis) ;      
	      if (i-1!=k2)
		{
		  contenu=is_a_no_stencil;
		};
	      lis  =call_arguments( syntax_call(std));
	      j=0;
	      MAP(EXPRESSION,exp,{ 
		ref =expression_reference(exp);
		if(k1==0)
		  first_array=reference_variable(ref);
		if(k1>0)
		  {
		    if (reference_variable(ref) !=sequen[k1-1].tableau)
		      contenu=is_a_no_stencil;
		  };
		lis2  =reference_indices(ref);
		i=1;
		sequen[k1].st[j]=matrix_new(k2,1);
		MAP(EXPRESSION,exp2,{
		  if (k2-i >=0) 
		    vt=sequen[k1].nd[k2-i].index;
		  else
		    contenu=is_a_no_stencil;
		  normalize_all_expressions_of(exp2);
		  norm=expression_normalized(exp2);
		  pv= normalized_linear(norm);
		  v1=vect_coeff(TCST,pv);
		  MATRIX_ELEM(sequen[k1].st[j],k2-i+1,1)=v1;
		  pvt = vect_make(VECTEUR_NUL, vt, VALUE_ONE,
				  TCST,v1);
		  if (!vect_equal(pvt, pv)){
		    contenu=is_a_no_stencil;
		  };
		  i++; 
		},lis2) ;  
		if (i-1!=k2){
		  contenu=is_a_no_stencil; };
		j++;
	      },lis) ; 
	      sequen[k1].nbr_stencil=j;
	    }
	  else
	    {    
              contenu=is_a_no_stencil; 
	    }; 
	}
      else
	{
	  contenu=is_a_no_stencil; 
	}
    }
  hash_put(context->contenu,stack_head(context->statement_stack), (void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), ( void *) 0);  
  sequen[k1].s= copy_statement( stack_head(context->statement_stack));
}


static void wl_rwt( context_p context)
{  
  contenu_t contenu;
  contenu=is_a_no_stencil; 
  hash_put(context->contenu,stack_head(context->statement_stack),(void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), ( void *) 0); 
} 

/* Cette fonction retourne true si  le vecteur 'a' est lexicographiquement superieur au vecteur 'b' */
static bool lexi_sup(Pmatrix a, Pmatrix b)
{
  int i;
  for (i=1;i<=depth;i++)
    {
      if (MATRIX_ELEM(a,i,1) >MATRIX_ELEM(b,i,1))
	return true;
      if (MATRIX_ELEM(a,i,1) < MATRIX_ELEM(b,i,1))
	return false;
    }
  return false;
} 

/* Cette fonction trie un tableau de stencil d'un  nid donne. st est le tableau 
   de stencils et length sa taille */

static void  trier(Pmatrix *st,int length)
{
  Value    temp;
  int   i, j,k;
  for (i=0;i<=length-2;i++)
    for(j=i+1;j<=length-1;j++)
      {
	if (lexi_sup(st[i],st[j]))
	  {
	    for (k=1;k<=depth;k++)
	      {   
		temp= MATRIX_ELEM(st[j],k,1);
		MATRIX_ELEM(st[j],k,1)=  MATRIX_ELEM(st[i],k,1);
		MATRIX_ELEM(st[i],k,1)=temp;
	      };
	  };
      };
}

/* Cette fonction calcule les delais qu'on doit ajouter aux differents nids */
 
static void compute_delay_merged_nest ()
{
  int i;
  for (i=k1-1;i>=0;i--)
    {
      sequen [i].delai=matrix_new(depth,1);
      if (i==k1-1)  matrix_nulle(sequen [i].delai);   /* le delaie  du dernier est le vecteur nul */
      else
	matrix_substract(sequen [i].delai, sequen [i+1].delai,   
			 sequen[i+1].st[sequen[i+1].nbr_stencil-1]);
      /*d_k=d_{k+1} - plus grand vecteur stencil du nid  k+1*/
    }
}
 
static void compute_delay_tiled_nest ()
{
  int i,j,k;
  Pmatrix max_st=matrix_new(depth,1);
  
  for (i=k1-1;i>=0;i--)
    {
      sequen [i].delai=matrix_new(depth,1);
      if (i==k1-1)  matrix_nulle(sequen [i].delai);   /* le delaie  du dernier est le vecteur nul */
      else
	{  
	  for(k=0;k<=depth-1;k++)
	    {
	      Value val;
	      val=MATRIX_ELEM(sequen[i+1].st[0],k+1,1);
	      for (j=1;j<=sequen[i+1].nbr_stencil-1;j++)
	      {
		if ( MATRIX_ELEM(sequen[i+1].st[j],k+1,1)>val)
		  val=MATRIX_ELEM(sequen[i+1].st[j],k+1,1);
		
	      };
	      MATRIX_ELEM(max_st,k+1,1)=val;
            };
	  matrix_substract(sequen [i].delai, sequen [i+1].delai, max_st);
	}
   
      /*d_k=d_{k+1} - plus grand vecteur stencil du nid  k+1*/
    }
  
}


/* Cette fonction calcule les bornes du nid fusionne */
static void compute_bound_merged_nest ()
{
  int i,j;
  merged_nest= (struct INFO_LOOP  *)malloc(depth *sizeof(struct INFO_LOOP));
  for(j=0;j<=depth-1;j++)
    {
      merged_nest[j].lower =sequen[0].nd[j].lower+MATRIX_ELEM( sequen[0].delai,j+1,1);
      merged_nest[j].upper=sequen[0].nd[j].upper+MATRIX_ELEM( sequen[0].delai,j+1,1);
      for (i=1;i<=k1-1;i++)
       {
	 if( merged_nest[j].lower > sequen[i].nd[j].lower+MATRIX_ELEM( sequen[i].delai,j+1,1)) 
	   merged_nest[j].lower =sequen[i].nd[j].lower+MATRIX_ELEM( sequen[i].delai,j+1,1 );
	 if( merged_nest[j].upper <sequen[i].nd[j].upper+MATRIX_ELEM( sequen[i].delai,j+1,1))
	   merged_nest[j].upper=sequen[i].nd[j].upper+MATRIX_ELEM( sequen[i].delai,j+1,1);
       };
    } 
}
static void compute_bound_tiled_nest ()
{
  int i,j;
  
  tiled_nest= (struct INFO_LOOP  *)malloc(depth *sizeof(struct INFO_LOOP));
  for(j=0;j<=depth-1;j++)
    {
      Value val;
      tiled_nest[j].lower =sequen[0].nd[j].lower+MATRIX_ELEM( sequen[0].delai,j+1,1);
      tiled_nest[j].upper=sequen[0].nd[j].upper+MATRIX_ELEM( sequen[0].delai,j+1,1);
      for (i=1;i<=k1-1;i++)
       {
	 if( tiled_nest[j].lower > sequen[i].nd[j].lower+MATRIX_ELEM( sequen[i].delai,j+1,1)) 
	   tiled_nest[j].lower =sequen[i].nd[j].lower+MATRIX_ELEM( sequen[i].delai,j+1,1 );
	 if( tiled_nest[j].upper <sequen[i].nd[j].upper+MATRIX_ELEM( sequen[i].delai,j+1,1))
	   tiled_nest[j].upper=sequen[i].nd[j].upper+MATRIX_ELEM( sequen[i].delai,j+1,1);
       };
      val =value_uminus(tiled_nest[j].lower);
      tiled_nest[j].lower =value_plus(tiled_nest[j].lower,val);
      tiled_nest[j].upper =value_plus(tiled_nest[j].upper,val);
      for (i=0;i<=k1-1;i++)
	MATRIX_ELEM(sequen[i].delai,j+1,1)=value_plus( MATRIX_ELEM(sequen[i].delai,j+1,1),val);
       

    } 
 
}



/* cette fonction donne le code fusionne */
static statement  fusion()
{ 
  int i,j;
  list lis=NULL;
  instruction ins;  
  sequence seq=NULL;
  statement s;
  loop ls=NULL;
  /* calculer les delais des differents nis */
  compute_delay_merged_nest();
  /* calculer les bornes du nid fusionne */
  compute_bound_merged_nest ();
  /* on construit la liste des instructions qui vont former le corps du nid fusionne */
  for (i=k1-1;i>=0;i--)
    {
      expression e1,e2,e,gauche=NULL,droite=NULL,delai_plus;
      test t;
      call c;
      int m;
      Pvecteur pv;
      e1=ge_expression(entity_to_expression ((entity)sequen[i].nd[0].index),
		       Value_to_expression( value_plus(sequen[i].nd[0].lower,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      e2=le_expression(entity_to_expression ((entity)sequen[i].nd[0].index),
		       Value_to_expression( value_plus(sequen[i].nd[0].upper,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      if (value_eq( value_plus(sequen[i].nd[0].lower,
			       MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].lower))
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,
				   MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].upper)) 
	    e=NULL;
	  else
	    e=e2;
	}
      else
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,
				   MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].upper)) 
	    e=e1;
	  else
	    e=and_expression(e1,e2);
	};
      for(j=0;j<=depth-1;j++)
	{
	  if (j>=1) 
	    {
	      e1=ge_expression(entity_to_expression((entity)sequen[i].nd[j].index),
			       Value_to_expression(value_plus(sequen[i].nd[j].lower,
							      MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      e2=le_expression(entity_to_expression ((entity)sequen[i].nd[j].index),
			       Value_to_expression( value_plus(sequen[i].nd[j].upper,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      if (value_eq( value_plus(sequen[i].nd[j].lower,
				       MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].lower))
		{
		  if (!value_eq( value_plus(sequen[i].nd[j].upper,
					    MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].upper)) 
		    {
		      if (e==NULL)
			e=e2;
		      else
			e=and_expression(e,e2);
		    };
		}
	      else
		{
		  if (value_eq( value_plus(sequen[i].nd[j].upper,
					   MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].upper)) 
		    {
		      if (e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		  else
		    {
		      e1=and_expression(e1,e2);
		      if(e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		};
	    };
	  c= instruction_call(statement_instruction((sequen[i].s )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    if (m==1)  droite=exp;
	    m++;
	  },call_arguments(c)) ;
	  pv = vect_make(VECTEUR_NUL, sequen[i].nd[j].index, VALUE_ONE,
			 TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	  delai_plus=Pvecteur_to_expression(pv);
      replace_entity_by_expression(gauche,(entity) sequen[i].nd[j].index,delai_plus);
      FOREACH(EXPRESSION,exp,call_arguments( syntax_call(expression_syntax(droite))))
          replace_entity_by_expression(exp,(entity) sequen[i].nd[j].index,delai_plus);
	};
      if (e==NULL)
	s=sequen[i].s;
      else
	{
	  t= make_test(e,sequen[i].s,make_block_statement(NIL));
	  s=test_to_statement(t);
	};
      if(i==k1-1)   lis=CONS(STATEMENT,s,NIL);
      else      lis=CONS(STATEMENT,s,lis);
    };
  seq= make_sequence(lis);
  ins= make_instruction_sequence(seq);
  s= instruction_to_statement(ins);
  ls=loop1;
  for(j=0;j<=depth-1;j++)
    {
      range range1;
      range1=loop_range(ls); 
      range_lower(range1)=int_to_expression(merged_nest[j].lower);
      range_upper(range1)= int_to_expression(merged_nest[j].upper);
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
      else loop_body(ls)=s;
    };
  s=loop_to_statement(loop1);
  return s;
} 

/*ON construit la coifficients des fonctions d'acces */
/*                   ->             ->                      */
/* si par exemple f( i )= (N, 1)^t( i- (1,1)^t) alors p= (N, 1)^t */
static void cons_coef(Pmatrix p, int nid)
{
  int i;
  Value temp;
  for(i=depth;i>=1 ;i--)
    {
      if (i==depth) MATRIX_ELEM(p,1,i)=1;
      else  
	{
	  temp= value_minus(sequen[nid].nd[i].upper,sequen[nid].nd[i].lower);
	  temp=value_plus(temp,VALUE_ONE);
	  MATRIX_ELEM(p,1,i)=  value_direct_multiply( MATRIX_ELEM(p,1,i+1),temp );
	} 
    }
}
/* On  construit  la foction d'acces  */
 /*                  ->             ->                        ->      */
/* si par exemple f( i )= (N, 1)^t( i- (1,1)^t) alors  pv= f( i ) */

static Pvecteur buffer_acces(int nid )
{
  Pvecteur pv=NULL,tmp;
  int j;
  for(j=0;j<=depth-1;j++)
    {
      if(j==0) 
	pv=vect_make(VECTEUR_NUL,sequen[nid].nd[j].index,MATRIX_ELEM(sequen[nid].coef,1,j+1), 
		     TCST,value_uminus(value_direct_multiply(MATRIX_ELEM(sequen[nid].coef,1,j+1),
		     value_plus( sequen[nid].nd[j].lower,
				 MATRIX_ELEM( sequen[nid].delai,j+1,1 ) ))));
      else 
	{
          tmp =pv;
	  pv = vect_make(tmp,sequen[nid].nd[j].index,MATRIX_ELEM(sequen[nid].coef,1,j+1),
	       TCST,value_uminus(value_direct_multiply(MATRIX_ELEM(sequen[nid].coef,1,j+1), 
	       value_plus(sequen[nid].nd[j].lower,
			  MATRIX_ELEM( sequen[nid].delai,j+1,1 ) ))));
	}
    };
  
  return pv;
} 



/* Cette fonction donne le code fusionne avec allocation des tampons */
static statement fusion_buffer()
{
  call c;
  int m;
  int i;
  int j;
  list lis,lis3,lisi=NULL;
                                 
  entity name;
 
  reference ref=NULL;
   Pmatrix temp1,temp2,temp3;
   sequence seq;
   instruction ins;
   statement s;
   loop ls;
  compute_delay_merged_nest();
  compute_bound_merged_nest ();
   

  temp1=matrix_new(1,1);
  temp2=matrix_new(depth,1);
  temp3=matrix_new(depth,1);
  for(i=0;i<=k1-1;i++)
    {
      expression e1,e2,e,exp,gauche=NULL,droite=NULL;
      test t;
      statement s;
      if (i < k1-1)
	{
	  sequen[i].coef=matrix_new(1,depth);
	  cons_coef(sequen[i].coef,i);
	  matrix_substract(temp2,sequen[i+1].delai, sequen[i].delai);
	  matrix_substract(temp3, temp2,sequen[i+1].st[0]);
	  matrix_multiply(sequen[i].coef,temp3,temp1);
	  sequen[i].surface=value_plus(MATRIX_ELEM(temp1,1,1),VALUE_ONE);
	  sequen[i].pv_acces= buffer_acces(i);
	  exp= binary_intrinsic_expression ("MOD",Pvecteur_to_expression(  sequen[i].pv_acces),
					    Value_to_expression( sequen[i].surface) );
	  lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(value_minus 
										      (sequen[i].surface,VALUE_ONE))), NIL);
	  name= internal_make_new_array_variable(i+1,-1,get_current_module_entity() , make_basic(is_basic_int, (void *) 4), lis);
	  
	  lis =CONS(EXPRESSION, exp, NIL);
	  ref=make_reference(name,lis);
	  sequen[i].ref=name; 
	}
      e1=ge_expression(entity_to_expression ((entity)sequen[i].nd[0].index),
		       Value_to_expression( value_plus(sequen[i].nd[0].lower,MATRIX_ELEM( sequen[i].delai,1,1))));
      e2=le_expression(entity_to_expression ((entity)sequen[i].nd[0].index),
		       Value_to_expression( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1))));

      if (value_eq( value_plus(sequen[i].nd[0].lower,MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].lower))
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].upper)) 
	     {
	       e=NULL;
	     }
	  else
	     {
	       e=e2;
	     }
	}
      else
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),merged_nest[0].upper)) 
	    {
	      e=e1;
	    }
	  else
	     {
	       e=and_expression(e1,e2);
	     }
	};
    
      for(j=1;j<=depth-1;j++)
       {
	 e1=ge_expression(entity_to_expression ((entity)sequen[i].nd[j].index),
	 		  Value_to_expression( sequen[i].nd[j].lower+MATRIX_ELEM( sequen[i].delai,j+1,1)));
	  
	 e2=le_expression(entity_to_expression ((entity)sequen[i].nd[j].index),
			  Value_to_expression( sequen[i].nd[j].upper+MATRIX_ELEM( sequen[i].delai,j+1,1)));
	 if (value_eq( value_plus(sequen[i].nd[j].lower,MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].lower))
	   {
	     if (!value_eq( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].upper)) 
	       {
		 if (e==NULL)
                   e=e2;
		 else
		   e=and_expression(e,e2);
	       };
	   }
	 else
	   {
	     if (value_eq( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),merged_nest[j].upper)) 
	       {
		 if (e==NULL)
		   e=e1;
		 else
		    e=and_expression(e,e1);
	       }
	     else
	       {
		 e1=and_expression(e1,e2);
		 if(e==NULL)
		   e=e1;
		 else
		   e=and_expression(e,e1);
	       }
	   };
	 

 
       };
      if(e==NULL)     
	s=sequen[i].s;
      else
	{
	  t= make_test(e,sequen[i].s,make_block_statement(NIL));
	  s=test_to_statement(t);
	};
      
     if(i==0)   lisi=CONS(STATEMENT,s,NIL);
     else      lisi= gen_nconc(lisi, CONS(STATEMENT,s,NIL));
     c= instruction_call(statement_instruction((sequen[i].s )));
      m=0;
      MAP(EXPRESSION,exp,{  
	if (m==0)  gauche=exp;
	if (m==1)  droite=exp;
	m++;
      },call_arguments(c)) ;
      if(i!=k1-1)   syntax_reference(expression_syntax(gauche))=ref;
      lis3=call_arguments( syntax_call(expression_syntax(droite)));
      if(i==0)
	{
	  for(j=0;j<=depth-1;j++)
	    {
	      Pvecteur pv;
              expression delai_plus;
	      pv = vect_make(VECTEUR_NUL, sequen[i].nd[j].index, VALUE_ONE,
			     TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	      delai_plus=Pvecteur_to_expression(pv);
	      
	      
          FOREACH(EXPRESSION,exp,call_arguments( syntax_call(expression_syntax(droite))))
              replace_entity_by_expression(exp,(entity) sequen[i].nd[j].index,delai_plus);
	      
	    };

	  
	}    
      else
	 {
	   for(j=0;j<=sequen[i].nbr_stencil-1;j++)
	     {
	       Pvecteur pvt; 
               expression expt;
	       pvt=vect_dup(sequen[i-1].pv_acces) ;
               matrix_multiply(sequen[i-1].coef,sequen[i].delai,temp1);
	       pvt=vect_make(pvt,TCST, value_uminus (MATRIX_ELEM(temp1,1,1)));
	       matrix_multiply(sequen[i-1].coef,sequen[i-1].delai,temp1);
	       pvt=vect_make(pvt,TCST, MATRIX_ELEM(temp1,1,1) );
	       matrix_multiply(sequen[i-1].coef,sequen[i].st[j],temp1);
	       pvt=vect_make(pvt,TCST,MATRIX_ELEM(temp1,1,1)  );
	       exp= binary_intrinsic_expression ("MOD",Pvecteur_to_expression(pvt), 
						 Value_to_expression( sequen[i-1].surface) );
	       lis =CONS(EXPRESSION, exp, NIL);
	       ref=make_reference(sequen[i-1].ref,lis);
	       expt=EXPRESSION( CAR(lis3));
	       lis3=CDR(lis3);
	       syntax_reference(expression_syntax(expt))=ref;
	     };
	 };
    
    };
  seq= make_sequence(lisi);
  ins= make_instruction_sequence(seq);
  s= instruction_to_statement(ins);
  ls=loop1;
  for(j=0;j<=depth-1;j++)
    {
      range range1;
    
      range1=loop_range(ls); 
      range_lower(range1)=
	int_to_expression(merged_nest[j].lower);
      range_upper(range1)=
       int_to_expression(merged_nest[j].upper);
      
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
      else loop_body(ls)=s;
      
    };
  
 s=loop_to_statement(loop1);
  
 return s;

}



static entity 
make_tile_index_entity_n(entity old_index,char *su)
{
    entity new_index;
    string old_name;
    char *new_name = (char*) malloc(33);
    
    old_name = entity_name(old_index);
    for (sprintf(new_name, "%s%s", old_name, su);
         gen_find_tabulated(new_name, entity_domain)!=entity_undefined; 

         old_name = new_name) {
        sprintf(new_name, "%s%s", old_name, su);
	pips_assert("Variable name cannot be longer than 32 characters",
		    strlen(new_name)<33);
    }
 
   new_index = make_entity(new_name,
			   copy_type(entity_type(old_index)),
			   /* Should be AddVariableToCommon(DynamicArea) or
			      something similar! */
			   copy_storage(entity_storage(old_index)),
			   copy_value(entity_initial(old_index)));

    return(new_index);
}
static void derive_new_basis_deux(Pbase base_oldindex, Pbase * base_newindex, entity (*new_entity)(entity, char*))
{
    Pbase pb;
    entity new_name;
    char *a;
    for (pb=base_oldindex; pb!=NULL; pb=pb->succ)
      { 
	a="1";
	new_name = new_entity((entity) pb->var,a);
        base_add_dimension(base_newindex, (Variable) new_name);
	a="2";
	new_name = new_entity((entity) pb->var,a);
	base_add_dimension(base_newindex, (Variable) new_name);
	a="3";
	new_name = new_entity((entity) pb->var,a);
        printf("%s \n",entity_name(new_name));
	base_add_dimension(base_newindex, (Variable) new_name);

    }
    /* *base_newindex = base_reversal(*base_newindex); */
}
static void derive_new_basis_une(Pbase base_oldindex, Pbase * base_newindex, entity (*new_entity)(entity, char*))
{
    Pbase pb;
    entity new_name;
    char *a;
    for (pb=base_oldindex; pb!=NULL; pb=pb->succ)
      { 
	a="1";
	new_name = new_entity((entity) pb->var,a);
        base_add_dimension(base_newindex, (Variable) new_name);
	a="2";
	new_name = new_entity((entity) pb->var,a);
	base_add_dimension(base_newindex, (Variable) new_name);
	

    }
    /* *base_newindex = base_reversal(*base_newindex); */
}


statement permutation( s, P)
     statement s;
     Pmatrix P;
{
  statement scopy;
  int i,j;
  loop ls1,ls2;
  bool found=false;
  scopy=copy_statement(s);
  ls1=instruction_loop(statement_instruction(scopy));
  for(i=1;i<=3*depth;i++)
    {
      j=1;
      found=false;
      ls2=instruction_loop(statement_instruction(s));
      while ((j<=3*depth)&& !found)
	{
	  if (value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
	    {
	      found =true;
	      loop_index(ls1)=loop_index(ls2);
	      loop_range(ls1)=loop_range(ls2);
	    };
	  ls2=instruction_loop(statement_instruction(loop_body(ls2)));
	  j++;
	};
      ls1=instruction_loop(statement_instruction(loop_body(ls1)));
    }
  
  return scopy;
}

statement permutation2( s, P)
     statement s;
     Pmatrix P;
{
  statement scopy;
  int i,j;
  loop ls1,ls2;
  bool found=false;
  scopy=copy_statement(s);
  ls1=instruction_loop(statement_instruction(scopy));
  for(i=1;i<=2*depth;i++)
    {
      j=1;
      found=false;
      ls2=instruction_loop(statement_instruction(s));
      while ((j<=2*depth)&& !found)
	{
	  if (value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
	    {
	      found =true;
	      loop_index(ls1)=loop_index(ls2);
	      loop_range(ls1)=loop_range(ls2);
	    };
	  ls2=instruction_loop(statement_instruction(loop_body(ls2)));
	  j++;
	};
      ls1=instruction_loop(statement_instruction(loop_body(ls1)));
    }
  
  return scopy;
}

statement Hierarchical_tiling ()
{
  FILE *infp;
  char *name_file;
  Pmatrix A,P;
  int rw,cl,i,j;
  list lis=NULL;
  instruction ins;
  sequence seq=NULL;
  statement s,s1;
  loop ls=NULL;
  cons *lls=NULL;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  Pbase pb;
  expression upper=NULL,lower=NULL;
  int plus;
  Value elem_mat1,elem_mat2;
  Psysteme  loop_iteration_domaine_to_sc();
  name_file = user_request("nom du fichier pour la matrice A ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&A,&rw,&cl);
  name_file = user_request("nom du fichier pour la matrice P ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&P,&cl,&cl);
  matrix_print(A);
  matrix_print(P); 
  compute_delay_tiled_nest ();
  compute_bound_tiled_nest();
  ls=loop1;
  for(j=0;j<=depth-1;j++)
    {
      range range1;  
      range1=loop_range(ls); 
      range_lower(range1)=int_to_expression(tiled_nest[j].lower);
      range_upper(range1)= int_to_expression(tiled_nest[j].upper);
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    };
  s1=loop_to_statement(loop1);
  while(instruction_loop_p(statement_instruction (s1))) {
    lls = CONS(STATEMENT,s1,lls);
    s1 = loop_body(instruction_loop(statement_instruction(s1)));
  }
  (void)loop_iteration_domaine_to_sc(lls, &base_oldindex);

  derive_new_basis_deux(base_oldindex, &base_newindex , make_tile_index_entity_n);
  tiling_indice= ( Pvecteur  *)malloc(depth *sizeof( Pvecteur*));
  plus=depth*3-1;
  for (pb =base_newindex; pb!=NULL; pb=pb->succ) {
    loop tl;
    switch(plus%3)
      {
      case 2: 
	{
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus);
	  elem_mat2=MATRIX_ELEM(A,plus/3+1,plus+1);
	  upper= int_to_expression(value_minus(elem_mat1,VALUE_ONE));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make(VECTEUR_NUL,vecteur_var(pb),elem_mat2,TCST,VALUE_ZERO);
	};
	break;
      case 1: 
	{ 
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus) ;
	  elem_mat2=MATRIX_ELEM(A,plus/3+1,plus+1  ) ;
	  upper= int_to_expression(value_minus(value_div(elem_mat1,elem_mat2),VALUE_ONE));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make( tiling_indice[plus/3],pb->var , elem_mat2,TCST,VALUE_ZERO);
	};
	break;   
      case 0:
	{ 
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus+1);
	  upper= int_to_expression(value_div(tiled_nest[plus/3].upper,elem_mat1));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make( tiling_indice[plus/3],pb->var ,elem_mat1,TCST,VALUE_ZERO);
	};
      default: break;
      }  
    tl = make_loop((entity) vecteur_var(pb),
		   make_range(copy_expression(lower), copy_expression(upper),
			      int_to_expression(1)),
		   s1,
		   entity_empty_label(),
		   make_execution(is_execution_sequential, UU),
		   NIL);
    s1 = instruction_to_statement(make_instruction(is_instruction_loop, tl));
    plus--;
  };
  
  printf(" azuuuul lalallalalal \n"); getchar();

  /******/
  for (i=k1-1;i>=0;i--)
    {
      expression e1,e2,e,gauche=NULL,droite=NULL,delai_plus;
      test t;
      call c;
      int m;
      Pvecteur pv;
      e1=ge_expression(Pvecteur_to_expression (tiling_indice[0]),
		       Value_to_expression( value_plus(sequen[i].nd[0].lower,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      e2=le_expression(Pvecteur_to_expression (tiling_indice[0]), 
		       Value_to_expression( value_plus(sequen[i].nd[0].upper,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      if (value_eq( value_plus(sequen[i].nd[0].lower,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].lower))
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=NULL;
	  else
	    e=e2;
	}
      else
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=e1;
	   
	  else
	    e=and_expression(e1,e2);
	    
	};
    
      for(j=0;j<=depth-1;j++)
	{
         
	  if (j>=1) 
	    { 
	       
             
              print_expression(Pvecteur_to_expression (tiling_indice[j])  );
               
              
	      e1=ge_expression( Pvecteur_to_expression (tiling_indice[j]),
				Value_to_expression(value_plus(sequen[i].nd[j].lower,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
             
	      e2=le_expression(Pvecteur_to_expression (tiling_indice[j]), 
			       Value_to_expression( value_plus(sequen[i].nd[j].upper,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      
	      if (value_eq( value_plus(sequen[i].nd[j].lower,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].lower))
		{
		  if (! (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
			 value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					     MATRIX_ELEM(A,j+1,3*j+1)),VALUE_ZERO)))
		    {
		      if (e==NULL)
			e=e2;
		      else
			e=and_expression(e,e2);
		    };
		}
	      else
		{
		  if (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
		      value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					  MATRIX_ELEM(A,j+1,3*j+1)),VALUE_ZERO))
		    {
		      if (e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		  else
		    {
		      e1=and_expression(e1,e2);
		      if(e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		};

                
	    }; 
	  c= instruction_call(statement_instruction((sequen[i].s )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    if (m==1)  droite=exp;
	    m++;
	  },call_arguments(c)) ;
	  pv = vect_make(tiling_indice[j],TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	  delai_plus=Pvecteur_to_expression(pv);
      replace_entity_by_expression(gauche,(entity) sequen[i].nd[j].index,delai_plus);
      FOREACH(EXPRESSION,exp,call_arguments( syntax_call(expression_syntax(droite))))
          replace_entity_by_expression(exp,(entity) sequen[i].nd[j].index,delai_plus);
	};
      t= make_test(e,sequen[i].s,make_block_statement(NIL));
      s=test_to_statement(t);
      if(i==k1-1)   lis=CONS(STATEMENT,s,NIL);
      else      lis=CONS(STATEMENT,s,lis);
    };
  seq= make_sequence(lis);
  ins= make_instruction_sequence(seq);
  s= instruction_to_statement(ins);
  ls=instruction_loop(statement_instruction(s1));
  for(j=0;j<=3*depth-1;j++)
    if  (j!=3*depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    else loop_body(ls)=s;
  s1=permutation(s1,P);
  return s1;
}



int position_one_element(Pmatrix P, int i)
{
  int j;
  for (j=1;j<=depth;j++)
    if(MATRIX_ELEM(P,i,j)==VALUE_ONE) return j;
  pips_internal_error("asking for a non existing element");
  return -1;
}

 
statement Tiling_buffer_allocation ()
{
  FILE *infp;
  char *name_file;
  Pmatrix A,P,ATE,ATEP,ATEPP,P1,P2,N[nid_nbr],NP[nid_nbr],G_PRO[nid_nbr], G_PROP[nid_nbr],G_PROPP[nid_nbr];
  int rw,cl,i,j,k,pos;
  list lis=NULL,lis1=NULL,lis2=NULL;
  instruction ins;
  sequence seq=NULL;
  statement s,s1,sequenp[nid_nbr],ps1,ps2/*,*pst1*/;
  loop ls=NULL;
  cons *lls=NULL;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  Pbase pb;
  expression upper=NULL,lower=NULL,exp=NULL;
  int plus;
  Value elem_mat1,elem_mat2;
  entity *ref[nid_nbr],name;
  reference *buf_ref[nid_nbr];
  statement *scopy[nid_nbr];
  Psysteme  loop_iteration_domaine_to_sc();
  Variable *iter, *itert;
  name_file = user_request("nom du fichier pour la matrice A ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&A,&rw,&cl);
  name_file = user_request("nom du fichier pour la matrice P ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&P,&cl,&cl);
  compute_delay_tiled_nest ();
  compute_bound_tiled_nest();
  P1=matrix_new(depth,depth);
  
  for(i=1;i<=depth;i++)
    for(j=1;j<=2*depth;j=j+2)
      {
	if ( value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
          MATRIX_ELEM(P1,i,j/2+1)=VALUE_ONE;
	else 
          MATRIX_ELEM(P1,i,j/2+1)=VALUE_ZERO;
      };
  
  
  P2=matrix_new(depth,depth);
  for(i=depth+1;i<=2*depth;i++)
    for(j=2;j<=2*depth;j=j+2)
      {
	if (value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
          MATRIX_ELEM(P2,(i-1)%depth+1,j/2)=VALUE_ONE;
	else 
          MATRIX_ELEM(P2,(i-1)%depth+1,j/2)=VALUE_ZERO;
      };
  
  ATE=matrix_new(depth,1);
  for(i=0;i<=depth-1;i++)
    MATRIX_ELEM(ATE,i+1,1)=MATRIX_ELEM(A,i+1,2*i+1);
 
  ATEP=matrix_new(depth,1);
  matrix_multiply(P1,ATE,ATEP);
  ATEPP=matrix_new(depth,1);
  matrix_multiply(P2,ATE,ATEPP);
 



  s1=loop_to_statement(loop1);
  while(instruction_loop_p(statement_instruction (s1))) {
    lls = CONS(STATEMENT,s1,lls);
    s1 = loop_body(instruction_loop(statement_instruction(s1)));
  }
 
  (void)loop_iteration_domaine_to_sc(lls, &base_oldindex);
  derive_new_basis_une(base_oldindex, &base_newindex , make_tile_index_entity_n);
 
  iter= ( Variable  *)malloc(depth *sizeof( Variable));
  itert=( Variable  *)malloc(depth *sizeof( Variable));
  i=2*depth-1;
  for (pb =base_newindex; pb!=NULL; pb=pb->succ) {
    if (i%2==1) iter[i/2]=(pb->var);
    else itert[i/2]=(pb->var);
    i--;
  };
  
    
  for(i=0;i<=k1-1;i++)
    {
      //statement s1;
      sequence seq;
      instruction ins;

      Pvecteur pv;
      Value con;
      expression e,gauche,cavite;
      test t;
      statement s;
      call c;
      int m;
      
      N[i]=matrix_new(depth,1);
      for (j=1;j<=depth;j++)
	MATRIX_ELEM(N[i],j,1)= value_plus(value_minus(sequen[i].nd[j-1].upper,sequen[i].nd[j-1].lower),VALUE_ONE);
      NP[i]=matrix_new(depth,1);
      matrix_multiply(P1,N[i],NP[i]);
   
      if (i<=k1-2)
	{ 
	  
	  
	  lis2=NULL;
	  G_PRO[i]=matrix_new(depth,1);
	  for(j=1;j<=depth;j++)
	    {
	      
	      Value temp;
	 
	      temp=value_minus(MATRIX_ELEM(sequen[i+1].delai,j,1),MATRIX_ELEM(sequen[i].delai,j,1));
	      MATRIX_ELEM( G_PRO[i],j,1)=value_minus(temp, MATRIX_ELEM(sequen[i+1].st[0],j,1));
	      for(k=1;k<=sequen[i].nbr_stencil-1;k++)
		{   
		  if (MATRIX_ELEM( G_PRO[i],j,1)< value_minus(temp, MATRIX_ELEM(sequen[i].st[k],j,1)))
		    MATRIX_ELEM( G_PRO[i],j,1)=value_minus(temp, MATRIX_ELEM(sequen[i].st[k],j,1));
		};
	    };
             
	  G_PROP[i]=matrix_new(depth,1);
	  matrix_multiply(P1,G_PRO[i],G_PROP[i]);
	  G_PROPP[i]=matrix_new(depth,1);
	  matrix_multiply(P2,G_PRO[i],G_PROPP[i]);
	  ref[i]= ( entity  *)malloc((depth+1) *sizeof( entity));
	  buf_ref[i]=( reference  *)malloc((depth+1) *sizeof( reference));
          scopy[i]=( statement   *)malloc((depth+1) *sizeof( statement));
	 
	  for(j=1;j<=depth;j++)
	    {	      

	      lis= NULL;
              lis1=NULL;
	     
	      lis=CONS(DIMENSION, make_dimension(int_to_expression(0), int_to_expression(1)), lis);
	      pos=position_one_element(P1,j);
	      pv= vect_make(VECTEUR_NUL ,itert[pos-1], VALUE_ONE,TCST,VALUE_ZERO);
	      exp= binary_intrinsic_expression ("MOD",Pvecteur_to_expression(pv ),Value_to_expression(2));
	      lis1 =CONS(EXPRESSION, exp,lis1);
	 
	      for (k=1;k<=j-1;k++)
		{
		 
		  lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(value_minus 
										      (MATRIX_ELEM(ATEP,k,1),VALUE_ONE))), lis);
		  pos=position_one_element(P1,k);
		  pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,TCST,VALUE_ZERO);
		  exp=Pvecteur_to_expression(pv);
		 
		  lis1 =CONS(EXPRESSION, exp,lis1);
		}

              if (value_minus (MATRIX_ELEM(G_PROP[i],j,1),VALUE_ONE)==VALUE_MONE)
		{
 
		  cavite=int_to_expression(0);
		}
	      
              else 
		{ 

		  cavite =Value_to_expression(value_minus (MATRIX_ELEM(G_PROP[i],j,1),VALUE_ONE));
		};
	      lis=CONS(DIMENSION, make_dimension(int_to_expression(0), cavite), 
		       lis);
	      pos=position_one_element(P1,k);
              con=value_minus(MATRIX_ELEM(G_PRO[i],pos,1), MATRIX_ELEM(ATE,pos,1));    
	      pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,TCST,con);
	      
	      exp=Pvecteur_to_expression(pv);
	     
	      lis1 =CONS(EXPRESSION, exp,lis1);
	      for (k=j+1;k<=depth;k++)
		{
		  Value v1;
		  lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(value_minus 
										      (MATRIX_ELEM(NP[i],k,1),VALUE_ONE))), lis);
		  pos=position_one_element(P1,k);
		 v1=sequen[i].nd[pos-1].lower+MATRIX_ELEM( sequen[i].delai,pos,1 );
		 
		 
		 pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,itert[pos-1], MATRIX_ELEM(ATE,pos,1) ,TCST,
				value_uminus(v1)  );
		 
		  vect_dump(pv);
		  exp=Pvecteur_to_expression(pv);
		  print_expression(exp);
		 
		  lis1 =CONS(EXPRESSION, exp,lis1);
		}
	      
	   
	      name= internal_make_new_array_variable(i+1,j,get_current_module_entity() , make_basic(is_basic_int, (void *) 4), lis);
	      buf_ref[i][j-1]=make_reference(name,lis1);
	      ref[i][j-1]=name;
              scopy[i][j-1]=copy_statement(sequen[i].s);

	      pos=position_one_element(P1,j);
	      con=value_minus(MATRIX_ELEM(G_PRO[i],pos,1), MATRIX_ELEM(ATE,pos,1));  
	      pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,TCST,con);
	      e=ge_expression( Pvecteur_to_expression (pv), Value_to_expression( VALUE_ZERO));
              t= make_test(e,scopy[i][j-1],make_block_statement(NIL));
	      s=test_to_statement(t);
	      lis2=CONS(STATEMENT,s,lis2);
	      c= instruction_call(statement_instruction((   scopy[i][j-1] )));
	      m=0;
	      MAP(EXPRESSION,exp,{  
		if (m==0)  gauche=exp;
		m++;
	      },call_arguments(c)) ;
	      syntax_reference(expression_syntax(gauche))= buf_ref[i][j-1];
	      

	    }
	 
	  lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(MATRIX_ELEM(G_PROPP[i],1,1))), NIL);
	  pos=position_one_element(P2,1);
	  pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,TCST,VALUE_ZERO);
	 

	  exp= binary_intrinsic_expression ("MOD",Pvecteur_to_expression(pv ),
					    Value_to_expression(value_plus(MATRIX_ELEM(G_PROPP[i],1,1),VALUE_ONE)));

	  
	  lis1 =CONS(EXPRESSION, exp,NIL);
	  for (k=2;k<=depth;k++)
	    {
	      lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(value_minus 
											  (MATRIX_ELEM(ATEPP,k,1),VALUE_ONE))),
		       lis);
	      pos=position_one_element(P2,k);
	      pv= vect_make(VECTEUR_NUL ,iter[pos-1], VALUE_ONE,TCST,VALUE_ZERO);
	      exp=Pvecteur_to_expression(pv);
	      
	      lis1 =CONS(EXPRESSION, exp,lis1);
	    }
	  name= internal_make_new_array_variable(i+1,depth+1,get_current_module_entity() , make_basic(is_basic_int, (void *) 4), lis);
	  buf_ref[i][depth]=make_reference(name,lis1);
	  ref[i][depth]=name;
	  scopy[i][depth]=copy_statement(sequen[i].s);
	  c= instruction_call(statement_instruction((   scopy[i][depth] )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    m++;
	  },call_arguments(c)) ;
	  syntax_reference(expression_syntax(gauche))= buf_ref[i][depth];
	  lis2=CONS(STATEMENT, scopy[i][depth],lis2);
	  seq= make_sequence(lis2);
	  ins= make_instruction_sequence(seq);
	  sequenp[i]= instruction_to_statement(ins);
	 
	};
    };
   
  for(i=1;i<=k1-2;i++)
    { 
      expression e=NULL;
      Pvecteur pv;
      Pmatrix *temp;
      reference *regi[nid_nbr];
      int m1;
     
      for(j=1;j<=depth;j++)
	{
	  Value v2;
	  v2=value_uminus(MATRIX_ELEM(G_PRO[i-1],j,1));
	  pv= vect_make(VECTEUR_NUL,iter[j-1], VALUE_ONE,TCST ,v2);
	  if (j==1)  e=ge_expression(Pvecteur_to_expression(pv),Value_to_expression(VALUE_ZERO));  
	  else
	    e=and_expression(e,ge_expression(Pvecteur_to_expression(pv),Value_to_expression(VALUE_ZERO)));
	}
      temp=( Pmatrix *)malloc( sequen[i].nbr_stencil *sizeof(Pmatrix));
      regi[i]=( reference *)malloc( sequen[i].nbr_stencil *sizeof(reference));;
      for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	{
	  entity name;
	  char buffer[20];  
 	  //expression exp;
	  //reference ref;
	  temp[k]=matrix_new(depth,1);
	  matrix_substract(temp[k],sequen[i].delai, sequen[i-1].delai);
	  matrix_substract(temp[k], temp[k],sequen[i].st[k]);
	  sprintf(buffer,"%s%d_%d", "R",i+1,k);
	  name= make_scalar_entity(buffer,  module_local_name(get_current_module_entity()),
				   make_basic(is_basic_int, (void *) 4) );
	  regi[i][k]=make_reference(name,NIL);
	}
      for(j=0;j<=depth;j++) 
	{
	  expression droite,expt;
	  statement s1,s2,s3;
          sequence seq;
	  instruction ins;
	  call c;
	  test t;
	  int m,l;
	  list lis,lisp;
	  reference ref;
	  s1=copy_statement(scopy[i][j]);
	  c= instruction_call(statement_instruction(( s1)));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==1)  droite=exp;
	    m++;
	  },call_arguments(c)) ;
	  lis=call_arguments( syntax_call(expression_syntax(droite)));
	  for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	    {
	      expression exp;
	      ref=copy_reference (buf_ref[i-1][depth]);  /**** ?????? ****/
	      exp=reference_to_expression(ref);
              for(l=0;l<=depth-1;l++)
		{
		  Pvecteur pv;
		  expression delai_plus;
		  pv = vect_make(VECTEUR_NUL,iter[l], VALUE_ONE, TCST,-MATRIX_ELEM(temp[k],l+1,1));
		  delai_plus=Pvecteur_to_expression(pv);
          replace_entity_by_expression(exp,(entity)iter[l],delai_plus);
		}
	      expt=EXPRESSION( CAR(lis));
	      lis=CDR(lis);
	      syntax_reference(expression_syntax(expt))=ref;
	    }
	  lis=NIL;
	  for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	    {
	      statement s,*stemp;
	      stemp=( statement  *)malloc( (depth+1) *sizeof(statement));
	      for(l=0;l<=depth;l++)
		{ 
		  int r;
		  expression exp;
		  reference ref;
		   
		  ref=copy_reference (buf_ref[i-1][l]);
		  exp=reference_to_expression(ref);
                  for (r=0;r<=depth-1;r++)
		    {
		      Pvecteur pv;
		      expression delai_plus;
		      
		      if(r==l)
			{
			  Value t;
			  int pos;
			  pos=position_one_element(P1,r+1);
			  t=value_minus(MATRIX_ELEM(ATEP,r+1,1),MATRIX_ELEM(temp[k],pos,1));
			  pv = vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE, TCST,t);
			  delai_plus=Pvecteur_to_expression(pv);
              replace_entity_by_expression(exp,(entity)iter[pos-1],delai_plus);
			}		   
		      
		      else	     
			{
			  int pos;
			  pos=position_one_element(P1,r+1);
			  pv = vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE, TCST,-MATRIX_ELEM(temp[k],pos,1));
			  delai_plus=Pvecteur_to_expression(pv);
              replace_entity_by_expression(exp,(entity)iter[pos-1],delai_plus);
			}
		    }

		  
		  r=0;
		  MAP(EXPRESSION,exp,{  
		   
		    if (r==depth)
		      {
			 
			int pos;
			expression delai_plus;
			pos=position_one_element(P1,l+1);
			pv= vect_make(VECTEUR_NUL ,itert[pos-1], VALUE_ONE,TCST,-VALUE_ONE);
			delai_plus=Pvecteur_to_expression(pv);
            replace_entity_by_expression(exp,(entity)itert[pos-1],delai_plus);

		      };
		    r++;
		  },reference_indices(ref)) 

		  stemp[l]=make_assign_statement(reference_to_expression(regi[i][k]),reference_to_expression(ref));
		};
	      s=  stemp[depth];
	      for(l=depth-1;l>=0;l--)
		{
		  int pos;
		  Pvecteur pv2;
		  //statement s1/*,s2*/;
		  test t;
		  expression exp;
		  pos=position_one_element(P1,l+1);
		  pv2= vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE,TCST , -MATRIX_ELEM(temp[k],pos,1));
		  exp=lt_expression(Pvecteur_to_expression(pv2 ),Value_to_expression(VALUE_ZERO));
		  t= make_test(exp,stemp[l],s);
		  s=test_to_statement(t);
		}
	      lis=CONS(STATEMENT,s,lis);

	    }
	  s3=copy_statement(s1);
	 
	  c= instruction_call(statement_instruction(( s3)));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==1)  droite=exp;
	    m++;
	  },call_arguments(c)) ;
	  lisp=call_arguments( syntax_call(expression_syntax(droite)));
	  for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	    {
	     
	      expression expt;
	     
	   
               
	      expt=EXPRESSION( CAR(lisp));
	       
	      lisp=CDR(lisp);
	   
	      syntax_reference(expression_syntax(expt))=regi[i][k];
	     
	    }
 


	  lis=CONS(STATEMENT,s3,lis);
	  lis=gen_nreverse(lis);
	  
	  seq= make_sequence(lis);
	  ins= make_instruction_sequence(seq);
	  s2= instruction_to_statement(ins);
	  t= make_test(e,s1,s2);
	  scopy[i][j]=test_to_statement(t);
	  
	  
	};

      m1=0; 

        CHUNK(CAR( sequence_statements(instruction_sequence(statement_instruction(sequenp[i])))))=   (gen_chunk *) scopy[i][depth];  
      MAP(STATEMENT,s,{  
	if (m1!=0)   
	  {
	    test_true(instruction_test(statement_instruction(s)) )=  scopy[i][depth-m1];
    
	  }
	m1++;
	}, sequence_statements(instruction_sequence(statement_instruction(sequenp[i]))) ) ; 




      
    };
  /*   **********************************   */

  for(i= k1-1 ;i<=k1-1;i++)
    { 
      expression e=NULL;
      Pvecteur pv;
      Pmatrix *temp;
      reference *regi[nid_nbr];
      //int m1;
      /* */

      expression droite,expt;
      statement s1,s2,s3;
      sequence seq;
      instruction ins;
      call c;
      test t;
      int m,l;
      list lis,lisp;
      reference ref;
      
     
      for(j=1;j<=depth;j++)
	{
	  
          Value v2;
	  
	  
	  
	  v2=value_uminus(MATRIX_ELEM(G_PRO[i-1],j,1));
	  
	  pv= vect_make(VECTEUR_NUL,iter[j-1], VALUE_ONE,TCST ,v2);
	  
	  if (j==1)  e=ge_expression(Pvecteur_to_expression(pv),Value_to_expression(VALUE_ZERO));  
	  else
	    e=and_expression(e,ge_expression(Pvecteur_to_expression(pv),Value_to_expression(VALUE_ZERO)));
	  
	}
      
      temp=( Pmatrix *)malloc( sequen[i].nbr_stencil *sizeof(Pmatrix));
      regi[i]=( reference *)malloc( sequen[i].nbr_stencil *sizeof(reference));;
      for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	{
	  entity name;
	  char buffer[20];  
 	  //expression exp;
	  //reference ref;
	  temp[k]=matrix_new(depth,1);
	  
	  matrix_substract(temp[k],sequen[i].delai, sequen[i-1].delai);
	  matrix_substract(temp[k], temp[k],sequen[i].st[k]);
	  
	  
	  
	  sprintf(buffer,"%s%d_%d", "R",i+1,k);
	  name= make_scalar_entity(buffer,  module_local_name(get_current_module_entity()),
				   make_basic(is_basic_int, (void *) 4) );
	  
	  regi[i][k]=make_reference(name,NIL);
	  
	  
	      
	}
      
      s1=copy_statement( sequen[i].s);
      ps1=s1;
      c= instruction_call(statement_instruction(( s1)));
      m=0;
      MAP(EXPRESSION,exp,{  
	if (m==1)  droite=exp;
	m++;
      },call_arguments(c)) ;
      lis=call_arguments( syntax_call(expression_syntax(droite)));
      for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	{
	  expression exp;
	  ref=copy_reference (buf_ref[i-1][depth]);
	  exp=reference_to_expression(ref);
	  for(l=0;l<=depth-1;l++)
	    {
	      Pvecteur pv;
	      expression delai_plus;
	      pv = vect_make(VECTEUR_NUL,iter[l], VALUE_ONE, TCST,-MATRIX_ELEM(temp[k],l+1,1));
	      delai_plus=Pvecteur_to_expression(pv);
          replace_entity_by_expression(exp,(entity)iter[l],delai_plus);
	    }
	  expt=EXPRESSION( CAR(lis));
	  lis=CDR(lis);
	  syntax_reference(expression_syntax(expt))=ref;
	} 
      lis=NIL;
      for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	{
	  statement s,*stemp;
	  stemp=( statement  *)malloc( (depth+1) *sizeof(statement));
	  for(l=0;l<=depth;l++)
	    { 
	      int r;
	      expression exp;
	      reference ref;
	      
	      ref=copy_reference (buf_ref[i-1][l]);
	      
	      
	      exp=reference_to_expression(ref);
	      for (r=0;r<=depth-1;r++)
		{
		  Pvecteur pv;
		  expression delai_plus;
		  
		     if(r==l)
			{
			  Value t;
			  int pos;
			  pos=position_one_element(P1,r+1);
			  t=value_minus(MATRIX_ELEM(ATEP,r+1,1),MATRIX_ELEM(temp[k],pos,1));
			  pv = vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE, TCST,t);
			  delai_plus=Pvecteur_to_expression(pv);
              replace_entity_by_expression(exp,(entity) iter[pos-1], delai_plus);
			}		   
		      
		      else	     
			{
			  int pos;
			  pos=position_one_element(P1,r+1);
			  pv = vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE, TCST,-MATRIX_ELEM(temp[k],pos,1));
			  delai_plus=Pvecteur_to_expression(pv);
              replace_entity_by_expression(exp,(entity) iter[pos-1], delai_plus);
			}
		    
		}
	      r=0;
	      MAP(EXPRESSION,exp,{  
		
		if (r==depth)
		  {
		    int pos;
		    expression delai_plus;
		    pos=position_one_element(P1,l+1);
                    
		 
		    pv= vect_make(VECTEUR_NUL ,itert[pos-1], VALUE_ONE,TCST,-VALUE_ONE);
		 
		    delai_plus=Pvecteur_to_expression(pv);
            replace_entity_by_expression(exp,(entity) itert[pos-1], delai_plus);
		    

		  
		  };
		r++;
	      },reference_indices(ref));
	      
	      stemp[l]=make_assign_statement(reference_to_expression(regi[i][k]),reference_to_expression(ref));
	    };
	  s=  stemp[depth];
	  for(l=depth-1;l>=0;l--)
	    {
	      int pos;
	      Pvecteur pv2;
	      //statement s1,s2;
	      test t;
	      expression exp;
	      pos=position_one_element(P1,l+1);
	      pv2= vect_make(VECTEUR_NUL,iter[pos-1], VALUE_ONE,TCST , -MATRIX_ELEM(temp[k],pos,1));
	      exp=lt_expression( Pvecteur_to_expression(pv2 ) ,Value_to_expression(VALUE_ZERO) );
	      t= make_test(exp,stemp[l],s);
	      s=test_to_statement(t);
	    }
	  lis=CONS(STATEMENT,s,lis);
	  
	}
      s3=copy_statement(s1);
      ps2=s3;
      c= instruction_call(statement_instruction(( s3)));
      m=0;
      MAP(EXPRESSION,exp,{  
	if (m==1)  droite=exp;
	m++;
      },call_arguments(c)) ;
      lisp=call_arguments( syntax_call(expression_syntax(droite)));
      for(k=0;k<=sequen[i].nbr_stencil-1;k++)
	{
	  expression expt;
	  expt=EXPRESSION( CAR(lisp));
	  lisp=CDR(lisp);
	  syntax_reference(expression_syntax(expt))=regi[i][k];
	}
      lis=CONS(STATEMENT,s3,lis);
      lis=gen_nreverse(lis);
      seq= make_sequence(lis);
      ins= make_instruction_sequence(seq);
      s2= instruction_to_statement(ins);
      t= make_test(e,s1,s2);
      sequen[i].s=test_to_statement(t);


      
    }; 




      

/* ************************** */







  ls=loop1;
  for(j=0;j<=depth-1;j++)
    {
      range range1;  
      range1=loop_range(ls); 
      range_lower(range1)=int_to_expression(tiled_nest[j].lower);
      range_upper(range1)= int_to_expression(tiled_nest[j].upper);
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    };
  tiling_indice= ( Pvecteur  *)malloc(depth *sizeof( Pvecteur*));
  plus=depth*2-1;
  
  for (pb =base_newindex; pb!=NULL; pb=pb->succ) {
    loop tl;
    switch(plus%2)
      {
      case 1: 
	{ 
	  
	  elem_mat2=MATRIX_ELEM(A,plus/2+1,plus  ) ;
	  
	  upper= int_to_expression(value_minus(elem_mat2,VALUE_ONE));
	  
	  lower= int_to_expression(0);
	  
	  tiling_indice[plus/2]=vect_make(VECTEUR_NUL ,pb->var , VALUE_ONE,TCST,VALUE_ZERO);
	        
	  vect_dump( tiling_indice[plus/2]);
	  
	 
	  break;
	}
	
      case 0:
	{ 
	  elem_mat1=MATRIX_ELEM(A,plus/2+1,plus+1);
	  upper= int_to_expression(value_div(tiled_nest[plus/2].upper,elem_mat1));
	  lower= int_to_expression(0);
	 
	  tiling_indice[plus/2]=vect_make( tiling_indice[plus/2],pb->var ,elem_mat1,TCST,VALUE_ZERO);
	  vect_dup( tiling_indice[plus/2]);
	 
	  break;
	};
	
      default: break;
      }  
    
    tl = make_loop((entity) vecteur_var(pb),
		   make_range(copy_expression(lower), copy_expression(upper),
			      int_to_expression(1)),
		   s1,
		   entity_empty_label(),
		   make_execution(is_execution_sequential, UU),
		   NIL);
    s1 = instruction_to_statement(make_instruction(is_instruction_loop, tl));
    
    
    plus--;
  };
  
  

  /******/
  for (i=k1-1;i>=0;i--)
    {
      expression e1,e2,e/*,gauche=NULL,droite=NULL*/,delai_plus;
      test t;
      //call c;
      //int m;
      //Pvecteur pv;
      e1=ge_expression(Pvecteur_to_expression (tiling_indice[0]),
		       Value_to_expression( value_plus(sequen[i].nd[0].lower,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      e2=le_expression(Pvecteur_to_expression (tiling_indice[0]), 
		       Value_to_expression( value_plus(sequen[i].nd[0].upper,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      if (value_eq( value_plus(sequen[i].nd[0].lower,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].lower))
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=NULL;
	  else
	    e=e2;
	}
      else
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=e1;
	  
	  else
	    e=and_expression(e1,e2);
	  
	};
     
      for(j=0;j<=depth-1;j++)
	{
	  if (j>=1) 
	    {
	      e1=ge_expression( Pvecteur_to_expression (tiling_indice[j]),
				Value_to_expression(value_plus(sequen[i].nd[j].lower,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      e2=le_expression(Pvecteur_to_expression (tiling_indice[j]), 
			       Value_to_expression( value_plus(sequen[i].nd[j].upper,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      if (value_eq( value_plus(sequen[i].nd[j].lower,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].lower))
		{
		  if (! (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
			 value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					     MATRIX_ELEM(A,j+1,2*j+1)),VALUE_ZERO)))
		    {
		      if (e==NULL)
			e=e2;
		      else
			e=and_expression(e,e2);
		    };
		}
	      else
		{
		  if (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
		      value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					  MATRIX_ELEM(A,j+1,2*j+1)),VALUE_ZERO))
		    {
		      if (e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		  else
		    {
		      e1=and_expression(e1,e2);
		      if(e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		};
	    };
	};
      if(i<= k1-2) 
	{
	  if(i==0)
	    {
	      call c;
	      int m ,j,l;
	      expression droite;
	      Pvecteur pv;

	      for(j=0;j<depth+1;j++)
		{
		  c= instruction_call(statement_instruction(( scopy[i][j])));
		  m=0;
		  MAP(EXPRESSION,exp,{  
		    if (m==1)  droite=exp;
		    m++;
		  },call_arguments(c)) ;
		  for(l=0;l<=depth-1;l++)
		    {
		      pv = vect_make(tiling_indice[l],TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,l+1,1)));
		      delai_plus=Pvecteur_to_expression(pv);
		   
		      FOREACH(EXPRESSION,exp,call_arguments( syntax_call(expression_syntax(droite))))
                  replace_entity_by_expression(exp,(entity)  sequen[i].nd[l].index, delai_plus);
		    }
		}
	    }	      
	  
	  t= make_test(e,sequenp[i],make_block_statement(NIL));
	}
      else 
	{
	  call c;
	  int m ,j;
	  expression gauche;
	  Pvecteur pv;
	  c= instruction_call(statement_instruction(( ps1 )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    m++;
	  },call_arguments(c)) ;
	  for(j=0;j<=depth-1;j++)
	    {  
	      pv = vect_make(tiling_indice[j],TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	      delai_plus=Pvecteur_to_expression(pv);
          replace_entity_by_expression(gauche,(entity)  sequen[i].nd[j].index, delai_plus);
	    }
	  c= instruction_call(statement_instruction(( ps2 )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    m++;
	  },call_arguments(c)) ;
	  for(j=0;j<=depth-1;j++)
	    {  
	      pv = vect_make(tiling_indice[j],TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	      delai_plus=Pvecteur_to_expression(pv);
          replace_entity_by_expression(gauche,(entity)  sequen[i].nd[j].index, delai_plus);
	    }


	  
	  t= make_test(e,sequen[i].s,make_block_statement(NIL));
	};
      s=test_to_statement(t);
 
      if(i==k1-1)   lis=CONS(STATEMENT,s,NIL);
      else      lis=CONS(STATEMENT,s,lis);
      
    };
 
  seq= make_sequence(lis);
  ins= make_instruction_sequence(seq);
  s= instruction_to_statement(ins);
  ls=instruction_loop(statement_instruction(s1));
  for(j=0;j<=2*depth-1;j++)
    if  (j!=2*depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    else loop_body(ls)=s;
 
   
  s1=permutation2(s1,P);
  
  return s1;
}










static bool array_overflow()
{ int i,j,k;

 for(i=0;i<=k1-1;i++)
   {
     trier (sequen[i].st,sequen[i].nbr_stencil);
     if(i==0)
       {
	 int m=0;
	 list lis;
	 expression lower, upper;
	 normalized norm1,norm2;
	 Pvecteur pv1,pv2;
	    Value v1,v2;
	    lis= variable_dimensions(type_variable(entity_type(first_array)));
          
	    MAP(DIMENSION,dim,{
	      lower= dimension_lower(dim);
	      upper=dimension_upper(dim);
	      normalize_all_expressions_of(lower);
	      norm1=expression_normalized(lower);
	      pv1= normalized_linear(norm1);
	      v1=vect_coeff(TCST,pv1);
	    

	     
	      normalize_all_expressions_of(upper);
	      norm2=expression_normalized(upper);
	      pv2= normalized_linear(norm2);
	      v2=vect_coeff(TCST,pv2);
	        
	      for(j=0;j<=sequen[i].nbr_stencil-1;j++)
		{    
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],depth-m,1), sequen[i].nd[depth-m-1].lower)
		      < v1 )
		    { 
		      overflow=true ;
		      printf(" Debordement dans le   tableau: %s\n",entity_name(first_array));


		    };
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],depth-m,1), sequen[i].nd[depth-m-1].upper)> 
		      v2 )
		    {   

		      overflow=true ;
		      printf(" Debordement dans le  tableau: %s\n",entity_name(first_array));
		    };
		}
	      m++;
	    },lis) ;
       }
     else
       { 
	 for(j=0;j<=sequen[i].nbr_stencil-1;j++)
	   for(k=0;k<=depth-1;k++)
		{
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],k+1,1), sequen[i].nd[k].lower)
		      <sequen[i-1].nd[k].lower )
		    {
		      overflow=true ;
		      printf(" Debordement dans le domaine du tableau: %s\n",entity_name(sequen[i-1].tableau));
		    };
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],k+1,1), sequen[i].nd[k].upper)> 
		      sequen[i-1].nd[k].upper )
		    {
		      overflow=true ;
		      printf(" Debordement dans le domaine du tableau: %s\n",entity_name(sequen[i-1].tableau));
		    };
		  
		};
       };
   };

 return overflow;
}

 

static void unroll_recursive(statement s,int n)
    
{
  if (n==1) 
    full_loop_unroll(s);
  else 
    {
 
      unroll_recursive(loop_body(instruction_loop(statement_instruction(s))), n-1);
      full_loop_unroll(s);
    };
}

statement  Tiling2_buffer()
{
  FILE *infp;
  char *name_file;
  Pmatrix A,A1,A2,P, P1,P2,P3;
  int rw,cl,i,j;
  list lis=NULL;
  instruction ins;
  sequence seq=NULL;
  statement s,s1,s2;
  loop ls=NULL;
  cons *lls=NULL;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  Pbase pb;
  expression upper=NULL,lower=NULL;
  int plus;
  Value elem_mat1,elem_mat2;
  Psysteme  loop_iteration_domaine_to_sc();
  //  void  full_loop_unroll();
  name_file = user_request("nom du fichier pour la matrice A ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&A,&rw,&cl);
  name_file = user_request("nom du fichier pour la matrice P ");
  infp = safe_fopen(name_file,"r");
  matrix_fscan(infp,&P,&cl,&cl);
  matrix_print(A);
  matrix_print(P); 
  P1=matrix_new(depth,depth);
  for(i=1;i<=depth;i++)
    for(j=1;j<=3*depth;j=j+3)
      {
	if ( value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
          MATRIX_ELEM(P1,i,j/3+1)=VALUE_ONE;
	else 
          MATRIX_ELEM(P1,i,j/3+1)=VALUE_ZERO;
      };

  P2=matrix_new(depth,depth);
  for(i=depth+1;i<=2*depth;i++)
    for(j=2;j<=3*depth;j=j+3)
      {
	if (value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
          MATRIX_ELEM(P2,(i-1)%depth+1,j/3+1)=VALUE_ONE;
	else 
          MATRIX_ELEM(P2,(i-1)%depth+1,j/3+1)=VALUE_ZERO;
      };
  
  P3=matrix_new(depth,depth);
  for(i=2*depth+1;i<=3*depth;i++)
    for(j=3;j<=3*depth;j=j+3)
      {
	if (value_eq(MATRIX_ELEM(P,i,j),VALUE_ONE))
          MATRIX_ELEM(P3,(i-1)%depth+1,j/3)=VALUE_ONE;
	else 
          MATRIX_ELEM(P3,(i-1)%depth+1,j/3)=VALUE_ZERO;
      };


  A1=matrix_new(depth,1);
  for(i=0;i<=depth-1;i++)
    MATRIX_ELEM(A1,i+1,1)=MATRIX_ELEM(A,i+1,3*i+1);

  A2=matrix_new(depth,1);
  for(i=0;i<=depth-1;i++)
    MATRIX_ELEM(A2,i+1,1)=MATRIX_ELEM(A,i+1,3*i+2);



 matrix_print(P1);
 matrix_print(P2);
 matrix_print(P3);
 matrix_print(A1);
 matrix_print(A2);

 getchar();
 getchar(); 

  compute_delay_tiled_nest ();
  compute_bound_tiled_nest();
  ls=loop1;
  for(j=0;j<=depth-1;j++)
    {
      range range1;  
      range1=loop_range(ls); 
      range_lower(range1)=int_to_expression(tiled_nest[j].lower);
      range_upper(range1)= int_to_expression(tiled_nest[j].upper);
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    };
  s1=loop_to_statement(loop1);
  while(instruction_loop_p(statement_instruction (s1))) {
    lls = CONS(STATEMENT,s1,lls);
    s1 = loop_body(instruction_loop(statement_instruction(s1)));
  }
  (void)loop_iteration_domaine_to_sc(lls, &base_oldindex);

  derive_new_basis_deux(base_oldindex, &base_newindex , make_tile_index_entity_n);
  tiling_indice= ( Pvecteur  *)malloc(depth *sizeof( Pvecteur*));
  plus=depth*3-1;
  for (pb =base_newindex; pb!=NULL; pb=pb->succ) {
    loop tl;
    switch(plus%3)
      {
      case 2: 
	{
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus);
	  elem_mat2=MATRIX_ELEM(A,plus/3+1,plus+1);
	  upper= int_to_expression(value_minus(elem_mat1,VALUE_ONE));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make(VECTEUR_NUL,vecteur_var(pb),elem_mat2,TCST,VALUE_ZERO);
	};
	break;
      case 1: 
	{ 
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus) ;
	  elem_mat2=MATRIX_ELEM(A,plus/3+1,plus+1  ) ;
	  upper= int_to_expression(value_minus(value_div(elem_mat1,elem_mat2),VALUE_ONE));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make( tiling_indice[plus/3],pb->var , elem_mat2,TCST,VALUE_ZERO);
	};
	break;   
      case 0:
	{ 
	  elem_mat1=MATRIX_ELEM(A,plus/3+1,plus+1);
	  upper= int_to_expression(value_div(tiled_nest[plus/3].upper,elem_mat1));
	  lower= int_to_expression(0);
	  tiling_indice[plus/3]=vect_make( tiling_indice[plus/3],pb->var ,elem_mat1,TCST,VALUE_ZERO);
	};
      default: break;
      }  
    tl = make_loop((entity) vecteur_var(pb),
		   make_range(copy_expression(lower), copy_expression(upper),
			      int_to_expression(1)),
		   s1,
		   entity_empty_label(),
		   make_execution(is_execution_sequential, UU),
		   NIL);
    s1 = instruction_to_statement(make_instruction(is_instruction_loop, tl));
    plus--;
  };
  
 

  /******/
  for (i=k1-1;i>=0;i--)
    {
      expression e1,e2,e,gauche=NULL,droite=NULL,delai_plus;
      test t;
      call c;
      int m;
      Pvecteur pv;
      e1=ge_expression(Pvecteur_to_expression (tiling_indice[0]),
		       Value_to_expression( value_plus(sequen[i].nd[0].lower,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      e2=le_expression(Pvecteur_to_expression (tiling_indice[0]), 
		       Value_to_expression( value_plus(sequen[i].nd[0].upper,
						       MATRIX_ELEM( sequen[i].delai,1,1))));
      if (value_eq( value_plus(sequen[i].nd[0].lower,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].lower))
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=NULL;
	  else
	    e=e2;
	}
      else
	{
	  if (value_eq( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)),tiled_nest[0].upper)&&
	      value_eq(value_mod( value_plus(sequen[i].nd[0].upper,MATRIX_ELEM( sequen[i].delai,1,1)), 
				  MATRIX_ELEM(A,1,1)),VALUE_ZERO))
	    e=e1;
	   
	  else
	    e=and_expression(e1,e2);
	    
	};
      for(j=0;j<=depth-1;j++)
	{
	  if (j>=1) 
	    {
	      e1=ge_expression( Pvecteur_to_expression (tiling_indice[j]),
				Value_to_expression(value_plus(sequen[i].nd[j].lower,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      e2=le_expression(Pvecteur_to_expression (tiling_indice[j]), 
			       Value_to_expression( value_plus(sequen[i].nd[j].upper,
							       MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      if (value_eq( value_plus(sequen[i].nd[j].lower,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].lower))
		{
		  if (! (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
			 value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					     MATRIX_ELEM(A,j+1,3*j+1)),VALUE_ZERO)))
		    {
		      if (e==NULL)
			e=e2;
		      else
			e=and_expression(e,e2);
		    };
		}
	      else
		{
		  if (value_eq(value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)),tiled_nest[j].upper)&&
		      value_eq(value_mod( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1)), 
					  MATRIX_ELEM(A,j+1,3*j+1)),VALUE_ZERO))
		    {
		      if (e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		  else
		    {
		      e1=and_expression(e1,e2);
		      if(e==NULL)
			e=e1;
		      else
			e=and_expression(e,e1);
		    }
		};
	    }; 
	  c= instruction_call(statement_instruction((sequen[i].s )));
	  m=0;
	  MAP(EXPRESSION,exp,{  
	    if (m==0)  gauche=exp;
	    if (m==1)  droite=exp;
	    m++;
	  },call_arguments(c)) ;
	  pv = vect_make(tiling_indice[j],TCST,value_uminus(MATRIX_ELEM(sequen[i].delai,j+1,1)));
	  delai_plus=Pvecteur_to_expression(pv);
      replace_entity_by_expression(gauche,(entity) sequen[i].nd[j].index,delai_plus);
      FOREACH(EXPRESSION,exp,call_arguments( syntax_call(expression_syntax(droite))))
        replace_entity_by_expression(exp,(entity) sequen[i].nd[j].index,delai_plus);
	};
      t= make_test(e,sequen[i].s,make_block_statement(NIL));
      s=test_to_statement(t);
      if(i==k1-1)   lis=CONS(STATEMENT,s,NIL);
      else      lis=CONS(STATEMENT,s,lis);
    };
  seq= make_sequence(lis);
  ins= make_instruction_sequence(seq);
  s= instruction_to_statement(ins);
  ls=instruction_loop(statement_instruction(s1));
  for(j=0;j<=3*depth-1;j++)
    if  (j!=3*depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
    else loop_body(ls)=s;
  s1=permutation(s1,P);
  
  i=1;
  s2=s1;
  while (i <=2*depth)
    {
      printf(" %d \n", i);
      s2= loop_body(instruction_loop(statement_instruction(s2)));
      i++;
    }
  print_statement(s2);
  printf(" tptpto \n");
  unroll_recursive(s2,depth);

  
  return s1;
}







bool tiling_sequence(string module)  
{
  statement stat,s1=NULL;
  
  context_t context;
 
  contenu_t contenu;
 
  
 void  module_reorder(); 
  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);
  set_current_module_entity(local_name_to_top_level_entity(module));
  
 
  
  stat = (statement) db_get_memory_resource(DBR_CODE, module, true); 
 
  
  context.contenu = hash_table_make(hash_pointer, 0);  
  context.depth   = hash_table_make(hash_pointer, 0);
  context.statement_stack = stack_make(statement_domain, 0, 0); 
   
  gen_context_multi_recurse
    (stat, & context, 
     statement_domain, stmt_flt, stmt_rwt,
     sequence_domain, seq_flt, seq_rwt, 
     test_domain, test_flt, test_rwt,
     loop_domain, loop_flt, loop_rwt,
     call_domain, call_flt, call_rwt,
     whileloop_domain, gen_true, wl_rwt,
     unstructured_domain, uns_flt, uns_rwt,
     expression_domain, gen_false, gen_null,
     NULL); 
  contenu = (contenu_t) hash_get(context.contenu, stat); 
  depth = (intptr_t ) hash_get(context.depth, stat);
    
  
  if (contenu!=is_a_stencil)
    {
      printf(" Le programme ne repond pas aux hypotheses  \n");
      array_overflow();
    }
  else
    {  
      if(!array_overflow())
	{
	  int choix ;
	  printf("--------------Choix de la transformation-------\n"); 
	  printf(" 1: Fusion  \n");
	  printf(" 2: Fusion avec allocation des tampons  \n");
	  printf(" 3: Tiling \n");
	  printf(" 4: Tiling a seul niveau avec  allocation des tampons \n");
	  printf(" 5: Tiling a deux niveaux avec  allocation des tampons \n");
	  printf(" Choix: ");
	  scanf("%d", &choix);
	  switch(choix)
	    {
	    case 1: s1=fusion(); 
	      break;
	    case 2:  s1= fusion_buffer();
	      break;
	    case 3: s1=Hierarchical_tiling(); 
	      break; 
	    case 4: s1=Tiling_buffer_allocation();
	      break;
            case 5: s1=Tiling2_buffer();
	      break;
	    default: break;
	      
            }  
	  
	  
	  module_reorder(s1); 
	  
	  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, s1);
	}
      
	   
    } 

      
    reset_current_module_entity();
    //  reset_current_module_statement();
  

  pips_debug(1, "done.\n");
  debug_off();
   
  return true;
}
 


