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
 
/*#include <sys/ddi.h>*/

#include "genC.h"

#include "linear.h"
#include "matrice.h"  
#include "sparse_sc.h"

#include "ri.h"
#include "database.h"
#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h" 
#include "control.h"
#include "conversion.h"
#include "hyperplane.h"

Psysteme sc_newbase;
Ptsg sg;
bool if1=false,if2=false; 

typedef struct 
{
  stack statement_stack;
} * context_p, context_t;

static Value myceil( k,m)
     Value  k,m; 
{
  Value  res;
  res = value_div(k,m);
  if(value_mod(k,m)==0) return res;
  else
    {
      if ( value_direct_multiply(k,m)>=0 ) return res+1;
      else return res;
    }
}

static Value myfloor ( k , m)
     Value  k,m; 
{
  Value  res;
  res = value_div(k,m);
  if(value_mod(k,m)==0) return res;
  else
    {
     if ( value_direct_multiply(k,m)>=0 )  return res;
     else return res-1;
    }
}

static Value eval(pv,val,var)
     Pvecteur pv;
     Value val;
     Variable var;
{
  Value  coef, cons;
  coef = vect_coeff(var,pv);
  cons = vect_coeff(TCST,pv);
  return (coef * val +cons ) ;
}

static Value eval2(val0,val1,val2,val)
     Value val0,val1,val2,val;
{
  return (  value_div(value_direct_multiply (val1,val)+val0,-val2  )) ;
}

 
static Value intersection (pv1, pv2,var,floor)
     Pvecteur pv1,pv2;
     Variable var;
     int  floor ;
{
  if (floor==1)
   return 
     myfloor(vect_coeff(TCST,pv2) -vect_coeff(TCST,pv1),
	     vect_coeff(var,pv1 )- vect_coeff(var,pv2 ) ); 
  else
    return 
      myceil(vect_coeff(TCST,pv2) -vect_coeff(TCST,pv1),
	     vect_coeff(var,pv1 )- vect_coeff(var,pv2 ) ); 
  
}

static bool stmt_flt(statement s,context_p context)
{
  stack_push(s, context->statement_stack); 
  return true; 
}
static void  stmt_rwt(statement s,context_p context)
{  
  stack_pop(context->statement_stack);
}
   
static bool loop_flt(loop l, context_p context)
{ 
                   /*DECLARATION */
  statement s1=NULL,s2=NULL,s_in_loop_in_side=NULL,st=NULL;
  int i=0,cas=0;
  bool first1=true,first2=true;
  Value val1=0,val2=0,f1_val1,f1_val2,f2_val1,f2_val2,e=0;
  loop copyloop,loop_in_side=NULL;
  Variable var1= (Variable)loop_index(l);
  instruction ins,ins1,ins2;
  sequence seq,seq1,seq2;
  list lis=NIL,lis1=NIL,lis1par=NIL,lis2=NIL,lis2par=NIL;
  range range;
  expression lower, upper,index=NULL ;
  normalized norm1, norm2;
  Pvecteur pv1 ,pv2;
  test t1,t2;
                /*END OF DECLARATION */

  copyloop =copy_loop(l);
  range=loop_range(l); 
  lower=range_lower(range);  
  upper=range_upper(range); 
  normalize_all_expressions_of(lower);
  normalize_all_expressions_of(upper);
  norm1=expression_normalized(lower);
  norm2=expression_normalized(upper);
  pips_assert("normalized are linear", normalized_linear_p(norm1)&&
	      normalized_linear_p(norm2));
  pv1= normalized_linear(norm1);
  pv2= normalized_linear(norm2); 
  val1= vect_coeff(TCST,pv1);
  val2=vect_coeff(TCST,pv2);
  ins= statement_instruction(loop_body(l));
  if (instruction_sequence_p(ins))
    {
      seq = instruction_sequence(ins);
      lis =sequence_statements(seq); 
      MAP(STATEMENT,s,{ instruction ins1=statement_instruction(s);
      i++;
      if(!(instruction_loop_p(ins1))&&first1 ) {
	if (instruction_call_p(ins1 )) {
	  call ca= instruction_call(ins1 ) ;
	  if(strcmp(entity_name(call_function(ca)),"TOP-LEVEL:CONTINUE" )!=0){
	    if (i==1)  { lis1=CONS(STATEMENT,s,NIL); lis1par=lis1;}
	    else {  
	      CDR(lis1par)=CONS(STATEMENT,s,NIL);
	      lis1par=CDR(lis1par);
	    };
	  };
	}
	else 
	  {
	    if (i==1)  { 
	      lis1=CONS(STATEMENT,s,NIL); 
	      lis1par=lis1;
	    }
	    else {
	      CDR(lis1par)=CONS(STATEMENT,s,NIL);
	      lis1par=CDR(lis1par);
	    };
	  }; 
      };
      if( (instruction_loop_p(ins1))&&first1 ) {
	loop_in_side=instruction_loop(ins1);
	index=  entity_to_expression(loop_index(loop_in_side));
	s_in_loop_in_side=loop_body( loop_in_side   );
	range=loop_range(loop_in_side);
	lower=range_lower(range);
	upper=range_upper(range); 
	normalize_all_expressions_of(lower);
	normalize_all_expressions_of(upper);
	norm1=expression_normalized(lower);
	norm2=expression_normalized(upper);
	pips_assert("normalized are linear", normalized_linear_p(norm1) &&
		    normalized_linear_p(norm2));
	pv1= normalized_linear(norm1);
	pv2= normalized_linear(norm2);
	f1_val1= eval(pv1,val1,var1);
	f1_val2= eval(pv1,val2,var1);
	f2_val1= eval(pv2,val1,var1);
	f2_val2= eval(pv2,val2,var1);
	if ( (f2_val1 >=f1_val1)&&(f2_val2 >=f1_val2)){
	  cas=1;
	  loop_body(l)=s;
	}
	else {
	  if ( (f2_val1 >=f1_val1)&&(f2_val2 <f1_val2)){
	    cas=2; 
	    loop_body(l)=s;
	    e=intersection(pv1,pv2,var1,1);
	  }
	  else{
	    if ( (f2_val1 < f1_val1)&&(f2_val2 >= f1_val2)){
	      e=intersection(pv1,pv2,var1,2);
	      cas=3; 
	      loop_body(l)=s;
	    }
	    else  {
	      cas=4;
	      CHUNK(CAR(lis))= (gen_chunk *) make_block_statement(NIL);
	    }
	  }
	}
	first1 =false; 
	i=0;
      }
      if ( (! first1 )&&(i !=0)) {
	if (instruction_call_p(ins1 )) {
	  call ca= instruction_call(ins1 ) ;
	  if(  strcmp(  entity_name(call_function(ca)),"TOP-LEVEL:CONTINUE" )!=0){
	    if (i==1)  { 
	      lis2=CONS(STATEMENT,s,NIL); 
	      lis2par=lis2;
	    }
	    else {
	      CDR(lis2par)=CONS(STATEMENT,s,NIL);
	      lis2par=CDR(lis2par);
	    };
	  };
	}
	else {
	  if (i==1) 
	    { 
	      lis2=CONS(STATEMENT,s,NIL);
	      lis2par=lis2;
	    }
	  else{
	    CDR(lis2par)=CONS(STATEMENT,s,NIL);
	    lis2par=CDR(lis2par);
	  };
	}; 
      };
      lis=CDR(lis);
      ;},lis);
      switch (cas) {
      case 1:{
	switch (gen_length(lis1)) {
	case 0 :{  break;}
	case 1 :{  s1=STATEMENT(CAR(lis1)); break;} 
	default :{ seq1= make_sequence(lis1);
	ins1= make_instruction_sequence(seq1);
	s1= instruction_to_statement(ins1); break;
	}
	};
	switch (gen_length(lis2)) {
	case 0 :{  break;}
	case 1 :{  s2=STATEMENT(CAR(lis2)); break;} 
	default :{ seq2= make_sequence(lis2);
	ins2= make_instruction_sequence(seq2);
	s2= instruction_to_statement(ins2); break;
	}
	};
	if (gen_length(lis1)!=0)
	  { expression copyindex, copylower;
	  copyindex=copy_expression(index);
          copylower=copy_expression(lower);
	  t1= make_test(eq_expression (copyindex,copylower),s1, 
			make_block_statement(NIL));
	  s1=test_to_statement(t1);
	  if1=true;
	  };
	if (gen_length(lis2)!=0)  
	  { expression copyindex, copyupper;
	    copyindex=copy_expression(index);
            copyupper=copy_expression(upper);
	    t2= make_test(eq_expression (copyindex,copyupper) ,s2, 
			  make_block_statement(NIL));
	    s2=test_to_statement(t2);
	    if2=true;
	  };
	if (gen_length(lis2)!=0)
	  lis=CONS(STATEMENT , s2,NIL);
	else{
	  s2=  make_block_statement(NIL);
	  lis=CONS(STATEMENT,s2,NIL);
	};
	lis=CONS(STATEMENT,s_in_loop_in_side,lis);
	if (gen_length(lis1)!=0)
	  lis=CONS(STATEMENT,s1,lis);
	else
	  lis=CONS(STATEMENT,   make_block_statement(NIL),lis);
	seq= make_sequence(lis);
	ins= make_instruction_sequence(seq);
	s_in_loop_in_side= instruction_to_statement(ins);
	loop_body(loop_in_side)=s_in_loop_in_side; 
	break ;
      }
      case 2:{
	switch (gen_length(lis1)) {
	case 0 :{  break;}
	case 1 :{  s1=STATEMENT(CAR(lis1)); break;} 
	default :{ seq1= make_sequence(lis1);
	ins1= make_instruction_sequence(seq1);
	s1= instruction_to_statement(ins1); break;
	}
	};
	switch (gen_length(lis2)) {
	case 0 :{  break;}
	case 1 :{  s2=STATEMENT(CAR(lis2)); break;} 
	default :{ seq2= make_sequence(lis2);
	ins2= make_instruction_sequence(seq2);
	s2= instruction_to_statement(ins2); break;
	}
	};
	if (gen_length(lis1)!=0)
	  {
	    expression copyindex, copylower;
	    copyindex=copy_expression(index);
	    copylower=copy_expression(lower); 
	    t1= make_test(eq_expression (copyindex,copylower),s1, 
			  make_block_statement(NIL));
	    s1=test_to_statement(t1);
	  };
	if (gen_length(lis2)!=0)  
	  { 
	    expression copyindex, copyupper;
	    copyindex=copy_expression(index);
            copyupper=copy_expression(upper);
	    t2= make_test(eq_expression (copyindex,copyupper) ,s2,
			  make_block_statement(NIL));
	    s2=test_to_statement(t2);
	  };
	if (gen_length(lis2)!=0)
	  lis=CONS(STATEMENT , s2,NIL);
	else{
	  s2=  make_block_statement(NIL);
	  lis=CONS(STATEMENT,s2,NIL);
	};
	lis=CONS(STATEMENT,s_in_loop_in_side,lis);
	if (gen_length(lis1)!=0)
	  lis=CONS(STATEMENT,s1,lis);
	else
	  lis=CONS(STATEMENT,   make_block_statement(NIL)    ,lis);
	seq= make_sequence(lis);
	ins= make_instruction_sequence(seq);
	s_in_loop_in_side= instruction_to_statement(ins);
	loop_body(loop_in_side)=s_in_loop_in_side;
	range_upper(loop_range(l))= int_to_expression(e);
	st=stack_head(context->statement_stack);
	s1=copy_statement (st);
	ins1 = statement_instruction(loop_body(copyloop));
	seq = instruction_sequence(ins1);
	lis =sequence_statements(seq); 
	MAP(STATEMENT,s,{ instruction ins1=statement_instruction(s);
	if( (instruction_loop_p(ins1))&&first2 ) {
	  CHUNK(CAR(lis))= (gen_chunk *) make_block_statement(NIL);
	  first2=false;
	};
	lis=CDR(lis);
	;},lis);
	range_lower(loop_range(copyloop))= int_to_expression(e+1);
	s2=loop_to_statement(copyloop); 
	lis=CONS(STATEMENT,s1,CONS(STATEMENT,s2,NIL));
	seq=make_sequence(lis);
	ins=make_instruction_sequence(seq);
	statement_instruction(st) = ins;
	break ;
      }
      
      case 3: {
	switch (gen_length(lis1)) {
	case 0 :{  break;}
	case 1 :{  s1=STATEMENT(CAR(lis1)); break;} 
	default :{ seq1= make_sequence(lis1);
	ins1= make_instruction_sequence(seq1);
	s1= instruction_to_statement(ins1); break;
	}
	};
	switch (gen_length(lis2)) {
	case 0 :{  break;}
	case 1 :{  s2=STATEMENT(CAR(lis2)); break;} 
	default :{ seq2= make_sequence(lis2);
	ins2= make_instruction_sequence(seq2);
	s2= instruction_to_statement(ins2); break;
	}
	};
	if (gen_length(lis1)!=0)
	  {
	    expression copyindex, copylower;
	    copyindex=copy_expression(index);
	    copylower=copy_expression(lower);
	    t1= make_test(eq_expression (copyindex,copylower),s1,
			  make_block_statement(NIL));
	    s1=test_to_statement(t1);
	    if1=true;
	  };
	if (gen_length(lis2)!=0)  
	  {
	    expression copyindex, copyupper;
	    copyindex=copy_expression(index);
	    copyupper=copy_expression(upper);
	    t2= make_test(eq_expression (copyindex,copyupper) ,s2, 
			  make_block_statement(NIL));
	    s2=test_to_statement(t2);
	    if2=true;
	  };
	if (gen_length(lis2)!=0)
	  lis=CONS(STATEMENT , s2,NIL);
	else{
	  s2=  make_block_statement(NIL);
	  lis=CONS(STATEMENT,s2,NIL);
	};
	lis=CONS(STATEMENT,s_in_loop_in_side,lis);
	if (gen_length(lis1)!=0)
	  lis=CONS(STATEMENT,s1,lis);
	else
	  lis=CONS(STATEMENT,   make_block_statement(NIL)    ,lis);
	seq= make_sequence(lis);
	ins= make_instruction_sequence(seq);
	s_in_loop_in_side= instruction_to_statement(ins);
	loop_body(loop_in_side)=s_in_loop_in_side;
	range_lower(loop_range(l))= int_to_expression(e);
	st=stack_head(context->statement_stack);
	s2=copy_statement (st);
	ins = statement_instruction(loop_body(copyloop));
	seq = instruction_sequence(ins);
	lis =sequence_statements(seq); 
	MAP(STATEMENT,s,{ instruction ins1=statement_instruction(s);
	if( (instruction_loop_p(ins1))&&first2 ) {
	  CHUNK(CAR(lis))= (gen_chunk *) make_block_statement(NIL);
	  first2=false;};
	lis=CDR(lis);
	;},lis);
	range_upper(loop_range(copyloop))= int_to_expression(e-1);
	s1=loop_to_statement(copyloop); 
	lis=CONS(STATEMENT,s1,CONS(STATEMENT,s2,NIL));
	seq=make_sequence(lis);
	ins=make_instruction_sequence(seq); 
	statement_instruction(st) = ins;
	break;}
      case 4 :{ break;}
      default :{break;}
      };
    }; 
  return false;
}
   
statement unimodular(s)
     statement s    ;
{
  FILE *infp;
  char *name_file;
  cons *lls=NULL;
  Psysteme sci;			/* sc initial */
  Psysteme scn;			/* sc nouveau */
  Psysteme sc_row_echelon;
  Pbase base_oldindex = NULL;
  Pbase base_newindex = NULL;
  matrice A;
  matrice G;
  matrice AG; 
  matrice G_inv;
  int n;				/* number of index */
  int m ;				/* number of constraints */
  statement s_lhyp;
  Pvecteur *pvg;
  Pbase pb;  
  expression lower, upper;
  Pvecteur pv1 ;
  loop l;
  while(instruction_loop_p(statement_instruction (s))) {
    lls = CONS(STATEMENT,s,lls);
    s = loop_body(instruction_loop(statement_instruction(s)));
  }
  sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);
  sc_dump(sci);
  n = base_dimension(base_oldindex);
  m = sci->nb_ineq;
  A = matrice_new(m,n);
  sys_matrice_index(sci, base_oldindex, A, n, m);
  name_file = user_request("nom du fichier pour la matrice T");
  infp = safe_fopen(name_file,"r");
  matrice_fscan(infp,&G,&n,&n); 
  safe_fclose(infp, name_file);
  free(name_file);
  G_inv = matrice_new(n,n);
  matrice_general_inversion(G,G_inv,n);
  AG = matrice_new(m,n);
  matrice_multiply(A,  G_inv, AG, m, n, n);
  printf( " tototototo \n");
  /* create the new system of constraintes (Psysteme scn) with  
     AG and sci */
  scn = sc_dup(sci);
  matrice_index_sys(scn, base_oldindex, AG, n,m );
  
  /* computation of the new iteration space in the new basis G */
  sc_row_echelon = new_loop_bound(scn,base_oldindex);

 
  /* change of basis for index */
  change_of_base_index(base_oldindex, &base_newindex);
  
  sc_newbase = sc_change_baseindex(sc_dup(sc_row_echelon), base_oldindex, base_newindex);

  
  sc_syst_debug(sc_newbase);
  sc_dump(sc_newbase);
  sg= sc_to_sg_chernikova(sc_newbase);  
  /* generation of hyperplane  code */
  /*  generation of bounds */
  for (pb=base_newindex; pb!=NULL; pb=pb->succ) {
    make_bound_expression(pb->var, base_newindex, sc_newbase, &lower, &upper);
  }
  /* loop body generation */
  pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
  scanning_base_to_vect(G_inv,n,base_newindex,pvg);
  pv1 = sc_row_echelon->inegalites->succ->vecteur;
  (void)vect_change_base(pv1,base_oldindex,pvg);    
  l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
  lower = range_upper(loop_range(l));
  upper= expression_to_expression_newbase(lower, pvg, base_oldindex);
  /* FI: I do not know if the last parameter should be true or false */
  s_lhyp = code_generation(lls, pvg, base_oldindex, base_newindex, sc_newbase, true);
  printf(" finn  \n");
  return(s_lhyp);
}
statement  free_guards( s)
     statement s;
{
  int isoler =-1,isoler2=-1;
  instruction ins;
  statement body1,body2,loop2=NULL;
  sequence seqi,seqj,seq;
  list lisi,lisj,lisjcopy,lis,lisjcopy2,lisp;
  statement first, last=NULL;
  expression  exp;
  normalized norm;
  Pvecteur pv1,pv2=NULL,pvif1=NULL,pvif2=NULL,pv_i_lower,pv_i_upper;
  Pcontrainte cif1=NULL,cif2=NULL;
  Value sommetg[4];
  loop tab_loop[6], loop1;
  instruction tab_instruction[6];
  statement tab_statement[6];
  int i=0,j;
  Psommet vertice,e;
  int nbr_vertice;
  Variable indice1,indice2;
  range range_i,range1,range2,range3;
  /* sg_dump(sg);  */
  /* fprint_lsom(stderr, vertice,variable_dump_name ); */
  ins =statement_instruction(s);
  loop1=instruction_loop(ins);
  body1=loop_body(instruction_loop(ins));
  indice1=(Variable )loop_index(instruction_loop(ins));
  range_i=loop_range(instruction_loop(ins));
  normalize_all_expressions_of(range_lower(range_i));
  normalize_all_expressions_of(range_upper(range_i));
  pv_i_lower=normalized_linear(expression_normalized(range_lower(range_i)));
  vect_add_elem(&pv_i_lower,indice1,-1);
  pv_i_upper=normalized_linear(expression_normalized(range_upper(range_i)));
  vect_add_elem(&pv_i_upper,indice1,-1);
  ins =statement_instruction(body1);
  seqi=instruction_sequence(ins);
  lisi =sequence_statements(seqi); 
  MAP(STATEMENT,s1,{loop2 =s1; },lisi) ; 
  ins=statement_instruction(loop2);
  indice2=(Variable )loop_index(instruction_loop(ins));
  body2=loop_body(instruction_loop(ins));
  ins =statement_instruction(body2); 
  seqj=instruction_sequence(ins);
  lisj =sequence_statements(seqj); 
  lisjcopy= gen_full_copy_list(lisj);
  first= (statement )CHUNK( CAR(lisj));
  nbr_vertice=sg_nbre_sommets(sg);
  printf(" le nombre de sommet est %d \n",  nbr_vertice);
  vertice= sg_sommets(sg); 
  for (e = vertice; e != NULL; e = e->succ) {
    pv1=e->vecteur;
    sommetg[i]=value_div (vect_coeff(indice1,pv1),e->denominateur);
    i++;
  };
  for(i=0;i<=nbr_vertice-1;i++)  
    { 
      for(j=i+1;j<= nbr_vertice;j++)  
	if( sommetg[j]< sommetg[i] )
	  {
	    Value permut;
	    permut= sommetg[i];
	    sommetg[i]= sommetg[j];
	    sommetg[j]=permut;
	  };
    };
  for(i=0;i<= nbr_vertice-1;i++)  
    {
      if( sommetg[i]== sommetg[i+1] )
	{
	  for (j=i;j<=nbr_vertice-1;j++)
	    sommetg[j]=sommetg[j+1];
	  nbr_vertice--;
	};
    };
  i=0;
  MAP(STATEMENT,s1, { last =s1; },lisjcopy) ; 
  if (if1){ 
    Pcontrainte peq;
    syntax syntax;
    call call;
    list lis;
    CHUNK(CAR(lisjcopy))= (gen_chunk *)
	test_true(instruction_test(statement_instruction(first))); 
    CHUNK(CAR(lisj))= (gen_chunk *) make_block_statement(NIL); 
    exp= test_condition (instruction_test(statement_instruction(first)));
    normalize_all_expressions_of(exp); 
    syntax=expression_syntax(exp);
    call=syntax_call(syntax);
    lis =call_arguments(call);
    exp=(expression )CHUNK(CAR(lis));
    norm=expression_normalized(exp);
    pv1=normalized_linear(norm);
    MAP(EXPRESSION,exp,{
      norm=expression_normalized(exp);
      pv2=normalized_linear(norm); 
    },lis) ; 
    pvif1=vect_substract(pv1,pv2);
    exp=Pvecteur_to_expression(pvif1);
    for (peq = sc_newbase->inegalites;peq!=NULL;peq=peq->succ){
      if(  vect_oppos(pvif1,peq->vecteur))
	pvif1=vect_substract(VECTEUR_NUL,vect_dup(pvif1));
    };  
    cif1=contrainte_make(pvif1);
    exp=Pvecteur_to_expression(pvif1);
    test_condition (instruction_test(statement_instruction(first)))= 
      eq_expression (exp, int_to_expression(0) );
  }
  if(if2){
    Pcontrainte peq;
    syntax syntax;
    call call;
    list lis;
    CHUNK(CAR(gen_last(lisjcopy)))= (gen_chunk *)
      test_true(instruction_test(statement_instruction(last))); 
    CHUNK( CAR(gen_last(lisj)))= (gen_chunk *) make_block_statement(NIL);   
    exp= test_condition (instruction_test(statement_instruction(last)));
    normalize_all_expressions_of(exp); 
    syntax=expression_syntax(exp);
    call=syntax_call(syntax);
    lis =call_arguments(call);
    exp=(expression) CHUNK(CAR(lis));
    norm=expression_normalized(exp);
    pv1=normalized_linear(norm);  
    MAP(EXPRESSION,exp,{
      norm=expression_normalized(exp);
      pv2=normalized_linear(norm); 
    },lis) ; 
    pvif2=vect_substract(pv1,pv2);
    for (peq = sc_newbase->inegalites;peq!=NULL;peq=peq->succ){
      if(  vect_oppos(pvif2,peq->vecteur))
	pvif2= vect_substract(VECTEUR_NUL,vect_dup(pvif2)) ; 
    }
    cif2=contrainte_make(pvif2);
    exp=Pvecteur_to_expression(pvif2);
    test_condition (instruction_test(statement_instruction(last)))= 
      eq_expression 
      (exp, int_to_expression(0) );
  }
  lis=NIL; 
  lisjcopy2= gen_full_copy_list(lisjcopy);
  if(if1) CHUNK( CAR(    (lisjcopy2)))= (gen_chunk *) make_block_statement(NIL);
  if(if2)CHUNK(CAR(gen_last((lisjcopy))))= (gen_chunk *) make_block_statement(NIL);
  for(i=0;i<=2*nbr_vertice-2;i++){ 
    Pcontrainte peq,cinf=NULL,csup=NULL;
    statement body1,body2;
    instruction ins;
    sequence seqi,seqj;
    list lisi,lisj;
    range range; 
    statement loop2=NULL;
    bool condition1=false,condition2=false;
    tab_loop[i]=copy_loop(loop1);
    tab_instruction[i] = make_instruction_loop(tab_loop[i]);
    tab_statement[i]=instruction_to_statement (  tab_instruction[i]);
    lis=CONS(STATEMENT,  tab_statement[i],lis);
    if (i%2==0){
      int indice;
      Value minjp=-100,maxjp=100;
      indice=i/2;
      range_lower(loop_range(tab_loop[i]))=
	int_to_expression( sommetg[indice]);
      range_upper(loop_range(tab_loop[i]))=
	int_to_expression( sommetg[indice]);
      for (peq = sc_newbase->inegalites;peq!=NULL;peq=peq->succ){
	Pvecteur v;
	Value  constante,val1,val2;
	v = contrainte_vecteur(peq);
	constante=vect_coeff(TCST,v);
	val1=vect_coeff(indice1,v);
	val2=vect_coeff(indice2,v);
	if (val2 < 0){
	  if (eval2(constante,val1,val2, sommetg[indice])>minjp ){
	    minjp=eval2(constante,val1,val2, sommetg[indice]);
	    cinf=peq;
	  }
	  else {
	    if (  eval2(constante,val1,val2, sommetg[indice])==minjp ){
	      Pvecteur vp;
	      Value  constantep,val1p,val2p;
	      vp=contrainte_vecteur(cinf);
	      constantep=vect_coeff(TCST,vp);
	      val1p=vect_coeff(indice1,vp);
	      val2p=vect_coeff(indice2,vp);
	      if(value_direct_multiply
		 ( value_direct_multiply(val1,sommetg[indice])+ constante ,-val2p)
		 > value_direct_multiply
		 (value_direct_multiply(val1p,sommetg[indice])+constantep,-val2))
		cinf=peq;
	      else
		if(value_direct_multiply
		   (value_direct_multiply(val1,sommetg[indice])+ constante,-val2p)
		   ==value_direct_multiply
		   (value_direct_multiply(val1p,sommetg[indice])+constantep,-val2))
		  if(egalite_equal(peq,cif1)||egalite_equal(peq,cif2) )
		    cinf=peq;
	    };
	  };
	};
	if (val2 > 0){
	  if (eval2(constante,val1,val2, sommetg[indice])<maxjp ){
	    maxjp=eval2(constante,val1,val2, sommetg[indice]);
	    csup=peq;
	  }
	  else {
	    if (eval2(constante,val1,val2, sommetg[indice])==maxjp ){  
	      Pvecteur vp;
	      Value  constantep,val1p,val2p;
	      vp=contrainte_vecteur(csup);
	      constantep=vect_coeff(TCST,vp);
	      val1p=vect_coeff(indice1,vp);
	      val2p=vect_coeff(indice2,vp);;
	      if(value_direct_multiply
		 (value_direct_multiply(val1,sommetg[indice])+ constante ,-val2p)
		 < value_direct_multiply
		 (value_direct_multiply (val1p,sommetg[indice])+ constantep,-val2))
		csup=peq; 
	      else
		if(value_direct_multiply
		   (value_direct_multiply(val1,sommetg[indice])+ constante ,-val2p)
		   ==value_direct_multiply
		   (value_direct_multiply(val1p,sommetg[indice])+constantep,-val2))
		  if( ( egalite_equal(peq,cif1) )||(egalite_equal(peq,cif2)) )  
		    csup=peq; 
	    };
	  };
	};
      };
      for (e = vertice; e != NULL; e = e->succ) {
	Value j2;
	Pvecteur pv1;
	pv1=e->vecteur;
	if (sommetg[indice]==vect_coeff(indice1,pv1)){
	  j2=vect_coeff(indice2,pv1);
	  if ( 
	      (value_direct_multiply(vect_coeff(indice1,pvif1),sommetg[indice])+
	       value_direct_multiply(vect_coeff(indice2,pvif1),j2)+
	       vect_coeff(TCST,pvif1)==0)&&
	      (value_direct_multiply(vect_coeff(indice1,pvif2),sommetg[indice])+
	       value_direct_multiply(vect_coeff(indice2,pvif2),j2)+
	       vect_coeff(TCST,pvif2)==0)){
	    if( vect_coeff(indice2,pvif2) < 0 )
	      condition1=true;
	    else{
	      if( vect_coeff(indice2,pvif1) > 0 ){
		condition2=true;
		
	      };
	    }
	  }
	}
      };
      
    }
    else{
      int indice;
      Value minjp1=-100,minjp2=-100,maxjp1=100,maxjp2=100;
      indice=i/2;
      range_lower(loop_range(tab_loop[i]))=
	int_to_expression( sommetg[indice]+1);
      range_upper(loop_range(tab_loop[i]))=
	int_to_expression( sommetg[indice+1]-1);
      for (peq = sc_newbase->inegalites;peq!=NULL;peq=peq->succ){
	Pvecteur v;
	Value constante,val1,val2;
	v = contrainte_vecteur(peq);   
	constante=vect_coeff(TCST,v);
	val1=vect_coeff(indice1,v);
	val2=vect_coeff(indice2,v);
	if (val2 < 0){
	  if (  (eval2(constante,val1,val2, sommetg[indice])>=minjp1 )
		&&( eval2(constante,val1,val2, sommetg[indice+1]) >=minjp2)){
	    if (  (eval2(constante,val1,val2, sommetg[indice])==minjp1 )
		  &&( eval2(constante,val1,val2, sommetg[indice+1]) ==minjp2))
	      {
		Pvecteur vp;
		Value  constantep,val1p,val2p;
		vp=contrainte_vecteur(cinf);
		constantep=vect_coeff(TCST,vp);
		val1p=vect_coeff(indice1,vp);
		val2p=vect_coeff(indice2,vp);
		if((value_direct_multiply
		    (value_direct_multiply(val1,sommetg[indice])+constante,-val2p)
		    > value_direct_multiply
		    (value_direct_multiply(val1p,sommetg[indice])+constantep,-val2))|| 
		   (value_direct_multiply
		    ( value_direct_multiply(val1,sommetg[indice+1])+ constante ,-val2p)
		    > value_direct_multiply
		    ( value_direct_multiply 
		      (val1p,sommetg[indice+1])+ constantep ,-val2)))
                  {
		    minjp1=eval2(constante,val1,val2, sommetg[indice]);
		    minjp2=eval2(constante,val1,val2, sommetg[indice+1]);
		    cinf=peq;
		  }
	      }
	    else{
	      minjp1=eval2(constante,val1,val2, sommetg[indice]);
	      minjp2=eval2(constante,val1,val2, sommetg[indice+1]);
	      cinf=peq;
	    }
	  };   
	};
	if (val2 > 0){
	  if (  (eval2(constante,val1,val2, sommetg[indice])<=maxjp1 )&&
		( eval2(constante,val1,val2, sommetg[indice+1]) <=maxjp2)){
	    if (  (eval2(constante,val1,val2, sommetg[indice])==maxjp1 )
		  &&( eval2(constante,val1,val2, sommetg[indice+1]) ==maxjp2))
	      {
		Pvecteur vp;
		Value  constantep,val1p,val2p;
		vp=contrainte_vecteur(csup);
		constantep=vect_coeff(TCST,vp);
		val1p=vect_coeff(indice1,vp);
		val2p=vect_coeff(indice2,vp);
		if((value_direct_multiply
		    (value_direct_multiply(val1,sommetg[indice])+ constante ,-val2p)
		    < value_direct_multiply
		    (value_direct_multiply(val1p,sommetg[indice])+constantep,-val2)) || 
		   (value_direct_multiply
		    (value_direct_multiply(val1,sommetg[indice+1])+ constante ,-val2p)
		    < value_direct_multiply
		    (value_direct_multiply (val1p,sommetg[indice+1])+constantep,-val2)))
		  {
		    maxjp1=eval2(constante,val1,val2, sommetg[indice]);
		    maxjp2=eval2(constante,val1,val2, sommetg[indice+1]);
		    csup=peq;
		  };
	      }
	    else{
	      maxjp1=eval2(constante,val1,val2, sommetg[indice]);
	      maxjp2=eval2(constante,val1,val2, sommetg[indice+1]);
	      csup=peq;
	    };
	  };
	};
      };
    }; 
    body1=loop_body( tab_loop[i]);
    ins =statement_instruction(body1);
    seqi=instruction_sequence(ins);
    lisi =sequence_statements(seqi); 
    MAP(STATEMENT,s,loop2=s,lisi);
    ins=statement_instruction(loop2);
    range =loop_range(instruction_loop(ins));
    range_lower(range)=make_contrainte_expression(cinf,indice2);
    range_upper(range)=make_contrainte_expression(csup,indice2);
    body2= loop_body(instruction_loop(ins));
    seqj=instruction_sequence(statement_instruction(body2));
    lisj=sequence_statements(seqj);
    if (condition1){
      list lisjcopy;
      Pvecteur v;
      Value constante;
      lisjcopy= gen_full_copy_list(lisj);
      CHUNK(CAR(gen_last(lisjcopy)))= (gen_chunk *)
	test_true(instruction_test(statement_instruction(last)));
      CHUNK(CAR(lisjcopy))= (gen_chunk *)
	test_true(instruction_test(statement_instruction(first)));
      lisi= gen_concatenate(lisjcopy,lisi);
      sequence_statements(seqi)=lisi;
      v = vect_dup (contrainte_vecteur(cinf));
      constante=vect_coeff(TCST,v);
      vect_chg_coeff(&v,TCST,value_plus(1,constante)); 
      cinf=contrainte_make(v);
      range_lower(range)=make_contrainte_expression(cinf,indice2); 
      
    }
    if (condition2){
      Pvecteur v;
      Value constante;
      list lisjcopy;
      lisjcopy= gen_full_copy_list(lisj);
      CHUNK(CAR(gen_last(lisjcopy)))= (gen_chunk *)
	test_true(instruction_test(statement_instruction(last)));
      CHUNK(CAR(lisjcopy))= (gen_chunk *)
	test_true(instruction_test(statement_instruction(first)));
      lisi= gen_concatenate(lisi,lisjcopy);
      sequence_statements(seqi)=lisi;
      v=vect_dup( contrainte_vecteur(csup));
      constante=vect_coeff(TCST,v);
      vect_chg_coeff(&v,TCST,value_plus(1,constante)); 
      csup=contrainte_make(v);
      range_upper(range)=make_contrainte_expression(csup,indice2);  
    }
    if (i==0){
      if ( vect_equal(pvif1,pv_i_lower) ||
	   vect_oppos(pvif1,pv_i_lower)){
	CHUNK( CAR((lisj)))= (gen_chunk * ) 
	  test_true(instruction_test(statement_instruction(first)));
	sequence_statements(seqj)=lisj;   
	isoler2=0;  
      }  
      else{
	if ( vect_equal(pvif2,pv_i_lower) ||
	     vect_oppos(pvif2,pv_i_lower)) {
	  CHUNK( CAR(gen_last((lisj))))= (gen_chunk * ) 
	    test_true(instruction_test(statement_instruction(last)));
	  sequence_statements(seqj)=lisj;  
	  isoler2=0; 
	};
      };
    };
    if (i==2*nbr_vertice-2){
      if ( vect_equal(pvif1,pv_i_upper) ||
	   vect_oppos(pvif1,pv_i_upper)) 
	{
	  CHUNK( CAR((lisj)))= (gen_chunk * ) 
	  test_true(instruction_test(statement_instruction(first)));
	  sequence_statements(seqj)=lisj;   
	  isoler=2*nbr_vertice-2;
	}  
      else{
	if ( vect_equal(pvif2,pv_i_upper) ||
	     vect_oppos(pvif2,pv_i_upper)) {
	  CHUNK( CAR(gen_last((lisj))))= (gen_chunk * ) 
	    test_true(instruction_test(statement_instruction(last)));
	  sequence_statements(seqj)=lisj;  
	  isoler=2*nbr_vertice-2;
	};
      };
    };
    
    if ((condition1==false) && (condition2==false)){
      
      /*  debut  cas1  */
      
      if ( egalite_equal(cinf,cif1)){
	if (value_abs  (vect_coeff(indice2,pvif1))==1){  
	  lisi=CONS(STATEMENT,
		    test_true(instruction_test(statement_instruction(first))),lisi);
	  sequence_statements(seqi)=lisi;  
	}  
	else{
	  list lexp;
	  call ca;
	  expression exp1,exp2;
	  Pvecteur pv;
	  statement s;
	  pv=vect_dup (pvif1); 
	  vect_erase_var( &pv,indice2);
	  exp1=Pvecteur_to_expression(pv);
	  s= copy_statement (first);
	  lexp = CONS(EXPRESSION, int_to_expression
		      (-vect_coeff(indice2,pvif1)), NIL);
	  lexp = CONS(EXPRESSION, exp1, lexp);
	  ca = make_call(entity_intrinsic(MODULO_OPERATOR_NAME), lexp);
	  exp1 = make_expression(make_syntax(is_syntax_call, ca),
				 normalized_undefined);
	  exp2=int_to_expression(0);
	  exp1=eq_expression(exp1,exp2);
	  test_condition(instruction_test(statement_instruction(s)))=exp1; 
	  lisi=CONS(STATEMENT,s,lisi);
	  sequence_statements(seqi)=lisi;
	};
      }; 
      
      /* fins cas1 */
      
      
      /*debut cas2 */
      
      if ( egalite_equal(csup,cif2)) {
	list listemp;
	if (value_abs  (vect_coeff(indice2,pvif2))==1){  
	  listemp=CONS(STATEMENT,
		       test_true(instruction_test(statement_instruction(last))),NIL);
	  lisi= gen_concatenate(lisi,listemp);
	  sequence_statements(seqi)=lisi;  
	}  
	else{
	  list lexp;
	  call ca;
	  expression exp1,exp2;
	  Pvecteur pv;
	  statement s;
	  pv=vect_dup (pvif2); 
	  vect_erase_var( &pv,indice2);
	  exp1=Pvecteur_to_expression(pv);
	  s= copy_statement (last);
	  lexp = CONS(EXPRESSION, int_to_expression
		      (-vect_coeff(indice2,pvif2)), NIL);
	  lexp = CONS(EXPRESSION, exp1, lexp);
	  ca = make_call(entity_intrinsic(MODULO_OPERATOR_NAME), lexp);
	  exp1 = make_expression(make_syntax(is_syntax_call, ca),
				 normalized_undefined);
	  exp2=int_to_expression(0);
	  exp1=eq_expression(exp1,exp2);
	  test_condition(instruction_test(statement_instruction(s)))=exp1; 
	  listemp=CONS(STATEMENT,s ,NIL);
	  lisi= gen_concatenate(lisi,listemp);
	  sequence_statements(seqi)=lisi;
	};
      }; 
    
      
      /* fin  cas2  */
      /* debut cas3 */
      
      if ( egalite_equal(cinf,cif2)) {
	statement sif1;
	sequence seqif1;
	instruction insif1;
	Pvecteur v;
	Value  constante;
	seqif1=make_sequence(lisjcopy2);
	insif1=make_instruction_sequence(seqif1);
	sif1=instruction_to_statement(insif1);
	if (value_abs  (vect_coeff(indice2,pvif2))==1){  
	  lisi=gen_concatenate(lisjcopy2,lisi);
	  sequence_statements(seqi)=lisi;  
	}  
	else{
	  list lexp;
	  call ca;
	  expression exp1,exp2;
	  Pvecteur pv;
	  Value val;
	  statement s/*,s1*/;
	  pv=vect_dup (pvif2);
	  val=vect_coeff(indice2,pvif2);
	  vect_erase_var( &pv,indice2);
	  exp1=Pvecteur_to_expression(pv);;
	    s= copy_statement (last);

	    lexp = CONS(EXPRESSION, int_to_expression
		      (-val), NIL);
	   lexp = CONS(EXPRESSION, exp1, lexp); 
	  ca = make_call(entity_intrinsic(MODULO_OPERATOR_NAME), lexp);
	  exp1 = make_expression(make_syntax(is_syntax_call, ca),
				 normalized_undefined);
	    exp2=int_to_expression(0);
	    exp1=eq_expression(exp1,exp2); 
	      test_condition(instruction_test(statement_instruction(s)))=exp1;    
		test_true(instruction_test(statement_instruction(s)))=copy_statement (sif1);   
	    lisi=CONS(STATEMENT,s,lisi);
	      sequence_statements(seqi)=lisi; 
	};
        v = vect_dup (contrainte_vecteur(cinf));
	constante=vect_coeff(TCST,v);
        vect_chg_coeff(&v,TCST,value_plus(1,constante)); 
	cinf=contrainte_make(v);
	range_lower(range)=make_contrainte_expression(cinf,indice2);  
	}  
      /*fin cas3 */
      /* debut cas4 */
        
      if ( egalite_equal(csup,cif1)){
	statement sif2;
	sequence seqif2;
	instruction insif2;
	Pvecteur v;
	list listemp;
	Value  constante;
	seqif2=make_sequence(lisjcopy);
	insif2=make_instruction_sequence(seqif2);
	sif2=instruction_to_statement(insif2);
	if (value_abs  (vect_coeff(indice2,pvif1))==1){  
	  lisi=gen_concatenate(lisi,lisjcopy);
	  sequence_statements(seqi)=lisi;  
	}  
	else{
	  list lexp;
	  call ca;
	  expression exp1,exp2;
	  Pvecteur pv;
	  statement s;
	  pv=vect_dup (pvif1); 
	  vect_erase_var( &pv,indice2);
	  exp1=Pvecteur_to_expression(pv);
	  s= copy_statement (first);
	  lexp = CONS(EXPRESSION, int_to_expression
		      (-vect_coeff(indice2,pvif1)), NIL);
	  lexp = CONS(EXPRESSION, exp1, lexp);
	  ca = make_call(entity_intrinsic(MODULO_OPERATOR_NAME), lexp);
	  exp1 = make_expression(make_syntax(is_syntax_call, ca),
				 normalized_undefined);
	  exp2=int_to_expression(0);
	  exp1=eq_expression(exp1,exp2);
	  test_condition(instruction_test(statement_instruction(s)))=exp1;    
	  test_true(instruction_test(statement_instruction(s)))=copy_statement(sif2);
	  listemp=CONS(STATEMENT,s ,NIL);
	  lisi= gen_concatenate(lisi,listemp);
	  sequence_statements(seqi)=lisi;  
	};
        v=vect_dup( contrainte_vecteur(csup));
	constante=vect_coeff(TCST,v);
	vect_chg_coeff(&v,TCST,value_plus(1,constante)); 
	csup=contrainte_make(v);
	range_upper(range)=make_contrainte_expression(csup,indice2);  
	
	}  
      
      /*fin cas4 */
    };
    if (i%2==0){
        if (condition1||condition2) isoler =i; 
    }

  };
  lis= gen_nreverse(lis);
  i=-1;
  lisp=lis; 
  FOREACH(STATEMENT,s,lis)
  {
      s=s;
      i++;if((i!=isoler)&&(i%2==0)&&(i!=isoler2)) {
    if(i==0) {
      CHUNK( CAR(lisp))= (gen_chunk *) make_block_statement(NIL); 
      range_lower(loop_range(tab_loop[i+1]))= 
	int_to_expression( sommetg[(i+1)/2]);  
    }
    else{
      if(i==2*nbr_vertice-2){
	CHUNK( CAR(lisp))= (gen_chunk *) make_block_statement(NIL); 
	range_upper(loop_range(tab_loop[i-1]))= 
	  int_to_expression(sommetg[(i-1)/2+1]);
      }
      else{   
	body1=loop_body(tab_loop[i]);
	ins =statement_instruction(body1);
	seq=instruction_sequence(ins);
	lisi =sequence_statements(seq); 
	MAP(STATEMENT,s1,{if ( instruction_loop_p(statement_instruction(s1)))
	  loop2 =s1; },lisi) ; 
	ins =statement_instruction(loop2);
	range1=loop_range(instruction_loop(ins)); 
	body1=loop_body(tab_loop[i-1]);
	ins =statement_instruction(body1);
	seq=instruction_sequence(ins);
	lisi =sequence_statements(seq); 
	MAP(STATEMENT,s1,{if ( instruction_loop_p(statement_instruction(s1)))
	  loop2 =s1; },lisi) ; 
	ins =statement_instruction(loop2);
	range2=loop_range(instruction_loop(ins)); 
	body1=loop_body(tab_loop[i+1]);
	ins =statement_instruction(body1);
	seq=instruction_sequence(ins);
	lisi =sequence_statements(seq); 
	MAP(STATEMENT,s1,{if ( instruction_loop_p(statement_instruction(s1)))
	  loop2 =s1; },lisi) ; 
	ins =statement_instruction(loop2);
	range3=loop_range(instruction_loop(ins)); 
	if (range_equal_p(range1,range2)) {
	  CHUNK( CAR(lisp))= (gen_chunk *) make_block_statement(NIL); 
	  range_upper(loop_range(tab_loop[i-1]))=
	    int_to_expression( sommetg[(i-1)/2+1]);
	}
	else {
	  CHUNK( CAR(lisp))= (gen_chunk *) make_block_statement(NIL); 
	  if (range_equal_p(range1,range3)) 
	    range_lower(loop_range(tab_loop[i+1]))= 
	      int_to_expression( sommetg[(i+1)/2]);  
	}
      };  
    };

  };
  lisp=CDR(lisp); 
   }
    seq=make_sequence (lis);
     ins= make_instruction_sequence(seq);
     return(  instruction_to_statement(ins)); 
}

bool guard_elimination(string module)
{

  statement stat,s1;
  string str ;
  int ordering ; 
  instruction ins;
  sequence seq;
  list lis;        
  context_t context;                              
  debug_on("GUARD_ELIMINATION_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);
  set_current_module_entity(local_name_to_top_level_entity(module));
  stat = (statement) db_get_memory_resource(DBR_CODE, module, true); 
  set_current_module_statement(stat);
  context.statement_stack = stack_make(statement_domain, 0, 0);
  
  // user_request
  // recherche le statement dans stat
  // appliquer...
  
  /* set_ordering_to_statement(stat); */
  ins=statement_instruction(stat);
  seq=instruction_sequence(ins);
  lis =sequence_statements(seq); 
  /* MAP(STATEMENT,s,{printf("%d \n",statement_ordering(s));},lis) */
  
  str = user_request("Which nid do you want to treat?\n"
		     "(give its ordering): ");
  ordering=atoi(str);  
  
  
  s1=ordering_to_statement(ordering);
  
  gen_context_multi_recurse(s1, &context,
			    statement_domain, stmt_flt, stmt_rwt,
			    loop_domain, loop_flt, gen_null,
			    NULL);  
  str = user_request("for Which nid do you want apply unimodular " 
                      "transformation ?\n" "(give its ordering): ");
  ordering=atoi(str); 
  	      
  s1=ordering_to_statement(ordering);
  MAP(STATEMENT,s,{if (statement_ordering(s)==ordering){
    s1=unimodular(s1); ; s1= free_guards(s1);  
    CHUNK(CAR(lis))= (gen_chunk *) s1;}
  ;lis=CDR(lis) ;},lis) 
    
    
    
  
    /*  gen_recurse(stat, statement_domain, gen_true,);  */
    module_reorder(stat); 
  
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, stat);
    reset_current_module_entity();
    reset_current_module_statement();
    debug_off();
    return true;  
}








































