#include <stdio.h>
#include "genC.h"    
#include "linear.h"
#include "ri.h"  
#include "database.h"  
#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
/* #include "statistics.h" */
#define DEFAULT_INT_PREFIX 	"I_"
#define DEFAULT_FLOAT_PREFIX 	"F_"
#define DEFAULT_LOGICAL_PREFIX 	"L_"
#define DEFAULT_COMPLEX_PREFIX	"C_"
#define DEFAULT_STRING_PREFIX	"S_" 

typedef struct INFO_LOOP {
  Variable index;
  Value upper, lower;  
} info_loop;


typedef struct NID { 
  entity tableau;
  Pmatrix  *st;
  info_loop *nd;
  int nbr_stencil;
  Pmatrix delai;
  statement s;
  Value surface;
  Pmatrix coef;
  Pvecteur pv_acces;
  entity ref;
} nid;

expression expt;
Pmatrix a;   
int k1=0,k2=0;
nid  sequen[10];
int depth;
info_loop  *ptr;     
int depth_max=40,st_max=40;
info_loop *merged_nest;
loop loop1;
bool overflow=FALSE;
entity first_array;
static int unique_integer_number = 0,
  unique_float_number = 0,
  unique_logical_number = 0,
  unique_complex_number = 0,
  unique_string_number = 0;


static int *nbr_no_perf_nest_loop_of_depth, *nbr_perf_nest_loop_of_depth,
  nbr_loop, nbr_perfectly_nested_loop, nbr_interested_loop,
  nbr_no_perfectly_nested_loop,nbr_nested_loop,
  max_depth_perfect, max_depth_no_perfect,cpt ;

static bool sequence_valide=TRUE, first_turn=FALSE;


static typedef enum {is_a_stencil, is_unknown, is_a_perf_nes_loop, is_a_no_perf_nes_loop, 
	      is_a_no_perf_nes_loop_t, is_a_call, is_a_continue,
                is_a_while, is_a_test,  is_a_no_stencil } contenu_t;
static typedef struct {
  hash_table contenu;
  hash_table depth;
  stack statement_stack;
} * context_p, context_t;

static bool loop_flt(loop l, context_p context )
{  
  range range1;
  expression lower, upper;
  normalized norm1, norm2;
  Variable  index;
  Pvecteur pv1,pv2;
  if( ! first_turn )
    {
      loop1=copy_loop(l);
      first_turn = TRUE; 
      sequen[k1].nd= (struct INFO_LOOP  *)malloc(depth_max *sizeof(struct INFO_LOOP));
      sequen[k1].st= ( Pmatrix *)malloc(st_max *sizeof(Pmatrix));
    }; 
  index =(Variable) loop_index(l);
  range1=loop_range(l); 
  lower=range_lower(range1);  
  upper=range_upper(range1); 
  expt=lower;
  normalize_all_expressions_of(lower);
  normalize_all_expressions_of(upper);
  norm1=expression_normalized(lower);
  norm2=expression_normalized(upper);
  pips_assert("normalized are linear", normalized_linear_p(norm1)&&
	     normalized_linear_p(norm2));
  pv1= normalized_linear(norm1);
  pv2= normalized_linear(norm2); 
  sequen[k1].nd[k2].index=index;
  sequen[k1].nd[k2].lower=vect_coeff(TCST,pv1);
  sequen[k1].nd[k2].upper=vect_coeff(TCST,pv2); 
  k2++; 
  return TRUE;
} 

static void loop_rwt(loop l, context_p context  ) 
{  
  contenu_t contenu;
  int depth;

   
    statement s = loop_body(l);
    contenu = (contenu_t) hash_get(context->contenu, s);
    depth= (int) hash_get(context->depth, s); 
    depth++;
  
    hash_put(context->depth,stack_head(context->statement_stack), 
             ( void *) depth); 
    hash_put(context->contenu,stack_head(context->statement_stack), 
	     (void *) contenu  );
    if (first_turn)
      {
	first_turn=FALSE;
	k2=0;
	k1++;
      }
}
static bool stmt_flt(statement s,context_p context )
{  
  stack_push(s, context->statement_stack); 
  return TRUE;
}


static void stmt_rwt( statement s, context_p context)
{  
 stack_pop(context->statement_stack);   
}
 
static bool seq_flt( sequence sq, context_p context  )
{
  return TRUE;
}

static void seq_rwt(sequence sq, context_p context)
{
  contenu_t contenu= is_unknown;
  int depth1=0, depth2=0;
  int max=0;
  int i=0;
  list l= sequence_statements(sq);
  contenu=is_a_stencil;
  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	       (void *) contenu);
  MAP(STATEMENT, s,
  {
    contenu = (contenu_t) hash_get(context->contenu, s);
    if (i==0) depth1 = (int ) hash_get(context->depth, s);
    if (contenu ==is_a_stencil)  
      {  
	depth2= (int ) hash_get(context->depth, s);
	if (depth1!=depth2)
	  {  
	    contenu=is_a_no_stencil;
	    hash_put(context->contenu,
		     stack_head(context->statement_stack),
		     (void *) contenu);
	  };
	depth1 =  depth2;
      }
    else 
      {
	if ( contenu !=is_a_continue)
	  { 
	    contenu=is_a_no_stencil;
	    hash_put(context->contenu,
		     stack_head(context->statement_stack),
		     (void *) contenu);
	  };
      };
    if (depth2 > max) max=depth2;
    i++;
  };, l); 
  hash_put(context->depth,
	   stack_head(context->statement_stack), 
	   (void *) max   );
} 

static bool uns_flt(unstructured u, context_p context   )
{
  return TRUE;
}

static void uns_rwt(unstructured u, context_p context)
{  contenu_t contenu;

  contenu=is_a_no_stencil; 
  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), 
	   ( void *) 0);  
}
 
static bool test_flt(test t, context_p context)
{
  return TRUE;
}

static void test_rwt(test t, context_p context)
{
  contenu_t contenu;
  contenu=is_a_no_stencil; 
  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), 
	   ( void *) 0);   
} 
static bool call_flt( call  ca, context_p context)
{
  return TRUE ;
}

static void call_rwt(call  ca, context_p context)
{ 
  contenu_t contenu=is_a_stencil;
  if (  (  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:CONTINUE" )==0) ||
	(  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:RETURN" )==0))
    {
      contenu=is_a_continue; 
    }
  else
    {
      if (  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:=")==0)
	{
	  list lis;
	  expression exp, gauche=NULL, droite=NULL;
	  syntax stg,std;
          normalized norm;
	  Pvecteur pv;       
	  int i=0,j=0;
	  lis =call_arguments(ca);
	  exp=(expression )CHUNK(CAR(lis));
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
  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *)contenu );
  hash_put(context->depth,stack_head(context->statement_stack), 
	   ( void *) 0);  
  /* if  (contenu==is_a_stencil) */    sequen[k1].s= copy_statement( stack_head(context->statement_stack));
}


static void wl_rwt(whileloop w, context_p context)
{  
  contenu_t contenu=is_a_while;
   hash_put(context->contenu,
	    stack_head(context->statement_stack), 
   (void *) contenu );
} 
  
static bool lexi_sup(Pmatrix a, Pmatrix b)
{int i;
 for (i=1;i<=depth;i++)
   {
     if (MATRIX_ELEM(a,i,1) >MATRIX_ELEM(b,i,1))
       return TRUE;
     if (MATRIX_ELEM(a,i,1) < MATRIX_ELEM(b,i,1))
       return FALSE;
    }
} 

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
 
static void compute_delay ()
{
  int i;
  for (i=k1-1;i>=0;i--)
    {
      sequen [i].delai=matrix_new(depth,1);
      if (i==k1-1)  matrix_nulle(sequen [i].delai);
      else
	matrix_substract(sequen [i].delai, sequen [i+1].delai ,sequen[i+1].st[sequen[i+1].nbr_stencil-1]  );
    }
  
}

static void compute_bound_merged_nest ()
{int i,j;
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
 
static statement  fusion()
{ int i,j;
 list lis;
 instruction ins;
 sequence seq;
  statement s;
  normalized norm;
  Pvecteur pv;
  loop ls;
  /* calcul des delais des differents nis */
  compute_delay();
  compute_bound_merged_nest ();
  /* on construit la liste des instructions qui vont former le corps du nid fusionne */
  normalize_all_expressions_of(expt);
  for (i=k1-1;i>=0;i--)
    {
      expression e1,e2,e,exp,gauche,droite,delai_plus;
      test t;
      call c;
      int m;
      Pvecteur pv;
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
      
      for(j=0;j<=depth-1;j++)
	{
	  if (j>=1) 
	    {
	      e1=ge_expression(entity_to_expression ((entity)sequen[i].nd[j].index),
			       Value_to_expression( value_plus(sequen[i].nd[j].lower,MATRIX_ELEM( sequen[i].delai,j+1,1))));
	      e2=le_expression(entity_to_expression ((entity)sequen[i].nd[j].index),
			       Value_to_expression( value_plus(sequen[i].nd[j].upper,MATRIX_ELEM( sequen[i].delai,j+1,1))));
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
	  ExpressionReplaceReference(gauche,
				     make_reference((entity*) sequen[i].nd[j].index,NIL),delai_plus);
	  MAP(EXPRESSION,exp,{  
	    ExpressionReplaceReference(exp,
				       make_reference((entity*) sequen[i].nd[j].index,NIL),delai_plus);
	  },call_arguments( syntax_call(expression_syntax(droite))));
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
      expression lower, upper;
      Pvecteur pv1,pv2;
      range1=loop_range(ls); 
      lower=range_lower(range1);  
      upper=range_upper(range1); 
      range_lower(range1)=make_integer_constant_expression(merged_nest[j].lower);
      range_upper(range1)= make_integer_constant_expression(merged_nest[j].upper);
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
      else loop_body(ls)=s;
    };
  s=loop_to_statement(loop1);
  return s;
} 

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

static Pvecteur buffer_acces(int nid )
{
  Pvecteur pv,tmp;
  int j;
  for(j=0;j<=depth-1;j++)
    {
      if (j==0) pv = vect_make(VECTEUR_NUL, sequen[nid].nd[j].index, 
			       MATRIX_ELEM(sequen[nid].coef,1,j+1), TCST,
			     value_uminus  (value_direct_multiply(MATRIX_ELEM(sequen[nid].coef,1,j+1),
								  value_plus( sequen[nid].nd[j].lower,
									      MATRIX_ELEM( sequen[nid].delai,j+1,1 ) ))));

   
      else 
	{
          tmp =pv;
	  pv = vect_make(tmp, sequen[nid].nd[j].index, 
	 	       MATRIX_ELEM(sequen[nid].coef,1,j+1),   TCST,
			      value_uminus(value_direct_multiply(MATRIX_ELEM(sequen[nid].coef,1,j+1), 
								 value_plus( sequen[nid].nd[j].lower,
									     MATRIX_ELEM( sequen[nid].delai,j+1,1 ) ))));

	}
     
    };

  return pv;
}

static entity make_array_entity(name, module_name, base,lis)
string name;
string module_name;
basic base;
list lis;
{
  string full_name;
  entity e, f, a;
  basic b = base;
 
 
  full_name =
    strdup(concatenate(module_name, MODULE_SEP_STRING, name, NULL));
  
  pips_debug(8, "name %s\n", full_name);
 
  message_assert("not already defined", 
		 gen_find_tabulated(full_name, entity_domain)==entity_undefined);
  
  e = make_entity(full_name, type_undefined, 
		  storage_undefined, value_undefined);
 
  entity_type(e) = (type) MakeTypeVariable(b, lis);
  f = local_name_to_top_level_entity(module_name);
  a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME); 
  
  entity_storage(e) = 
    make_storage(is_storage_ram,
		 make_ram(f, a,
			  (basic_tag(base)!=is_basic_overloaded)?
			  (add_variable_to_area(a, e)):(0),
			  NIL));

    
  entity_initial(e) = make_value(is_value_constant,
				 MakeConstantLitteral());
  
  return(e);
}

static entity make_new_array_variable_with_prefix(string prefix,
				     entity module,
				     basic b,list lis)
{
  string module_name = module_local_name(module);
  char buffer[20];
  entity e;
  int number = 0;
  bool empty_prefix = (strlen(prefix) == 0);
  
  /* let's assume positive int stored on 4 bytes */
  pips_assert("make_new_scalar_variable_with_prefix", strlen(prefix)<=10);

  do {
    if (empty_prefix) {
      switch(basic_tag(b)) {
      case is_basic_int:
	sprintf(buffer,"%s%d", DEFAULT_INT_PREFIX,
		unique_integer_number++);
	break;
      case is_basic_float:
	sprintf(buffer,"%s%d", DEFAULT_FLOAT_PREFIX, 
		unique_float_number++);
	break;
      case is_basic_logical:
	sprintf(buffer,"%s%d", DEFAULT_LOGICAL_PREFIX,
		unique_logical_number++);
	break;
      case is_basic_complex:
	sprintf(buffer,"%s%d", DEFAULT_COMPLEX_PREFIX,
		unique_complex_number++);
	break;
      case is_basic_string:
	sprintf(buffer, "%s%d", DEFAULT_STRING_PREFIX,
		unique_string_number++);
	break;
      default:
	pips_error("make_new_scalar_variable_with_prefix", 
		   "unknown basic tag: %d\n",
		   basic_tag(b));
	break;
      }
    }
    else {
      sprintf(buffer,"%s%d", prefix, number++);
      pips_assert ("make_new_scalar_variable_with_prefix",
		   strlen (buffer) < 19);
    }
  }
  while(gen_find_tabulated(concatenate(module_name,
				       MODULE_SEP_STRING,
				       buffer,
				       NULL),
			   entity_domain) != entity_undefined);
   
  pips_debug(9, "var %s, tag %d\n", buffer, basic_tag(b));
   
  e = make_array_entity(&buffer[0], module_name, b, lis);
  AddEntityToDeclarations(e, module);
   
  return e;
}

static entity make_new_array_variable(entity module,
                         basic b,list lis)
{
  
  return make_new_array_variable_with_prefix("B", module, b,lis);
}
static statement fusion_buffer()
{
  call c;
  int m;
  int i;
  int j;
  list lis,lis2,lis3,lisi;
  expression exp,gauche, droite; 
  entity name;
  Variable v1;
  reference ref;
   Pmatrix temp1,temp2,temp3;
   sequence seq;
   instruction ins;
   statement s;
   loop ls;
  compute_delay();
  compute_bound_merged_nest ();
   

  temp1=matrix_new(1,1);
  temp2=matrix_new(depth,1);
  temp3=matrix_new(depth,1);
  for(i=0;i<=k1-1;i++)
    {
      expression e1,e2,e,exp,gauche,droite;
      test t;
      statement s;
      if (i < k1-1){
      sequen[i].coef=matrix_new(1,depth);
      cons_coef(sequen[i].coef,i);
      matrix_substract(temp2,sequen[i+1].delai, sequen[i].delai);
      matrix_substract(temp3, temp2,sequen[i+1].st[0]);
      matrix_multiply(sequen[i].coef,temp3,temp1);
      sequen[i].surface=value_plus(MATRIX_ELEM(temp1,1,1),VALUE_ONE);
      sequen[i].pv_acces= buffer_acces(i);
      exp= binary_intrinsic_expression ("MOD",Pvecteur_to_expression(  sequen[i].pv_acces),
					Value_to_expression( sequen[i].surface) );
      lis=CONS(DIMENSION, make_dimension(int_to_expression(0),Value_to_expression(value_minus (sequen[i].surface,VALUE_ONE))), NIL);
      name= make_new_array_variable(get_current_module_entity() , make_basic(is_basic_int, 4), lis);
      
      lis =CONS(EXPRESSION, exp, NIL);
      ref=make_reference(name,lis);
      sequen[i].ref=name; }
     
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
	      
	      
	      MAP(EXPRESSION,exp,{  
		ExpressionReplaceReference(exp,
					   make_reference((entity*) sequen[i].nd[j].index,NIL),delai_plus);
		
	      },call_arguments( syntax_call(expression_syntax(droite))));
	      
	      
	      
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
      expression lower, upper;
      Pvecteur pv1,pv2;
      range1=loop_range(ls); 
      lower=range_lower(range1);  
      upper=range_upper(range1); 
      range_lower(range1)=
	make_integer_constant_expression(merged_nest[j].lower);
      range_upper(range1)=
       make_integer_constant_expression(merged_nest[j].upper);
      
      if  (j!=depth-1) ls=instruction_loop(statement_instruction(loop_body(ls)));
      else loop_body(ls)=s;
      
    };
  
 s=loop_to_statement(loop1);
  
 return s;

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
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],m+1,1), sequen[i].nd[m].lower)
		      < v1 )
		    {
		      overflow=TRUE ;
		      printf(" Debordement dans le   tableau: %s\n",entity_name(first_array));
		    };
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],m+1,1), sequen[i].nd[m].upper)> 
		      v2 )
		    {  
		      overflow=TRUE ;
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
		      overflow=TRUE ;
		      printf(" Debordement dans le domaine du tableau: %s\n",entity_name(sequen[i-1].tableau));
		    };
		  if (value_plus(MATRIX_ELEM(sequen[i].st[j],k+1,1), sequen[i].nd[k].upper)> 
		      sequen[i-1].nd[k].upper )
		    {
		      overflow=TRUE ;
		      printf(" Debordement dans le domaine du tableau: %s\n",entity_name(sequen[i-1].tableau));
		    };
		  
		};
       };
   };

 return overflow;
}
bool tiling_sequence(string module)  
{
  statement stat,s1;
  int i,j,k,debordement;
  context_t context;
 
  contenu_t contenu;
  list lis;
  instruction ins;
  sequence seq;
  
  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module);
  set_current_module_entity(local_name_to_top_level_entity(module));
  
 
  
  stat = (statement) db_get_memory_resource(DBR_CODE, module, TRUE); 
 
  
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
  depth = (int ) hash_get(context.depth, stat);
  
  
  if (contenu!=is_a_stencil)
    {
      printf(" Le programme ne repond pas aux hypotheses  \n");
      array_overflow();
    }
  else
    {
      if(!array_overflow())
	{
	  
	  //s1=fusion(); 
	  s1= fusion_buffer();
	  
	  module_reorder(s1); 
	  
	  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module, s1);
	} 
    }
      
    reset_current_module_entity();
    //  reset_current_module_statement();
  

  pips_debug(1, "done.\n");
  debug_off();
   
  return TRUE;
}
 


