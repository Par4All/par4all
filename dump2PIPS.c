#include "dump2PIPS.h"

/*
 *
 */
char * str2upper(char s[]){
	return strn2upper(s,strlen(s));
}
char * strn2upper(char s[], size_t n){
	while(n){
		s[n-1] = toupper(s[n-1]);
		n--;
	}
	return s;
}

//an enum to know what kind of main entity we are dealing with
typedef enum main_entity_type{PROG,SUB,FUNC,MOD} main_entity_type;
entity gfc2pips_main_entity = entity_undefined;
/*
 * Dump a namespace
 */
void gfc2pips_namespace(gfc_namespace* ns){
	gfc_symtree * current = NULL;
	gfc_symbol * root_sym;
	gfc_formal_arglist *formal;
	newgen_list args_list = NULL;
	//string prefix = string_undefined;

	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				x
	  type _entity_type_;				x
	  storage _entity_storage_;			x
	  value _entity_initial_;			x
	};*/
	instruction icf = instruction_undefined;


	debug(2, "gfc2pips_namespace", "Starting gfc 2 pips dumping\n");
	message_assert("No namespace to dump.",		ns);
	message_assert("No symtree root.",			ns->sym_root);
	message_assert("No symbol for the root.",	ns->sym_root->n.sym);


	current = getSymtreeByName( ns->proc_name->name, ns->sym_root );
	message_assert("No current symtree to match the name of the namespace",current);
	root_sym = current->n.sym;



	CurrentPackage = str2upper(strdup(ns->proc_name->name));
	/*
	 * We have to create a PIPS entity which is a program/function/other
	 * according to the type we have to create different values as parameters for the creation of the PIPS entity by:
	 * MakeCurrentFunction(type_undefined, TK_PROGRAM , CurrentPackage, NULL);
	 */
	main_entity_type bloc_token = -1;
	////type returned
	type bloc_type = make_type(is_type_void, UU);
	if( root_sym->attr.is_main_program ){
		debug(3, "gfc2pips_namespace", "main program founded %s\n",ns->proc_name->name);
		gfc2pips_main_entity = make_empty_program(
			str2upper((ns->proc_name->name))
		);
		message_assert("Main entity no created ! fuck you !",gfc2pips_main_entity!=entity_undefined);
		bloc_token=PROG;
	}else if( root_sym->attr.subroutine ){
		debug(3, "gfc2pips_namespace", "subroutine founded %s\n",ns->proc_name->name);
		gfc2pips_main_entity = make_empty_subroutine(
				str2upper((ns->proc_name->name))
		);
		bloc_token = SUB;
	}else if( root_sym->attr.function ){
		debug(3, "gfc2pips_namespace", "function founded %s\n",ns->proc_name->name);
		gfc2pips_main_entity = make_empty_function(
			str2upper(ns->proc_name->name),
			gfc2pips_symbol2type(root_sym)
		);
		bloc_token = FUNC;
	}else{
		debug(3, "gfc2pips_namespace", "not yet dumpable %s\n",ns->proc_name->name);
		if(root_sym->attr.procedure){
			fprintf(stdout,"procedure\n");
		}
		return;
		gfc2pips_main_entity = make_empty_blockdata(
			str2upper((ns->proc_name->name))
		);
	}
	message_assert("Main entity no created !",gfc2pips_main_entity!=entity_undefined);
	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				ok
	  type _entity_type_;				x
	  storage _entity_storage_;			x
	  value _entity_initial_;			x
	};*/

	debug(2, "gfc2pips_namespace", "main entity object initialized\n");

	////list of parameters
	debug(2, "gfc2pips_namespace", "dump the list of parameters\n");
	newgen_list parameters = NULL,parameters_p = NULL;
	if(bloc_token == FUNC || bloc_token == SUB){
		parameters = gfc2pips_args(ns);
	}
	debug(2, "gfc2pips_namespace", "List of parameters done: %s\n",parameters?"there is/are argument(s)":"none");

	/*if( bloc_token==FUNC ){
		switch(root_sym->ts.type){

		}
		//bloc_type =
	}*/

	////type of entity we are creating : a function (except maybe for module ?
	////a functional type is made of a list of parameters and the type returned
	entity_type(gfc2pips_main_entity) = make_type(is_type_functional, make_functional(parameters, bloc_type));

	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				ok
	  type _entity_type_;				ok
	  storage _entity_storage_;			x
	  value _entity_initial_;			x
	};*/



	//can it be removed ?
	//init_ghost_variable_entities();

	//can it be removed ?
	//BeginingOfProcedure();

	entity_storage( gfc2pips_main_entity ) = MakeStorageRom();
	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				ok
	  type _entity_type_;				ok
	  storage _entity_storage_;			ok
	  value _entity_initial_;			x
	};*/


	//can it be removed ?
	//common_size_map = hash_table_make(hash_pointer, 0);

	//// declare variables
	newgen_list variables_p,variables;
	variables_p = variables = gfc2pips_vars(ns);
	//the list of variables have to contains the args of the function/subroutine and those have to be of formal type
	/*int i=0;
	while(variables_p){
		if(gfc2pips_test_variable2(ns,getSymtreeByName(entity_name((entity)variables_p->car.e),ns))){
			entity_storage((entity)variables_p->car.e) = make_storage_formal(make_formal(gfc2pips_main_entity,i++));
		}
		POP(variables_p);
	}*/


	//we have to add the list of variables declared in the initial value of the main entity
	entity_initial(gfc2pips_main_entity) = make_value(
		is_value_code,
		make_code(
			variables,
			strdup(""),
			make_sequence(NIL),
			NULL
		)
	);
	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				ok
	  type _entity_type_;				ok
	  storage _entity_storage_;			ok
	  value _entity_initial_;			ok
	};*/
	set_current_module_entity(gfc2pips_main_entity);
	debug(2, "gfc2pips_namespace", "main entity creation finished\n");

	//can it be removed ?
	//InitAreas();//Syntax


	//// declare code
	debug(2, "gfc2pips_namespace", "dumping code ...\n");
	icf = gfc2pips_code2instruction(ns->code,true);
	message_assert("Dumping instruction failed\n",icf!=instruction_undefined);
	//we have got another problem with empty modules, we need an instruction for the last statement



	//can it be removed ?
	//PushBlock(icf, "INITIAL");//Syntax
	/*gfc_function_body = make_statement(
		entity_empty_label(),
		//ceci merde une fois les character implémentés
		STATEMENT_NUMBER_UNDEFINED,//lien number
		STATEMENT_ORDERING_UNDEFINED,
		empty_comments,
		icf,
		NULL,//variables ? pour fortran2C ?
		NULL,
		empty_extensions()
	);*/
	gfc_function_body = make_stmt_of_instr(icf);



	//we automatically add a return statement
	//we have got a problem with multiple return in the function
	insure_return_as_last_statement(gfc2pips_main_entity,&gfc_function_body);



	debug(2, "gfc2pips_namespace", "dumping done\n");

}










/*
 * Retrieve the names of every argument of the function, if exists
 */
newgen_list gfc2pips_args(gfc_namespace* ns){
	gfc_symtree * current = NULL;
	gfc_formal_arglist *formal;
	newgen_list args_list = NULL,args_list_p = NULL;

	if( ns && ns->proc_name ){

		current = getSymtreeByName( ns->proc_name->name, ns->sym_root );
		if( current && current->n.sym ){
			if (current->n.sym->formal){
				formal = current->n.sym->formal;
				if(formal){
					args_list = args_list_p = CONS(ENTITY,
						gfc2pips_symbol2entity(
							getSymtreeByName(
								str2upper(formal->sym->name),
								ns->sym_root
							)->n.sym
						),
						NULL
					);
					formal = formal->next;
					while(formal){
						if (formal->sym != NULL){
							CDR(args_list) = CONS(ENTITY,
								gfc2pips_symbol2entity(
									getSymtreeByName(
										str2upper(formal->sym->name),
										ns->sym_root
									)->n.sym
								),
								NULL
							);
							args_list = CDR(args_list);
						}
						formal = formal->next;
					}
				}
			}
		}
	}
	return args_list_p;
}

/*
 * Look for a specific symbol in a tree
 */
gfc_symtree* getSymtreeByName (char* name, gfc_symtree *st){
  gfc_symtree *return_value = NULL;
  if(!name) return NULL;

  if(!st) return NULL;
  if(!st->n.sym) return NULL;
  if(!st->name) return NULL;

  debug(9,"getSymtreeByName","Lookin for the symtree called: %s(%d) %s(%d)\n", name, strlen(name), st->name, strlen(st->name) );

  if(
		strcmp(str2upper(st->name),str2upper(name))==0
		/*&&(
				st->n.sym->attr.subroutine
			|| st->n.sym->attr.function
		)*/
  ){
	  debug(9,"getSymtreeByName","symbol %s founded\n",name);
	  return st;
  }
  return_value = getSymtreeByName (name, st->left  );

  if( return_value != NULL) return return_value;
  return_value = getSymtreeByName (name, st->right  );
  if(return_value != NULL) return return_value;

  //fprintf(stdout,"NULL\n");
  return NULL;
}

/*
 * Extract every and each variable from a namespace
 */
newgen_list gfc2pips_vars(gfc_namespace *ns){
	if(ns){
		return gfc2pips_vars_(ns,gen_nreverse(getSymbolBy(ns,ns->sym_root, gfc2pips_test_variable)));
	}
	return NULL;
}

/*
 * Convert the list of gfc symbol into a list of pips entities
 */
newgen_list gfc2pips_vars_(gfc_namespace *ns,newgen_list variables_p){
	newgen_list variables = NULL;
	//variables_p = gen_nreverse(getSymbolBy(ns,ns->sym_root, gfc2pips_test_variable));
	variables = variables_p;//balancer la suite dans une fonction à part afin de pouvoir la réutiliser pour les calls
	newgen_list arguments,arguments_p;
	arguments = arguments_p = gfc2pips_args(ns);
	while(variables_p){
		type Type = type_undefined;
		//create entities here
		gfc_symtree *current_symtree = (gfc_symtree*)variables_p->car.e ;
		if(current_symtree && current_symtree->n.sym){
			debug(3, "gfc2pips_vars", "translation of entity gfc2pips start\n");
			debug(4, "gfc2pips_vars", " %s %d\r\n", str2upper(current_symtree->name), current_symtree->n.sym->ts.kind );
			int TypeSize = gfc2pips_symbol2size(current_symtree->n.sym);
			value Value;// = MakeValueUnknown();
			Type = gfc2pips_symbol2type(current_symtree->n.sym);
			debug(3, "gfc2pips_vars", "Type done\n");


			//handle the value
			//don't ask why it is is_value_constant
			if(
				Type!=type_undefined
				&& current_symtree->n.sym->ts.type==BT_CHARACTER
			){
				Value = make_value_constant(
					MakeConstantLitteral()//MakeConstant(current_symtree->n.sym->value->value.character.string,is_basic_string)
				);
			}else{
				Value = make_value_constant(
					make_constant(is_constant_int,(void *) TypeSize)
				);
			}

			/*declaration: entity_name decl_tableau lg_fortran_type
			basic b;
			type t = CurrentType;
			b = variable_basic(type_variable(CurrentType));

			if(basic_string_p(b)){
				t = value_intrinsic_p($3)?
					copy_type(t):
					MakeTypeVariable(make_basic(is_basic_string, $3), NIL);
			}

			DeclareVariable($1, t, $2,storage_undefined, value_undefined);//DeclareVariable(entity e, type t, list d, storage s, value v);
			*/

			newgen_list list_of_dimensions = NULL;
			if(current_symtree->n.sym->attr.dimension){
				gfc_array_spec *as = current_symtree->n.sym->as;
				const char *c;
				if ( as!=NULL && as->rank != 0){
					int i,j;
					/*fprintf(stdout,"toto %d\n",as);
					fprintf(stdout,"toto %d\n",as->upper);
					fprintf(stdout,"toto %d\n",as->upper[i]);
					fprintf(stdout,"toto %d\n",as->upper[i]->value);
					fprintf(stdout,"toto %d\n",as->upper[i]->value.integer);
					fprintf(stdout,"toto %d\n",as->upper[i]->value.integer[0]);
					fprintf(stdout,"toto %d\n",as->upper[i]->value.integer[0]._mp_size);
					fprintf(stdout,"toto %d\n",as->upper[i]->value.integer[0]._mp_alloc);
					fprintf(stdout,"toto %d\n",as->upper[i]->value.integer[0]._mp_d);
					fprintf(stdout,"\n");
					*/
					/*
					 * show_expr (as->upper[i]);
					 * mpz_out_str (stdout, 10, p->value.integer);
					 * if (p->ts.kind != gfc_default_integer_kind)
					 *   fprintf (dumpfile, "_%d", p->ts.kind);
					 */

					//le switch nous intéresse-t-il ? => oui très important il faut traduire chaque valeur de dimension du tableau en "truc" approprié newgen/PIPS
					switch (as->type){
						case AS_EXPLICIT:
						c = "AS_EXPLICIT";
						//create the list of dimensions
						for (i=as->rank-1,j = 0; i >= j; i--){
							list_of_dimensions = gen_int_cons(
									gfc2pips_int2dimension(*as->upper[i]->value.integer[0]._mp_d),
										list_of_dimensions
									);
						}
						break;
						case AS_DEFERRED:      c = "AS_DEFERRED";      break;
						case AS_ASSUMED_SIZE:  c = "AS_ASSUMED_SIZE";  break;
						case AS_ASSUMED_SHAPE: c = "AS_ASSUMED_SHAPE"; break;
						default:
						  gfc_internal_error ("show_array_spec(): Unhandled array shape "
									  "type.");
					}
				}
			}
			int i,j;
			i=0;j=1;
			arguments_p = arguments;
			while(arguments_p){
				fprintf(stdout,"%s %s\n",entity_local_name((entity)arguments_p->car.e),current_symtree->name);
				if(strcmp(str2upper(entity_local_name((entity)arguments_p->car.e)),str2upper(current_symtree->name))==0){
					i=j;
					break;
				}
				j++;
				POP(arguments_p);
			}


			variables_p->car.e = FindOrCreateEntity(CurrentPackage, current_symtree->name);
			entity_type((entity)variables_p->car.e) = Type;
			entity_storage((entity)variables_p->car.e) = current_symtree->n.sym->attr.dummy ?
				make_storage_formal(
					make_formal(
						gfc2pips_main_entity,
						i
					)
				) : MakeStorageRom() ;
			entity_initial((entity)variables_p->car.e) = Value;//make_value(is_value_code, make_code(NULL, strdup(""), make_sequence(NIL),NIL));
			if(Type!=type_undefined)variable_dimensions(type_variable(entity_type((entity)variables_p->car.e))) = list_of_dimensions;
			debug(3, "gfc2pips_vars", "translation of entity gfc2pips end\n");
		}else{
			variables_p->car.e = NULL;
		}
		POP(variables_p);
	}
	return variables;
}


/*
 * Look for a set of symbol filtered by a predicate function
 */
newgen_list getSymbolBy(gfc_namespace* ns, gfc_symtree *st, bool (*func)(gfc_namespace*, gfc_symtree *)){
  newgen_list args_list = NULL;

  if(!ns) return NULL;
  if(!st) return NULL;
  if(!func) return NULL;

  if( func(ns,st)){
    args_list = gen_cons(st,args_list);
  }
  args_list = gen_nconc(args_list, getSymbolBy (ns, st->left, func) );
  args_list = gen_nconc(args_list, getSymbolBy (ns, st->right, func) );

  return args_list;
}

/*
 * Predicate functions
 */
bool gfc2pips_test_variable(gfc_namespace* ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE;// && !st->n.sym->attr.dummy;
}
bool gfc2pips_test_variable2(gfc_namespace* ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && !st->n.sym->attr.dummy;
}
bool gfc2pips_test_arg(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && st->n.sym->attr.dummy;
}

entity gfc2pips_symbol2entity(gfc_symbol* s){
	entity e = FindOrCreateEntity(CurrentPackage,s->name);
	if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
	if(entity_type(e)==type_undefined) entity_type(e) = gfc2pips_symbol2type(s);
	//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
	return e;
}









/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */

dimension gfc2pips_int2dimension(int n){
	return make_dimension(MakeIntegerConstantExpression("1"),gfc2pips_int2expression(n));
}

/*
 * expression = sous_expression:
   atom	{
			$$ = make_expression($1, normalized_undefined);
		}
 | unsigned_const_simple
    {
	    $$ = MakeNullaryCall($1);
    }
 | signe expression  %prec TK_STAR
    {
	    if ($1 == -1)
		$$ = MakeFortranUnaryCall(CreateIntrinsic("--"), $2);
	    else
		$$ = $2;
    }
 | expression TK_PLUS expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("+"), $1, $3);
    }
 | expression TK_MINUS expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("-"), $1, $3);
    }
 | expression TK_STAR expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("*"), $1, $3);
    }
 | expression TK_SLASH expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("/"), $1, $3);
    }
 | expression TK_POWER expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("**"),$1, $3);
    }
 | expression oper_rela expression  %prec TK_EQ
    {
	    $$ = MakeFortranBinaryCall($2, $1, $3);
    }
 | expression TK_EQV expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic(".EQV."),$1, $3);
    }
 | expression TK_NEQV expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic(".NEQV."),$1, $3);
    }
 | expression TK_OR expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic(".OR."),
				       fix_if_condition($1),
				       fix_if_condition($3));
    }
 | expression TK_AND expression
    {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic(".AND."),
				       fix_if_condition($1),
				       fix_if_condition($3));
    }
 | TK_NOT expression
    {
	    $$ = MakeFortranUnaryCall(CreateIntrinsic(".NOT."),
				      fix_if_condition($2));
    }
 | expression TK_CONCAT expression
        {
	    $$ = MakeFortranBinaryCall(CreateIntrinsic("//"),
				      $1, $3);
    }
 *
 */
expression gfc2pips_int2expression(int n){
	return MakeNullaryCall(gfc2pips_int_const2entity(n));
}
expression gfc2pips_real2expression(double r){
	return MakeNullaryCall(gfc2pips_real2entity(r));
}
expression gfc2pips_logical2expression(bool b){
	return MakeNullaryCall(gfc2pips_logical2entity(b));
}


/*
 * If a number is used, such as 3, 3.14 in the program, it'll be translated into an entity
 */
entity gfc2pips_int_const2entity(int n){
	char str[30];
	sprintf(str,"%d",n);
	return MakeConstant(str, is_basic_int);
}
entity gfc2pips_int2label(int n){
	char str[60];
	sprintf(str,"%s%s%s%d",TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);//fprintf(stdout,"new label: %s %s %s %s %d\n",str,TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);
	return make_label(str);
}
//on a un grave problème de traduction des valeurs réelles
//code  => réel en valeur interne
//16.53 => 16.530001
//16.56 => 16.559999
entity gfc2pips_real2entity(double r){
	char str[30];
	sprintf(str,"%f",r);//fprintf(stdout,"copy of the entity name(real) %s\n",str);
	return MakeConstant(str, is_basic_float);
}

entity gfc2pips_logical2entity(bool b){
	return MakeConstant(b?"1":"0",is_basic_logical);
}



/*
 * enum basic_utype {
  is_basic_int,
  is_basic_float,
  is_basic_logical,
  is_basic_overloaded,
  is_basic_complex,
  is_basic_string,
  is_basic_bit,
  is_basic_pointer,
  is_basic_derived,
  is_basic_typedef
};
 */
type gfc2pips_symbol2type(gfc_symbol *s){
	//beware the size of strings
	//type t = gfc2pips_type2type(s->ts.type,s->ts.kind);//make_value(is_value_constant, make_constant(is_constant_int, (void *) CurrentTypeSize))

	enum basic_utype ut;
	switch(s->ts.type){
		case BT_INTEGER:	ut = is_basic_int;		break;
		case BT_REAL:		ut = is_basic_float;	break;
		case BT_COMPLEX:	ut = is_basic_complex;	break;
		case BT_LOGICAL:	ut = is_basic_logical;	break;
		case BT_CHARACTER:	ut = is_basic_string;	break;
		case BT_UNKNOWN:
		case BT_DERIVED:
		case BT_PROCEDURE:
		case BT_HOLLERITH:
		case BT_VOID:
		default:
			pips_error("gfc2pips_symbol2type","An error occured in the type to type translation: impossible to translate the symbol.\n");
			return type_undefined;
	}
	debug(5,"gfc2pips_type2type","Basic type is : %d\n",ut);
	if(ut!=is_basic_string){
		return MakeTypeVariable(
			make_basic(
				ut,
				(void*) gfc2pips_symbol2size(s)
			),
			NIL
		);
	}else{
		//fprintf(stdout,"STRING !!! %d\n",s->value);
		//il faut créer impérativement une "value" contenant un "constant" qui contient un "litteral"
		return MakeTypeVariable(
			make_basic(
				ut,
				(void*) make_value(
					is_value_constant,
					//don't use litteral, it's a trap !
					make_constant_int(gfc2pips_symbol2size(s))//it is here we have to specify the length of the character symbol
				)
			),
			NIL
		);
	}
	debug( 5, "gfc2pips_type2type", "WARNING: no type\n" );
	return type_undefined;
}
int gfc2pips_symbol2size(gfc_symbol *s){
	if(s->ts.type==BT_CHARACTER){
		return mpz_get_ui(s->ts.cl->length->value.integer);
	}else{
		return s->ts.kind;
	}
}

//only for AR_ARRAY references
newgen_list gfc2pips_array_ref2indices(gfc_array_ref *ar){
	int i;
	newgen_list indices=NULL,indices_p=NULL;

	if(!ar->start[0])return NULL;
	indices_p = CONS( EXPRESSION, gfc2pips_expr2expression(ar->start[0]), NIL );
	indices=indices_p;
	for( i=1 ; ar->start[i] ;i++){
		CDR(indices_p) = CONS( EXPRESSION, gfc2pips_expr2expression(ar->start[i]), NIL );
		indices_p = CDR(indices_p);
	}
	return indices;
	/*
	switch (ar->type)
	{
		case AR_FULL:
			fputs ("FULL", dumpfile);
		break;

		case AR_SECTION:
			for (i = 0; i < ar->dimen; i++)
			{
				There are two types of array sections: either the
				elements are identified by an integer array ('vector'),
				or by an index range. In the former case we only have to
				print the start expression which contains the vector, in
				the latter case we have to print any of lower and upper
				bound and the stride, if they're present.

				if (ar->start[i] != NULL)
					show_expr (ar->start[i]);

				if (ar->dimen_type[i] == DIMEN_RANGE)
				{
					fputc (':', dumpfile);

					if (ar->end[i] != NULL)
						show_expr (ar->end[i]);

					if (ar->stride[i] != NULL)
					{
						fputc (':', dumpfile);
						show_expr (ar->stride[i]);
					}
				}

				if (i != ar->dimen - 1)
					fputs (" , ", dumpfile);
			}
		break;

		case AR_ELEMENT:
			for (i = 0; i < ar->dimen; i++)
			{
				show_expr (ar->start[i]);
				if (i != ar->dimen - 1)
					fputs (" , ", dumpfile);
			}
		break;

		case AR_UNKNOWN:
			fputs ("UNKNOWN", dumpfile);
		break;

		default:
			gfc_internal_error ("show_array_ref(): Unknown array reference");
	}

*/
	return NULL;
}

/*
 * Declaration of instructions
 *
instruction:
	[...]
        | { check_in_declarations();} inst_spec
	| { check_first_statement();} inst_exec
	    {
		if ($2 != instruction_undefined)
			LinkInstToCurrentBlock($2, TRUE);
	    }
	|  format_inst
	    {
		LinkInstToCurrentBlock($1, TRUE);
	    }
	;

inst_spec: parameter_inst
	| implicit_inst
	| dimension_inst
	| pointer_inst
	| equivalence_inst
	| common_inst {}
	| type_inst
	| external_inst
	| intrinsic_inst
	| save_inst
	;

inst_exec: assignment_inst
	    { $$ = $1; }
	| goto_inst
	    { $$ = $1; }
	| arithmif_inst
	    { $$ = $1; }
	| logicalif_inst
	    { $$ = $1; }
	| blockif_inst
	    { $$ = $1; }
	| elseif_inst
	    { $$ = $1; }
	| else_inst
	    { $$ = instruction_undefined; }
	| endif_inst
	    { $$ = instruction_undefined; }
	| enddo_inst
	    { $$ = instruction_undefined; }
	| do_inst
	    { $$ = instruction_undefined; }
	| bdo_inst
	    { $$ = instruction_undefined; }
	| wdo_inst
	    { $$ = instruction_undefined; }
	| continue_inst
	    { $$ = $1; }
	| stop_inst
	    { $$ = $1; }
	| pause_inst
	    { $$ = $1; }
	| call_inst
	    { $$ = $1; }
	| return_inst
	    { $$ = $1; }
	| io_inst
	    { $$ = $1; }
        | entry_inst
            { $$ = $1; }
	;
 */


/*
 * Assignement of a value, simple case
 * assignment_inst: TK_ASSIGN icon TK_TO atom
            {
                if(get_bool_property("PARSER_SUBSTITUTE_ASSIGNED_GOTO")) {
		    expression e = entity_to_expression($2);
		    $$ = MakeAssignInst($4, e);
		}
		else {
		    ParserError("parser", "ASSIGN statement prohibited by PIPS"
				" unless property PARSER_SUBSTITUTE_ASSIGNED_GOTO is set\n");
		}
	    }
	| atom TK_EQUALS expression
	    {
		syntax s = $1;
		syntax new_s = syntax_undefined;

	        if(expression_implied_do_p($3))
		  ParserError("Syntax", "Unexpected implied DO\n");

		new_s = CheckLeftHandSide(s);

		$$ = MakeAssignInst(new_s, $3);
	    }
	;
 */

/*
 * Build an instruction sequence
 */
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence){
	if(!c){
		if(force_sequence){
			fprintf(stdout,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
			return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
		}else{
			fprintf(stdout,"Undefined code\n");
			return make_instruction_block(NULL);
		}
	}
	//No block, only one instruction
	if(!c->next && !force_sequence )return gfc2pips_code2instruction_(c);

	//create a sequence and put everything into it ? is it right ?
	newgen_list list_of_instructions,list_of_instructions_p;
	list_of_instructions= NULL;

	//entity l = gfc2pips_code2get_label(c);
	list_of_instructions_p = list_of_instructions = CONS(STATEMENT, make_statement(
		gfc2pips_code2get_label(c),
		STATEMENT_NUMBER_UNDEFINED,
		STATEMENT_ORDERING_UNDEFINED,
		empty_comments,
		gfc2pips_code2instruction_(c),
		NIL,
		NULL,
		empty_extensions ()
	), list_of_instructions);
	//statement_label((statement)list_of_instructions->car.e) = gfc2pips_code2get_label(c);

	for( c=c->next ; c ; c=c->next ){
		statement s = statement_undefined;
		//l = gfc2pips_code2get_label(c);
		//on lie l'instruction suivante à la courante
		//fprintf(stdout,"Dump the following instructions\n");
		//CONS(STATEMENT,instruction_to_statement(gfc2pips_code2instruction_(c)),list_of_instructions);
		s = make_statement(
			gfc2pips_code2get_label(c),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			gfc2pips_code2instruction_(c),
			NULL,
			NULL,
			empty_extensions ()
		);
		CDR(list_of_instructions) = CONS(STATEMENT, s, NIL);
		list_of_instructions = CDR(list_of_instructions);
		//statement_label((statement)list_of_instructions->car.e) = gfc2pips_code2get_label(c);
		/*
		 * if we have got a label like a
		 * ----------------------
		 * do LABEL while (expr)
		 *    statement
		 * LABEL continue
		 * ----------------------
		 * we do need to make a continue statement BUT this will crash the program in some cases
		 * PARSED_PRINTED_FILE is okay, but not PRINTED_FILE so we have to find out why
		 */
		/*if( c->label && (c->op==EXEC_DO || c->op==EXEC_DO_WHILE) ){
			s = make_statement(
					gfc2pips_code2get_label2(c),
					c->here?c->here->where.lb->file->inclusion_line:0,
					STATEMENT_ORDERING_UNDEFINED,
					empty_comments,
					make_instruction(is_instruction_call, make_call(CreateIntrinsic("CONTINUE"), NULL)),
					NULL,
					NULL,
					empty_extensions ()
				);
			CDR(list_of_instructions) = CONS(STATEMENT, s, NIL);
			list_of_instructions = CDR(list_of_instructions);
		}*/

	}
	//CONS(STATEMENT, make_return_statement(FindOrCreateEntity(CurrentPackage,"TEST")),list_of_instructions);
	return make_instruction_block(list_of_instructions_p);//make a sequence <=> make_instruction_sequence(make_sequence(list_of_instructions));
}
//never call this function except in gfc2pips_code2instruction or in recursive mode
instruction gfc2pips_code2instruction_(gfc_code* c){
	//do we have a label ?
	//if(c->here){}
	//debug(5,"gfc2pips_code2instruction","Start function\n");
	switch (c->op){
		/*case EXEC_NOP:
	      fputs ("NOP", dumpfile);
	      break;
*/
		case EXEC_CONTINUE:
			return make_instruction(is_instruction_call, make_call(CreateIntrinsic("CONTINUE"), NULL));
		break;
/*	    case EXEC_ENTRY:
	      fprintf (dumpfile, "ENTRY %s", c->ext.entry->sym->name);
	      break;
*/
		case EXEC_INIT_ASSIGN:
		case EXEC_ASSIGN:{
			debug(5,"gfc2pips_code2instruction","Translation of assign\n");
			return make_assign_instruction(
				gfc2pips_expr2expression(c->expr),
				gfc2pips_expr2expression(c->expr2)
			);
		}
		break;

		/*case EXEC_LABEL_ASSIGN:
	      fputs ("LABEL ASSIGN ", dumpfile);
	      show_expr (c->expr);
	      fprintf (dumpfile, " %d", c->label->value);
	      break;

	    case EXEC_POINTER_ASSIGN:
	      fputs ("POINTER ASSIGN ", dumpfile);
	      show_expr (c->expr);
	      fputc (' ', dumpfile);
	      show_expr (c->expr2);
	      break;
*/
	    case EXEC_GOTO:{
	    	instruction i = make_instruction(is_instruction_goto,
	    			make_continue_statement(
	    				gfc2pips_code2get_label2(c)
	    			)
	    	);
	    	return i;
/*	    	statement s = make_statement(
	    		gfc2pips_code2get_label2(c),
	    		c->here?c->here->where.lb->file->inclusion_line:0,
	    		STATEMENT_ORDERING_UNDEFINED,
				empty_comments,
				gfc2pips_code2instruction_(c),
				NULL,
				NULL,
				empty_extensions ()
			);
			if (c->label){
				fprintf (dumpfile, "%d", c->label->value);
			}else{
				show_expr (c->expr);
				d = c->block;//on a qqch de plus d'attendu, qu'est-ce ?
				if (d != NULL){
					fputs (", (", dumpfile);
					for (; d; d = d ->block){
						code_indent (level, d->label);
						if (d->block != NULL)
							fputc (',', dumpfile);
						else
							fputc (')', dumpfile);
					}
				}
			}

*/	    }break;

/*	    case EXEC_CALL:
	    case EXEC_ASSIGN_CALL:
	      if (c->resolved_sym)
		fprintf (dumpfile, "CALL %s ", c->resolved_sym->name);
	      else if (c->symtree)
		fprintf (dumpfile, "CALL %s ", c->symtree->name);
	      else
		fputs ("CALL ?? ", dumpfile);

	      show_actual_arglist (c->ext.actual);
	      break;

	    case EXEC_COMPCALL:
	      fputs ("CALL ", dumpfile);
	      show_compcall (c->expr);
	      break;
*/
		case EXEC_RETURN:{
			debug(5,"gfc2pips_code2instruction","Translation of return\n");
			expression e = expression_undefined;
			if(c->expr!=NULL){
				//traitement de la variable de retour
			}
			return MakeReturn(e);
		}

	    /*case EXEC_PAUSE:
	      fputs ("PAUSE ", dumpfile);

	      if (c->expr != NULL)
		show_expr (c->expr);
	      else
		fprintf (dumpfile, "%d", c->ext.stop_code);
30
	      break;

	    case EXEC_STOP:
	      fputs ("STOP ", dumpfile);

	      if (c->expr != NULL)
		show_expr (c->expr);
	      else
		fprintf (dumpfile, "%d", c->ext.stop_code);

	      break;

	    case EXEC_ARITHMETIC_IF:
	      fputs ("IF ", dumpfile);
	      show_expr (c->expr);
	      fprintf (dumpfile, " %d, %d, %d",
			  c->label->value, c->label2->value, c->label3->value);
	      break;

*/
		case EXEC_IF:{
	    	debug(5,"gfc2pips_code2instruction","Translation of IF\n");
	    	if(!c->block)return make_instruction_block(NULL);
			gfc_code* d=c->block;//fprintf(stdout,"%d %d %d %d\n",d, d?d->expr:d,c,c->expr);
			//next est le code pointé par la sortie IF
			//block est le code pointé par la sortie ELSE

			if(!d->expr){
	    		fprintf(stdout,"No condition ???\n");
	    		//we are at the last ELSE statement for an ELSE IF
	    		if(d->next){
	    			return gfc2pips_code2instruction(d->next,false);
	    		}else{
	    			return make_instruction_block(NULL);
	    		}
	    	}
	    	expression e = gfc2pips_expr2expression(d->expr);
	    	//transformer ce petit bout en une boucle ou une fonction récursive pour dump les elsif
	    	if(e==expression_undefined){
	    		fprintf(stderr,"An error occured: impossible to translate a IF expression condition\n");
	    		return make_instruction_block(NULL);
	    	}
	    	statement s_if=statement_undefined;
	    	statement s_else=statement_undefined;
	    	//IF
	    	if(d->next){
	    		gfc2pips_code2get_label(c);gfc2pips_code2get_label(d);gfc2pips_code2get_label(d->next);
	    		s_if = instruction_to_statement(gfc2pips_code2instruction(d->next,false));
	    		statement_label(s_if) = gfc2pips_code2get_label(d->next);
	    		//ELSE + ?
	    		if(d->block){
		    		//s_else = instruction_to_statement(gfc2pips_code2instruction(d->block,false));
	    			//ELSE IF
		    		if(d->block->expr){
		    			fprintf(stdout,"d->block->expr %d\n",d->block->expr);
		    			s_else = instruction_to_statement(gfc2pips_code2instruction_(d));
			    		statement_label(s_else) = gfc2pips_code2get_label(d);
		    		//ELSE
		    		}else{
		    			fprintf(stdout,"d->block %d\n",d->block);
		    			s_else = instruction_to_statement(gfc2pips_code2instruction(d->block->next,false));//no condition therefore we are in the last ELSE statement
			    		statement_label(s_else) = gfc2pips_code2get_label(d->block->next);
		    		}
	    		}
    		}else{
    			return make_instruction_block(NULL);
    		}
	    	if( s_if==statement_undefined || statement_instruction(s_if)==instruction_undefined ){
	    		 s_if = make_empty_block_statement();
	    	}
	    	if( s_else==statement_undefined || statement_instruction(s_else)==instruction_undefined ){
	    		 s_else = make_empty_block_statement();
	    	}
	    	return test_to_instruction(
				make_test(
					e,
					s_if,
					s_else
				)
			);

			//IF ( expr ){ next } block
			//block= ELSE IF ( expr ){ next } block
	    	//block = ELSE { next }
		}break;

	    /*case EXEC_SELECT:
	      d = c->block;
	      fputs ("SELECT CASE ", dumpfile);
	      show_expr (c->expr);
	      fputc ('\n', dumpfile);

	      for (; d; d = d->block)
		{
		  code_indent (level, 0);

		  fputs ("CASE ", dumpfile);
		  for (cp = d->ext.case_list; cp; cp = cp->next)
		    {
		      fputc ('(', dumpfile);
		      show_expr (cp->low);
		      fputc (' ', dumpfile);
		      show_expr (cp->high);
		      fputc (')', dumpfile);
		      fputc (' ', dumpfile);
		    }
		  fputc ('\n', dumpfile);

		  show_code (level + 1, d->next);
		}

	      code_indent (level, c->label);
	      fputs ("END SELECT", dumpfile);
	      break;

	    case EXEC_WHERE:
	      fputs ("WHERE ", dumpfile);

	      d = c->block;
	      show_expr (d->expr);
	      fputc ('\n', dumpfile);

	      show_code (level + 1, d->next);

	      for (d = d->block; d; d = d->block)
		{
		  code_indent (level, 0);
		  fputs ("ELSE WHERE ", dumpfile);
		  show_expr (d->expr);
		  fputc ('\n', dumpfile);
		  show_code (level + 1, d->next);
		}

	      code_indent (level, 0);
	      fputs ("END WHERE", dumpfile);
	      break;


	    case EXEC_FORALL:
	      fputs ("FORALL ", dumpfile);
	      for (fa = c->ext.forall_iterator; fa; fa = fa->next)
		{
		  show_expr (fa->var);
		  fputc (' ', dumpfile);
		  show_expr (fa->start);
		  fputc (':', dumpfile);
		  show_expr (fa->end);
		  fputc (':', dumpfile);
		  show_expr (fa->stride);

		  if (fa->next != NULL)
		    fputc (',', dumpfile);
		}

	      if (c->expr != NULL)
		{
		  fputc (',', dumpfile);
		  show_expr (c->expr);
		}
	      fputc ('\n', dumpfile);

	      show_code (level + 1, c->block->next);

	      code_indent (level, 0);
	      fputs ("END FORALL", dumpfile);
	      break;
*/
	    case EXEC_DO:{
	    	debug(5,"gfc2pips_code2instruction","Translation of DO\n");
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,false));
	    	statement_label(s)=gfc2pips_code2get_label(c->block->next);
	    	loop w = make_loop(
	    		gfc2pips_expr2entity(c->ext.iterator->var),//variable incremented
	    		make_range(
    				gfc2pips_expr2expression(c->ext.iterator->start),
    				gfc2pips_expr2expression(c->ext.iterator->end),
    				gfc2pips_expr2expression(c->ext.iterator->step)
	    		),//lower, upper, increment
	    		s,
	    		gfc2pips_code2get_label2(c),
	    		make_execution_sequential(),//sequential/parallel //beware gfc parameters to say it is a parallel or a sequential loop
	    		NULL
	    	);
	    	return make_instruction_loop(w);

	    }break;

	    case EXEC_DO_WHILE:{
	    	debug(5,"gfc2pips_code2instruction","Translation of DO WHILE\n");
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,false));
	    	statement_label(s) = gfc2pips_code2get_label(c->block->next);
	    	whileloop w = make_whileloop(
	    		gfc2pips_expr2expression(c->expr),
	    		s,
	    		gfc2pips_code2get_label2(c),
	    		make_evaluation_before()
	    	);
	    	return make_instruction_whileloop(w);
	    }break;

/*	    case EXEC_CYCLE:
	      fputs ("CYCLE", dumpfile);
	      if (c->symtree)
		fprintf (dumpfile, " %s", c->symtree->n.sym->name);
	      break;

	    case EXEC_EXIT:
	      fputs ("EXIT", dumpfile);
	      if (c->symtree)
		fprintf (dumpfile, " %s", c->symtree->n.sym->name);
	      break;

	    case EXEC_ALLOCATE:
	      fputs ("ALLOCATE ", dumpfile);
	      if (c->expr)
		{
		  fputs (" STAT=", dumpfile);
		  show_expr (c->expr);
		}

	      for (a = c->ext.alloc_list; a; a = a->next)
		{
		  fputc (' ', dumpfile);
		  show_expr (a->expr);
		}

	      break;

	    case EXEC_DEALLOCATE:
	      fputs ("DEALLOCATE ", dumpfile);
	      if (c->expr)
		{
		  fputs (" STAT=", dumpfile);
		  show_expr (c->expr);
		}

	      for (a = c->ext.alloc_list; a; a = a->next)
		{
		  fputc (' ', dumpfile);
		  show_expr (a->expr);
		}

	      break;

	    case EXEC_OPEN:
	      fputs ("OPEN", dumpfile);
	      open = c->ext.open;

	      if (open->unit)
		{
		  fputs (" UNIT=", dumpfile);
		  show_expr (open->unit);
		}
	      if (open->iomsg)
		{
		  fputs (" IOMSG=", dumpfile);
		  show_expr (open->iomsg);
		}
	      if (open->iostat)
		{
		  fputs (" IOSTAT=", dumpfile);
		  show_expr (open->iostat);
		}
	      if (open->file)
		{
		  fputs (" FILE=", dumpfile);
		  show_expr (open->file);
		}
	      if (open->status)
		{
		  fputs (" STATUS=", dumpfile);
		  show_expr (open->status);
		}
	      if (open->access)
		{
		  fputs (" ACCESS=", dumpfile);
		  show_expr (open->access);
		}
	      if (open->form)
		{
		  fputs (" FORM=", dumpfile);
		  show_expr (open->form);
		}
	      if (open->recl)
		{
		  fputs (" RECL=", dumpfile);
		  show_expr (open->recl);
		}
	      if (open->blank)
		{
		  fputs (" BLANK=", dumpfile);
		  show_expr (open->blank);
		}
	      if (open->position)
		{
		  fputs (" POSITION=", dumpfile);
		  show_expr (open->position);
		}
	      if (open->action)
		{
		  fputs (" ACTION=", dumpfile);
		  show_expr (open->action);
		}
	      if (open->delim)
		{
		  fputs (" DELIM=", dumpfile);
		  show_expr (open->delim);
		}
	      if (open->pad)
		{
		  fputs (" PAD=", dumpfile);
		  show_expr (open->pad);
		}
	      if (open->decimal)
		{
		  fputs (" DECIMAL=", dumpfile);
		  show_expr (open->decimal);
		}
	      if (open->encoding)
		{
		  fputs (" ENCODING=", dumpfile);
		  show_expr (open->encoding);
		}
	      if (open->round)
		{
		  fputs (" ROUND=", dumpfile);
		  show_expr (open->round);
		}
	      if (open->sign)
		{
		  fputs (" SIGN=", dumpfile);
		  show_expr (open->sign);
		}
	      if (open->convert)
		{
		  fputs (" CONVERT=", dumpfile);
		  show_expr (open->convert);
		}
	      if (open->asynchronous)
		{
		  fputs (" ASYNCHRONOUS=", dumpfile);
		  show_expr (open->asynchronous);
		}
	      if (open->err != NULL)
		fprintf (dumpfile, " ERR=%d", open->err->value);

	      break;

	    case EXEC_CLOSE:
	      fputs ("CLOSE", dumpfile);
	      close = c->ext.close;

	      if (close->unit)
		{
		  fputs (" UNIT=", dumpfile);
		  show_expr (close->unit);
		}
	      if (close->iomsg)
		{
		  fputs (" IOMSG=", dumpfile);
		  show_expr (close->iomsg);
		}
	      if (close->iostat)
		{
		  fputs (" IOSTAT=", dumpfile);
		  show_expr (close->iostat);
		}
	      if (close->status)
		{
		  fputs (" STATUS=", dumpfile);
		  show_expr (close->status);
		}
	      if (close->err != NULL)
		fprintf (dumpfile, " ERR=%d", close->err->value);
	      break;

	    case EXEC_BACKSPACE:
	      fputs ("BACKSPACE", dumpfile);
	      goto show_filepos;

	    case EXEC_ENDFILE:
	      fputs ("ENDFILE", dumpfile);
	      goto show_filepos;

	    case EXEC_REWIND:
	      fputs ("REWIND", dumpfile);
	      goto show_filepos;

	    case EXEC_FLUSH:
	      fputs ("FLUSH", dumpfile);

	    show_filepos:
	      fp = c->ext.filepos;

	      if (fp->unit)
		{
		  fputs (" UNIT=", dumpfile);
		  show_expr (fp->unit);
		}
	      if (fp->iomsg)
		{
		  fputs (" IOMSG=", dumpfile);
		  show_expr (fp->iomsg);
		}
	      if (fp->iostat)
		{
		  fputs (" IOSTAT=", dumpfile);
		  show_expr (fp->iostat);
		}
	      if (fp->err != NULL)
		fprintf (dumpfile, " ERR=%d", fp->err->value);
	      break;

	    case EXEC_INQUIRE:
	      fputs ("INQUIRE", dumpfile);
	      i = c->ext.inquire;

	      if (i->unit)
		{
		  fputs (" UNIT=", dumpfile);
		  show_expr (i->unit);
		}
	      if (i->file)
		{
		  fputs (" FILE=", dumpfile);
		  show_expr (i->file);
		}

	      if (i->iomsg)
		{
		  fputs (" IOMSG=", dumpfile);
		  show_expr (i->iomsg);
		}
	      if (i->iostat)
		{
		  fputs (" IOSTAT=", dumpfile);
		  show_expr (i->iostat);
		}
	      if (i->exist)
		{
		  fputs (" EXIST=", dumpfile);
		  show_expr (i->exist);
		}
	      if (i->opened)
		{
		  fputs (" OPENED=", dumpfile);
		  show_expr (i->opened);
		}
	      if (i->number)
		{
		  fputs (" NUMBER=", dumpfile);
		  show_expr (i->number);
		}
	      if (i->named)
		{
		  fputs (" NAMED=", dumpfile);
		  show_expr (i->named);
		}
	      if (i->name)
		{
		  fputs (" NAME=", dumpfile);
		  show_expr (i->name);
		}
	      if (i->access)
		{
		  fputs (" ACCESS=", dumpfile);
		  show_expr (i->access);
		}
	      if (i->sequential)
		{
		  fputs (" SEQUENTIAL=", dumpfile);
		  show_expr (i->sequential);
		}

	      if (i->direct)
		{
		  fputs (" DIRECT=", dumpfile);
		  show_expr (i->direct);
		}
	      if (i->form)
		{
		  fputs (" FORM=", dumpfile);
		  show_expr (i->form);
		}
	      if (i->formatted)
		{
		  fputs (" FORMATTED", dumpfile);
		  show_expr (i->formatted);
		}
	      if (i->unformatted)
		{
		  fputs (" UNFORMATTED=", dumpfile);
		  show_expr (i->unformatted);
		}
	      if (i->recl)
		{
		  fputs (" RECL=", dumpfile);
		  show_expr (i->recl);
		}
	      if (i->nextrec)
		{
		  fputs (" NEXTREC=", dumpfile);
		  show_expr (i->nextrec);
		}
	      if (i->blank)
		{
		  fputs (" BLANK=", dumpfile);
		  show_expr (i->blank);
		}
	      if (i->position)
		{
		  fputs (" POSITION=", dumpfile);
		  show_expr (i->position);
		}
	      if (i->action)
		{
		  fputs (" ACTION=", dumpfile);
		  show_expr (i->action);
		}
	      if (i->read)
		{
		  fputs (" READ=", dumpfile);
		  show_expr (i->read);
		}
	      if (i->write)
		{
		  fputs (" WRITE=", dumpfile);
		  show_expr (i->write);
		}
	      if (i->readwrite)
		{
		  fputs (" READWRITE=", dumpfile);
		  show_expr (i->readwrite);
		}
	      if (i->delim)
		{
		  fputs (" DELIM=", dumpfile);
		  show_expr (i->delim);
		}
	      if (i->pad)
		{
		  fputs (" PAD=", dumpfile);
		  show_expr (i->pad);
		}
	      if (i->convert)
		{
		  fputs (" CONVERT=", dumpfile);
		  show_expr (i->convert);
		}
	      if (i->asynchronous)
		{
		  fputs (" ASYNCHRONOUS=", dumpfile);
		  show_expr (i->asynchronous);
		}
	      if (i->decimal)
		{
		  fputs (" DECIMAL=", dumpfile);
		  show_expr (i->decimal);
		}
	      if (i->encoding)
		{
		  fputs (" ENCODING=", dumpfile);
		  show_expr (i->encoding);
		}
	      if (i->pending)
		{
		  fputs (" PENDING=", dumpfile);
		  show_expr (i->pending);
		}
	      if (i->round)
		{
		  fputs (" ROUND=", dumpfile);
		  show_expr (i->round);
		}
	      if (i->sign)
		{
		  fputs (" SIGN=", dumpfile);
		  show_expr (i->sign);
		}
	      if (i->size)
		{
		  fputs (" SIZE=", dumpfile);
		  show_expr (i->size);
		}
	      if (i->id)
		{
		  fputs (" ID=", dumpfile);
		  show_expr (i->id);
		}

	      if (i->err != NULL)
		fprintf (dumpfile, " ERR=%d", i->err->value);
	      break;

	    case EXEC_IOLENGTH:
	      fputs ("IOLENGTH ", dumpfile);
	      show_expr (c->expr);
	      goto show_dt_code;
	      break;
*/
	    case EXEC_READ:
	    case EXEC_WRITE:
	    {
	    	debug(5,"gfc2pips_code2instruction","Translation of %s\n",c->op==EXEC_WRITE?"PRINT":"READ");
	    	//yeah ! we've got an intrinsic
	    	gfc_code *d=c;
	    	entity e = entity_undefined;
	    	if(c->op==EXEC_WRITE){

	    		e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup("write")));//print or write ? print is only a particular case of write
	    	}else{
	    		e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup("read")));
	    	}


	    	newgen_list list_of_arguments,list_of_arguments_p;
	    	list_of_arguments = list_of_arguments_p = NULL;
			for (c = c->block->next; c; c = c->next){
				if(c->expr){
					expression ex = gfc2pips_expr2expression(c->expr);
					if( ex!=entity_undefined && ex!=NULL ){
						if(list_of_arguments_p){
							CDR(list_of_arguments_p) = CONS(EXPRESSION,ex,NULL);
							list_of_arguments_p = CDR(list_of_arguments_p);
						}else{
							list_of_arguments = list_of_arguments_p = CONS(EXPRESSION,ex,NULL);
						}
					}
				}
			}
			entity_initial(e) = make_value( is_value_intrinsic, e );
			entity_type(e) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,list_of_arguments);
			/*return make_instruction_call(call_);*/
/*
 * FROM pips parser
**************************************************
   char buffer[20];
		(void) strcpy(buffer, (syn_vsp[(1) - (3)].string));
		free((syn_vsp[(1) - (3)].string));

		if(strcmp(buffer,"UNIT")!=0 && strcmp(buffer,"FMT")!=0) {
		  ParserError("parser",
			  "Illegal default option '*' in IO control list\n");
		}

		(void) strcat(buffer, "=");

		(syn_val.liste) = CONS(EXPRESSION,
			  MakeCharacterConstantExpression(buffer),
			  CONS(EXPRESSION,
		 MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME))
			       , NULL));
		ici += 2;
**************************************************
FROM gram.y
io_inst:  io_keyword io_f_u_id
	    {
                $$ = MakeSimpleIoInst1($1, $2);
	    }
        | io_keyword io_f_u_id TK_COMMA opt_lio_elem
            {
		$$ = MakeSimpleIoInst2($1, $2, $4);
	    }
	| io_keyword TK_LPAR lci TK_RPAR opt_virgule opt_lio_elem
	    { $$ = MakeIoInstA($1, $3, $6); }
        | iobuf_keyword TK_LPAR io_f_u_id TK_COMMA io_f_u_id TK_RPAR
                        TK_LPAR unpar_io_expr TK_COMMA unpar_io_expr TK_RPAR
	    { $$ = MakeIoInstB($1, $3, $5, $8, $10); }

**************************************************
FROM instruction MakeSimpleIoInst2(int keyword, expression f, list list_of_arguments(io_list)))
*/
			expression std, format, unite,f;
			cons * lci;

			f = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));
			//f = make_expression(<syntax(atom)>, normalized_undefined);
			std = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));
			//balancer la valeur de UNIT et de FMT d'une façon ou d'une autre ?
			unite = MakeCharacterConstantExpression("UNIT=");
			format = MakeCharacterConstantExpression("FMT=");

			lci = CONS(EXPRESSION, unite,
				CONS(EXPRESSION, std,
					CONS(EXPRESSION, format,
						CONS(EXPRESSION, f, NULL)
					)
				)
			);
/*	inst = MakeIoInstA((keyword==TK_PRINT)?TK_WRITE:TK_READ ,lci, list_of_arguments);

**************************************************
instruction
MakeIoInstA(keyword, lci, lio)
int keyword;
cons *lci;
cons *lio;
{
  cons *l;
  instruction io = instruction_undefined;
  instruction io_call = instruction_undefined;
  statement io_err = statement_undefined;
  statement io_end = statement_undefined;
  expression unit = expression_undefined;

  //on fait une copie et on passe de deux en deux :  attribut1=, valeur1, attribut2=, valeur2
  for (l = lci; l != NULL; l = CDR(CDR(l))) {
    syntax s1;
    entity e1;

    s1 = expression_syntax(EXPRESSION(CAR(l)));

    e1 = call_function(syntax_call(s1));

    if (strcmp(entity_local_name(e1), "UNIT=") == 0) {
      if( ! expression_undefined_p(unit) )
	free_expression(unit);
      unit = copy_expression(EXPRESSION(CAR(CDR(l))));
    }
  }

  for (l = lci; l != NULL; l = CDR(CDR(l))) {
    syntax s1, s2;
    entity e1, e2;

    s1 = expression_syntax(EXPRESSION(CAR(l)));
    s2 = expression_syntax(EXPRESSION(CAR(CDR(l))));

    pips_assert("syntax is a call", syntax_call_p(s1));
    e1 = call_function(syntax_call(s1));
    pips_assert("value is constant", value_constant_p(entity_initial(e1)));
    pips_assert("constant is not int (thus litteral or call)",
		!constant_int_p(value_constant(entity_initial(e1))));

    if (strcmp(entity_local_name(e1), "ERR=") == 0 ||
	strcmp(entity_local_name(e1), "END=") == 0 ||
	strcmp(entity_local_name(e1), "FMT=") == 0) {
      if (syntax_call_p(s2)) {
	e2 = call_function(syntax_call(s2));
	if (value_constant_p(entity_initial(e2))) {
	  if (constant_int_p(value_constant(entity_initial(e2)))) {
	    call_function(syntax_call(s2)) =
	      MakeLabel(entity_local_name(e2));
	  }
	}
	e2 = call_function(syntax_call(s2));
	if (strcmp(entity_local_name(e1), "FMT=") != 0
	    && expression_undefined_p(unit)) {
	  unit = int_to_expression(0);
	}
	if (strcmp(entity_local_name(e1), "ERR=") == 0) {
	  io_err = make_check_io_statement(IO_ERROR_ARRAY_NAME, unit, e2);
	}
	else if (strcmp(entity_local_name(e1), "END=") == 0) {
	  io_end = make_check_io_statement(IO_EOF_ARRAY_NAME, unit, e2);
	}
	else
	  ;
      }
    }
  }

  lci = gen_nconc(lci, lio);

  io_call = make_instruction(is_instruction_call,
			     make_call(CreateIntrinsic(NameOfToken(keyword)),
				       lci));

  if(statement_undefined_p(io_err) && statement_undefined_p(io_end)) {
    io = io_call;
  }
  else {
    list ls = NIL;
    if(!statement_undefined_p(io_err)) {
      ls = CONS(STATEMENT, io_err, ls);
    }
    if(!statement_undefined_p(io_end)) {
      ls = CONS(STATEMENT, io_end, ls);
    }
    ls = CONS(STATEMENT, MakeStatement(entity_empty_label(), io_call), ls);
    io = make_instruction(is_instruction_sequence, make_sequence(ls));
    instruction_consistent_p(io);
  }

  return io;
}

**************************************************
 */
		return make_instruction(is_instruction_call,
			make_call(e,
				//CreateIntrinsic(d->op==EXEC_WRITE?"PRINT":"READ"),
				gen_nconc(lci, NULL)
			)
		);

	    /*show_dt:
	      dt = c->ext.dt;
	      if (dt->io_unit)
		{
		  fputs (" UNIT=", dumpfile);
		  show_expr (dt->io_unit);
		}

	      if (dt->format_expr)
		{
		  fputs (" FMT=", dumpfile);
		  show_expr (dt->format_expr);
		}

	      if (dt->format_label != NULL)
		fprintf (dumpfile, " FMT=%d", dt->format_label->value);
	      if (dt->namelist)
		fprintf (dumpfile, " NML=%s", dt->namelist->name);

	      if (dt->iomsg)
		{
		  fputs (" IOMSG=", dumpfile);
		  show_expr (dt->iomsg);
		}
	      if (dt->iostat)
		{
		  fputs (" IOSTAT=", dumpfile);
		  show_expr (dt->iostat);
		}
	      if (dt->size)
		{
		  fputs (" SIZE=", dumpfile);
		  show_expr (dt->size);
		}
	      if (dt->rec)
		{
		  fputs (" REC=", dumpfile);
		  show_expr (dt->rec);
		}
	      if (dt->advance)
		{
		  fputs (" ADVANCE=", dumpfile);
		  show_expr (dt->advance);
		}
	      if (dt->id)
		{
		  fputs (" ID=", dumpfile);
		  show_expr (dt->id);
		}
	      if (dt->pos)
		{
		  fputs (" POS=", dumpfile);
		  show_expr (dt->pos);
		}
	      if (dt->asynchronous)
		{
		  fputs (" ASYNCHRONOUS=", dumpfile);
		  show_expr (dt->asynchronous);
		}
	      if (dt->blank)
		{
		  fputs (" BLANK=", dumpfile);
		  show_expr (dt->blank);
		}
	      if (dt->decimal)
		{
		  fputs (" DECIMAL=", dumpfile);				make_syntax(is_syntax_call,call_),
								normalized_undefined

		  show_expr (dt->decimal);
		}
	      if (dt->delim)
		{
		  fputs (" DELIM=", dumpfile);
		  show_expr (dt->delim);
		}
	      if (dt->pad)
		{
		  fputs (" PAD=", dumpfile);
		  show_expr (dt->pad);
		}
	      if (dt->round)
		{
		  fputs (" ROUND=", dumpfile);
		  show_expr (dt->round);
		}
	      if (dt->sign)
		{
		  fputs (" SIGN=", dumpfile);
		  show_expr (dt->sign);
		}

	    show_dt_code:
	      fputc ('\n', dumpfile);
	      for (c = c->block->next; c; c = c->next)
		show_code_node (level + (c->next != NULL), c);
	      //return;
*/

	    }break;
/*
	    case EXEC_TRANSFER:
	      fputs ("TRANSFER ", dumpfile);
	      show_expr (c->expr);
	      break;

	    case EXEC_DT_END:
	      fputs ("DT_END", dumpfile);
	      dt = c->ext.dt;

	      if (dt->err != NULL)
		fprintf (dumpfile, " ERR=%d", dt->err->value);
	      if (dt->end != NULL)
		fprintf (dumpfile, " END=%d", dt->end->value);
	      if (dt->eor != NULL)
		fprintf (dumpfile, " EOR=%d", dt->eor->value);
	      break;

	    case EXEC_OMP_ATOMIC:
	    case EXEC_OMP_BARRIER:
	    case EXEC_OMP_CRITICAL:
	    case EXEC_OMP_FLUSH:
	    case EXEC_OMP_DO:
	    case EXEC_OMP_MASTER:
	    case EXEC_OMP_ORDERED:
	    case EXEC_OMP_PARALLEL:
	    case EXEC_OMP_PARALLEL_DO:
	    case EXEC_OMP_PARALLEL_SECTIONS:
	    case EXEC_OMP_PARALLEL_WORKSHARE:
	    case EXEC_OMP_SECTIONS:
	    case EXEC_OMP_SINGLE:
	    case EXEC_OMP_TASK:
	    case EXEC_OMP_TASKWAIT:
	    case EXEC_OMP_WORKSHARE:
	      show_omp_node (level, c);
	      break;
	    */
	    default:
	    	debug(3, "gfc2pips_code2instruction", "not yet dumpable %d\n",c->op);
	      //gfc_internal_error ("show_code_node(): Bad statement code");
	}
	//return instruction_undefined;
	return make_instruction_block(NULL);;
}

entity gfc2pips_code2get_label(gfc_code *c){
	debug(9,"gfc2pips_code2get_label",
		"test label: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if( c && c->here ) return gfc2pips_int2label(c->here->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label2(gfc_code *c){
	debug(9,"gfc2pips_code2get_label2",
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if(c&&c->label )return gfc2pips_int2label(c->label->value);
	return entity_empty_label() ;
}

/*
 * Translate an expression from a RI to another
 * for the moment only care about (a+b)*c   + - * / ** // .AND. .OR. .EQV. .NEQV. .NOT.
 */
expression gfc2pips_expr2expression(gfc_expr *expr){
	//GFC
	//p->value.op.op opérateur de l'expression
	//p->value.op.op1 premier membre de l'expression
	//p->value.op.op2 second membre de l'expression

	//PIPS
	//expression => sous_expression | TK_LPAR sous_expression TK_RPAR
	//MakeFortranBinaryCall(CreateIntrinsic("+"), expression 1, expression 2);
	expression e = expression_undefined;
	message_assert("No expr\n",expr);
	if(!expr->symtree){
		//fprintf(stdout,"No symtree\n");
	}else if(!expr->symtree->n.sym){
		//fprintf(stdout,"No symbol\n");
	}else if(!expr->symtree->n.sym->name){
		//fprintf(stdout,"No name\n");
	}else{
		//fprintf(stdout,"gfc2pips_expr2expression: dumping %s\n",expr->symtree->n.sym->name);
	}

	//fprintf(stdout,"type: %d\n",expr->expr_type);
	//fprintf(stdout,"kind: %d\n",expr->ts.kind);
	switch(expr->expr_type){
		case EXPR_OP:{
			const char *c;
			debug(5,"gfc2pips_expr2expression","op\n");
			switch(expr->value.op.op){
				case INTRINSIC_UPLUS: case INTRINSIC_PLUS: c="+"; break;
				case INTRINSIC_UMINUS: case INTRINSIC_MINUS: c="-"; break;
				case INTRINSIC_TIMES: c="*"; break;
				case INTRINSIC_DIVIDE: c="/"; break;
				case INTRINSIC_POWER: c="**"; break;
				case INTRINSIC_CONCAT: c="//"; break;
				case INTRINSIC_AND: c=".AND."; break;
				case INTRINSIC_OR: c=".OR."; break;
				case INTRINSIC_EQV: c=".EQV."; break;
				case INTRINSIC_NEQV: c=".NEQV."; break;

				case INTRINSIC_EQ: case INTRINSIC_EQ_OS: c=".EQ."; break;
				case INTRINSIC_NE: case INTRINSIC_NE_OS: c=".NE."; break;
				case INTRINSIC_GT: case INTRINSIC_GT_OS: c=".GT."; break;
				case INTRINSIC_GE: case INTRINSIC_GE_OS: c=".GE."; break;
				case INTRINSIC_LT: case INTRINSIC_LT_OS: c=".LT."; break;
				case INTRINSIC_LE: case INTRINSIC_LE_OS: c=".LE."; break;

				case INTRINSIC_NOT: c=".NOT."; break;

				case INTRINSIC_PARENTHESES: return gfc2pips_expr2expression(expr->value.op.op1); break;
				default:
					fprintf(stdout,"gfc2pips_expr2expression: intrinsic not yet recognized: %d\n",expr->value.op.op);
					c="";
				break;
			}
			//c = gfc_op2string(expr->value.op.op);
			if(strlen(c)>0){
				debug(6,"gfc2pips_expr2expression","intrinsic recognized: %s\n",c);
				expression e1 = gfc2pips_expr2expression(expr->value.op.op1);
				expression e2 = gfc2pips_expr2expression(expr->value.op.op2);
				if(e1 && e2){
					return MakeBinaryCall(
						CreateIntrinsic(c),
						e1,
						e2
					);
				}else{
					debug(6,"gfc2pips_expr2expression","e1 or e2 is null\n");
				}
			}
		}break;
		case EXPR_VARIABLE:{
			debug(5,"gfc2pips_expr2expression","var\n");
			//use ri-util only functions
		    //add array recognition (multi-dimension variable)
		    //créer une liste de tous les indices et la balancer en deuxième arg  gen_nconc($1, CONS(EXPRESSION, $3, NIL))
			newgen_list ref = NULL;
			if(expr->ref){
				/*//assign statement
				$$ = MakeAssignInst(
					CheckLeftHandSide(
						MakeAtom(
							string,
							newgen_list<expression>,
							expression_undefined,
							expression_undefined,
							TRUE
						)
					),
					[second membre de l'expression]
				);
				 */
				gfc_ref *r =expr->ref;
				while(r){
					fprintf(stdout,"^^^^^^^^^^^^^^^^^^^^^\n");
					if(r->type==REF_ARRAY){
						ref = gfc2pips_array_ref2indices(&r->u.ar);
					}
					r=r->next;
				}
			}
			syntax s =make_syntax_reference(
				make_reference(
					FindOrCreateEntity(CurrentPackage, expr->symtree->n.sym->name),
					ref
				)
			);
			e = make_expression( s, normalized_undefined );
			return e;
		}break;
		case EXPR_CONSTANT:
			debug(5,"gfc2pips_expr2expression","cst %d\n",expr);
			switch(expr->ts.type){
				case BT_INTEGER: e = gfc2pips_int2expression(mpz_get_ui(expr->value.integer)); break;
				case BT_LOGICAL: e = gfc2pips_logical2expression(expr->value.logical); break;
				case BT_REAL:
					/*fprintf(stdout,"%f\n",mpfr_get_d(expr->value.real,GFC_RND_MODE));
					fprintf(stdout,"expr->where.nextc\t %s\n",expr->where.nextc);
					fprintf(stdout,"expr->where.lb->dbg_emitted\t %d\n",expr->where.lb->dbg_emitted);

					fprintf(stdout,"expr->where.lb->file->filename\t %s\n",expr->where.lb->file->filename);
					fprintf(stdout,"expr->where.lb->file->inclusion_line\t %d\n",expr->where.lb->file->inclusion_line);
					fprintf(stdout,"expr->where.lb->file->line\t %d\n",expr->where.lb->file->line);
					fprintf(stdout,"expr->ref\t %d\n",expr->ref);


					fprintf(stdout,"expr->where.lb->line\t %s\n",expr->where.lb->line);
					fprintf(stdout,"expr->where.lb->location\t %d\n",expr->where.lb->location);//c'est un chiffre qui dépend de la ligne et de  ??? => trouver comment il est calculé
					fprintf(stdout,"expr->where.lb->truncated\t %d\n",expr->where.lb->truncated);//établi si la ligne a été coupée lors du parse
*/
					//convertir le real en qqch de correct au niveau sortie
					e = gfc2pips_real2expression(mpfr_get_d(expr->value.real,GFC_RND_MODE));
				break;
				case BT_CHARACTER: e = MakeCharacterConstantExpression(expr->value.character.string); break;
				case BT_COMPLEX:
					e = MakeComplexConstantExpression(
						gfc2pips_real2expression(mpfr_get_d(expr->value.complex.r,GFC_RND_MODE)),
						gfc2pips_real2expression(mpfr_get_d(expr->value.complex.i,GFC_RND_MODE))
					);
				break;
				case BT_HOLLERITH:
				default:break;

			}
			//if(expr->ref)
			return e;
		break;
		case EXPR_FUNCTION:
			debug(5,"gfc2pips_expr2expression","func\n");
			//beware the automatic conversion here, some conversion functions may be automatically called here, and we do not want them in the code
			if(strncmp(str2upper(expr->symtree->n.sym->name),str2upper(strdup("__convert_")),strlen("__convert_"))==0){
				//fprintf(stdout,"gfc2pips_expr2expression: auto-convert detected %s\n",expr->symtree->n.sym->name);
				if(expr->value.function.actual->expr){
					debug(6,"gfc2pips_expr2expression","expression not null !\n");
					//show_expr(expr->value.function.actual->expr);
					return gfc2pips_expr2expression(expr->value.function.actual->expr);
				}else{
					debug(6,"gfc2pips_expr2expression","expression null !\n");
				}
			}else{
				//this is a regular call
				//on dump l'appel de fonction

				//actual est la liste
				newgen_list list_of_arguments = NULL,list_of_arguments_p = NULL;
				gfc_actual_arglist *act=expr->value.function.actual;

				do{
					expression ex = gfc2pips_expr2expression(act->expr);

					if(ex!=entity_undefined){

						if(list_of_arguments_p){
							CDR(list_of_arguments_p) = CONS(EXPRESSION,ex,NIL);
							list_of_arguments_p = CDR(list_of_arguments_p);
						}else{
							list_of_arguments_p = CONS(EXPRESSION,e,NIL);
						}
					}
					if(list_of_arguments==NULL)list_of_arguments = list_of_arguments_p;

				}while(act = act->next);


				entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, expr->value.function.name);
				entity_initial(e) = make_value(is_value_intrinsic, e );
				//entity_initial(e) = make_value(is_value_constant,make_constant(is_constant_int, (void *) CurrentTypeSize));
				entity_type(e) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
				call call_ = make_call(e,list_of_arguments);
				return make_expression(
					make_syntax(is_syntax_call,call_),
					normalized_undefined
				);
			}
		break;
	default:
			fprintf(stdout,"gfc2pips_expr2expression: dump not yet implemented, type of gfc_expr not recognized %d\n",expr->expr_type);
		break;
	}
	return expression_undefined;
}

bool gfc2pips_exprIsVariable(gfc_expr * expr){
	return expr && expr->expr_type==EXPR_VARIABLE;
}

entity gfc2pips_expr2entity(gfc_expr *expr){
	message_assert("No expression to dump.",expr);

	if(expr->expr_type==EXPR_VARIABLE){
		message_assert("No symtree in the expression.",expr->symtree);
		message_assert("No symbol in the expression.",expr->symtree->n.sym);
		message_assert("No name in the expression.",expr->symtree->n.sym->name);
		return FindOrCreateEntity(CurrentPackage, expr->symtree->n.sym->name);
	}

	if(expr->expr_type==EXPR_CONSTANT){
		if( expr->ts.type==BT_INTEGER ){
			return gfc2pips_int_const2entity(mpz_get_ui(expr->value.integer));
		}
		if( expr->ts.type==BT_LOGICAL ){
			return gfc2pips_logical2entity(expr->value.logical);
		}
		if( expr->ts.type==BT_REAL ){
			return gfc2pips_real2entity(mpfr_get_d(expr->value.real,GFC_RND_MODE));
		}
	}

	message_assert("No entity to extract from this expression",0);
}


statement gfc2pips_bloc2statement(gfc_code *c){
	//entity l, instruction i
	if(!c){
		return statement_undefined;
	}
	entity l = gfc2pips_code2get_label(c);
	instruction i = gfc2pips_code2instruction(c,false);
	if(i==instruction_undefined){
		fprintf(stderr,"Dumping of instruction failed\n");
		return statement_undefined;
	}
	debug(4,"gfc2pips_bloc2statement","Instruction created\n");
	statement s = make_statement(
		l,
		c->here->where.lb->file->inclusion_line,//(instruction_goto_p(i)||instruction_block_p(i))?STATEMENT_NUMBER_UNDEFINED : get_statement_number(),//provient de Syntax :S
		STATEMENT_ORDERING_UNDEFINED,
		empty_comments,
		i,
		NIL,
		NULL,
		empty_extensions ()
	);
	//boucle de dump de chaque instruction
	return s;
}

