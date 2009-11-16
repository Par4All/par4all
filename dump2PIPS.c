/*
 * The naming of functions is based on the following syntax :
 * - every and each function used to manipulate gfc or pips entities begin with gfc2pips_
 * - if the function is a translation from one RI to the other it will be:
 * 		gfc2pips_<type_of_the_source_entity>2<type_of_the_target_entity>(arguments)
 */
#include "dump2PIPS.h"



newgen_list gfc2pips_list_of_declared_code = NULL;
newgen_list gfc2pips_list_of_loops = NULL;
static int gfc2pips_last_created_label = 95000;
static int gfc2pips_last_created_label_step = 2;

/*
 * Add initialization at the begining and at the end of the dumping
 * Put information in a table, in the "the_actual_gfc_parser"
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
char * strrcpy(char *dest, __const char *src){
	int i = strlen(src);
	while(i--) dest[i] = src[i];
	return dest;
}

int fcopy(char* old,char* new){
	if(!old||!new) return 0;
	FILE * o = fopen(old,"r");
	if(o){
		FILE * n = fopen(new,"w");
		if(n){
			int c = fgetc(o);
			while(c!=EOF){
				fputc(c,n);
				c = fgetc(o);
			}
			fclose(n);
			fclose(o);
			return 1;
		}
		fclose(o);
		return 0;
	}
	return 0;
}

//an enum to know what kind of main entity we are dealing with
typedef enum main_entity_type{MET_PROG,MET_SUB,MET_FUNC,MET_MOD,MET_BLOCK} main_entity_type;
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

	//gfc_get_code();
	gfc2pips_shift_comments();
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
		bloc_token=MET_PROG;
	}else if( root_sym->attr.subroutine ){
		debug(3, "gfc2pips_namespace", "subroutine founded %s\n",ns->proc_name->name);
		gfc2pips_main_entity = make_empty_subroutine(
				str2upper((ns->proc_name->name))
		);
		bloc_token = MET_SUB;
	}else if( root_sym->attr.function ){
		debug(3, "gfc2pips_namespace", "function founded %s\n",ns->proc_name->name);
		gfc2pips_main_entity = make_empty_function(
			str2upper(ns->proc_name->name),
			gfc2pips_symbol2type(root_sym)
		);
		bloc_token = MET_FUNC;
	}else if(root_sym->attr.flavor == FL_BLOCK_DATA){
		debug(3, "gfc2pips_namespace", "block data founded \n");
		gfc2pips_main_entity = make_empty_blockdata(
			str2upper((ns->proc_name->name))
		);
		bloc_token = MET_BLOCK;
	}else{
		debug(3, "gfc2pips_namespace", "not yet dumpable %s\n",ns->proc_name->name);
		if(root_sym->attr.procedure){
			fprintf(stderr,"procedure\n");
		}
		//set_current_module_entity(gfc2pips_main_entity);
		return;
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
	newgen_list parameters = NULL, parameters_p = NULL, parameters_name = NULL;
	int formal_offset = 1;
	if(bloc_token == MET_FUNC || bloc_token == MET_SUB){
		parameters_p = parameters = gfc2pips_args(ns);
		parameters_name = gen_copy_seq(parameters);//we need a copy of the list of the entities of the parameters
		while(parameters_p){
			entity ent = parameters_p->car.e;
			debug(8,"gfc2pips_namespace", "parameter founded: %s", entity_name(ent));
			entity_storage(ent) = make_storage(is_storage_formal, make_formal(gfc2pips_main_entity, formal_offset));
			/*if(formal_label_replacement_p(ent)){
				entity_type(ent) = make_type(
					is_type_variable,
					make_variable(
						make_basic(is_basic_string, MakeValueUnknown()),
						NIL,
						NIL
					)
				);
			} else if(SubstituteAlternateReturnsP() && ReturnCodeVariableP(ent)) {
				entity_type(ent) = MakeTypeVariable(make_basic(is_basic_int, (void *) 4), NIL);
			} else {
				entity_type(ent) = ImplicitType(ent);
			}*/
			entity_initial(ent) = MakeValueUnknown();
			type formal_param_type = entity_type(ent);//is the format ok ?
			parameters_p->car.e = make_parameter( formal_param_type, make_mode_reference(), make_dummy_identifier(ent) );
			formal_offset ++;
			POP(parameters_p);
		}
	}
	int stack_offset = 0;
	debug(2, "gfc2pips_namespace", "List of parameters done: %s\n",parameters?"there is/are argument(s)":"none");


	////type of entity we are creating : a function (except maybe for module ?
	////a functional type is made of a list of parameters and the type returned
	if(bloc_token != MET_BLOCK){
		entity_type(gfc2pips_main_entity) = make_type(is_type_functional, make_functional(parameters, bloc_type));
	}

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

	set_current_module_entity(gfc2pips_main_entity);
	gfc2pips_initAreas();


	//// declare commons
	newgen_list commons,commons_p;
	commons = commons_p = getSymbolBy(ns, ns->common_root,gfc2pips_get_commons);
	while(commons_p){
		gfc_symtree *st = (gfc_symtree*)commons_p->car.e;
		debug(4,"gfc2pips_namespace","common founded: /%s/\n",st->name);
		entity com = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, st->name);
		//com = make_common_entity(com);
		entity_type(com) = make_type(is_type_area, make_area(0, NIL));
		entity_storage(com) = make_storage(
			is_storage_ram,
			(make_ram(get_current_module_entity(),StaticArea, 0, NIL))
		);
		entity_initial(com) = make_value_code(make_code(NIL,string_undefined,make_sequence(NIL),NIL));
		AddEntityToDeclarations(com, get_current_module_entity());
		commons_p->car.e = com;
		gfc_symbol *s = st->n.common->head;
		int indice_common = stack_offset;
		while(s){
			//faire une fonction récursive qui permet de déclarer à l'envers la liste des élements du common ?
			debug(5,"gfc2pips_namespace","element in common founded: %s\n",s->name);
			entity in_common_entity = gfc2pips_symbol2entity(s);
			entity_storage(in_common_entity) = make_storage(
				is_storage_ram,
				make_ram(
					get_current_module_entity(),
					com,
					indice_common,
				    NIL
			   )
			);
			int size;SizeOfArray(in_common_entity,&size);
			//size = gfc2pips_symbol2sizeArray(s) * gfc2pips_symbol2size(s);
			indice_common += size;
			area_layout(type_area(entity_type(com))) = gen_nconc(area_layout(type_area(entity_type(com))), CONS(ENTITY, in_common_entity, NIL));
			s = s->common_next;
		}
		//area_layout(type_area(entity_type(com))) = gen_nreverse(area_layout(type_area(entity_type(com))));
		debug(4,"gfc2pips_namespace","nb of elements in the common: %d\n",
			gen_length(
				area_layout(type_area(entity_type(com)))
			)
		);
		POP(commons_p);
	}

	//// declare DIMENSIONS
	//newgen_list dimensions_p,dimensions;
	//dimensions_p = dimensions = getSymbolBy( ns, ns->sym_root, gfc2pips_test_dimensions );
	// trouver comment démmêler les implicit/explicit des dimensions ? est-ce nécessaire ?
	//=> choix de mettre les dimensions uniquement des variables utilisées dans les commons
	//=> revoir la taille des commons à cause de leur foutues dimensions supplémentaires



	//// declare variables
	newgen_list variables_p,variables;
	variables_p = variables = gfc2pips_vars(ns);

	//we concatenate the entities from variables, commons and parameters and make sure they are declared only once
	//it seems parameters cannot be declared implicitly and have to be part of the list
	newgen_list complete_list_of_entities = NULL,complete_list_of_entities_p = NULL;
	complete_list_of_entities_p = variables_p;
	/*while(variables_p){
		if(complete_list_of_entities){
			CDR(complete_list_of_entities) = gen_cons(ENTITY(CAR(variables_p)),NULL);
			complete_list_of_entities = CDR(complete_list_of_entities);
		}else{
			complete_list_of_entities_p = complete_list_of_entities = gen_cons(ENTITY(CAR(variables_p)),NULL);
		}
		//complete_list_of_entities = gen_once(CAR(variables_p).e,complete_list_of_entities);
		POP(variables_p);
	}*/
	commons_p = commons;
	complete_list_of_entities_p = gen_concatenate( commons_p, complete_list_of_entities_p );
	/*while(commons_p){
		if(complete_list_of_entities){
			CDR(complete_list_of_entities) = gen_cons(ENTITY(CAR(commons_p)),NULL);
			complete_list_of_entities = CDR(complete_list_of_entities);
		}else{
			complete_list_of_entities_p = complete_list_of_entities = gen_cons(ENTITY(CAR(commons_p)),NULL);
		}
		//complete_list_of_entities = gen_once(CAR(commons_p).e,complete_list_of_entities);
		POP(commons_p);
	}*/

	complete_list_of_entities_p = gen_concatenate(complete_list_of_entities_p, parameters_name);
	/*while(parameters_name){
		if(complete_list_of_entities){
			CDR(complete_list_of_entities) = gen_cons(CAR(parameters_name).e,NULL);
		}else{
			complete_list_of_entities_p = complete_list_of_entities = gen_cons(ENTITY(CAR(parameters_name)),NULL);
			complete_list_of_entities = CDR(complete_list_of_entities);
		}
		//complete_list_of_entities = gen_once(CAR(parameters_name).e,complete_list_of_entities);
		POP(parameters_name);
	}*/

	complete_list_of_entities = complete_list_of_entities_p;
	while(complete_list_of_entities_p){
		if(entity_initial(ENTITY(CAR(complete_list_of_entities_p)))==value_undefined){
			entity_initial(ENTITY(CAR(complete_list_of_entities_p))) = MakeValueUnknown();
		}
		//fprintf(stderr,"all entities: %s \n",((entity)CAR(complete_list_of_entities_p).e)->_entity_name_);
		POP(complete_list_of_entities_p);
	}


	//we have to add the list of variables declared in the initial value of the main entity
	entity_initial(gfc2pips_main_entity) = make_value(
		is_value_code,
		make_code(
			complete_list_of_entities,
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
	debug(2, "gfc2pips_namespace", "main entity creation finished\n");

	//can it be removed ?
	//InitAreas();//Syntax


	//get symbols with value, data and explicit-save
	//sym->value is an expression to build the save
	//create data $var /$val/
	//save si explicit save, rien sinon
	instruction data_inst = instruction_undefined;

	//// declare code
	debug(2, "gfc2pips_namespace", "dumping code ...\n");
	//make a 3rd function code2instruction to have a TOP-LEVEL dumping with each and every data instantiation
	//data_inst: TK_DATA ldatavar TK_SLASH ldataval TK_SLASH
	//	MakeDataStatement($2, $4);
	icf = gfc2pips_code2instruction__TOP(ns,ns->code);
	message_assert("Dumping instruction failed\n",icf!=instruction_undefined);


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

	/*gfc2pips_comments com;
	gfc_code *current_code_linked_to_comments=NULL;
		fprintf(stderr,"gfc2pips_comments_stack: %d\n",gfc2pips_comments_stack);
		if( com=gfc2pips_pop_comment() ){
			while(1){
				fprintf(stderr,"comment: %d ",com);
				if(com){
					fprintf(stderr,"linked %s\n",com->done?"yes":"no");
					current_code_linked_to_comments = com->num;
					do{
						fprintf(stderr,"\t %d > %s\n", com->num, com->s );
						com=gfc2pips_pop_comment();
					}while(
						com
						&& current_code_linked_to_comments == com->num
					);

				}else{
					break;
				}
				fprintf(stderr,"\n");
			}
			fprintf(stderr,"\n");
		}

		fprintf(stderr,"gfc2pips_list_of_declared_code: %d\n",gfc2pips_list_of_declared_code);
		while( gfc2pips_list_of_declared_code ){
			if(gfc2pips_list_of_declared_code->car.e){
				fprintf(stderr,"gfc_code: %d %d %d %d %d\n",
					gfc2pips_list_of_declared_code->car.e,
					((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc,
					*((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc,
					*((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.lb->line,
					((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.lb->location
				);
				fprintf(stderr,"%s\n",
					gfc2pips_gfc_char_t2string2( ((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc )
				);
				fprintf(stderr,"\n");
			}
			POP(gfc2pips_list_of_declared_code);
		}
		fprintf(stderr,"\n");
*/

}










/*
 * Retrieve the names of every argument of the function, if exists
 *///pb here in the creation of the parameters
newgen_list gfc2pips_args(gfc_namespace* ns){
	gfc_symtree * current = NULL;
	gfc_formal_arglist *formal;
	newgen_list args_list = NULL,args_list_p = NULL;
	entity e = entity_undefined;
	//return NULL;//en attendant de pouvoir créer correctement les parameters

	if( ns && ns->proc_name ){

		current = getSymtreeByName( ns->proc_name->name, ns->sym_root );
		if( current && current->n.sym ){
			if (current->n.sym->formal){
				formal = current->n.sym->formal;
				if(formal){
					e = gfc2pips_symbol2entity(
						getSymtreeByName(
							str2upper(formal->sym->name),
							ns->sym_root
						)->n.sym
					);
					//e = FindOrCreateEntity(CurrentPackage,formal->sym->name);
					//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
					args_list = args_list_p = CONS(ENTITY, e, NULL );//fprintf(stderr,"%s\n",formal->sym->name);
					formal = formal->next;
					while(formal){
						if (formal->sym != NULL){
							e = gfc2pips_symbol2entity(
								getSymtreeByName(
									str2upper(formal->sym->name),
									ns->sym_root
								)->n.sym
							);
							//fprintf(stderr,"gfc2pips_args: arg founded %s\n",formal->sym->name);fflush(stderr);
							//e = FindOrCreateEntity(CurrentPackage,formal->sym->name);
							//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
							CDR(args_list) = CONS(ENTITY, e, NULL );
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

  debug(10,"getSymtreeByName","Lookin for the symtree called: %s(%d) %s(%d)\n", name, strlen(name), st->name, strlen(st->name) );

  if(
		strcmp(str2upper(st->name),str2upper(name))==0
		/*&&(
				st->n.sym->attr.subroutine
			|| st->n.sym->attr.function
		)*/
  ){
	  debug(10,"getSymtreeByName","symbol %s founded\n",name);
	  return st;
  }
  return_value = getSymtreeByName (name, st->left  );

  if( return_value != NULL) return return_value;
  return_value = getSymtreeByName (name, st->right  );
  if(return_value != NULL) return return_value;

  //fprintf(stderr,"NULL\n");
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
	variables_p;//balancer la suite dans une fonction à part afin de pouvoir la réutiliser pour les calls
	newgen_list arguments,arguments_p;
	arguments = arguments_p = gfc2pips_args(ns);
	while(variables_p){
		type Type = type_undefined;
		//create entities here
		gfc_symtree *current_symtree = (gfc_symtree*)variables_p->car.e ;
		if(current_symtree && current_symtree->n.sym){
			debug(3, "gfc2pips_vars", "translation of entity gfc2pips start\n");
			if( current_symtree->n.sym->attr.in_common ){
				debug(4, "gfc2pips_vars", " %s is in a common\r\n", str2upper(current_symtree->name) );
				POP(variables_p);
				continue;
			}
			debug(4, "gfc2pips_vars", " symbol: %s size: %d\r\n", str2upper(current_symtree->name), current_symtree->n.sym->ts.kind );
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
				debug(5, "gfc2pips_vars","the symbol is a string\n");
				Value = make_value_constant(
					MakeConstantLitteral()//MakeConstant(current_symtree->n.sym->value->value.character.string,is_basic_string)
				);
			}else{
				debug(5, "gfc2pips_vars","the symbol is a constant\n");
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
			int i,j=0;
			//newgen_list list_of_dimensions = gfc2pips_get_list_of_dimensions(current_symtree);
			//si allocatable alors on fait qqch d'un peu spécial

			/*TK_STRUCT id_or_typename TK_LBRACE
			                        {
						  code c = make_code(NIL,$2,sequence_undefined,NIL);
						  stack_push((char *) c, StructNameStack);
						}
			    struct_decl_list TK_RBRACE
			                        {
						  /* Create the struct entity
						  entity ent = MakeDerivedEntity($2,$5,is_external,is_type_struct);
						  /* Specify the type of the variable that follows this declaration specifier

						  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
						  /* Take from $5 the struct/union entities
						  list le = TakeDerivedEntities($5);
						  $$ = gen_nconc(le,CONS(ENTITY,ent,NIL));
						  c_parser_context_type(ycontext) = make_type_variable(v);
						  stack_pop(StructNameStack);
						}*/
			//we look into the list of arguments to know if the entity is in and thus the offset in the stack
			i=0;j=1;
			arguments_p = arguments;
			while(arguments_p){
				//fprintf(stderr,"%s %s\n",entity_local_name((entity)arguments_p->car.e),current_symtree->name);
				if(strcmp(str2upper(entity_local_name( (entity)arguments_p->car.e )),str2upper(current_symtree->name))==0){
					i=j;
					break;
				}
				j++;
				POP(arguments_p);
			}
			//fprintf(stderr,"%s %d\n",current_symtree->name,i);

			variables = CONS(ENTITY, FindOrCreateEntity(CurrentPackage, current_symtree->name), variables);
			entity_type((entity)variables->car.e) = Type;
			entity_initial((entity)variables->car.e) = Value;//make_value(is_value_code, make_code(NULL, strdup(""), make_sequence(NIL),NIL));
			if(current_symtree->n.sym->attr.dummy){
				//we have a formal parameter (argument of the function/subroutine)
				entity_storage((entity)variables->car.e) = make_storage_formal(
					make_formal(
						gfc2pips_main_entity,
						i
					)
				);
			}else if( current_symtree->n.sym->attr.flavor==FL_PARAMETER ){//fprintf(stderr,"ROM storage for %s\n",current_symtree->n.sym->name);
				//we have a parameter, we rewrite some attributes of the entity
				entity_type((entity)variables->car.e) = make_type(is_type_functional, make_functional(NIL, entity_type((entity)variables->car.e)));
				entity_initial((entity)variables->car.e) = MakeValueSymbolic(gfc2pips_expr2expression(current_symtree->n.sym->value));
				entity_storage((entity)variables->car.e) = MakeStorageRom();
			}else{
				//we have a variable
				ram _r_ = make_ram(
					get_current_module_entity(),
					current_symtree->n.sym->value?DynamicArea:StackArea,
					UNKNOWN_RAM_OFFSET,
					NULL
				);
				entity_storage( (entity)variables->car.e ) = make_storage( is_storage_ram, _r_ );
			}

			//if(Type!=type_undefined){
				//variable_dimensions(type_variable(entity_type( (entity)variables->car.e ))) = gfc2pips_get_list_of_dimensions(current_symtree);
				/*if(current_symtree->n.sym->attr.pointer){
					basic b = make_basic(is_basic_pointer, Type);
					type newType = make_type(is_type_variable, make_variable(b, NIL, NIL));
					entity_type((entity)variables->car.e) = newType;
				}*/
			//}
			debug(3, "gfc2pips_vars", "translation of entity gfc2pips end\n");
		}else{
			variables_p->car.e = NULL;
		}
		POP(variables_p);
	}
	return variables;
}

newgen_list gfc2pips_get_data_vars(gfc_namespace *ns){
	return getSymbolBy(ns,ns->sym_root,gfc2pips_test_data);
}


newgen_list gfc2pips_get_list_of_dimensions(gfc_symtree *st){
	if(st){
		return gfc2pips_get_list_of_dimensions2(st->n.sym);
	}else{
		return NULL;
	}
}
newgen_list gfc2pips_get_list_of_dimensions2(gfc_symbol *s){
	newgen_list list_of_dimensions = NULL;
	int i=0,j=0;
	if( s && s->attr.dimension ){
		gfc_array_spec *as = s->as;
		const char *c;
		debug(4,"gfc2pips_get_list_of_dimensions2","%s is an array\n",s->name);
		if ( as!=NULL && as->rank != 0){

			/*fprintf(stderr,"toto %d\n",as);
			fprintf(stderr,"toto %d\n",as->upper);
			fprintf(stderr,"toto %d\n",as->upper[i]);
			fprintf(stderr,"toto %d\n",as->upper[i]->value);
			fprintf(stderr,"toto %d\n",as->upper[i]->value.integer);
			fprintf(stderr,"toto %d\n",as->upper[i]->value.integer[0]);
			fprintf(stderr,"toto %d\n",as->upper[i]->value.integer[0]._mp_size);
			fprintf(stderr,"toto %d\n",as->upper[i]->value.integer[0]._mp_alloc);
			fprintf(stderr,"toto %d\n",as->upper[i]->value.integer[0]._mp_d);
			fprintf(stderr,"\n");
			*/
			/*
			 * show_expr (as->upper[i]);
			 * mpz_out_str (stderr, 10, p->value.integer);
			 * if (p->ts.kind != gfc_default_integer_kind)
			 *   fprintf (dumpfile, "_%d", p->ts.kind);
			 */

			//le switch nous intéresse-t-il ? => oui très important il faut traduire chaque valeur de dimension du tableau en "truc" approprié newgen/PIPS
			switch (as->type){
				case AS_EXPLICIT:
					c = strdup("AS_EXPLICIT");
					//create the list of dimensions
					i = as->rank-1;
					do{
						list_of_dimensions = gen_int_cons(
							make_dimension(
									gfc2pips_expr2expression(as->lower[i]),
									gfc2pips_expr2expression(as->upper[i])
							),
							list_of_dimensions
						);
					}while(--i >= j);
				break;
				case AS_DEFERRED:
					c = strdup("AS_DEFERRED");
					i = as->rank-1;
					do{
						list_of_dimensions = gen_int_cons(
								make_dimension(
									MakeIntegerConstantExpression("1"),
									MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
								),
							list_of_dimensions
						);
					}while(--i >= j);
				break;
				case AS_ASSUMED_SIZE:  c = strdup("AS_ASSUMED_SIZE");  break;
				case AS_ASSUMED_SHAPE: c = strdup("AS_ASSUMED_SHAPE"); break;
				default:
				  gfc_internal_error ("show_array_spec(): Unhandled array shape "
							  "type.");
			}
		}
		debug(4,"gfc2pips_get_list_of_dimensions2","%d dimensions detected for %s\n",gen_length(list_of_dimensions),s->name);
	}

	return list_of_dimensions;
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
//get variables who are not implicit or are needed to be declared for data statements
bool gfc2pips_test_variable(gfc_namespace* ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return ( st->n.sym->attr.flavor == FL_VARIABLE || st->n.sym->attr.flavor == FL_PARAMETER )
		&& (
			!st->n.sym->attr.implicit_type
			|| st->n.sym->value//very important
		)
		//&& !st->n.sym->attr.in_common
		&& !st->n.sym->attr.pointer ;// && !st->n.sym->attr.dummy;
}
bool gfc2pips_test_variable2(gfc_namespace* ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && !st->n.sym->attr.dummy;
}
bool gfc2pips_test_allocatable(gfc_namespace *ns, gfc_symtree *st){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.allocatable;
}
bool gfc2pips_test_arg(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && st->n.sym->attr.dummy;
}
bool gfc2pips_test_data(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->value && st->n.sym->attr.flavor != FL_PARAMETER;
}
bool gfc2pips_get_commons(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* __attribute__ ((__unused__)) st ){
	return true;
}
bool gfc2pips_test_dimensions(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.dimension;
}

entity gfc2pips_symbol2entity(gfc_symbol* s){
	entity e = FindOrCreateEntity(CurrentPackage,s->name);
	if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
	if(entity_type(e)==type_undefined) entity_type(e) = gfc2pips_symbol2type(s);
	//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
	return e;
}
entity gfc2pips_char2entity(char* package, char* s){
	entity e = FindOrCreateEntity(package, str2upper(strdup(s)));
	if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
	if(entity_type(e)==type_undefined) entity_type(e) = make_type_unknown();
	return e;
}









/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */

dimension gfc2pips_int2dimension(int n){
	return make_dimension(MakeIntegerConstantExpression("1"),gfc2pips_int2expression(n));
}

expression gfc2pips_int2expression(int n){
	//return int_expr(n);
	if(n<0){
		return MakeFortranUnaryCall(CreateIntrinsic("--"), gfc2pips_int2expression(-n));
	}else{
		return entity_to_expression(gfc2pips_int_const2entity(n));
	}
}
expression gfc2pips_real2expression(double r){
	return entity_to_expression(gfc2pips_real2entity(r));
}
expression gfc2pips_logical2expression(bool b){
	//return int_expr(b!=false);
	return entity_to_expression(gfc2pips_logical2entity(b));
}
expression gfc2pips_string2expression(char* s){
	return entity_to_expression(FindOrCreateEntity(CurrentPackage,s));
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
	//return make_loop_label(n,concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,NULL));
	char str[60];
	sprintf(str,"%s%s%s%d",TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);//fprintf(stderr,"new label: %s %s %s %s %d\n",str,TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);
	return make_label(str);
}
//on a un grave problème de traduction des valeurs réelles
//code  => réel en valeur interne
//16.53 => 16.530001
//16.56 => 16.559999
entity gfc2pips_real2entity(double r){
	char str[30];
	sprintf(str,"%f",r);//fprintf(stderr,"copy of the entity name(real) %s\n",str);
	return MakeConstant(str, is_basic_float);
}

entity gfc2pips_logical2entity(bool b){
	return MakeConstant(b?"1":"0",is_basic_logical);
}
//escape all ' in the string
char* gfc2pips_gfc_char_t2string(gfc_char_t *c, int nb){
	if(nb){
		gfc_char_t *p=c;
		while(*p){
			if(*p=='\'')nb++;
			p++;
		}
		return gfc2pips_gfc_char_t2string_(c, nb);
	}else{
		return NULL;
	}
}
char* gfc2pips_gfc_char_t2string2(gfc_char_t *c){
	gfc_char_t *p=c;
	char *s = NULL;
	int nb,i;
	//fprintf(stderr,"try gfc2pips_gfc_char_t2string2");

	nb=0;
	while( p && *p && nb< 132){
		nb++;
		p++;
	}
	p=c;
	//fprintf(stderr,"continue gfc2pips_gfc_char_t2string2 %d\n",nb);

	if(nb){

		s = malloc(sizeof(char)*(nb+1));
		i=0;
		while(i<nb && *p){
			//fprintf(stderr,"i:%d *p:(%d=%c)\n",i,*p,*p);
			s[i++]=*p;
			p++;
		}
		s[i]='\0';
		//fprintf(stderr,"end gfc2pips_gfc_char_t2string2");
		return s;
	}else{
		return NULL;
	}
}

//very important question: do we have to put the ' before and after the string ? this function add them
char* gfc2pips_gfc_char_t2string_(gfc_char_t *c, int nb){
	char *s = malloc(sizeof(char)*(nb+1+2));
	gfc_char_t *p=c;
	int i=1;
	s[0]='\'';
	while(i<=nb){
		if(*p=='\''){
			s[i++]='\'';
		}
		s[i++]=*p;
		p++;

	}
	s[i++]='\'';
	s[i]='\0';
	return s;
}

/*
gfc_traverse_symtree (ns->common_root, show_common);
static void
show_common (gfc_symtree *st){
	gfc_symbol *s;
	show_indent ();
	fprintf (dumpfile, "common: /%s/ ", st->name);
	s = st->n.common->head;
	while (s){
		fprintf (dumpfile, "%s", s->name);
		s = s->common_next;
		if (s)
		fputs (", ", dumpfile);
	}
}
<entity:common> NameToCommon(global_name)
AddVariableToCommon(<entity:common>,<entity:variable of the list>)
 *
 */


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

	enum basic_utype ut;
	switch(s->ts.type){
		case BT_INTEGER:	ut = is_basic_int;		break;
		case BT_REAL:		ut = is_basic_float;	break;
		case BT_COMPLEX:	ut = is_basic_complex;	break;
		case BT_LOGICAL:	ut = is_basic_logical;	break;
		case BT_CHARACTER:	ut = is_basic_string;	break;
		case BT_UNKNOWN:
			return make_type_unknown();
		break;
		case BT_DERIVED:
		case BT_PROCEDURE:
		case BT_HOLLERITH:
		case BT_VOID:
		default:
			pips_error("gfc2pips_symbol2type","An error occured in the type to type translation: impossible to translate the symbol.\n");
			return type_undefined;
			//return make_type_unknown();
	}
	debug(5,"gfc2pips_type2type","Basic type is : %d\n",ut);
	//variable_dimensions(type_variable(entity_type( (entity)CAR(variables) ))) = gfc2pips_get_list_of_dimensions(current_symtree)
	if(ut!=is_basic_string){
		return MakeTypeVariable(
			make_basic(
				ut,
				(void*) (gfc2pips_symbol2size(s))// * gfc2pips_symbol2sizeArray(s))
			),
			gfc2pips_get_list_of_dimensions2(s)
		);
	}else{
		//fprintf(stderr,"STRING !!! %d\n",s->value);
		return MakeTypeVariable(
			make_basic(
				ut,
				(void*) make_value(
					is_value_constant,
					//don't use litteral, it's a trap !
					make_constant_int(
						gfc2pips_symbol2size(s)
					)//it is here we have to specify the length of the character symbol
				)
			),
			NIL//gfc2pips_get_list_of_dimensions2(s)
		);
	}
	debug( 5, "gfc2pips_type2type", "WARNING: no type\n" );
	return type_undefined;
	//return make_type_unknown();
}

int gfc2pips_symbol2size(gfc_symbol *s){
	if(
		s->ts.type==BT_CHARACTER
		&& s->ts.cl
		&& s->ts.cl->length
	){
		debug(9,"gfc2pips_symbol2size","size of %s: %d\n",s->name,mpz_get_si(s->ts.cl->length->value.integer));
		return mpz_get_si(s->ts.cl->length->value.integer);
	}else{
		debug(9,"gfc2pips_symbol2size","size of %s: %d\n",s->name,s->ts.kind);
		return s->ts.kind;
	}
}
int gfc2pips_symbol2sizeArray(gfc_symbol *s){
	int retour = 1;
	newgen_list list_of_dimensions = NULL;
	int i=0,j=0;
	if( s && s->attr.dimension ){
		gfc_array_spec *as = s->as;
		const char *c;
		if ( as!=NULL && as->rank != 0 && as->type == AS_EXPLICIT){
			i = as->rank-1;
			do{
				retour *= gfc2pips_expr2int(as->upper[i]) - gfc2pips_expr2int(as->lower[i]) +1;
			}while(--i >= j);
		}
	}
	debug(9,"gfc2pips_symbol2sizeArray","size of %s: %d\n",s->name,retour);
	return retour;
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

//this is to know if we have to add a continue statement just after the loop statement for (exit/break)
bool gfc2pips_last_statement_is_loop = false;

/*
 * Declaration of instructions
 */

instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c){
	newgen_list list_of_data_symbol,list_of_data_symbol_p;
	list_of_data_symbol_p = list_of_data_symbol = gfc2pips_get_data_vars(ns);

	//create a sequence and put everything into it ? is it right ?
	newgen_list list_of_instructions,list_of_instructions_p;
	list_of_instructions_p = list_of_instructions= NULL;

	instruction i = instruction_undefined;


	//dump DATA
	//create a list of statements and dump them in one go
	//test for each if there is an explicit save statement
	//fprintf(stderr,"nb of data statements: %d\n",gen_length(list_of_data_symbol_p));
	if(list_of_data_symbol_p){
		//int add,min,is,tr,at,e,ur;
		do{
			instruction ins = gfc2pips_symbol2data_instruction(
				( (gfc_symtree*)list_of_data_symbol_p->car.e )->n.sym
			);
			if(ins!=instruction_undefined){
				//fprintf(stderr,"Got a data !\n");
				//PIPS doesn't tolerate comments here
				//we should shift endlessly the comments number to the first "real" statement
				//string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")

				newgen_list lst = CONS(STATEMENT, make_statement(
					entity_empty_label(),
					STATEMENT_NUMBER_UNDEFINED,
					STATEMENT_ORDERING_UNDEFINED,
					//comments,
					empty_comments,
					ins,
					NULL,
					NULL,
					empty_extensions ()
				), NULL);
				code mc = entity_code(get_current_module_entity());
				sequence_statements(code_initializations(mc)) = gen_nconc(sequence_statements(code_initializations(mc)), lst);

				/*if(list_of_instructions){
					CDR(list_of_instructions) = lst;
					list_of_instructions = CDR(list_of_instructions);
				}else{
					list_of_instructions_p = list_of_instructions = lst;
				}*/
			}else{
				break;
			}
			POP(list_of_data_symbol_p);
		}while( list_of_data_symbol_p );
	}

	//dump equivalence statements
	/*gfc_equiv * eq, *eq2;
	for (eq = ns->equiv; eq; eq = eq->next){
		//show_indent ();
		//fputs ("Equivalence: ", dumpfile);

		//gfc2pips_handleEquiv(eq);
		eq2 = eq;
		while (eq2){
			fprintf(stderr,"eq: %d %s %d ",eq2->expr, eq2->module, eq2->used);
			//show_expr (eq2->expr);
			eq2 = eq2->eq;
			if(eq2)fputs (", ", stderr);
			else fputs ("\n", stderr);
		}
	}*/
	//StoreEquivChain(<atom>)

	if( !c ){
		//fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
		return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
	}

	//dump other
	//we know we have at least one instruction, otherwise we would have returned an empty list of statements
	do{
		i = gfc2pips_code2instruction_(c);
		if(i!=instruction_undefined){
			//string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
			newgen_list lst = CONS(STATEMENT, make_statement(
				gfc2pips_code2get_label(c),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				//comments,
				empty_comments,
				i,
				NULL,
				NULL,
				empty_extensions ()
			), NULL);
			//unlike the classical method, we don't know if we have had a first statement (data inst)
			if(list_of_instructions){
				CDR(list_of_instructions) = lst;
				list_of_instructions = CDR(list_of_instructions);
			}else{
				list_of_instructions_p = list_of_instructions = lst;
			}
		}
		c=c->next;
	}while( i==instruction_undefined && c);

	//compter le nombre de statements décalés
	unsigned long first_comment_num  = gfc2pips_get_num_of_gfc_code(c);
	unsigned long last_code_num = gen_length(gfc2pips_list_of_declared_code);
	unsigned long curr_comment_num=first_comment_num;
	//for( ; curr_comment_num<first_comment_num ; curr_comment_num++ ) gfc2pips_replace_comments_num( curr_comment_num, first_comment_num );
	for( ; curr_comment_num<=last_code_num ; curr_comment_num++ ) gfc2pips_replace_comments_num( curr_comment_num, curr_comment_num +1 - first_comment_num );

	gfc2pips_assign_gfc_code_to_num_comments(c,0 );
	/*
	unsigned long first_comment_num  = gfc2pips_get_num_of_gfc_code(c);
	unsigned long curr_comment_num=0;
	for( ; curr_comment_num<first_comment_num ; curr_comment_num++ ){
		gfc2pips_replace_comments_num( curr_comment_num, first_comment_num );
	}
	gfc2pips_assign_gfc_code_to_num_comments(c,first_comment_num );
	*/

	for( ; c ; c=c->next ){
		statement s = statement_undefined;
		i = gfc2pips_code2instruction_(c);
		if(i!=instruction_undefined){
			string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
			s = make_statement(
				gfc2pips_code2get_label(c),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				comments,
				//empty_comments,
				i,
				NULL,
				NULL,
				empty_extensions ()
			);
			//if(s!=statement_undefined){
				CDR(list_of_instructions) = CONS(STATEMENT, s, NIL);
				list_of_instructions = CDR(list_of_instructions);
			//}
		}
	}
	if(list_of_instructions_p){
		return make_instruction_block(list_of_instructions_p);//make a sequence <=> make_instruction_sequence(make_sequence(list_of_instructions));
	}else{
		fprintf(stderr,"Warning ! no instruction dumped => very bad\n");
		return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
	}
}
/*
 * Build an instruction sequence
 */
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence){
	force_sequence = true;
	if(!c){
		if(force_sequence){
			fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
			return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
		}else{
			fprintf(stderr,"Undefined code\n");
			return make_instruction_block(NULL);
		}
	}
	//No block, only one instruction
	if(!c->next && !force_sequence )return gfc2pips_code2instruction_(c);

	//create a sequence and put everything into it ? is it right ?
	newgen_list list_of_instructions,list_of_instructions_p;
	list_of_instructions= NULL;

	//entity l = gfc2pips_code2get_label(c);
	instruction i = instruction_undefined;
	do{
		i = gfc2pips_code2instruction_(c);
		if(i!=instruction_undefined){
			string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
			list_of_instructions_p = list_of_instructions = CONS(STATEMENT, make_statement(
				gfc2pips_code2get_label(c),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				comments,
				//empty_comments,
				i,
				NIL,
				NULL,
				empty_extensions ()
			), list_of_instructions);
		}
		c=c->next;
	}while( i==instruction_undefined && c);

	//statement_label((statement)list_of_instructions->car.e) = gfc2pips_code2get_label(c);

	for( ; c ; c=c->next ){
		statement s = statement_undefined;
		//l = gfc2pips_code2get_label(c);
		//on lie l'instruction suivante à la courante
		//fprintf(stderr,"Dump the following instructions\n");
		//CONS(STATEMENT,instruction_to_statement(gfc2pips_code2instruction_(c)),list_of_instructions);
		int curr_label_num = gfc2pips_last_created_label;
		i = gfc2pips_code2instruction_(c);
		//si dernière boucle == c alors on doit ajouter un statement continue sur le label <curr_label_num>
		if(i!=instruction_undefined){
			string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
			s = make_statement(
				gfc2pips_code2get_label(c),
				STATEMENT_NUMBER_UNDEFINED,
				STATEMENT_ORDERING_UNDEFINED,
				comments,
				//empty_comments,
				i,
				NULL,
				NULL,
				empty_extensions ()
			);
			if(s!=statement_undefined){
				CDR(list_of_instructions) = CONS(STATEMENT, s, NIL);
				list_of_instructions = CDR(list_of_instructions);
			}
			if(gfc2pips_get_last_loop()==c){
				s = make_continue_statement(gfc2pips_int2label(curr_label_num-1));
				CDR(list_of_instructions) = CONS(STATEMENT, s, NIL);
				list_of_instructions = CDR(list_of_instructions);
			}
		}
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
		case EXEC_NOP://an instruction without anything => continue statement
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

*/
		case EXEC_POINTER_ASSIGN:{
			debug(5,"gfc2pips_code2instruction","Translation of assign pointer\n");
			newgen_list list_of_arguments = CONS(EXPRESSION,gfc2pips_expr2expression(c->expr2),NIL);


			entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, ADDRESS_OF_OPERATOR_NAME);
			entity_initial(e) = make_value(is_value_intrinsic, e );
			//entity_initial(e) = make_value(is_value_constant,make_constant(is_constant_int, (void *) CurrentTypeSize));
			entity_type(e) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,list_of_arguments);
			expression ex = make_expression(
				make_syntax(is_syntax_call,call_),
				normalized_undefined
			);
			return make_assign_instruction(
				gfc2pips_expr2expression(c->expr),
				ex
			);
		}break;
		case EXEC_GOTO:{
			debug(5,"gfc2pips_code2instruction","Translation of GOTO\n");
			instruction i = make_instruction(is_instruction_goto,
				make_continue_statement(
					gfc2pips_code2get_label2(c)
				)
			);
			return i;
		}break;

		case EXEC_CALL:
		case EXEC_ASSIGN_CALL:{
			debug(5,"gfc2pips_code2instruction","Translation of CALL\n");
			entity called_function = entity_undefined;
			char * str = NULL;
			if(c->resolved_sym){
				str = c->resolved_sym->name;
			}else if(c->symtree){
				str = c->symtree->name;
			}else{
				//erreur
				return instruction_undefined;
			}
			if(strncmp("_gfortran_",str,strlen("_gfortran_"))==0){
				str = str2upper(strdup("exit"));
			}
			called_function = FindOrCreateEntity(CurrentPackage, str);
			newgen_list list_of_arguments = NULL,list_of_arguments_p = NULL;
			gfc_actual_arglist *act=c->ext.actual;

			while(act){
				expression ex = gfc2pips_expr2expression(act->expr);

				if(ex!=entity_undefined){

					if(list_of_arguments_p){
						CDR(list_of_arguments_p) = CONS(EXPRESSION,ex,NIL);
						list_of_arguments_p = CDR(list_of_arguments_p);
					}else{
						list_of_arguments_p = CONS(EXPRESSION,ex,NIL);
					}
				}
				if(list_of_arguments==NULL)list_of_arguments = list_of_arguments_p;

				act = act->next;
			}
			entity_initial(called_function) = make_value(is_value_intrinsic, called_function );
			//entity_initial(e) = make_value(is_value_constant,make_constant(is_constant_int, (void *) CurrentTypeSize));
			entity_type(called_function) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(called_function, list_of_arguments);

/*
	    $$ = MakeCallInst(
			<entity: called function>,
			parameters
		);
		reset_alternate_returns();
 */
			return make_instruction(is_instruction_call, call_);
		}break;
/*		if (c->resolved_sym)
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
		case EXEC_RETURN:{//we shouldn't even dump that for main entities
			debug(5,"gfc2pips_code2instruction","Translation of return\n");
			return instruction_undefined;
			expression e = expression_undefined;
			if(c->expr!=NULL){
				//traitement de la variable de retour
			}
			return MakeReturn(e);//Syntax !
		}

		case EXEC_PAUSE:
			debug(5,"gfc2pips_code2instruction","Translation of PAUSE\n");
			return make_instruction(
				is_instruction_call,
				make_call(
					CreateIntrinsic("PAUSE"),
					c->expr?CONS(EXPRESSION, gfc2pips_expr2expression(c->expr), NULL):NULL
				)
			);
		break;

	    case EXEC_STOP:
	    	debug(5,"gfc2pips_code2instruction","Translation of STOP\n");
	    	return make_instruction(
				is_instruction_call,
				make_call(
					CreateIntrinsic("STOP"),
					c->expr?CONS(EXPRESSION, gfc2pips_expr2expression(c->expr), NULL):NULL
				)
			);
		break;

	    case EXEC_ARITHMETIC_IF:{
	    	debug(5,"gfc2pips_code2instruction","Translation of ARITHMETIC IF\n");
	    	expression e = gfc2pips_expr2expression(c->expr);
	    	expression e1 = MakeBinaryCall(CreateIntrinsic(".LT."), e, MakeIntegerConstantExpression("0"));
	    	expression e2 = MakeBinaryCall(CreateIntrinsic(".EQ."), e, MakeIntegerConstantExpression("0"));
	    	statement s1 = instruction_to_statement(
				make_instruction(
					is_instruction_goto,
					make_continue_statement(
						gfc2pips_code2get_label2(c)
					)
				)
			);
	    	statement s2 = instruction_to_statement(
				make_instruction(
					is_instruction_goto,
					make_continue_statement(
						gfc2pips_code2get_label3(c)
					)
				)
			);
	    	statement s3 = instruction_to_statement(
				make_instruction(
					is_instruction_goto,
					make_continue_statement(
						gfc2pips_code2get_label4(c)
					)
				)
			);
	    	statement s = instruction_to_statement(
	    		make_instruction(
	    			is_instruction_test,
	    			make_test(e2,s2,s3)
	    		)
	    	);
	    	//statement_number(s) = get_statement_number();
	    	return make_instruction(is_instruction_test, make_test(e1,s1,s));

	    }break;


		case EXEC_IF:{
			debug(5,"gfc2pips_code2instruction","Translation of IF\n");
			if(!c->block){
				return make_instruction_block(NULL);
			}
			//next est le code pointé par la sortie IF
			//block est le code pointé par la sortie ELSE
			gfc_code* d = c->block;//fprintf(stderr,"%d %d %d %d\n",d, d?d->expr:d,c,c->expr);
			//gfc2pips_code2get_label(c);gfc2pips_code2get_label(c->next);gfc2pips_code2get_label(d);gfc2pips_code2get_label(d->next);

			if(!d->expr){
				fprintf(stderr,"No condition ???\n");
				//we are at the last ELSE statement for an ELSE IF
				if(d->next){
					return gfc2pips_code2instruction(d->next,true);
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
				s_if = instruction_to_statement(gfc2pips_code2instruction(d->next,false));
				statement_label(s_if) = gfc2pips_code2get_label(d->next);
				//ELSE + ?
				if(d->block){
					//s_else = instruction_to_statement(gfc2pips_code2instruction(d->block,false));
					//ELSE IF
					if(d->block->expr){
						//fprintf(stderr,"d->block->expr %d\n",d->block->expr);
						s_else = instruction_to_statement(gfc2pips_code2instruction_(d));
						statement_label(s_else) = gfc2pips_code2get_label(d);
					//ELSE
					}else{
						//fprintf(stderr,"d->block %d\n",d->block);
						s_else = instruction_to_statement(gfc2pips_code2instruction(d->block->next,false));//no condition therefore we are in the last ELSE statement
						statement_label(s_else) = gfc2pips_code2get_label(d->block->next);
					}
				}
			}else{
				return make_instruction_block(NULL);
			}
			if( s_if==statement_undefined || statement_instruction(s_if)==instruction_undefined ){
				s_if = make_empty_block_statement();
				statement_label(s_if) = gfc2pips_code2get_label(d->next);
			}
			if( s_else==statement_undefined || statement_instruction(s_else)==instruction_undefined ){
				s_else = make_empty_block_statement();
				if(d && d->block){
					if(d->block->expr){
						statement_label(s_else) = gfc2pips_code2get_label(d);
					}else{
						statement_label(s_else) = gfc2pips_code2get_label(d->block->next);
					}
				}
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
	    	gfc2pips_push_loop(c);
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,false));

	    	//add to s a continue statement at the end to make cycle/continue statements
	    	newgen_list list_of_instructions = sequence_statements(instruction_sequence(statement_instruction(s)));
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	list_of_instructions = gen_cons(make_continue_statement(gfc2pips_int2label(gfc2pips_last_created_label)),list_of_instructions);
	    	gfc2pips_last_created_label-=gfc2pips_last_created_label_step;
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	sequence_statements(instruction_sequence(statement_instruction(s))) = list_of_instructions;

	    	statement_label(s)=gfc2pips_code2get_label(c->block->next);
	    	pips_loop w = make_loop(
	    		gfc2pips_expr2entity(c->ext.iterator->var),//variable incremented
	    		make_range(
    				gfc2pips_expr2expression(c->ext.iterator->start),
    				gfc2pips_expr2expression(c->ext.iterator->end),
    				gfc2pips_expr2expression(c->ext.iterator->step)///ajouter des tests afin de pouvoir créer des do avec un incrément négatif
	    		),//lower, upper, increment
	    		s,
	    		gfc2pips_code2get_label2(c),
	    		make_execution_sequential(),//sequential/parallel //beware gfc parameters to say it is a parallel or a sequential loop
	    		NULL
	    	);
	    	gfc2pips_pop_loop();
	    	return make_instruction_loop(w);

	    }break;

	    case EXEC_DO_WHILE:{
	    	debug(5,"gfc2pips_code2instruction","Translation of DO WHILE\n");
	    	gfc2pips_push_loop(c);
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,false));

	    	//add to s a continue statement at the end to make cycle/continue statements
	    	newgen_list list_of_instructions = sequence_statements(instruction_sequence(statement_instruction(s)));
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	list_of_instructions = gen_cons(make_continue_statement(gfc2pips_int2label(gfc2pips_last_created_label)),list_of_instructions);
	    	gfc2pips_last_created_label-=gfc2pips_last_created_label_step;
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	sequence_statements(instruction_sequence(statement_instruction(s))) = list_of_instructions;

	    	statement_label(s) = gfc2pips_code2get_label(c->block->next);
	    	whileloop w = make_whileloop(
	    		gfc2pips_expr2expression(c->expr),
	    		s,
	    		gfc2pips_code2get_label2(c),
	    		make_evaluation_before()
	    	);
	    	gfc2pips_pop_loop();
	    	return make_instruction_whileloop(w);
	    }break;

	    case EXEC_CYCLE:{
	    	debug(5,"gfc2pips_code2instruction","Translation of CYCLE\n");
	    	gfc_code* loop_c = gfc2pips_get_last_loop();
	    	entity label = entity_undefined;
	    	if(true){
	    		//label = gfc2pips_code2get_label2(loop_c->block->next->label);
	    		label = gfc2pips_int2label(gfc2pips_last_created_label);
	    	}else{
				int num = gfc2pips_get_num_of_gfc_code(loop_c->block->next);
				fprintf(stderr,"%d\n",gen_length(gfc2pips_list_of_loops));
				string lab;
				asprintf(&lab,"%s%s%s1%d",CurrentPackage, MODULE_SEP_STRING,LABEL_PREFIX,num);
				fprintf(stderr,"%s",lab);
				label = make_label(lab);
				free(lab);
	    	}
			instruction i = make_instruction(is_instruction_goto,
				make_continue_statement(
						label
				)
			);
			return i;
		}break;
	    case EXEC_EXIT:{
	    	debug(5,"gfc2pips_code2instruction","Translation of EXIT\n");
	    	gfc_code* loop_c = gfc2pips_get_last_loop();
	    	entity label = entity_undefined;
	    	if(true){//loop_c->block->next->label){
	    		//label = gfc2pips_code2get_label2(loop_c->block->next->label);
	    		label = gfc2pips_int2label(gfc2pips_last_created_label-1);
	    	}else{
				int num = gfc2pips_get_num_of_gfc_code(loop_c->block->next);
				fprintf(stderr,"%d\n",gen_length(gfc2pips_list_of_loops));
				string lab;
				asprintf(&lab,"%s%s%s1%d",CurrentPackage, MODULE_SEP_STRING,LABEL_PREFIX,num);
				fprintf(stderr,"%s",lab);
				label = make_label(lab);
				free(lab);
	    	}
			instruction i = make_instruction(is_instruction_goto,
				make_continue_statement(
						label
				)
			);
			return i;
	    }break;
/*	      fputs ("EXIT", dumpfile);
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
*/	    case EXEC_OPEN:{
			debug(5,"gfc2pips_code2instruction","Translation of OPEN\n");
			entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup("open")));
			entity_initial(e) = make_value( is_value_intrinsic, e );
			entity_type(e) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,NULL);
			newgen_list lci = NULL;
			gfc_open * o = c->ext.open;



			//We have to build the list in the opposite order it should be displayed

			if(o->err)
				lci = gfc2pips_exprIO2("ERR=", o->err->value, lci );
			if(o->asynchronous)
				lci = gfc2pips_exprIO("ASYNCHRONOUS=", o->asynchronous, lci );
			if(o->convert)
				lci = gfc2pips_exprIO("CONVERT=", o->convert, lci );
			if(o->sign)
				lci = gfc2pips_exprIO("SIGN=", o->sign, lci );
			if(o->round)
				lci = gfc2pips_exprIO("ROUND=", o->round, lci );
			if(o->encoding)
				lci = gfc2pips_exprIO("ENCODING=", o->encoding, lci );
			if(o->decimal)
				lci = gfc2pips_exprIO("DECIMAL=", o->decimal, lci );
			if(o->pad)
				lci = gfc2pips_exprIO("PAD=", o->pad, lci );
			if(o->delim)
				lci = gfc2pips_exprIO("DELIM=", o->delim, lci );
			if(o->action)
				lci = gfc2pips_exprIO("ACTION=", o->action, lci );
			if(o->position)
				lci = gfc2pips_exprIO("POSITION=", o->position, lci );
			if(o->blank)
				lci = gfc2pips_exprIO("BLANK=", o->blank, lci );
			if(o->recl)
				lci = gfc2pips_exprIO("RECL=", o->recl, lci );
			if(o->form)
				lci = gfc2pips_exprIO("FORM=", o->form, lci );
			if(o->access)
				lci = gfc2pips_exprIO("ACCESS=", o->access, lci );
			if(o->status)
				lci = gfc2pips_exprIO("STATUS=", o->status, lci );
			if(o->file)
				lci = gfc2pips_exprIO("FILE=", o->file, lci );
			if(o->iostat)
				lci = gfc2pips_exprIO("IOSTAT=", o->iostat, lci );
			if(o->iomsg)
				lci = gfc2pips_exprIO("IOMSG=", o->iomsg, lci );
			if(o->unit)
				lci = gfc2pips_exprIO("UNIT=", o->unit, lci );


			return make_instruction(is_instruction_call,
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);

	    }break;

		case EXEC_CLOSE:{
			debug(5,"gfc2pips_code2instruction","Translation of CLOSE\n");
			entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup("close")));
			entity_initial(e) = make_value( is_value_intrinsic, e );
			entity_type(e) = make_type(is_type_functional,make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,NULL);
			newgen_list lci = NULL;
			gfc_close * o = c->ext.close;

			if(o->err)
				lci = gfc2pips_exprIO2("ERR=", o->err->value, lci );
			if(o->status)
				lci = gfc2pips_exprIO("STATUS=", o->status, lci );
			if(o->iostat)
				lci = gfc2pips_exprIO("IOSTAT=", o->iostat, lci );
			if(o->iomsg)
				lci = gfc2pips_exprIO("IOMSG=", o->iomsg, lci );
			if(o->unit)
				lci = gfc2pips_exprIO("UNIT=", o->unit, lci );
	    }break;
/*
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
	    case EXEC_WRITE:{
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
			list_of_arguments = gen_nreverse(list_of_arguments);
			entity_initial(e) = make_value( is_value_intrinsic, e );
			entity_type(e) = make_type(is_type_functional, make_functional(NIL, MakeOverloadedResult()) );
			call call_ = make_call(e, list_of_arguments);
			expression std, format, unite, f;
			newgen_list lci;

			//creation of the representation of the default channels for IO
			f = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));
			std = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));


			//we check if the chan is standard and if not put the right value
			//ajouter une property pour que * soit prioritaire sur 5/6 pour les canaux :  GFC_IOSTAR_IS_PRIOTITY
			if( d->ext.dt ){
				//if no format it is standard
				if( d->ext.dt->format_expr ){
					f = gfc2pips_expr2expression(d->ext.dt->format_expr);
				}else if(
					d->ext.dt->format_label
					&& d->ext.dt->format_label->value != -1
				){
					if(d->ext.dt->format_label->format){
						f = gfc2pips_int2expression(d->ext.dt->format_label->value);
						//do we have to change the string ? we've got parentheses around the expression for the moment
						f = gfc2pips_expr2expression(d->ext.dt->format_label->format);
					}else{
						//error or warning: we have bad code
						pips_error("gfc2pips_code2instruction","No format for label\n");
					}
				}
				if(
					//if GFC_IOSTAR_IS_PRIOTITY = TRUE
					d->ext.dt->io_unit
					&& (
						d->ext.dt->io_unit->expr_type!=EXPR_CONSTANT
						//if the canal is 6, it is standard
						|| (
							d->ext.dt->io_unit->expr_type==EXPR_CONSTANT
							&& (
								(d->op==EXEC_READ && mpz_get_si(d->ext.dt->io_unit->value.integer)!=5 )
								|| (d->op==EXEC_WRITE && mpz_get_si(d->ext.dt->io_unit->value.integer)!=6 )
							)
						)
					)
				){
					std = gfc2pips_expr2expression(d->ext.dt->io_unit);
				}
			}

			unite = MakeCharacterConstantExpression("UNIT=");
			format = MakeCharacterConstantExpression("FMT=");

			lci = CONS(EXPRESSION, unite,
				CONS(EXPRESSION, std,
					CONS(EXPRESSION, format,
						CONS(EXPRESSION, f, NULL)
					)
				)
			);
			if(d->ext.dt){
				if(d->ext.dt->err){
					lci = gfc2pips_exprIO2("ERR=", d->ext.dt->err->value, lci );
				}
				if(d->ext.dt->end){
					lci = gfc2pips_exprIO2("END=", d->ext.dt->end->value, lci );
				}
				if(d->ext.dt->eor){
					lci = gfc2pips_exprIO2("EOR=", d->ext.dt->end->value, lci );
				}
			}

			//we have to have a peer number of elements in the list, so we need to insert an element between each and every elements of our arguments list
			newgen_list pc,lr;
			lr = NULL;

			pc = list_of_arguments;
			while (pc != NULL) {
				expression e = MakeCharacterConstantExpression(IO_LIST_STRING_NAME);
				newgen_list p = CONS(EXPRESSION, e, NULL);

				CDR(p) = pc;
				pc = CDR(pc);
				CDR(CDR(p)) = NULL;

				lr = gen_nconc(p, lr);
			}


			return make_instruction(is_instruction_call,
				make_call(e,
					gen_nconc(lci, lr)
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
	    	//pips_warning_handler("gfc2pips_code2instruction", "not yet dumpable %d\n",c->op);
	    	debug(3, "gfc2pips_code2instruction", "not yet dumpable %d\n",c->op);
	      //gfc_internal_error ("show_code_node(): Bad statement code");
	}
	//return instruction_undefined;
	return make_instruction_block(NULL);;
}

instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym){
	/*data_inst: TK_DATA ldatavar TK_SLASH ldataval TK_SLASH
	MakeDataStatement(
		CONS(
			EXPRESSION,
			make_expression(
				MakeAtom(
					FindOrCreateEntity(CurrentPackage, name),
					NULL,
					expression_undefined,
					expression_undefined,
					FALSE
				),
				normalized_undefined
			),
			NIL
		),
		ldataval
	);
	*/
	/*
	void MakeDataStatement(list ldr, list ldv)
	{
	  statement ds = statement_undefined;
	  code mc = entity_code(get_current_module_entity());
	  entity dl = global_name_to_entity(TOP_LEVEL_MODULE_NAME, DATA_LIST_FUNCTION_NAME);
	  expression pldr = expression_undefined;

	  pips_assert("The static initialization pseudo-intrinsic is defined",
			  !entity_undefined_p(dl));

	  pldr = make_call_expression(dl, ldr);
	  ds = make_call_statement(STATIC_INITIALIZATION_NAME,
				   gen_nconc(CONS(EXPRESSION, pldr, NIL), ldv),
				   entity_undefined,
				   strdup(PrevComm));
	  PrevComm[0] = '\0';
	  iPrevComm = 0;

	  sequence_statements(code_initializations(mc)) = gen_nconc(sequence_statements(code_initializations(mc)), CONS(STATEMENT, ds, NIL));
	}
	 *
	 */
	/*
	ds = make_call_statement(
		STATIC_INITIALIZATION_NAME,
		gen_nconc(CONS(EXPRESSION, make_call_expression(
			global_name_to_entity(TOP_LEVEL_MODULE_NAME, DATA_LIST_FUNCTION_NAME),
			ldr
		), NIL), ldv),
		entity_undefined,
		empty_comments
	);
	*/
	entity e1 = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, DATA_LIST_FUNCTION_NAME);
	entity_initial(e1) = MakeValueUnknown();

	/*newgen_list args1 = CONS(EXPRESSION,
		gfc2pips_string2expression(sym->name),
		CONS(EXPRESSION,
			gfc2pips_expr2expression(sym->value),
			NULL
		)
	);/**/
	newgen_list args1 = CONS(EXPRESSION,
		gfc2pips_string2expression(sym->name),
		NULL
	);/**/

	//list of variables used int the data statement
	newgen_list init = CONS( EXPRESSION, make_call_expression( e1, args1 ), NULL );
	//list of values
	newgen_list values = NULL;
	if(sym->value && sym->value->expr_type==EXPR_ARRAY){
		gfc_constructor *constr = sym->value->value.constructor;
		for (; constr; constr = constr->next){
			values = CONS( EXPRESSION, gfc2pips_expr2expression(constr->expr), values );
			/*if (constr->iterator == NULL){
				show_expr (c->expr);
			}else{
				fputc ('(', dumpfile);
				show_expr (c->expr);
				fputc (' ', dumpfile);
				show_expr (c->iterator->var);
				fputc ('=', dumpfile);
				show_expr (c->iterator->start);
				fputc (',', dumpfile);
				show_expr (c->iterator->end);
				fputc (',', dumpfile);
				show_expr (c->iterator->step);

				fputc (')', dumpfile);
			}

			if (c->next != NULL)
			fputs (" , ", dumpfile);*/
		}
		values = gen_nreverse(values);
	}else{
		values = CONS( EXPRESSION, gfc2pips_expr2expression(sym->value), NULL );
	}


	entity e2 = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STATIC_INITIALIZATION_FUNCTION_NAME);
	entity_initial(e2) = MakeValueUnknown();
	newgen_list args2 = gen_nconc(init, values);
	call call_ = make_call( e2, args2 );
	return make_instruction(is_instruction_call,call_);
}




entity gfc2pips_code2get_label(gfc_code *c){
	if(!c) return entity_empty_label() ;
	debug(9,"gfc2pips_code2get_label",
		"test label: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if( c->here ) return gfc2pips_int2label(c->here->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label2(gfc_code *c){
	if(!c) return entity_empty_label() ;
	debug(9,"gfc2pips_code2get_label2",
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if( c->label )return gfc2pips_int2label(c->label->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label3(gfc_code *c){
	if(!c) return entity_empty_label() ;
	debug(9,"gfc2pips_code2get_label3",
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if( c->label )return gfc2pips_int2label(c->label2->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label4(gfc_code *c){
	if(!c) return entity_empty_label() ;
	debug(9,"gfc2pips_code2get_label4",
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(c->label?c->label->value:0),
		(c->label2?c->label2->value:0),
		(c->label3?c->label3->value:0),
		(c->here?c->here->value:0),
		c->next,c->block,c->expr
	);
	if( c->label )return gfc2pips_int2label(c->label3->value);
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
		//fprintf(stderr,"No symtree\n");
	}else if(!expr->symtree->n.sym){
		//fprintf(stderr,"No symbol\n");
	}else if(!expr->symtree->n.sym->name){
		//fprintf(stderr,"No name\n");
	}else{
		//fprintf(stderr,"gfc2pips_expr2expression: dumping %s\n",expr->symtree->n.sym->name);
	}

	//fprintf(stderr,"type: %d\n",expr->expr_type);
	//fprintf(stderr,"kind: %d\n",expr->ts.kind);
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
					fprintf(stderr,"gfc2pips_expr2expression: intrinsic not yet recognized: %d\n",expr->value.op.op);
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
			syntax s = syntax_undefined;
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
				//revoir : pourquoi une boucle ?decl_tableau
				gfc_ref *r =expr->ref;
				while(r){
					//fprintf(stderr,"^^^^^^^^^^^^^^^^^^^^^\n");
					if(r->type==REF_ARRAY){
						ref = gfc2pips_array_ref2indices(&r->u.ar);
						break;
					/*}else if(r->type==REF_COMPONENT){
						fprintf (dumpfile, " %% %s", p->u.c.component->name);
					*/}else if(r->type==REF_SUBSTRING){
						entity ent = FindOrCreateEntity(CurrentPackage,str2upper(strdup(expr->symtree->n.sym->name)));
						entity_type(ent) = gfc2pips_symbol2type(expr->symtree->n.sym);
						entity_storage(ent) = MakeStorageRom();fprintf(stderr,"expr2expression ROM %s\n",expr->symtree->n.sym->name);
						entity_initial(ent) = MakeValueUnknown();
						expression ref = make_expression(
							make_syntax(is_syntax_reference,
								make_reference(ent, NULL)
							),
						    normalized_undefined
						);

						entity substr = entity_intrinsic(SUBSTRING_FUNCTION_NAME);
						newgen_list lexpr = CONS(EXPRESSION, ref,
							CONS(EXPRESSION,
								gfc2pips_expr2expression(r->u.ss.start),
								CONS(EXPRESSION,
									r->u.ss.end ? gfc2pips_expr2expression(r->u.ss.end) : MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)),
									NULL
								)
							)
						);
						s = make_syntax(is_syntax_call, make_call(substr, lexpr));
						return make_expression( s, normalized_undefined );
					}else{
						fprintf(stderr,"Unable to understand the ref %d\n",r->type);
					}
					r=r->next;
				}
			}
			entity ent_ref = FindOrCreateEntity(CurrentPackage, str2upper(strdup(expr->symtree->n.sym->name)));
			entity_type(ent_ref) = gfc2pips_symbol2type(expr->symtree->n.sym);
			if(entity_storage(ent_ref)==storage_undefined){
				entity_storage(ent_ref) = MakeStorageRom();//fprintf(stderr,"expr2expression ROM %s\n",expr->symtree->n.sym->name);
			}
			entity_initial(ent_ref) = MakeValueUnknown();
			s = make_syntax_reference(
				make_reference(
					ent_ref,
					ref
				)
			);

			return make_expression( s, normalized_undefined );
		}break;
		case EXPR_CONSTANT:
			debug(5,"gfc2pips_expr2expression","cst %d %d\n",expr,expr->ts.type);
			switch(expr->ts.type){
				case BT_INTEGER: e = gfc2pips_int2expression(mpz_get_si(expr->value.integer)); break;
				case BT_LOGICAL: e = gfc2pips_logical2expression(expr->value.logical); break;
				case BT_REAL:
					/*fprintf(stderr,"%f\n",mpfr_get_d(expr->value.real,GFC_RND_MODE));
					fprintf(stderr,"expr->where.nextc\t %s\n",expr->where.nextc);
					fprintf(stderr,"expr->where.lb->dbg_emitted\t %d\n",expr->where.lb->dbg_emitted);

					fprintf(stderr,"expr->where.lb->file->filename\t %s\n",expr->where.lb->file->filename);
					fprintf(stderr,"expr->where.lb->file->inclusion_line\t %d\n",expr->where.lb->file->inclusion_line);
					fprintf(stderr,"expr->where.lb->file->line\t %d\n",expr->where.lb->file->line);
					fprintf(stderr,"expr->ref\t %d\n",expr->ref);


					fprintf(stderr,"expr->where.lb->line\t %s\n",expr->where.lb->line);
					fprintf(stderr,"expr->where.lb->location\t %d\n",expr->where.lb->location);//c'est un chiffre qui dépend de la ligne et de  ??? => trouver comment il est calculé
					fprintf(stderr,"expr->where.lb->truncated\t %d\n",expr->where.lb->truncated);//établi si la ligne a été coupée lors du parse
*/
					//convertir le real en qqch de correct au niveau sortie
					e = gfc2pips_real2expression(mpfr_get_d(expr->value.real,GFC_RND_MODE));
				break;
				case BT_CHARACTER:{
					char *char_expr = gfc2pips_gfc_char_t2string(expr->value.character.string,expr->value.character.length);
					e = MakeCharacterConstantExpression(char_expr);
					/*fprintf(
						stderr,
						"string(%d) %s\n",
						strlen(expr->value.character.string),
						char_expr
					);*/
					//free(char_expr);
				}break;
				case BT_COMPLEX:
					e = MakeComplexConstantExpression(
						gfc2pips_real2expression(mpfr_get_d(expr->value.complex.r,GFC_RND_MODE)),
						gfc2pips_real2expression(mpfr_get_d(expr->value.complex.i,GFC_RND_MODE))
					);
				break;
				case BT_HOLLERITH:
				default:
					debug(5,"gfc2pips_expr2expression","type not implemented %d",expr->ts.type);
				break;

			}
			//if(expr->ref)
			return e;
		break;
		case EXPR_FUNCTION:
			debug(5,"gfc2pips_expr2expression","func\n");
			//beware the automatic conversion here, some conversion functions may be automatically called here, and we do not want them in the code
			if(strncmp(str2upper(expr->symtree->n.sym->name),str2upper(strdup("__convert_")),strlen("__convert_"))==0){
				//fprintf(stderr,"gfc2pips_expr2expression: auto-convert detected %s\n",expr->symtree->n.sym->name);
				if(expr->value.function.actual->expr){
					debug(6,"gfc2pips_expr2expression","expression not null !\n");
					//show_expr(expr->value.function.actual->expr);
					return gfc2pips_expr2expression(expr->value.function.actual->expr);
				}else{
					debug(6,"gfc2pips_expr2expression","expression null !\n");
				}
			}else{
				//functions whose name begin with __ should be used by gfc only therefore we put the old name back
				if(strncmp(str2upper(expr->value.function.name),str2upper(strdup("__")),strlen("__"))==0){
					expr->value.function.name = expr->symtree->n.sym->name;
				}
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
							list_of_arguments_p = CONS(EXPRESSION,ex,NIL);
						}
					}
					if(list_of_arguments==NULL)list_of_arguments = list_of_arguments_p;

				}while(act = act->next);


				entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup(expr->value.function.name)));
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
		case EXPR_ARRAY:
			debug(5,"gfc2pips_expr2expression","array\n");

			//show_constructor (p->value.constructor);
			//show_ref (p->ref);

			//MakeDataStatement($2, $4);
			//return CONS(EXPRESSION, $1, NIL);
		//break;
		default:
			fprintf(stderr,"gfc2pips_expr2expression: dump not yet implemented, type of gfc_expr not recognized %d\n",expr->expr_type);
		break;
	}
	return expression_undefined;
}

/*
 * int gfc2pips_expr2int(gfc_expr *expr)
 *
 * we assume we have an expression representing an integer, and we translate it
 * this function consider everything is all right: i.e. the expression represent an integer
 */
int gfc2pips_expr2int(gfc_expr *expr){
	return mpz_get_si(expr->value.integer);
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
		entity e = FindOrCreateEntity(CurrentPackage, str2upper(strdup(expr->symtree->n.sym->name)));
		entity_type(e) = gfc2pips_symbol2type(expr->symtree->n.sym);
		entity_initial(e) = MakeValueUnknown();
		return e;
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

newgen_list gfc2pips_exprIO(char* s, gfc_expr* e, newgen_list l){
	return CONS(EXPRESSION, MakeCharacterConstantExpression(s),
		CONS(EXPRESSION, gfc2pips_expr2expression(e), l )
	);
}
newgen_list gfc2pips_exprIO2(char* s, int e, newgen_list l){
	return CONS(EXPRESSION, MakeCharacterConstantExpression(s),
		CONS(EXPRESSION, MakeNullaryCall(gfc2pips_int2label(e)), l )
	);
}

void gfc2pips_initAreas(){
    DynamicArea = FindOrCreateEntity(CurrentPackage, DYNAMIC_AREA_LOCAL_NAME);
    entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(DynamicArea) = MakeStorageRom();
    entity_initial(DynamicArea) = MakeValueUnknown();
//    AddEntityToDeclarations(DynamicArea, get_current_module_entity());
//    set_common_to_size(DynamicArea, 0);

    StaticArea = FindOrCreateEntity(CurrentPackage, STATIC_AREA_LOCAL_NAME);
    entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(StaticArea) = MakeStorageRom();
    entity_initial(StaticArea) = MakeValueUnknown();
//    AddEntityToDeclarations(StaticArea, get_current_module_entity());
//  set_common_to_size(StaticArea, 0);

    HeapArea = FindOrCreateEntity(CurrentPackage, HEAP_AREA_LOCAL_NAME);
    entity_type(HeapArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(HeapArea) = MakeStorageRom();
    entity_initial(HeapArea) = MakeValueUnknown();
//    AddEntityToDeclarations(HeapArea, get_current_module_entity());
//  set_common_to_size(HeapArea, 0);

    StackArea = FindOrCreateEntity(CurrentPackage, STACK_AREA_LOCAL_NAME);
    entity_type(StackArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(StackArea) = MakeStorageRom();
    entity_initial(StackArea) = MakeValueUnknown();
//    AddEntityToDeclarations(StackArea, get_current_module_entity());
//    set_common_to_size(StackArea, 0);

}



void gfc2pips_handleEquiv(gfc_equiv *eq){
	//StoreEquivChain(chain c){

/*
latom: atom
		    {
			$$ = make_chain(CONS(ATOM, MakeEquivAtom($1), (cons*) NULL));
		    }
		| latom TK_COMMA atom
		    {
			chain_atoms($1) = CONS(ATOM, MakeEquivAtom($3),
					     chain_atoms($1));
			$$ = $1;
		    }
		;
<syntax>atom: entity_name
	    {
		$$ = MakeAtom($1, NIL, expression_undefined,
				expression_undefined, FALSE);
	    }
	| entity_name indices
	    {
		$$ = MakeAtom($1, $2, expression_undefined,
				expression_undefined, TRUE);
	    }
	| entity_name TK_LPAR opt_expression TK_COLON opt_expression TK_RPAR
	    {
		$$ = MakeAtom($1, NIL, $3, $5, TRUE);
	    }
	| entity_name indices TK_LPAR opt_expression TK_COLON opt_expression TK_RPAR
	    {
		$$ = MakeAtom($1, $2, $4, $6, TRUE);
	    }
	;

*/
    int maxoff;

    maxoff = 0;
    //pour chaque chaine ? ou chaque élément de la chaine ?
    gfc_equiv * save = eq;
    for( ; eq ; eq=eq->eq ){
    	//ce n'est pas du tout ce qu'il faut
    	printf("test of offset: %d\n", expression_syntax_(gfc2pips_expr2expression(eq->expr) ) );
    	/*
struct _newgen_struct_atom_ {
  intptr_t _type_;
  entity _atom_equivar_;
  intptr_t _atom_equioff_;
};
    	 */
		//int o = atom_equioff(ATOM(CAR(pc)));
    	int o = atom_equioff( (atom)expression_syntax_(gfc2pips_expr2expression(eq->expr) ));

		if (o > maxoff)	maxoff = o;
	}

	pips_debug(9, "maxoff %d\n", maxoff);

	eq = save;
	if (maxoff > 0) {
	    for( ; eq ; eq=eq->eq ){
			atom a = expression_syntax_(gfc2pips_expr2expression(eq->expr));

			atom_equioff(a) = abs(atom_equioff(a)-maxoff);
		}
	}

	/*
	if (TempoEquivSet == equivalences_undefined) {
	TempoEquivSet = make_equivalences(NIL);
	}
     */
	//pips_assert("The TempoEquivSet is defined", !equivalences_undefined_p(TempoEquivSet) );

	//equivalences_chains(TempoEquivSet) = CONS( CHAIN, c, equivalences_chains(TempoEquivSet) );
}






//we need to copy the content of the locus
void gfc2pips_push_comment(locus l, unsigned long num, char s){
	if(gfc2pips_comments_stack){
		if(gfc2pips_check_already_done(l)){
			return;
		}
		gfc2pips_comments_stack->next = malloc(sizeof(struct _gfc2pips_comments_));
		gfc2pips_comments_stack->next->prev = gfc2pips_comments_stack;
		gfc2pips_comments_stack->next->next = NULL;

		gfc2pips_comments_stack = gfc2pips_comments_stack->next;
	}else{
		gfc2pips_comments_stack = malloc(sizeof(struct _gfc2pips_comments_));
		gfc2pips_comments_stack->prev = NULL;
		gfc2pips_comments_stack->next = NULL;
		gfc2pips_comments_stack_ = gfc2pips_comments_stack;
	}
	//fprintf(stderr,"push comments\n");

	gfc2pips_comments_stack->l = l;
	gfc2pips_comments_stack->num = num;
	gfc2pips_comments_stack->gfc = NULL;
	gfc2pips_comments_stack->done = false;


	gfc2pips_comments_stack->s = gfc2pips_gfc_char_t2string2(l.nextc);
	gfc2pips_comments_stack->s[ strlen(gfc2pips_comments_stack->s)-2 ] = '\0';
	strrcpy(gfc2pips_comments_stack->s+1,gfc2pips_comments_stack->s);
	*gfc2pips_comments_stack->s = s;

}

bool gfc2pips_check_already_done(locus l){
	gfc2pips_comments retour = gfc2pips_comments_stack;
	while(retour){
		if(retour->l.nextc==l.nextc)return true;
		retour = retour->prev;
	}
	return false;
}

unsigned long gfc2pips_get_num_of_gfc_code(gfc_code *c){
	unsigned long retour = 0;
	gfc2pips_comments curr = gfc2pips_comments_stack_;
	while(curr){
		if(curr->gfc == c){
			return retour+1;
		}
		curr = curr->next;
		retour++;
	}
	if(retour)return retour+1;
	return retour;// 0
}
string gfc2pips_get_comment_of_code(gfc_code *c){
	gfc2pips_comments retour = gfc2pips_comments_stack_;
	char *a,*b;
	while(retour){
		if(retour->gfc == c){
			a = retour->s;
			retour = retour->next;
			while(retour && retour->gfc==c){
				if(a && retour->s){
					b = malloc(
						sizeof(char)*(
							strlen(a) + strlen(retour->s) + 2
						)
					);
					strcpy(b,a);
					strcpy(b+strlen(b),"\n");
					strcpy(b+strlen(b),retour->s);
					free(a);
					a=b;
				}else if(retour->s){
					a=retour->s;
				}
				retour = retour->next;
			}
			if(a){
				b = malloc(sizeof(char)*(strlen(a) + 2));
				strcpy(b,a);
				strcpy(b+strlen(b),"\n");
				free(a);
				return b;
			}else{
				return empty_comments;
			}
		}
		retour = retour->next;
	}
	return empty_comments;
}

gfc2pips_comments gfc2pips_pop_comment(){
	if(gfc2pips_comments_stack){
		gfc2pips_comments retour = gfc2pips_comments_stack;
		gfc2pips_comments_stack = gfc2pips_comments_stack->prev;
		if(gfc2pips_comments_stack){
			gfc2pips_comments_stack->next = NULL;
		}else{
			gfc2pips_comments_stack_ = NULL;
		}
		return retour;
	}else{
		return NULL;
	}
}

//changer en juste un numéro, sans que ce soit "done"
//puis faire une étape similaire qui assigne un statement à la première plage non "done" et la met à "done"
void gfc2pips_set_last_comments_done(unsigned long nb){
	//printf("gfc2pips_set_last_comments_done\n");
	gfc2pips_comments retour = gfc2pips_comments_stack;
	while(retour){
		if(retour->done)return;
		retour->num = nb;
		retour->done = true;
		retour = retour->prev;
	}
}
void gfc2pips_assign_num_to_last_comments(unsigned long nb){
	gfc2pips_comments retour = gfc2pips_comments_stack;
	while(retour){
		if(retour->done||retour->num)return;
		retour->num = nb;
		retour = retour->prev;
	}
}
void gfc2pips_assign_gfc_code_to_last_comments(gfc_code *c){
	gfc2pips_comments retour = gfc2pips_comments_stack_;
	if(c){
		while(retour && retour->done ){
			retour = retour->next;
		}
		if(retour){
			unsigned long num_plage = retour->num;
			while( retour && retour->num==num_plage ){
				retour->gfc = c;
				retour->done = true;
				retour = retour->next;
			}
		}
	}
}

void gfc2pips_replace_comments_num(unsigned long old, unsigned long new){
	gfc2pips_comments retour = gfc2pips_comments_stack;
	bool if_changed = false;
	fprintf(stderr,"gfc2pips_replace_comments_num: replace %d by %d\n", old, new );
	while(retour){
		if(retour->num==old){
			if_changed = true;
			retour->num = new;
		}
		retour = retour->prev;
	}
	//if(if_changed) gfc2pips_nb_of_statements--;
}

void gfc2pips_assign_gfc_code_to_num_comments(gfc_code *c, unsigned long num){
	gfc2pips_comments retour = gfc2pips_comments_stack_;
	while( retour ){
		if( retour->num==num ) retour->gfc = c;
		retour = retour->next;
	}
}
bool gfc2pips_comment_num_exists(unsigned long num){
	gfc2pips_comments retour = gfc2pips_comments_stack;
	fprintf(stderr,"gfc2pips_comment_num_exists: %d\n", num );
	while(retour){
		if(retour->num==num)return true;
		retour = retour->prev;
	}
	return false;
}

void gfc2pips_pop_not_done_comments(){
	while(gfc2pips_comments_stack && gfc2pips_comments_stack->done==false){
		gfc2pips_pop_comment();
	}
}
void gfc2pips_shift_comments(){
	/*
	 * We assign a gfc_code depending to a list of comments if any depending of the number of the statement
	 */
	gfc2pips_comments retour = gfc2pips_comments_stack;
	newgen_list l = gen_nreverse(gfc2pips_list_of_declared_code);
	while(retour){

		newgen_list curr = gen_nthcdr(retour->num,l);
		if(curr){
			retour->gfc = (gfc_code*)curr->car.e;
		}
		retour = retour->prev;
	}
	return;
}


void gfc2pips_push_last_code(gfc_code *c){
	if(gfc2pips_list_of_declared_code==NULL) gfc2pips_list_of_declared_code = gen_cons(NULL,NULL);
	//gfc2pips_list_of_declared_code =
	gfc2pips_list_of_declared_code = gen_cons(c,gfc2pips_list_of_declared_code);
}


gfc_code* gfc2pips_get_last_loop(){
	if(gfc2pips_list_of_loops) return gfc2pips_list_of_loops->car.e;
	return NULL;
}
void gfc2pips_push_loop(gfc_code *c){
	gfc2pips_list_of_loops = gen_cons(c,gfc2pips_list_of_loops);
}
void gfc2pips_pop_loop(){
	POP(gfc2pips_list_of_loops);
}




