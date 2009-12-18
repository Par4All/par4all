/*
 * The naming of functions is based on the following syntax :
 * - every and each function used to manipulate gfc or pips entities begin with gfc2pips_
 * - if the function is a translation from one RI to the other it will be:
 * 		gfc2pips_<type_of_the_source_entity>2<type_of_the_target_entity>(arguments)
 */

#include "dump2PIPS.h"



newgen_list gfc2pips_list_of_declared_code = NULL;
newgen_list gfc2pips_list_of_loops = NULL;

newgen_list gfc2pips_format = NULL;//list of expression format
newgen_list gfc2pips_format2 = NULL;//list of labels for above

static int gfc2pips_last_created_label = 95000;
static int gfc2pips_last_created_label_step = 2;



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
int strcmp_ (__const char *__s1, __const char *__s2){
	char *a = str2upper(strdup(__s1));
	char *b = str2upper(strdup(__s2));
	int ret = strcmp(a,b);
	free(a);
	free(b);
	return ret;
}

int strncmp_ (__const char *__s1, __const char *__s2, size_t __n){
	char *a = str2upper(strdup(__s1));
	char *b = str2upper(strdup(__s2));
	int ret = strncmp(a,b,__n);
	free(a);
	free(b);
	return ret;
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

/**
 * @brief epurate a string representing a REAL, could be a pre-prettyprinter processing
 */
void gfc2pips_truncate_useless_zeroes(char *s){
	char *start = s;
	bool has_dot = false;
	char *end_sci = NULL;//scientific output ?
	while(*s){
		if(*s=='.'){
			has_dot=true;
			pips_debug(9,"found [dot] at %d\n",s-start);
			s++;
			while(*s){
				if(*s=='e'){
					end_sci = s;
					break;
				}
				s++;
			}
			break;
		}
		s++;
	}
	if(has_dot){
		int nb=0;
		if(end_sci){
			s=end_sci-1;
		}else{
			s=start+strlen(start);
		}

		while(s>start){
			if(*s=='0'){
				*s='\0';
				nb++;
			}else{
				break;
			}
			s--;
		}
		pips_debug(9,"%d zero(s) retrieved\n", nb);
		/*if(*s=='.'){
			*s='\0';
			s--;
			pips_debug(9,"final dot retrieved\n");
		}*/
		if(end_sci){
			if(strcmp(end_sci,"e+00")==0){
				*(s+1)='\0';
			}else if(s!=end_sci-1){
				strcpy(s+1,end_sci);
			}
		}
	}
}

//an enum to know what kind of main entity we are dealing with
typedef enum gfc2pips_main_entity_type{MET_PROG,MET_SUB,MET_FUNC,MET_MOD,MET_BLOCK} gfc2pips_main_entity_type;
entity gfc2pips_main_entity = entity_undefined;

/**
 * @brief Dump a namespace
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
	gfc2pips_format = gfc2pips_format2 = NULL;


	pips_debug(2, "Starting gfc 2 pips dumping\n");
	message_assert("No namespace to dump.",		ns);
	message_assert("No symtree root.",			ns->sym_root);
	message_assert("No symbol for the root.",	ns->sym_root->n.sym);

	//gfc_get_code();
	gfc2pips_shift_comments();
	current = gfc2pips_getSymtreeByName( ns->proc_name->name, ns->sym_root );
	message_assert("No current symtree to match the name of the namespace",current);
	root_sym = current->n.sym;



	CurrentPackage = str2upper(strdup(ns->proc_name->name));
	/*
	 * We have to create a PIPS entity which is a program/function/other
	 * according to the type we have to create different values as parameters for the creation of the PIPS entity by:
	 * MakeCurrentFunction(type_undefined, TK_PROGRAM , CurrentPackage, NULL);
	 */
	gfc2pips_main_entity_type bloc_token = -1;
	////type returned
	type bloc_type = make_type_void();

	/* WRONG
	string prefix;
	if( root_sym->attr.is_main_program ){
		prefix = MAIN_PREFIX;
	}else if( root_sym->attr.subroutine || root_sym->attr.function ){
		prefix = BLOCKDATA_PREFIX;
	}else{
		prefix = "";
	}*/

	//maybe right
	string full_name = concatenate(
		TOP_LEVEL_MODULE_NAME,
		MODULE_SEP_STRING,
		root_sym->attr.is_main_program?MAIN_PREFIX:"",
		ns->proc_name->name,
		NULL
	);
	gfc2pips_main_entity = gen_find_tabulated(full_name, entity_domain);
	if(gfc2pips_main_entity!=entity_undefined){
		//we have defined this entity already. Nothing to do ? we need at least to know the bloc_token
		//force temporarily the type for the set_curren_module_entity(<entity>)
		entity_initial(gfc2pips_main_entity) = make_value_code(
			make_code(
				NIL, strdup(""), make_sequence(NIL),NIL, make_language_unknown()
			)
		);

		//value v = entity_initial(gfc2pips_main_entity);
		//fprintf(stderr,"value_tag: %d %d\n",is_value_code,value_tag(v));
		//value_tag(v) = is_value_code;
	    //value_code_p(v);
	}
	if( root_sym->attr.is_main_program ){
		pips_debug(3, "main program %s\n",ns->proc_name->name);
		//gfc2pips_main_entity = make_empty_program(str2upper(strdup(ns->proc_name->name)));
		//message_assert("Main entity not created !\n",gfc2pips_main_entity!=entity_undefined);
		bloc_token = MET_PROG;
	}else if( root_sym->attr.subroutine ){
		pips_debug(3, "subroutine %s\n",ns->proc_name->name);
		//gfc2pips_main_entity = make_empty_subroutine(str2upper(strdup(ns->proc_name->name)));
		bloc_token = MET_SUB;
	}else if( root_sym->attr.function ){
		pips_debug(3, "function %s\n",ns->proc_name->name);
		//gfc2pips_main_entity = make_empty_function(str2upper(strdup(ns->proc_name->name)),gfc2pips_symbol2type(root_sym));
		//gfc2pips_main_entity = FindOrCreateEntity(CurrentPackage,ns->proc_name->name);
		bloc_type = gfc2pips_symbol2type(root_sym);
		bloc_token = MET_FUNC;
	}else if(root_sym->attr.flavor == FL_BLOCK_DATA){
		pips_debug(3, "block data \n");
		//gfc2pips_main_entity = make_empty_blockdata(str2upper(strdup(ns->proc_name->name)));
		bloc_token = MET_BLOCK;
	}else{
		pips_debug(3, "not yet dumpable %s\n",ns->proc_name->name);
		if(root_sym->attr.procedure){
			fprintf(stderr,"procedure\n");
		}
		return;
	}
	gfc2pips_main_entity = gfc2pips_symbol2entity(ns->proc_name);

	message_assert("Main entity no created !",gfc2pips_main_entity!=entity_undefined);
	/*struct _newgen_struct_entity_ {
	  intptr_t _type_;
	  intptr_t _entity_index__;
	  string _entity_name_;				ok
	  type _entity_type_;				x
	  storage _entity_storage_;			x
	  value _entity_initial_;			x
	};*/

	pips_debug(2, "main entity object initialized\n");


	//
	////list of parameters
	pips_debug(2, "dump the list of parameters\n");
	newgen_list parameters = NULL, parameters_name = NULL;
	if(bloc_token == MET_FUNC || bloc_token == MET_SUB){
		parameters = gfc2pips_args(ns);//change it to put both name and namespace in order to catch the parameters of any subroutine ? or create a sub-function for gfc2pips_args
		parameters_name = gen_copy_seq(parameters);//we need a copy of the list of the entities of the parameters
		gfc2pips_generate_parameters_list(parameters);
		//fprintf(stderr,"formal created ?? %d\n", storage_formal_p(entity_storage(ENTITY(CAR(parameters_name)))));

	    //ScanFormalParameters(gfc2pips_main_entity, add_formal_return_code(parameters));
	}
	pips_debug(2, "List of parameters done\t %d parameters(s)\n", gen_length(parameters_name) );



	////type of entity we are creating : a function (except maybe for module ?
	////a functional type is made of a list of parameters and the type returned
	if(bloc_token != MET_BLOCK){
		entity_type( gfc2pips_main_entity ) = make_type_functional( make_functional(parameters, bloc_type));//entity_type( gfc2pips_main_entity)
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

	int stack_offset = 0;
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

	set_current_module_entity( gfc2pips_main_entity );
	gfc2pips_initAreas();//Even if it is initialized by the defaults functions of PIPS, it is not the way we want it to be => still true ? if not remove the comment


	//// declare commons
	newgen_list commons, commons_p, unnamed_commons, unnamed_commons_p, common_entities;
	commons = commons_p = getSymbolBy(ns, ns->common_root, gfc2pips_get_commons);
	unnamed_commons = unnamed_commons_p = getSymbolBy(ns, ns->sym_root, gfc2pips_get_incommon);

	common_entities = NULL;

	pips_debug(2, "%d explicit common(s) founded\n",gen_length(commons));

	while(commons_p){
		gfc_symtree *st = (gfc_symtree*)commons_p->car.e;
		pips_debug(3, "common founded: /%s/\n",st->name);
		entity com = FindOrCreateEntity(
			strdup(TOP_LEVEL_MODULE_NAME),
			strdup(str2upper(concatenate(strdup(COMMON_PREFIX),(st->name), NULL)))
		);
		entity_type(com) = make_type_area( make_area(0, NIL));

		entity_storage(com) = make_storage_ram(
			make_ram(get_current_module_entity(),StaticArea, 0, NIL)
		);
		entity_initial(com) = make_value_code(make_code(NIL,string_undefined,make_sequence(NIL),NIL,make_language_fortran()));
		AddEntityToDeclarations(com, get_current_module_entity());
		commons_p->car.e = com;//we need in the final state a list of entities

		gfc_symbol *s = st->n.common->head;
		int offset_common = stack_offset;
		while(s){
			unnamed_commons_p = unnamed_commons;
			while(unnamed_commons_p){
				st = unnamed_commons_p->car.e;
				if( strcmp_(st->n.sym->name, s->name )==0 ){
					gen_remove(&unnamed_commons,st);
					break;
				}
				POP(unnamed_commons_p);
			}
			pips_debug(4, "element in common founded: %s\t\toffset: %d\n", s->name, offset_common );
			entity in_common_entity = gfc2pips_symbol2entity(s);
			entity_storage(in_common_entity) = make_storage_ram(
				make_ram(
					get_current_module_entity(),
					com,
					offset_common,
				    NIL
			   )
			);
			int size;
			SizeOfArray(in_common_entity,&size);
			offset_common += size;
			area_layout(type_area(entity_type(com))) = gen_nconc(area_layout(type_area(entity_type(com))), CONS(ENTITY, in_common_entity, NIL));
			common_entities = gen_cons(in_common_entity,common_entities);
			s = s->common_next;
		}
		set_common_to_size(com, offset_common);
		pips_debug(3, "nb of elements in the common: %d\n\t\tsize of the common: %d\n",
			gen_length(
				area_layout(type_area(entity_type(com)))
			),
			offset_common
		);
		POP(commons_p);
	}

	int unnamed_commons_nb = gen_length(unnamed_commons);
	if(unnamed_commons_nb){
		pips_debug(2, "nb of elements in %d unnamed common(s) founded\n",unnamed_commons_nb);

		entity com = FindOrCreateEntity(
			strdup(TOP_LEVEL_MODULE_NAME),
			strdup(str2upper(concatenate(strdup(COMMON_PREFIX),strdup(BLANK_COMMON_LOCAL_NAME), NULL)))
		);
		entity_type(com) = make_type_area(make_area(0, NIL));

		entity_storage(com) = make_storage_ram(
			make_ram(get_current_module_entity(),StaticArea, 0, NIL)
		);
		entity_initial(com) = make_value_code(make_code(NIL,string_undefined,make_sequence(NIL),NIL,make_language_fortran()));
		AddEntityToDeclarations(com, get_current_module_entity());
		int offset_common = stack_offset;
		unnamed_commons_p = unnamed_commons;
		while(unnamed_commons_p){
			gfc_symtree* st = unnamed_commons_p->car.e;
			gfc_symbol *s = st->n.sym;
			pips_debug(4, "element in common founded: %s\t\toffset: %d\n", s->name, offset_common );
			entity in_common_entity = gfc2pips_symbol2entity(s);
			entity_storage(in_common_entity) = make_storage_ram(
				make_ram(
					get_current_module_entity(),
					com,
					offset_common,
				    NIL
			   )
			);
			int size;
			SizeOfArray(in_common_entity,&size);
			offset_common += size;
			area_layout(type_area(entity_type(com))) = gen_nconc(area_layout(type_area(entity_type(com))), CONS(ENTITY, in_common_entity, NIL));
			common_entities = gen_cons(in_common_entity,common_entities);
			POP(unnamed_commons_p);
		}
		set_common_to_size(com, offset_common);
		pips_debug(3, "nb of elements in the common: %d\n\t\tsize of the common: %d\n",
			gen_length(
				area_layout(type_area(entity_type(com)))
			),
			offset_common
		);
		commons = gen_cons(com,commons);
	}
	pips_debug(2, "%d common(s) founded for %d entities\n", gen_length(commons), gen_length(common_entities) );
	//add variables who are not in a named common, their name is then the same as the common


	//// declare DIMENSIONS => information transfered to each single entity

	//// declare variables
	newgen_list variables_p,variables;
	variables_p = variables = gfc2pips_vars(ns);
	pips_debug(2, "%d variable(s) founded\n",gen_length(variables));

	//we concatenate the entities from variables, commons and parameters and make sure they are declared only once
	//it seems parameters cannot be declared implicitly and have to be part of the list
	newgen_list complete_list_of_entities = NULL,complete_list_of_entities_p = NULL;

	if(bloc_token == MET_FUNC){
		//we add a special entity called  func:func which is the return variable of the function
		entity ent = FindOrCreateEntity(CurrentPackage,get_current_module_name());
		entity_type(ent) = copy_type(entity_type(gfc2pips_main_entity));
		entity_initial(ent) = copy_value(entity_initial(gfc2pips_main_entity));
		//don't know were to put it hence StackArea
		entity_storage(ent) = make_storage_ram(
			make_ram(
				get_current_module_entity(),
				StackArea,
				UNKNOWN_RAM_OFFSET,
				NULL
			)
		);

	}

	complete_list_of_entities_p = gen_union(complete_list_of_entities_p,variables_p);
	commons_p = commons;

	complete_list_of_entities_p = gen_union( commons_p, complete_list_of_entities_p );

	complete_list_of_entities_p = gen_union(complete_list_of_entities_p, parameters_name);

	complete_list_of_entities = complete_list_of_entities_p;
	while(complete_list_of_entities_p){
		//force value
		if(entity_initial(ENTITY(CAR(complete_list_of_entities_p)))==value_undefined){
			entity_initial(ENTITY(CAR(complete_list_of_entities_p))) = MakeValueUnknown();
		}
		POP(complete_list_of_entities_p);
	}

	newgen_list list_of_declarations = code_declarations(EntityCode(gfc2pips_main_entity));
	pips_debug(2, "%d declaration(s) founded\n",gen_length(list_of_declarations));
	complete_list_of_entities = gen_union(complete_list_of_entities,list_of_declarations);


	newgen_list list_of_extern_entities = gfc2pips_get_extern_entities(ns);
	newgen_list list_of_extern_entities_p = list_of_extern_entities;
	while(list_of_extern_entities_p){
		//force storage
		if(entity_storage(ENTITY(CAR(list_of_extern_entities_p)))==storage_undefined){
			entity_storage(ENTITY(CAR(list_of_extern_entities_p))) = MakeStorageRom();
		}
		POP(list_of_extern_entities_p);
	}

	pips_debug(2, "%d extern(s) founded\n",gen_length(list_of_extern_entities));

	pips_debug(2, "nb of entities: %d\n",gen_length(complete_list_of_entities));


	newgen_list list_of_subroutines,list_of_subroutines_p;
	list_of_subroutines_p = list_of_subroutines = getSymbolBy(ns,ns->sym_root, gfc2pips_test_subroutine);
	while(list_of_subroutines_p){
		gfc_symtree* st = list_of_subroutines_p->car.e;

		list_of_subroutines_p->car.e = gfc2pips_symbol2entity(st->n.sym);
		//list_of_subroutines_p->car.e = FindOrCreateEntity( TOP_LEVEL_MODULE_NAME, str2upper(strdup(st->name)) );
		entity check_sub_entity = (entity)list_of_subroutines_p->car.e;
		if(type_functional_p(entity_type(check_sub_entity)) && strcmp(st->name,ns->proc_name->name)!=0 ){
			//check list of parameters;
			newgen_list check_sub_parameters = functional_parameters(type_functional(entity_type(check_sub_entity)));
			if( check_sub_parameters==NULL ){

			}
			pips_debug(9,"sub %s has %d parameters\n", entity_name(check_sub_entity), gen_length(check_sub_parameters) );
		}
		POP(list_of_subroutines_p);
	}
	pips_debug(2, "%d subroutine(s) encountered\n", gen_length(list_of_subroutines) );


	complete_list_of_entities = gen_union(
		complete_list_of_entities,
		common_entities//we need the variables in the common to be in the list too
	);

	//refaire un tri sur la liste afin de retirer les IMPLICIT, BUT beware arguments with the current method we have a pb with the ouput of subroutines/functions arguments even if they are of the right type
	/*complete_list_of_entities_p = complete_list_of_entities;
	while( complete_list_of_entities_p ){
		entity ent = complete_list_of_entities_p->car.e;
		if( ent ){
			pips_debug(9,"Look for %s %d\n", entity_local_name(ent), gen_length(complete_list_of_entities_p) );
			POP(complete_list_of_entities_p);
			gfc_symtree* sort_entities = gfc2pips_getSymtreeByName(entity_local_name(ent),ns->sym_root);
			if(
				sort_entities && sort_entities->n.sym
				&& (
					sort_entities->n.sym->attr.in_common
					|| sort_entities->n.sym->attr.implicit_type
				)
			){
				gen_remove( &complete_list_of_entities , ent );
				pips_debug(9,"Remove %s from list of entities, element\n",entity_local_name(ent));
			}
		}else{
			POP(complete_list_of_entities_p);
		}
	}*/

	ifdebug(9){
		complete_list_of_entities_p = complete_list_of_entities;
		entity ent = entity_undefined;
		while( complete_list_of_entities_p ){
			ent=complete_list_of_entities_p->car.e;
			if(ent)
				fprintf(stderr,"Complete list of entities, element: %s\n",entity_local_name(ent));
			POP(complete_list_of_entities_p);
		}
	}
	//sort by alphabetic order
	//gen_sort_list(complete_list_of_entities,(gen_cmp_func_t)compare_entities);


	entity_initial(gfc2pips_main_entity) = make_value_code(
		make_code(
			gen_union(
				list_of_extern_entities,
				complete_list_of_entities
			),
			strdup(""),
			make_sequence(NIL),
			gen_union(
				list_of_extern_entities,
				list_of_subroutines
			),
			make_language_fortran()
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
	pips_debug(2, "main entity creation finished\n");


	//get symbols with value, data and explicit-save
	//sym->value is an expression to build the save
	//create data $var /$val/
	//save si explicit save, rien sinon
	instruction data_inst = instruction_undefined;

	/*for (eq = ns->equiv; eq; eq = eq->next){
		show_equiv (eq);
	}*/

	//// declare code
	pips_debug(2, "dumping code ...\n");
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

	SetChains();
	//using ComputeAddresses() point a problem: entities in *STATIC* are computed two times
	//however we have to use it !
	//we have a conflict in storage
	//ComputeAddresses();
	gfc2pips_computeAdresses();

	//gfc2pips_computeEquiv(ns->equiv);
/*show_equiv (gfc_equiv *eq)
	{
	  //quand une ou plusieurs variables sont équivalentes, l'adresse de la première en mémoire deviens celles de toutes. Les adresses de toutes les variables qui s'ensuivent sont décalées si la variable mise plus en avant est plus large que la précédente.
	  show_indent ();
	  fputs ("Equivalence: ", dumpfile);
	  while (eq)
	    {
	      show_expr (eq->expr);
	      eq = eq->eq;
	      if (eq)
		fputs (", ", dumpfile);
	    }
	}
*/
	//SaveChains();//Syntax !! handle equiv in some way, look into it, need to use SetChains(); at the beginning to initiate
	//update_common_sizes();//Syntax !!//we have the job done 2 times if debug is at 9, one time if at 8
	//print_common_layout(stderr,StaticArea,true);
	pips_debug(2, "dumping done\n");

	//bad construction when  parameter = subroutine
	//text t = text_module(gfc2pips_main_entity,gfc_function_body);
	//dump_text(t);

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










/**
 * @brief Retrieve the list of names of every argument of the function, if any
 *
 * Since alternate returns are obsoletes in F90 we do not dump them, still there is a start of dump (but crash if some properties are not activated)
 */
newgen_list gfc2pips_args(gfc_namespace* ns){
	gfc_symtree * current = NULL;
	gfc_formal_arglist *formal;
	newgen_list args_list = NULL, args_list_p = NULL;
	entity e = entity_undefined;
	set_current_number_of_alternate_returns();
	type type_alt_return = make_type_variable(
		make_variable(
			make_basic_overloaded(),
			NULL,
			NULL
		)
	);

	if( ns && ns->proc_name ){

		current = gfc2pips_getSymtreeByName( ns->proc_name->name, ns->sym_root );
		//découper ce bout pour en faire une sous-fonction appelable pour n'importe quel gfc_symtree ?
		if( current && current->n.sym ){
			if (current->n.sym->formal){
				//we have a pb with alternate returns
				formal = current->n.sym->formal;
				if(formal){
					if(formal->sym){
						e = gfc2pips_symbol2entity(
							gfc2pips_getSymtreeByName(
								formal->sym->name,
								ns->sym_root
							)->n.sym
						);
					}else{
						return NULL;//alternate returns are obsolete in F90 (and since we only want it)
						uses_alternate_return(true);
						e = generate_pseudo_formal_variable_for_formal_label(
							CurrentPackage,
							get_current_number_of_alternate_returns()
						);
						if(entity_type(e)==type_undefined)
							entity_type(e) = type_alt_return;
					}
					args_list = args_list_p = CONS(ENTITY, e, NULL );//fprintf(stderr,"%s\n",formal->sym->name);


					formal = formal->next;
					while(formal){
						pips_debug(9,"alt return %s\n", formal->sym?"no":"yes");
						if( formal->sym ){
							e = gfc2pips_symbol2entity(
								gfc2pips_getSymtreeByName(
									formal->sym->name,
									ns->sym_root
								)->n.sym
							);
							CDR(args_list) = CONS( ENTITY, e, NULL );
							args_list = CDR(args_list);
						}else{
							return args_list_p;
							//return args_list_p;//alternate returns are obsolete in F90 (and since we only want it)
							uses_alternate_return(true);
							e = generate_pseudo_formal_variable_for_formal_label(
								CurrentPackage,
								get_current_number_of_alternate_returns()
							);
							if(entity_type(e)==type_undefined)
								entity_type(e) = type_alt_return;
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

/**
 * @brief replace a list of entities by a list of parameters to those entities
 */
void gfc2pips_generate_parameters_list(newgen_list parameters){
	int formal_offset = 1;
	while(parameters){
		entity ent = parameters->car.e;
		pips_debug(8, "parameter founded: %s\n\t\tindice %d\n", entity_local_name(ent), formal_offset );
		entity_storage(ent) = make_storage_formal( make_formal(gfc2pips_main_entity, formal_offset));
		//entity_initial(ent) = MakeValueUnknown();
		type formal_param_type = entity_type(ent);//is the format ok ?
		parameters->car.e = make_parameter( formal_param_type, make_mode_reference(), make_dummy_identifier(ent) );
		formal_offset ++;
		POP(parameters);
	}
}

/**
 * @brief Look for a specific symbol in a tree
 */
gfc_symtree* gfc2pips_getSymtreeByName (char* name, gfc_symtree *st){
  gfc_symtree *return_value = NULL;
  if(!name) return NULL;

  if(!st) return NULL;
  if(!st->n.sym) return NULL;
  if(!st->name) return NULL;

  //much much more information, BUT useless (cause recursive)
  pips_debug(10, "Looking for the symtree called: %s(%d) %s(%d)\n", name, strlen(name), st->name, strlen(st->name) );

  if( strcmp_( st->name, name )==0 ){
	  //much much more information, BUT useless (cause recursive)
	  pips_debug(9, "symbol %s founded\n",name);
	  return st;
  }
  return_value = gfc2pips_getSymtreeByName (name, st->left  );

  if( return_value != NULL) return return_value;
  return_value = gfc2pips_getSymtreeByName (name, st->right  );
  if(return_value != NULL) return return_value;

  //fprintf(stderr,"NULL\n");
  return NULL;
}

/**
 * @brief Extract every and each variable from a namespace
 */
newgen_list gfc2pips_vars(gfc_namespace *ns){
	if(ns){
		return gfc2pips_vars_(ns,gen_nreverse(getSymbolBy(ns,ns->sym_root, gfc2pips_test_variable)));
	}
	return NULL;
}

/**
 * @brief Convert the list of gfc symbols into a list of pips entities with storage, type, everything
 */
newgen_list gfc2pips_vars_(gfc_namespace *ns,newgen_list variables_p){
	newgen_list variables = NULL;
	//variables_p = gen_nreverse(getSymbolBy(ns,ns->sym_root, gfc2pips_test_variable));
	variables_p;//balancer la suite dans une fonction à part afin de pouvoir la réutiliser pour les calls
	//newgen_list arguments,arguments_p;
	//arguments = arguments_p = gfc2pips_args(ns);
	while(variables_p){
		type Type = type_undefined;
		//create entities here
		gfc_symtree *current_symtree = (gfc_symtree*)variables_p->car.e ;
		if(current_symtree && current_symtree->n.sym){
			pips_debug(3, "translation of entity gfc2pips start\n");
			if( current_symtree->n.sym->attr.in_common ){
				pips_debug(4, " %s is in a common\r\n", (current_symtree->name) );
				//we have to skip them, they don't have any place here
				POP(variables_p);
				continue;
			}
			pips_debug(4, " symbol: %s size: %d\r\n", (current_symtree->name), current_symtree->n.sym->ts.kind );
			int TypeSize = gfc2pips_symbol2size(current_symtree->n.sym);
			value Value;// = MakeValueUnknown();
			Type = gfc2pips_symbol2type(current_symtree->n.sym);
			pips_debug(3, "Type done\n");


			//handle the value
			//don't ask why it is is_value_constant
			if(
				Type!=type_undefined
				&& current_symtree->n.sym->ts.type==BT_CHARACTER
			){
				pips_debug(5, "the symbol is a string\n");
				Value = make_value_constant(
					MakeConstantLitteral()//MakeConstant(current_symtree->n.sym->value->value.character.string,is_basic_string)
				);
			}else{
				pips_debug(5, "the symbol is a constant\n");
				Value = make_value_constant(
					make_constant_int((void *) TypeSize)
				);
			}

			int i,j=0;
			//newgen_list list_of_dimensions = gfc2pips_get_list_of_dimensions(current_symtree);
			//si allocatable alors on fait qqch d'un peu spécial

			//we look into the list of arguments to know if the entity is in and thus the offset in the stack
			/*i=0;j=1;
			arguments_p = arguments;
			while(arguments_p){
				//fprintf(stderr,"%s %s\n",entity_local_name((entity)arguments_p->car.e),current_symtree->name);
				if(strcmp_( entity_local_name( (entity)arguments_p->car.e ), current_symtree->name )==0 ){
					i=j;
					break;
				}
				j++;
				POP(arguments_p);
			}*/
			//fprintf(stderr,"%s %d\n",current_symtree->name,i);

			variables = CONS(ENTITY, FindOrCreateEntity(CurrentPackage, str2upper(gfc2pips_get_safe_name(current_symtree->name)) ), variables);
			entity_type((entity)variables->car.e) = Type;
			entity_initial((entity)variables->car.e) = Value;//make_value(is_value_code, make_code(NULL, strdup(""), make_sequence(NIL),NIL));
			if(current_symtree->n.sym->attr.dummy){
				pips_debug(9,"formal parameter \"%s\" put in FORMAL\n",current_symtree->n.sym->name);
				//we have a formal parameter (argument of the function/subroutine)
				/*if(entity_storage((entity)variables->car.e)==storage_undefined)
				entity_storage((entity)variables->car.e) = make_storage_formal(
					make_formal(
						gfc2pips_main_entity,
						i
					)
				);*/
			}else if( current_symtree->n.sym->attr.flavor==FL_PARAMETER ){
				pips_debug(9,"Variable \"%s\" (PARAMETER) put in ROM\n",current_symtree->n.sym->name);
				//we have a parameter, we rewrite some attributes of the entity
				entity_type((entity)variables->car.e) = make_type_functional( make_functional(NIL, entity_type((entity)variables->car.e)));
				entity_initial((entity)variables->car.e) = MakeValueSymbolic(gfc2pips_expr2expression(current_symtree->n.sym->value));
				if(entity_storage((entity)variables->car.e)==storage_undefined)
				entity_storage((entity)variables->car.e) = MakeStorageRom();
			}else{
				//we have a variable
				entity area = entity_undefined;
				if(gfc2pips_test_save(NULL, current_symtree)){
					area = FindOrCreateEntity(CurrentPackage,STATIC_AREA_LOCAL_NAME);
					pips_debug(9,"Variable \"%s\" put in RAM \"%s\"\n",entity_local_name((entity)variables->car.e),STATIC_AREA_LOCAL_NAME);
					//set_common_to_size(StaticArea,CurrentOffsetOfArea(StaticArea,(entity)variables->car.e));
				}else{
					if(
						current_symtree->n.sym->as
						&& current_symtree->n.sym->as->type!=AS_EXPLICIT
						&& !current_symtree->n.sym->value
					){//some other criteria is needed
						if(current_symtree->n.sym->attr.allocatable){
							//we do know this entity is allocatable, it's place is in the heap. BUT in order to prettyprint the ALLOCATABLE statement, we need an other means to differenciate allocatables from the others.
							area = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,"*ALLOCATABLE*");
						}else{
							area = StackArea;
						}
					}else{
						area = DynamicArea;
					}
					pips_debug(
						9,
						"Variable \"%s\" put in RAM \"%s\"\n",
						entity_local_name((entity)variables->car.e),
						area==DynamicArea ? DYNAMIC_AREA_LOCAL_NAME:(area==StackArea ?STACK_AREA_LOCAL_NAME:"*ALLOCATABLE*")
					);
				}
				ram _r_ = make_ram(
					get_current_module_entity(),
					area,
					UNKNOWN_RAM_OFFSET,
					NULL
				);
				if(entity_storage((entity)variables->car.e)==storage_undefined)
				entity_storage( (entity)variables->car.e ) = make_storage_ram( _r_ );
			}

			//code for pointers
			//if(Type!=type_undefined){
				//variable_dimensions(type_variable(entity_type( (entity)variables->car.e ))) = gfc2pips_get_list_of_dimensions(current_symtree);
				/*if(current_symtree->n.sym->attr.pointer){
					basic b = make_basic(is_basic_pointer, Type);
					type newType = make_type(is_type_variable, make_variable(b, NIL, NIL));
					entity_type((entity)variables->car.e) = newType;
				}*/
			//}
			pips_debug(3, "translation of entity gfc2pips end\n");
		}else{
			variables_p->car.e = NULL;
		}
		POP(variables_p);
	}
	return variables;
}

/**
 * @brief build a list of externals entities
 */
newgen_list gfc2pips_get_extern_entities(gfc_namespace *ns){
	newgen_list list_of_extern,list_of_extern_p;
	list_of_extern_p = list_of_extern = getSymbolBy(ns,ns->sym_root,gfc2pips_test_extern);
	while(list_of_extern_p){
		gfc_symtree* curr = list_of_extern_p->car.e;
		entity e = gfc2pips_symbol2entity(curr->n.sym);
		//don't modify e, it will be automatically updated when the information is known
		//entity_storage(e) = NULL;
		list_of_extern_p->car.e = e;
		POP(list_of_extern_p);
	}
	return list_of_extern;
}
/**
 * @brief return a list of elements needing a DATA statement
 */
newgen_list gfc2pips_get_data_vars(gfc_namespace *ns){
	return getSymbolBy(ns,ns->sym_root,gfc2pips_test_data);
}
/**
 * @brief return a list of SAVE elements
 */
newgen_list gfc2pips_get_save(gfc_namespace *ns){
	return getSymbolBy(ns,ns->sym_root,gfc2pips_test_save);
}


/**
 * @brief build a list - if any - of dimension elements from the gfc_symtree given
 */
newgen_list gfc2pips_get_list_of_dimensions(gfc_symtree *st){
	if(st){
		return gfc2pips_get_list_of_dimensions2(st->n.sym);
	}else{
		return NULL;
	}
}
/**
 * @brief build a list - if any - of dimension elements from the gfc_symbol given
 */
newgen_list gfc2pips_get_list_of_dimensions2(gfc_symbol *s){
	newgen_list list_of_dimensions = NULL;
	int i=0,j=0;
	if( s && s->attr.dimension ){
		gfc_array_spec *as = s->as;
		const char *c;
		pips_debug(4, "%s is an array\n",s->name);
		if ( as!=NULL && as->rank != 0){
			//according to the type of array we create different types of dimensions parameters
			switch (as->type){
				case AS_EXPLICIT:
					c = strdup("AS_EXPLICIT");
					//create the list of dimensions
					i = as->rank-1;
					do{
						//check lower ou upper n'est pas une variable dont la valeur est inconnue
						list_of_dimensions = gen_cons(
							make_dimension(
									gfc2pips_expr2expression(as->lower[i]),
									gfc2pips_expr2expression(as->upper[i])
							),
							list_of_dimensions
						);
					}while(--i >= j);
				break;
				case AS_DEFERRED://beware allocatable !!!
					c = strdup("AS_DEFERRED");
					i = as->rank-1;
					if(s->attr.allocatable){
						do{
							list_of_dimensions = gen_cons(
								make_dimension(
									MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)),
									MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
								),
								list_of_dimensions
							);
						}while(--i >= j);
					}else{
						do{
							list_of_dimensions = gen_cons(
								make_dimension(
									MakeIntegerConstantExpression("1"),
									MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
								),
								list_of_dimensions
							);
						}while(--i >= j);
					}
				break;
				//AS_ASSUMED_...  means information come from a dummy argument and the property is inherited from the call
				case AS_ASSUMED_SIZE://means only the last set of dimensions is unknown
					j=1;
					c = strdup("AS_ASSUMED_SIZE");
					//create the list of dimensions
					i = as->rank-1;
					while(i>j){
						//check lower ou upper n'est pas une variable dont la valeur est inconnue
						list_of_dimensions = gen_int_cons(
							make_dimension(
									gfc2pips_expr2expression(as->lower[i]),
									gfc2pips_expr2expression(as->upper[i])
							),
							list_of_dimensions
						);
						i--;
					}

					list_of_dimensions = gen_int_cons(
						make_dimension(
								gfc2pips_expr2expression(as->lower[i]),
								MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))
						),
						list_of_dimensions
					);
				break;
				case AS_ASSUMED_SHAPE: c = strdup("AS_ASSUMED_SHAPE"); break;
				default:
				  gfc_internal_error ("show_array_spec(): Unhandled array shape "
							  "type.");
			}
		}
		pips_debug(4, "%d dimensions detected for %s\n",gen_length(list_of_dimensions),s->name);
	}

	return list_of_dimensions;
}


/**
 * @brief Look for a set of symbols filtered by a predicate function
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

/**
 * @brief get variables who are not implicit or are needed to be declared for data statements hence variable that should be explicit in PIPS
 */
bool gfc2pips_test_variable(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return ( st->n.sym->attr.flavor == FL_VARIABLE || st->n.sym->attr.flavor == FL_PARAMETER )
		/*&& (
			(!st->n.sym->attr.implicit_type||st->n.sym->attr.save==SAVE_EXPLICIT)
			|| st->n.sym->value//very important
		)*/
		&& !st->n.sym->attr.external
		//&& !st->n.sym->attr.in_common
		&& !st->n.sym->attr.pointer
		&& !st->n.sym->attr.dummy;
}
/*
 * @brief test if it is a variable
 */
bool gfc2pips_test_variable2(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && !st->n.sym->attr.dummy;
}
/**
 * @brief test if it is an external function
 */
bool gfc2pips_test_extern(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.external || st->n.sym->attr.proc == PROC_EXTERNAL;
}
bool gfc2pips_test_subroutine(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return (
		st->n.sym->attr.flavor == FL_PROCEDURE
		&& (st->n.sym->attr.subroutine||st->n.sym->attr.function)
		&& strncmp(st->n.sym->name,"__",strlen("__"))!=0
	);
	//return st->n.sym->attr.subroutine && strcmp(str2upper(strdup(ns->proc_name->name)), str2upper(strdup(st->n.sym->name)))!=0;
}

/**
 * @brief test if it is a allocatable entity
 */
bool gfc2pips_test_allocatable(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.allocatable;
}
/**
 * @brief test if it is a dummy parameter (formal parameter)
 */
bool gfc2pips_test_arg(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.flavor == EXPR_VARIABLE && st->n.sym->attr.dummy;
}
/**
 * @brief test if there is a value to stock
 */
bool gfc2pips_test_data(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->value && st->n.sym->attr.flavor != FL_PARAMETER && st->n.sym->attr.flavor != FL_PROCEDURE;
}
/**
 * @brief test if there is a SAVE to do
 */
bool gfc2pips_test_save(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.save != SAVE_NONE;
}
/**
 * @brief test function to know if it is a common, always true because the tree is completely separated therefore the function using it only create a list
 */
bool gfc2pips_get_commons(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree __attribute__ ((__unused__)) *st ){
	return true;
}
bool gfc2pips_get_incommon(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree __attribute__ ((__unused__)) *st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.in_common;
}
/**
 *
 */
bool gfc2pips_test_dimensions(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree* st ){
	if(!st || !st->n.sym) return false;
	return st->n.sym->attr.dimension;
}

entity gfc2pips_check_entity_doesnt_exists(char *s){
	entity e = entity_undefined;
	string full_name;
	//main program
	full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, MAIN_PREFIX, str2upper(strdup(s)), NULL);
	e = gen_find_tabulated(full_name, entity_domain);

	//module
	if(e==entity_undefined){
		full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, str2upper(strdup(s)), NULL);
		e = gen_find_tabulated(full_name, entity_domain);
	}

	//simple entity
	if(e==entity_undefined){
		full_name = concatenate(CurrentPackage, MODULE_SEP_STRING, str2upper(strdup(s)), NULL);
		e = gen_find_tabulated(full_name, entity_domain);
	}
	return e;
}
entity gfc2pips_check_entity_program_exists(char *s){
	string full_name;
	//main program
	full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, MAIN_PREFIX, str2upper(strdup(s)), NULL);
	return gen_find_tabulated(full_name, entity_domain);
}
entity gfc2pips_check_entity_module_exists(char *s){
	string full_name;
	//module
	full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, str2upper(strdup(s)), NULL);
	return gen_find_tabulated(full_name, entity_domain);
}
entity gfc2pips_check_entity_block_data_exists(char *s){
	string full_name;
	full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING,BLOCKDATA_PREFIX, str2upper(strdup(s)), NULL);
	return gen_find_tabulated(full_name, entity_domain);
}
entity gfc2pips_check_entity_exists(char *s){
	string full_name;
	//simple entity
	full_name = concatenate(CurrentPackage, MODULE_SEP_STRING, str2upper(strdup(s)), NULL);
	return gen_find_tabulated(full_name, entity_domain);
}

/**
 * @brief translate a gfc symbol to a PIPS entity, check if it is a function, program, subroutine or else
 */
//add declarations of parameters
entity gfc2pips_symbol2entity( gfc_symbol* s ){
	char* name = gfc2pips_get_safe_name(s->name);
	entity e = entity_undefined;//gfc2pips_check_entity_doesnt_exists(name);
	bool module = false;
	/*bool non_exists = (e==entity_undefined);

	if( non_exists == false ){
		pips_debug(9,"Entity %s already exists\n",name);
		free(name);
		return e;
	}*/

	if( s->attr.flavor==FL_PROGRAM||s->attr.is_main_program ){
		if((e=gfc2pips_check_entity_program_exists(name))==entity_undefined){
			pips_debug(9, "create main program %s\n",name);
			e = make_empty_program(str2upper((name)));
		}
		module = true;
	}else if( s->attr.function ){
		if((e=gfc2pips_check_entity_module_exists(name))==entity_undefined){
			pips_debug(9, "create function %s\n",name);
			e = make_empty_function(str2upper((name)),gfc2pips_symbol2type(s));
		}
		module = true;
	}else if( s->attr.subroutine ){
		if((e=gfc2pips_check_entity_module_exists(name))==entity_undefined){
			pips_debug(9, "create subroutine %s\n",name);
			e = make_empty_subroutine(str2upper((name)));
		}
		module = true;
	}else if( s->attr.flavor == FL_BLOCK_DATA){
		if((e=gfc2pips_check_entity_block_data_exists(name))==entity_undefined){
			pips_debug(9, "block data \n");
			e = make_empty_blockdata(str2upper((name)));
		}
		module = true;
	}else{
		pips_debug(9, "create entity %s\n",name);
		e = FindOrCreateEntity(CurrentPackage,str2upper((name)));
		if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
		if(entity_type(e)==type_undefined) entity_type(e) = gfc2pips_symbol2type(s);
		//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
		free(name);
		return e;
	}
	//it is a module and we do not know it yet, so we put an empty content in it
	if( module ){
		//message_assert("arg ! bad handling",entity_initial(e)==value_undefined);
		//fprintf(stderr,"value ... ... ... %s\n",entity_initial(e)==value_undefined?"ok":"nok");
		if(entity_initial(e)==value_undefined){
			entity_initial(e) = make_value_code(
				make_code(NULL,strdup(""),make_sequence(NIL),NULL, make_language_fortran())
			);
		}
	}
	free(name);
	return e;
}

/**
 * @brief translate a gfc symbol to a top-level entity
 */
entity gfc2pips_symbol2entity2(gfc_symbol* s){
	char* name = gfc2pips_get_safe_name(s->name);
	entity e = gfc2pips_check_entity_doesnt_exists(name);
	if(e!=entity_undefined){
		pips_debug(9,"Entity %s already exists\n",name);
		free(name);
		return e;
	}
	e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME),str2upper((name)));
	if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
	if(entity_type(e)==type_undefined) entity_type(e) = gfc2pips_symbol2type(s);
	//if(entity_storage(e)==storage_undefined) entity_storage(e) = MakeStorageRom();
	free(name);
	return e;
}

/**
 * @brief a little bit more elaborated FindOrCreateEntity
 */
entity gfc2pips_char2entity(char* package, char* s){
	s = gfc2pips_get_safe_name(s);
	entity e = FindOrCreateEntity(package, str2upper(s));
	if(entity_initial(e)==value_undefined) entity_initial(e) = MakeValueUnknown();
	if(entity_type(e)==type_undefined) entity_type(e) = make_type_unknown();
	free(s);
	return e;
}

/**
 * @brief gfc replace some functions by an homemade one, we check and return a copy of the original one if it is the case
 */
char* gfc2pips_get_safe_name(char* str){
	if(strncmp_("_gfortran_exit_", str, strlen("_gfortran_exit_") )==0){
			return strdup("exit");
	}else if(strncmp_("_gfortran_float", str, strlen("_gfortran_float") )==0){
			return strdup("float");
	}else{
		return strdup(str);
	}
}








/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */
/**
 * @brief create a <dimension> from the integer value given
 */
dimension gfc2pips_int2dimension(int n){
	return make_dimension(MakeIntegerConstantExpression("1"),gfc2pips_int2expression(n));
}

/**
 * @brief translate a int to an expression
 */
expression gfc2pips_int2expression(int n){
	//return int_expr(n);
	if(n<0){
		return MakeFortranUnaryCall(CreateIntrinsic("--"), entity_to_expression(gfc2pips_int_const2entity(-n)));
	}else{
		return entity_to_expression(gfc2pips_int_const2entity(n));
	}
}
/**
 * @brief translate a real to an expression
 */
expression gfc2pips_real2expression(double r){
	if(r<0.){
		return MakeFortranUnaryCall(CreateIntrinsic("--"), entity_to_expression(gfc2pips_real2entity(-r)));
	}else{
		return entity_to_expression(gfc2pips_real2entity(r));
	}
}
/**
 * @brief translate a bool to an expression
 */
expression gfc2pips_logical2expression(bool b){
	//return int_expr(b!=false);
	return entity_to_expression(gfc2pips_logical2entity(b));
}


/**
 * @brief translate an integer to a PIPS constant, assume n is positive (or it will not be handled properly)
 */
entity gfc2pips_int_const2entity(int n){
	char str[30];
	sprintf(str,"%d",n);
	return MakeConstant(str, is_basic_int);
}
/**
 * @brief dump an integer to a PIPS label entity
 * @param n the value of the integer
 */
entity gfc2pips_int2label(int n){
	//return make_loop_label(n,concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,NULL));
	char str[60];
	sprintf(str,"%s%s%s%d",TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);//fprintf(stderr,"new label: %s %s %s %s %d\n",str,TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);
	return make_label(str);
}

/**
 * @brief dump reals to PIPS entities
 * @param r the double to create
 * @return the corresponding entity
 *
 * we have a big issue with reals:
 * 16.53 => 16.530001
 * 16.56 => 16.559999
 */
entity gfc2pips_real2entity(double r){
	//create a more elaborate function to output a fortran format or something like it ?
	char str[60];
	if( r==0. || r==(double)((int)r) ){//if the real represent an integer, display a string representing an integer then
		sprintf(str,"%d",(int)r);
	}else{
		//we need a test to know if we output in scientific or normal mode
		//sprintf(str,"%.32f",r);
		sprintf(str,"%.6e",r);
		//fprintf(stderr,"copy of the entity name(real) %s\n",str);
	}
	gfc2pips_truncate_useless_zeroes(str);
	return MakeConstant(str, is_basic_float);
}

/**
 * @brief translate a boolean to a PIPS/fortran entity
 */
entity gfc2pips_logical2entity(bool b){
	return MakeConstant(b?".TRUE.":".FALSE.",is_basic_logical);
}

/**
 * @brief translate a string from a table of integers in gfc to one of chars in PIPS, escape all ' in the string
 * @param c the table of integers in gfc
 * @param nb the size of the table
 *
 * The function only calculate the number of ' to escape and give the information to gfc2pips_gfc_char_t2string_
 * TODO: optimize to know if we should put " or ' quotes
 */
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

/**
 * @brief translate a string from a table of integers in gfc to one of chars in PIPS, escape all ' in the string
 * @param c the table of integers in gfc
 * @param nb the size of the table
 *
 * Stupidly add ' before ' and add ' at the beginning and the end of the string
 */
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

/**
 * @brief translate the <nb> first elements of <c> from a wide integer representation to a char representation
 * @param c the gfc integer table
 */
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

/**
 * @brief try to create the PIPS type that would be associated by the PIPS default parser
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
			pips_debug( 5, "Type unknown\n" );
			return make_type_unknown();
		break;
		case BT_DERIVED:
		case BT_PROCEDURE:
		case BT_HOLLERITH:
		case BT_VOID:
		default:
			pips_error("gfc2pips_symbol2type","An error occurred in the type to type translation: impossible to translate the symbol.\n");
			return type_undefined;
			//return make_type_unknown();
	}
	pips_debug(5, "Basic type is : %d\n",(int)ut);
	//variable_dimensions(type_variable(entity_type( (entity)CAR(variables) ))) = gfc2pips_get_list_of_dimensions(current_symtree)
	if(ut!=is_basic_string){
		return MakeTypeVariable(
			make_basic(
				ut,
				(void*) ( (ut==is_basic_complex?2:1) * gfc2pips_symbol2size(s) )// * gfc2pips_symbol2sizeArray(s))
			),
			gfc2pips_get_list_of_dimensions2(s)
		);
	}else{
		if(s){
			if( s->ts.cl && s->ts.cl->length ){
				return MakeTypeVariable(
					make_basic(
						ut,
						(void*) make_value_constant(
							//don't use litteral, it's a trap !
							make_constant_int(
								gfc2pips_symbol2size(s)
							)//it is here we have to specify the length of the character symbol
						)
					),
					gfc2pips_get_list_of_dimensions2(s)
				);
			}else{
				//CHARACTER * (*) texte
				return MakeTypeVariable(
					make_basic(
						ut,
						 MakeValueUnknown()
					),
					NULL//gfc2pips_get_list_of_dimensions2(s)
				);
			}
		}
	}
	pips_debug( 5, "WARNING: no type\n" );
	return type_undefined;
	//return make_type_unknown();
}

/**
 * @brief return the size of an elementary element:  REAL*16 A    CHARACTER*17 B
 * @param s symbol of the entity
 */
int gfc2pips_symbol2size(gfc_symbol *s){
	if(
		s->ts.type==BT_CHARACTER
		&& s->ts.cl
		&& s->ts.cl->length
	){
		pips_debug(
			9,
			"size of %s: %d\n",
			s->name,
			mpz_get_si(s->ts.cl->length->value.integer)
		);
		return mpz_get_si(s->ts.cl->length->value.integer);
	}else{
		pips_debug(9, "size of %s: %d\n",s->name,s->ts.kind);
		return s->ts.kind;
	}
}
/**
 * @brief calculate the total size of the array whatever the bounds are:  A(-5,5)
 * @param s symbol of the array
 */
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
	pips_debug(9, "size of %s: %d\n",s->name,retour);
	return retour;
}

/**
 * @brief convert a list of indices from gfc to PIPS, assume there is no range (dump only the min range element)
 * @param ar the struct with indices
 * only for AR_ARRAY references
 */
newgen_list gfc2pips_array_ref2indices(gfc_array_ref *ar){
	int i;
	newgen_list indices=NULL,indices_p=NULL;

	if(!ar->start[0]){
		pips_debug(9,"no indice\n");
		return NULL;
	}
	//expression ex = gfc2pips_mkRangeExpression(ar->start[0],ar->end[0]);
	//reference_entity(syntax_reference(expression_syntax(ex))) = NULL;
	//indices_p = CONS( EXPRESSION, gfc2pips_expr2expression(ar->end[0]), indices_p );
	indices_p = CONS( EXPRESSION, gfc2pips_expr2expression(ar->start[0]), NIL );
	//indices_p = CONS( EXPRESSION, gfc2pips_mkRangeExpression(ar->start[0],ar->end[0]), NIL );
	indices=indices_p;
	for( i=1 ; ar->start[i] ;i++){
		//indices_p = CONS( EXPRESSION, gfc2pips_expr2expression(ar->end[i]), indices_p );
		CDR(indices_p) = CONS( EXPRESSION, gfc2pips_expr2expression(ar->start[i]), NIL );
		indices_p = CDR(indices_p);
		//indices_p = CONS( EXPRESSION, gfc2pips_mkRangeExpression(ar->start[i],ar->end[i]), indices_p );
	}
	pips_debug(9,"%d indice(s)\n", gen_length(indices) );
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

/**
 * @brief Test if there is a range:  A( 1, 2, 3:5 )
 * @param ar the gfc structure containing the information about range
 */
bool gfc2pips_there_is_a_range(gfc_array_ref *ar){
	int i;
	if( !ar || !ar->start || !ar->start[0] ) return false;
	for( i=0 ; ar->start[i] ;i++){
		if(ar->end[i])return true;
	}
	return false;
}

/**
 * @brief Create an expression similar to the substring implementation, but with a couple of parameters(min-max) for each indice
 * @param ent the entity refered by the indices
 * @param ar the gfc structure containing the information
 */
expression gfc2pips_mkRangeExpression(entity ent, gfc_array_ref *ar){
	expression ref = make_expression(
		make_syntax_reference(
			make_reference(ent, NULL)
		),
	    normalized_undefined
	);

	entity substr = entity_intrinsic(SUBSTRING_FUNCTION_NAME);
	newgen_list lexpr = NULL;
	int i;
	for( i=0 ; ar->start[i] ;i++){
		lexpr = CONS(EXPRESSION,
			ar->end[i] ? gfc2pips_expr2expression(ar->end[i]) : MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)),
			CONS(EXPRESSION,
				gfc2pips_expr2expression(ar->start[i]),
				lexpr
			)
		);
	}
	lexpr = CONS(EXPRESSION, ref, gen_nreverse(lexpr) );
	syntax s = make_syntax_call( make_call(substr, lexpr));
	return make_expression( s, normalized_undefined );
}


//this is to know if we have to add a continue statement just after the loop statement for (exit/break)
bool gfc2pips_last_statement_is_loop = false;

/**
 * Declaration of instructions
 * @brief We need to differentiate the instructions at the very top of the module from those in other blocks because of the declarations of DATA, SAVE, or simply any variable
 * @param ns the top-level entity from gfc. We need it to retrieve some more informations
 * @param c the struct containing information about the instruction
 */
instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c){
	newgen_list list_of_data_symbol,list_of_data_symbol_p;
	list_of_data_symbol_p = list_of_data_symbol = gfc2pips_get_data_vars(ns);

	//create a sequence and put everything into it ? is it right ?
	newgen_list list_of_statements,list_of_statements_p;
	list_of_statements_p = list_of_statements= NULL;

	instruction i = instruction_undefined;


	//dump DATA
	//create a list of statements and dump them in one go
	//test for each if there is an explicit save statement
	//fprintf(stderr,"nb of data statements: %d\n",gen_length(list_of_data_symbol_p));
	if(list_of_data_symbol_p){
		//int add,min,is,tr,at,e,ur;
		do{
			//if there are parts in the DATA statement, we have got a pb !
			instruction ins = instruction_undefined;
			//do{
				ins = gfc2pips_symbol2data_instruction(
					( (gfc_symtree*)list_of_data_symbol_p->car.e )->n.sym
				);
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

				/*if(list_of_statements){
					CDR(list_of_statements) = lst;
					list_of_statements = CDR(list_of_statements);
				}else{
					list_of_statements_p = list_of_statements = lst;
				}*/
			//}while(ins!=instruction_undefined);
			POP(list_of_data_symbol_p);
		}while( list_of_data_symbol_p );
	}

	//dump equivalence statements
	//int OffsetOfReference(reference r)
	//int CurrentOffsetOfArea(entity a, entity v)

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

	newgen_list list_of_save = gfc2pips_get_save(ns);
	//save_all_entities();//Syntax !!!
	pips_debug(3,"%d SAVE founded\n",gen_length(list_of_save));
	while(list_of_save){
		static int offset_area = 0;
		//we should know the current offset of every and each memory area or are equivalence not yet dumped ?
		// ProcessSave(<entity>); <=> MakeVariableStatic(<entity>,true)
		// => balance le storage dans RAM, ram_section(r) = StaticArea
		//fprintf(stderr,"%d\n",list_of_save->car.e);
		pips_debug(4,"entity to SAVE %s\n",((gfc_symtree*)list_of_save->car.e)->n.sym->name);
		entity curr_save = gfc2pips_symbol2entity(((gfc_symtree*)list_of_save->car.e)->n.sym);
		pips_debug(9,"Size of %s %d\n",STATIC_AREA_LOCAL_NAME, CurrentOffsetOfArea(StaticArea ,curr_save));//Syntax !
		//entity_type(curr_save) = make_type_area(<area>);
		//entity g = local_name_to_top_level_entity(entity_local_name(curr_save));
		if(	entity_storage(curr_save) == storage_undefined ){
			//int offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
			entity_storage(curr_save) = make_storage_ram(
				make_ram(get_current_module_entity(),StaticArea, UNKNOWN_RAM_OFFSET, NIL)
			);
			//AddVariableToCommon(StaticArea,curr_save);
			//SaveEntity(curr_save);
			//offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
			//set_common_to_size(StaticArea,offset_area);
		}else if(
			storage_ram_p(entity_storage(curr_save))
			&& ram_section(storage_ram(entity_storage(curr_save))) == DynamicArea
		){
			//int offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
			ram_section(storage_ram(entity_storage(curr_save))) = StaticArea;
			ram_offset(storage_ram(entity_storage(curr_save))) = UNKNOWN_RAM_OFFSET;
			//ram_offset(storage_ram(entity_storage(curr_save))) = UNKNOWN_RAM_OFFSET;
			//AddVariableToCommon(StaticArea,curr_save);
			//SaveEntity(curr_save);
			//offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
			//set_common_to_size(StaticArea,offset_area);
		}else if(storage_ram_p(entity_storage(curr_save))
			&& ram_section(storage_ram(entity_storage(curr_save))) == StaticArea
		){
			//int offset_area = CurrentOffsetOfArea(StackArea,curr_save);
			//set_common_to_size(StaticArea,offset_area);
			pips_debug(9,"Entity %s already in the Static area\n",entity_name(curr_save));
		}else{
			pips_user_warning("Static entity(%s) not in the correct memory Area: %s\n",entity_name(curr_save),storage_ram_p(entity_storage(curr_save))?entity_name(ram_section(storage_ram(entity_storage(curr_save)))):"?");
		}
		POP(list_of_save);
	}
	if( !c ){
		//fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
		return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
	}


	//dump other
	//we know we have at least one instruction, otherwise we would have returned an empty list of statements
	do{
		if(c && c->op==EXEC_SELECT){
			list_of_statements  = gen_nconc(
				list_of_statements,
				gfc2pips_dumpSELECT(c)
			);
			c=c->next;
			if(list_of_statements)break;
		}else{
			i = gfc2pips_code2instruction_(c);
			if(i!=instruction_undefined){
				//string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
				statement s = make_statement(
					gfc2pips_code2get_label(c),
					STATEMENT_NUMBER_UNDEFINED,
					STATEMENT_ORDERING_UNDEFINED,
					//comments,
					empty_comments,
					i,
					NULL,
					NULL,
					empty_extensions()
				);
				//unlike the classical method, we don't know if we have had a first statement (data inst)
				list_of_statements = gen_nconc(
					list_of_statements,
					CONS(STATEMENT, s, NIL)
				);
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
		if(c && c->op==EXEC_SELECT){
			list_of_statements  = gen_nconc(
				list_of_statements,
				gfc2pips_dumpSELECT(c)
			);
		}else{
			i = gfc2pips_code2instruction_(c);
			if(i!=instruction_undefined){
				string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
				s = make_statement(
					instruction_sequence_p(i)?entity_empty_label():gfc2pips_code2get_label(c),
					STATEMENT_NUMBER_UNDEFINED,
					STATEMENT_ORDERING_UNDEFINED,
					comments,
					//empty_comments,
					i,
					NULL,
					NULL,
					empty_extensions ()
				);
				list_of_statements = gen_nconc(
					list_of_statements,
					CONS(STATEMENT, s, NIL)
				);

			}
		}
	}

	//FORMAT
	//we have the informations only at the end, (<=>now)
	newgen_list gfc2pips_format_p = gfc2pips_format;
	newgen_list gfc2pips_format2_p = gfc2pips_format2;fprintf(stderr,"list of formats: 0x%e %d\n",gfc2pips_format,gen_length(gfc2pips_format));
	newgen_list list_of_statements_format = NULL;
	while(gfc2pips_format_p){
		i = MakeZeroOrOneArgCallInst("FORMAT", (expression)gfc2pips_format_p->car.e);
		statement s = make_statement(
			gfc2pips_int2label((int)gfc2pips_format2_p->car.e),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			//comments,
			empty_comments,
			i,
			NULL,
			NULL,
			empty_extensions()
		);
		//unlike the classical method, we don't know if we have had a first statement (data inst)
		list_of_statements_format = gen_nconc(
				list_of_statements_format,
			CONS(STATEMENT, s, NIL)
		);
		POP(gfc2pips_format_p);
		POP(gfc2pips_format2_p);
	}
	list_of_statements = gen_nconc(list_of_statements_format,list_of_statements);

	if(list_of_statements){
		return make_instruction_block(list_of_statements);//make a sequence <=> make_instruction_sequence(make_sequence(list_of_statements));
	}else{
		fprintf(stderr,"Warning ! no instruction dumped => very bad\n");
		return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
	}
}
/**
 * Build an instruction sequence
 * @brief same as {func}__TOP but without the declarations
 * @param ns the top-level entity from gfc. We need it to retrieve some more informations
 * @param c the struct containing information about the instruction
 */
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence){
	newgen_list list_of_statements;
	instruction i = instruction_undefined;
	force_sequence = true;
	if(!c){
		if(force_sequence){
			//fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
			return make_instruction_block(CONS(STATEMENT, make_stmt_of_instr(make_instruction_block(NULL)), NIL));
		}else{
			//fprintf(stderr,"Undefined code\n");
			return make_instruction_block(NULL);
		}
	}
	//No block, only one instruction
	//if(!c->next && !force_sequence )return gfc2pips_code2instruction_(c);

	//create a sequence and put everything into it ? is it right ?
	list_of_statements= NULL;

	//entity l = gfc2pips_code2get_label(c);
	do{
		if(c && c->op==EXEC_SELECT){
			list_of_statements  = gen_nconc(
				list_of_statements,
				gfc2pips_dumpSELECT(c)
			);
			c=c->next;
			if(list_of_statements)break;
		}else{
			i = gfc2pips_code2instruction_(c);
			if(i!=instruction_undefined){
				string comments  = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
				list_of_statements = CONS(
					STATEMENT,
					make_statement(
						gfc2pips_code2get_label(c),
						STATEMENT_NUMBER_UNDEFINED,
						STATEMENT_ORDERING_UNDEFINED,
						comments,
						//empty_comments,
						i,
						NIL,
						NULL,
						empty_extensions ()
					),
					list_of_statements
				);
			}
		}
		c=c->next;
	}while( i==instruction_undefined && c);

	//statement_label((statement)list_of_statements->car.e) = gfc2pips_code2get_label(c);

	for( ; c ; c=c->next ){
		statement s = statement_undefined;
		//l = gfc2pips_code2get_label(c);
		//on lie l'instruction suivante à la courante
		//fprintf(stderr,"Dump the following instructions\n");
		//CONS(STATEMENT,instruction_to_statement(gfc2pips_code2instruction_(c)),list_of_statements);
		int curr_label_num = gfc2pips_last_created_label;
		if(c && c->op==EXEC_SELECT){
			list_of_statements  = gen_nconc(
				list_of_statements,
				gfc2pips_dumpSELECT(c)
			);
		}else{
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
					list_of_statements = gen_nconc(
						list_of_statements,
						CONS(STATEMENT, s, NIL)
					);
				}
				if(gfc2pips_get_last_loop()==c){
					s = make_continue_statement(gfc2pips_int2label(curr_label_num-1));
					list_of_statements = gen_nconc(
						list_of_statements,
						CONS(STATEMENT, s, NIL)
					);
				}
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
	//CONS(STATEMENT, make_return_statement(FindOrCreateEntity(CurrentPackage,"TEST")),list_of_statements);
	return make_instruction_block(list_of_statements);//make a sequence <=> make_instruction_sequence(make_sequence(list_of_statements));
}

/**
 * @brief this function create an atomic statement, no block of data
 * @param c the instruction to translate from gfc
 * @return the statement equivalent in PIPS
 * never call this function except in gfc2pips_code2instruction or in recursive mode
 */
instruction gfc2pips_code2instruction_(gfc_code* c){
	//do we have a label ?
	//if(c->here){}
	//debug(5,"gfc2pips_code2instruction","Start function\n");
	switch (c->op){
		case EXEC_NOP://an instruction without anything => continue statement
		case EXEC_CONTINUE:
			pips_debug(5, "Translation of CONTINUE\n");
			return make_instruction_call( make_call(CreateIntrinsic("CONTINUE"), NULL));
		break;
/*	    case EXEC_ENTRY:
	      fprintf (dumpfile, "ENTRY %s", c->ext.entry->sym->name);
	      break;
*/
		case EXEC_INIT_ASSIGN:
		case EXEC_ASSIGN:{
			pips_debug(5, "Translation of ASSIGN\n\t%d %d\n", c->expr, c->expr2 );
			//if(c->expr->expr_type==EXPR_FUNCTION)
			/*call _call_ = make_call(
				entity_intrinsic(ASSIGN_OPERATOR_NAME),
				CONS(EXPRESSION, gfc2pips_expr2expression(c->expr), CONS(EXPRESSION, gfc2pips_expr2expression(c->expr2), NIL))
			);
			instruction i = make_instruction(is_instruction_call, _call_);
			*/
			instruction i = make_assign_instruction(//useless assert in make_assign_instruction ? check if it has an impact on the PIPS analyses !
				gfc2pips_expr2expression(c->expr),//beware, cannot be a TOP-LEVEL entity IF it is an assignment to a function, some complications are to be expected
				gfc2pips_expr2expression(c->expr2)
			);
			//fprintf(stderr,"bouh !\n");
			return i;
		}
		break;

		/*case EXEC_LABEL_ASSIGN:
			fputs ("LABEL ASSIGN ", dumpfile);
			show_expr (c->expr);
			fprintf (dumpfile, " %d", c->label->value);
		break;

*/
		case EXEC_POINTER_ASSIGN:{
			pips_debug(5, "Translation of assign POINTER ASSIGN\n");
			newgen_list list_of_arguments = CONS(EXPRESSION,gfc2pips_expr2expression(c->expr2),NIL);


			entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), ADDRESS_OF_OPERATOR_NAME);
			entity_initial(e) = make_value(is_value_intrinsic, e );
			//entity_initial(e) = make_value(is_value_constant,make_constant(is_constant_int, (void *) CurrentTypeSize));
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,list_of_arguments);
			expression ex = make_expression(
				make_syntax_call(call_),
				normalized_undefined
			);
			return make_assign_instruction(
				gfc2pips_expr2expression(c->expr),
				ex
			);
		}break;
		case EXEC_GOTO:{
			pips_debug(5, "Translation of GOTO\n");
			instruction i = make_instruction_goto(
				make_continue_statement(
					gfc2pips_code2get_label2(c)
				)
			);
			return i;
		}break;

		case EXEC_CALL:
		case EXEC_ASSIGN_CALL:{
			pips_debug(5, "Translation of %s\n",c->op==EXEC_CALL?"CALL":"ASSIGN_CALL");
			entity called_function = entity_undefined;
			//char * str = NULL;
			gfc_symbol* symbol = NULL;
			if(c->resolved_sym){
				symbol = c->resolved_sym;
			}else if(c->symtree){
				symbol = c->symtree->n.sym;
			}else{
				//erreur
				return instruction_undefined;
			}
			newgen_list list_of_arguments = gfc2pips_arglist2arglist(c->ext.actual);
			/*str = symbol->name;
			if(strncmp_("_gfortran_", str, strlen("_gfortran_") )==0){
				str = str2upper(strdup("exit"));
			}*/

			//called_function = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(strdup(str)) );
			called_function = gfc2pips_symbol2entity(symbol);
			if(entity_storage(called_function) == storage_undefined)
				entity_storage(called_function) = MakeStorageRom();

			if( entity_initial(called_function)==entity_undefined )
				entity_initial(called_function) = make_value(is_value_intrinsic, called_function );

			if(entity_type(called_function)==type_undefined){
				//fprintf(stderr,"type is already undefined %s\n",entity_name(called_function));
				//invent list of parameters
				newgen_list param_of_call = gen_copy_seq(list_of_arguments);
				newgen_list param_of_call_p = param_of_call;
				//need to translate list_of_arguments in sthg
				while(param_of_call_p){
					entity _new = gen_copy_tree((entity)param_of_call_p->car.e);
					entity_name(_new) = "toto";
					fprintf(stderr,"%s %s",entity_name((entity)param_of_call_p->car.e), entity_name(_new));
					param_of_call_p->car.e = _new;
					POP(param_of_call_p);
				}

				entity_type(called_function) = make_type_functional( make_functional(NULL, MakeOverloadedResult()) );
			}else{
				//fprintf(stderr,"type is already defined ? %s %d\n",entity_name(called_function), gen_length(functional_parameters(type_functional(entity_type(called_function)))));
			}

			/*if(type_functional_p(entity_type(called_function)) && strcmp(symbol->name, symbol->ns->proc_name->name)!=0 ){
				//check list of parameters;
				newgen_list check_sub_parameters = functional_parameters(type_functional(entity_type(called_function)));
				if( check_sub_parameters==NULL ){
					//on récupère les types des paramètres pour changer la liste des paramètres
				}
			}*/

			call call_ = make_call(called_function, list_of_arguments);

/*
		set_alternate_returns();
	    $$ = MakeCallInst(
			<entity: called function>,
			parameters
		);
		reset_alternate_returns();

 */
			return make_instruction_call( call_);
		}break;
/*		if (c->resolved_sym)
			fprintf (dumpfile, "CALL %s ", c->resolved_sym->name);
		else if (c->symtree)
			fprintf (dumpfile, "CALL %s ", c->symtree->name);
		else
			fputs ("CALL ?? ", dumpfile);

		show_actual_arglist (c->ext.actual);
		break;
*/
	    case EXEC_COMPCALL:{
	    	//Function (or subroutine) call of a procedure pointer component or type-bound procedure
	    	pips_debug(5, "Translation of COMPCALL\n");
	      /*
	      fputs ("CALL ", dumpfile);
	      //show_compcall (c->expr);
		  fprintf (dumpfile, "%s", p->symtree->n.sym->name);
		  show_ref (p->ref);
		  fprintf (dumpfile, "%s", p->value.compcall.name);
	      show_actual_arglist (p->value.compcall.actual);
	      */
	   	    }break;

		case EXEC_RETURN:{//we shouldn't even dump that for main entities
			pips_debug(5, "Translation of RETURN\n");
			return instruction_undefined;
			expression e = expression_undefined;
			if(c->expr){
				//traitement de la variable de retour
				e = gfc2pips_expr2expression(c->expr);
			}else{

			}
			return MakeReturn(e);//Syntax !
		}

		case EXEC_PAUSE:
			pips_debug(5, "Translation of PAUSE\n");
			return make_instruction_call(
				make_call(
					CreateIntrinsic("PAUSE"),
					c->expr?CONS(EXPRESSION, gfc2pips_expr2expression(c->expr), NULL):NULL
				)
			);
		break;

	    case EXEC_STOP:
	    	pips_debug(5, "Translation of STOP\n");
	    	return make_instruction_call(
				make_call(
					CreateIntrinsic("STOP"),
					c->expr?CONS(EXPRESSION, gfc2pips_expr2expression(c->expr), NULL):NULL
				)
			);
		break;

	    case EXEC_ARITHMETIC_IF:{
	    	pips_debug(5, "Translation of ARITHMETIC IF\n");
	    	expression e = gfc2pips_expr2expression(c->expr);
			expression e1 = MakeBinaryCall(CreateIntrinsic(".LT."), e, MakeIntegerConstantExpression("0"));
			expression e2 = MakeBinaryCall(CreateIntrinsic(".EQ."), e, MakeIntegerConstantExpression("0"));
			expression e3 = MakeBinaryCall(CreateIntrinsic(".LE."), e, MakeIntegerConstantExpression("0"));
			//we handle the labels doubled because it will never be checked afterwards to combine/fuse
	    	if(c->label->value==c->label2->value){
	    		if(c->label->value==c->label3->value){
	    			//goto only
	    			return make_instruction_goto(make_continue_statement(gfc2pips_code2get_label2(c)));
	    		}else{
		    		//.LE. / .GT.
					statement s2 = instruction_to_statement(
						make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c)))
					);
					statement s3 = instruction_to_statement(
						make_instruction_goto(make_continue_statement(gfc2pips_code2get_label4(c)))
					);
					return make_instruction_test(make_test(e3,s2,s3));
	    		}
	    	}else if(c->label2->value==c->label3->value){
	    		//.LT. / .GE.
				statement s2 = instruction_to_statement(
					make_instruction_goto(make_continue_statement(gfc2pips_code2get_label2(c)))
				);
				statement s3 = instruction_to_statement(
					make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c)))
				);
				return make_instruction_test(make_test(e1,s2,s3));
	    	}else{
				statement s1 = instruction_to_statement(
					make_instruction_goto(make_continue_statement(gfc2pips_code2get_label2(c)))
				);
				statement s2 = instruction_to_statement(
					make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c)))
				);
				statement s3 = instruction_to_statement(
					make_instruction_goto(make_continue_statement(gfc2pips_code2get_label4(c)))
				);
				statement s = instruction_to_statement(
					make_instruction_test(make_test(e2,s2,s3))
				);
				return make_instruction_test(make_test(e1,s1,s));
	    	}
	    }break;


		case EXEC_IF:{
			pips_debug(5, "Translation of IF\n");
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

		//we HAVE TO create a list of instructions so we shouldn't put it here but in gfc2pips_code2instruction() (done)
/*		case EXEC_SELECT:{//it is a switch or several elseif
			pips_debug(5, "Translation of SELECT into IF\n");
			newgen_list list_of_instructions_p = NULL, list_of_instructions = NULL;
			gfc_case *cp;
			gfc_code *d = c->block;

			expression tested_variable = gfc2pips_expr2expression(c->expr);

			for (; d; d = d->block){
				//create a function with low/high returning a test in one go
				expression test_expr;
				for (cp = d->ext.case_list; cp; cp = cp->next){
					test_expr = gfc2pips_buildCaseTest(tested_variable,cp);
					//transform add a list of OR to follow the list as in gfc
				}

				instruction s_if = gfc2pips_code2instruction(d->next,false);
				//boucle//s_if = instruction_to_statement(gfc2pips_code2instruction(d->next,false));
				instruction select_case = test_to_instruction(
					make_test(
						test_expr,
						s_if,
						instruction_undefined
					)
				);
				list_of_instructions = gen_nconc(list_of_instructions, CONS(INSTRUCTION, select_case, NULL));
			}

			return make_instruction_block(list_of_instructions);
		}break;
*/

		/*case EXEC_WHERE:
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
	    	pips_debug(5, "Translation of DO\n");
	    	gfc2pips_push_loop(c);
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,true));

	    	//it would be perfect if we new there is a EXIT or a CYCLE in the loop, do not add if already one (then how to stock the label ?)
	    	//add to s a continue statement at the end to make cycle/continue statements
	    	newgen_list list_of_instructions = sequence_statements(instruction_sequence(statement_instruction(s)));
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	list_of_instructions = gen_cons(make_continue_statement(gfc2pips_int2label(gfc2pips_last_created_label)),list_of_instructions);
	    	gfc2pips_last_created_label-=gfc2pips_last_created_label_step;
	    	list_of_instructions = gen_nreverse(list_of_instructions);
	    	sequence_statements(instruction_sequence(statement_instruction(s))) = list_of_instructions;

	    	//statement_label(s)=gfc2pips_code2get_label(c->block->next);//should'nt be any label here, if any, put on first instruction, and if no instruction is there no need to really dump the loop ?
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
	    	pips_debug(5, "Translation of DO WHILE\n");
	    	gfc2pips_push_loop(c);
	    	statement s = instruction_to_statement(gfc2pips_code2instruction(c->block->next,true));

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
	    	pips_debug(5, "Translation of CYCLE\n");
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
			instruction i = make_instruction_goto(
				make_continue_statement(
						label
				)
			);
			return i;
		}break;
	    case EXEC_EXIT:{
	    	pips_debug(5, "Translation of EXIT\n");
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
			instruction i = make_instruction_goto(
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
*/
	    case EXEC_ALLOCATE:
	    case EXEC_DEALLOCATE:{
	    	pips_debug(5, "Translation of %s\n",c->op==EXEC_ALLOCATE?"ALLOCATE":"DEALLOCATE");
			newgen_list lci = NULL;
	    	gfc_alloc *a;
	    	entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), str2upper(strdup(c->op==EXEC_ALLOCATE?"allocate":"deallocate")));
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
			//some problem inducted by the prettyprinter output become DEALLOCATE (variable, STAT=, I)
			if(c->expr)
				lci = gfc2pips_exprIO("STAT=", c->expr, NULL );

			for (a = c->ext.alloc_list; a; a = a->next){
				lci = CONS(EXPRESSION, gfc2pips_expr2expression(a->expr), lci );//DATA_LIST_FUNCTION_NAME, IO_LIST_STRING_NAME, or sthg else ?
				//show_expr (a->expr);
			}
			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);
	    }break;
	    case EXEC_OPEN:{
			pips_debug(5, "Translation of OPEN\n");
			entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), str2upper(strdup("open")));
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
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


			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);

	    }break;

		case EXEC_CLOSE:{
			pips_debug(5, "Translation of CLOSE\n");
			entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), str2upper(strdup("close")));
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
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
			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);
		}break;

		case EXEC_BACKSPACE:
		case EXEC_ENDFILE:
		case EXEC_REWIND:
		case EXEC_FLUSH:
		{
			const char* str;
			if(c->op==EXEC_BACKSPACE) str = "backspace";
			else if(c->op==EXEC_ENDFILE) str = "endfile";
			else if(c->op==EXEC_REWIND) str = "rewind";
			else if(c->op==EXEC_FLUSH) str = "flush";
			else pips_user_error("Your computer is mad\n");//no other possibility

			pips_debug(5, "Translation of %s\n",str);
			entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), str2upper(strdup(str)));
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,NULL);
			gfc_filepos *fp;
			fp = c->ext.filepos;

			newgen_list lci = NULL;
			if(fp->err)
				lci = gfc2pips_exprIO2("ERR=", fp->err->value, lci );
			if(fp->iostat)
				lci = gfc2pips_exprIO("UNIT=", fp->iostat, lci );
			if(fp->iomsg)
				lci = gfc2pips_exprIO("UNIT=", fp->iomsg, lci );
			if(fp->unit)
				lci = gfc2pips_exprIO("UNIT=", fp->unit, lci );
			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);
		}break;

		case EXEC_INQUIRE:{
			pips_debug(5, "Translation of INQUIRE\n");
			entity e = FindOrCreateEntity(strdup(TOP_LEVEL_MODULE_NAME), str2upper(strdup("inquire")));
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
			call call_ = make_call(e,NULL);
			newgen_list lci = NULL;
			gfc_inquire *i = c->ext.inquire;

			if(i->err)
				lci = gfc2pips_exprIO2("ERR=", i->err->value, lci );
			if(i->id)
				lci = gfc2pips_exprIO("ID=", i->id, lci );
			if(i->size)
				lci = gfc2pips_exprIO("SIZE=", i->size, lci );
			if(i->sign)
				lci = gfc2pips_exprIO("SIGN=", i->sign, lci );
			if(i->round)
				lci = gfc2pips_exprIO("ROUND=", i->round, lci );
			if(i->pending)
				lci = gfc2pips_exprIO("PENDING=", i->pending, lci );
			if(i->encoding)
				lci = gfc2pips_exprIO("ENCODING=", i->encoding, lci );
			if(i->decimal)
				lci = gfc2pips_exprIO("DECIMAL=", i->decimal, lci );
			if(i->asynchronous)
				lci = gfc2pips_exprIO("ASYNCHRONOUS=", i->asynchronous, lci );
			if(i->convert)
				lci = gfc2pips_exprIO("CONVERT=", i->convert, lci );
			if(i->pad)
				lci = gfc2pips_exprIO("PAD=", i->pad, lci );
			if(i->delim)
				lci = gfc2pips_exprIO("DELIM=", i->delim, lci );
			if(i->readwrite)
				lci = gfc2pips_exprIO("READWRITE=", i->readwrite, lci );
			if(i->write)
				lci = gfc2pips_exprIO("WRITE=", i->write, lci );
			if(i->read)
				lci = gfc2pips_exprIO("READ=", i->read, lci );
			if(i->action)
				lci = gfc2pips_exprIO("ACTION=", i->action, lci );
			if(i->position)
				lci = gfc2pips_exprIO("POSITION=", i->position, lci );
			if(i->blank)
				lci = gfc2pips_exprIO("BLANK=", i->blank, lci );
			if(i->nextrec)
				lci = gfc2pips_exprIO("NEXTREC=", i->nextrec, lci );
			if(i->recl)
				lci = gfc2pips_exprIO("RECL=", i->recl, lci );
			if(i->unformatted)
				lci = gfc2pips_exprIO("UNFORMATTED=", i->unformatted, lci );
			if(i->formatted)
				lci = gfc2pips_exprIO("FORMATTED=", i->formatted, lci );
			if(i->form)
				lci = gfc2pips_exprIO("FORM=", i->form, lci );
			if(i->direct)
				lci = gfc2pips_exprIO("DIRECT=", i->direct, lci );
			if(i->sequential)
				lci = gfc2pips_exprIO("SEQUENTIAL=", i->sequential, lci );
			if(i->access)
				lci = gfc2pips_exprIO("ACCESS=", i->access, lci );
			if(i->name)
				lci = gfc2pips_exprIO("NAME=", i->name, lci );
			if(i->named)
				lci = gfc2pips_exprIO("NAMED=", i->named, lci );
			if(i->number)
				lci = gfc2pips_exprIO("NUMBER=", i->number, lci );
			if(i->opened)
				lci = gfc2pips_exprIO("OPENED=", i->opened, lci );
			if(i->exist)
				lci = gfc2pips_exprIO("EXIST=", i->exist, lci );
			if(i->iostat)
				lci = gfc2pips_exprIO("IOSTAT=", i->iostat, lci );
			if(i->iomsg)
				lci = gfc2pips_exprIO("IOMSG=", i->iomsg, lci );
			if(i->file)
				lci = gfc2pips_exprIO("FILE=", i->file, lci );
			if(i->unit)
				lci = gfc2pips_exprIO("UNIT=", i->unit, lci );
			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, NULL)
				)
			);
		}break;

/*	    case EXEC_IOLENGTH:
	      fputs ("IOLENGTH ", dumpfile);
	      show_expr (c->expr);
	      goto show_dt_code;
	      break;
*/
	    case EXEC_READ:
	    case EXEC_WRITE:{
	    	pips_debug(5, "Translation of %s\n",c->op==EXEC_WRITE?"PRINT":"READ");
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
	    	if(c->block->next->expr == NULL){
	    		//check we have a DO
	    	}
			for (c = c->block->next; c ; c = c->next){
				if( c->op==EXEC_DO ){
					newgen_list do_list_of_arguments, do_list_of_arguments_p;
					do_list_of_arguments = do_list_of_arguments_p = NULL;
					//this block represent
					// ( val1 , val2 , ... , valn , iterator=init_val , increment )
					pips_debug(9,"\t\twe have a loop for a variable\n");

					// add the range to list_of_arguments
					//construction of the range expression
					expression ex = expression_undefined;

					newgen_list ref = NULL;
					if(c->ext.iterator->var->ref){
						ref = gfc2pips_array_ref2indices(&c->ext.iterator->var->ref->u.ar);
					}
					//peut-être faux
					syntax synt = make_syntax_reference(
						make_reference(
							gfc2pips_symbol2entity(c->ext.iterator->var->symtree->n.sym),
							ref
						)
					);
					range r = make_range(
						gfc2pips_expr2expression(c->ext.iterator->start),
						gfc2pips_expr2expression(c->ext.iterator->end),
						gfc2pips_expr2expression(c->ext.iterator->step)
					);



					gfc_code * transfer_code = c->block->next;
					for (; transfer_code ; transfer_code = transfer_code->next){
						ex = gfc2pips_expr2expression(transfer_code->expr);
						if( ex!=entity_undefined && ex!=NULL ){
							if(do_list_of_arguments_p){
								CDR(do_list_of_arguments_p) = CONS(EXPRESSION,ex,NULL);
								do_list_of_arguments_p = CDR(do_list_of_arguments_p);
							}else{
								do_list_of_arguments = do_list_of_arguments_p = CONS(EXPRESSION,ex,NULL);
							}
						}
					}
					expression er = make_expression(make_syntax_range( r), normalized_undefined);
					do_list_of_arguments = CONS(EXPRESSION,
						make_expression(synt, normalized_undefined),
						CONS(EXPRESSION,
							er,
							do_list_of_arguments
						)
					);
					call call_ = make_call(CreateIntrinsic(IMPLIED_DO_NAME), do_list_of_arguments);
					ex = make_expression(make_syntax_call( call_ ), normalized_undefined);
					if(list_of_arguments_p){
						CDR(list_of_arguments_p) = CONS(EXPRESSION,ex,NULL);
						list_of_arguments_p = CDR(list_of_arguments_p);
					}else{
						list_of_arguments = list_of_arguments_p = CONS(EXPRESSION,ex,NULL);
					}
				}else if(c->op==EXEC_TRANSFER){
					expression ex = gfc2pips_expr2expression(c->expr);
					if( ex!=entity_undefined && ex!=NULL ){
						if(list_of_arguments_p){
							CDR(list_of_arguments_p) = CONS(EXPRESSION,ex,NULL);
							list_of_arguments_p = CDR(list_of_arguments_p);
						}else{
							list_of_arguments = list_of_arguments_p = CONS(EXPRESSION,ex,NULL);
						}
					}//else{fprintf(stderr,"boum1\n");}
				}//else{fprintf(stderr,"boum2\n");}
				//fprintf(stderr,"boum3\n");
			}

			list_of_arguments = gen_nreverse(list_of_arguments);
			entity_initial(e) = make_value(is_value_intrinsic, e );
			entity_type(e) = make_type_functional( make_functional(NIL, MakeOverloadedResult()) );
			//call call_ = make_call(e, list_of_arguments);
			expression std, format, unite, f;
			newgen_list lci = NULL;

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

						//we have to push the current FORMAT in a list, we will dump it at the very, very TOP
						//we need to change the expression, a FORMAT statement doesn't have quotes around it
						expression fmt_expr = gfc2pips_expr2expression(d->ext.dt->format_label->format);
						//delete too much quotes
						char* str = entity_name(call_function(syntax_call(expression_syntax(fmt_expr))));
						//fprintf(stderr,"new format: %s\n",str);
						int curr_char_indice = 0,curr_char_indice_cible = 0, length_curr_format=strlen(str) ;
						//str[0] = str[1];
						for(;
							curr_char_indice_cible<length_curr_format-1 ;
							curr_char_indice++,curr_char_indice_cible++
						){
							if(str[curr_char_indice_cible]=='\'') curr_char_indice_cible++;
							str[curr_char_indice] = str[curr_char_indice_cible];
						}
						str[curr_char_indice] = '\0';

						gfc2pips_format = gen_cons(fmt_expr,gfc2pips_format);
						gfc2pips_format2 = gen_cons(d->ext.dt->format_label->value,gfc2pips_format2);
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

			if(d->ext.dt){

				if(d->ext.dt->err)
					lci = gfc2pips_exprIO2("ERR=", d->ext.dt->err->value, lci );
				if(d->ext.dt->end)
					lci = gfc2pips_exprIO2("END=", d->ext.dt->end->value, lci );
				if(d->ext.dt->eor)
					lci = gfc2pips_exprIO2("EOR=", d->ext.dt->end->value, lci );


				if(d->ext.dt->sign)
					lci = gfc2pips_exprIO("SIGN=", d->ext.dt->sign, lci );
				if(d->ext.dt->round)
					lci = gfc2pips_exprIO("ROUND=", d->ext.dt->round, lci );
				if(d->ext.dt->pad)
					lci = gfc2pips_exprIO("PAD=", d->ext.dt->pad, lci );
				if(d->ext.dt->delim)
					lci = gfc2pips_exprIO("DELIM=", d->ext.dt->delim, lci );
				if(d->ext.dt->decimal)
					lci = gfc2pips_exprIO("DECIMAL=", d->ext.dt->decimal, lci );
				if(d->ext.dt->blank)
					lci = gfc2pips_exprIO("BLANK=", d->ext.dt->blank, lci );
				if(d->ext.dt->asynchronous)
					lci = gfc2pips_exprIO("ASYNCHRONOUS=", d->ext.dt->asynchronous, lci );
				if(d->ext.dt->pos)
					lci = gfc2pips_exprIO("POS=", d->ext.dt->pos, lci );
				if(d->ext.dt->id)
					lci = gfc2pips_exprIO("ID=", d->ext.dt->id, lci );
				if(d->ext.dt->advance)
					lci = gfc2pips_exprIO("ADVANCE=", d->ext.dt->advance, lci );
				if(d->ext.dt->rec)
					lci = gfc2pips_exprIO("REC=", d->ext.dt->rec, lci );
				if(d->ext.dt->size)
					lci = gfc2pips_exprIO("SIZE=", d->ext.dt->size, lci );
				if(d->ext.dt->iostat)
					lci = gfc2pips_exprIO("IOSTAT=", d->ext.dt->iostat, lci );
				if(d->ext.dt->iomsg)
					lci = gfc2pips_exprIO("IOMSG=", d->ext.dt->iomsg, lci );


				if(d->ext.dt->namelist)
					lci = gfc2pips_exprIO3("NML=", d->ext.dt->namelist, lci );


			}
			lci = CONS(EXPRESSION, unite,
				CONS(EXPRESSION, std,
					CONS(EXPRESSION, format,
						CONS(EXPRESSION, f, lci)
					)
				)
			);


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


			return make_instruction_call(
				make_call(e,
					gen_nconc(lci, lr)
				)
			);


/*	    show_dt_code:
	      fputc ('\n', dumpfile);
	      for (c = c->block->next; c; c = c->next)
		show_code_node (level + (c->next != NULL), c);
	      //return;
*/

	    }break;
	    //this should be never dumped because only used in a WRITE block of gfc
	    /*case EXEC_TRANSFER:
	      fputs ("TRANSFER ", dumpfile);
	      show_expr (c->expr);
	      break;*/


/*	    case EXEC_DT_END:
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
	    	pips_user_warning("not yet dumpable %d\n", (int)c->op);
	      //gfc_internal_error ("show_code_node(): Bad statement code");
	}
	//return instruction_undefined;
	return make_instruction_block(NULL);;
}


expression gfc2pips_buildCaseTest(gfc_expr *test, gfc_case *cp){
	expression tested_variable = gfc2pips_expr2expression(test);
	expression bound1 = gfc2pips_expr2expression(cp->low);
	expression bound2 = gfc2pips_expr2expression(cp->high);
	return MakeBinaryCall(
		CreateIntrinsic(".EQ."),
		tested_variable,
		bound1
	);
	/*
	expression low_test = MakeBinaryCall(
		CreateIntrinsic(".GE."),
		tested_variable,
		bound1
	);
	expression high_test = MakeBinaryCall(
		CreateIntrinsic(".LE."),
		tested_variable,
		bound2
	);
	return MakeBinaryCall(
		CreateIntrinsic(".OR."),
		low_test,
		high_test
	);*/
}

newgen_list gfc2pips_dumpSELECT(gfc_code *c){
	newgen_list list_of_statements = NULL;
	gfc_case *cp;
	gfc_code *d = c->block;
	pips_debug(5,"dump of SELECT\n");

	if(c->here){
		list_of_statements = gen_nconc(
			CONS(STATEMENT,
				make_statement(
					gfc2pips_code2get_label(c),
					STATEMENT_NUMBER_UNDEFINED,
					STATEMENT_ORDERING_UNDEFINED,
					empty_comments,
					make_instruction_call( make_call(CreateIntrinsic("CONTINUE"), NULL)),
					NULL,
					NULL,
					empty_extensions ()
				),
				NULL
			),
			list_of_statements
		);
	}

	/*list_of_statements = CONS(STATEMENT,
		make_stmt_of_instr(
				make_assign_instruction(
				gfc2pips_expr2expression(c->expr),
				gfc2pips_expr2expression(c->expr2)
			)
		),
		NULL
	);*/
	for (; d; d = d->block){
		//create a function with low/high returning a test in one go
		expression test_expr;
		pips_debug(5,"dump of SELECT CASE\n");
		for (cp = d->ext.case_list; cp; cp = cp->next){
			test_expr = gfc2pips_buildCaseTest(c->expr,cp);
			//transform add a list of OR to follow the list as in gfc
		}

		instruction s_if = gfc2pips_code2instruction(d->next,false);
		//boucle//s_if = instruction_to_statement(gfc2pips_code2instruction(d->next,false));
		instruction select_case = test_to_instruction(
			make_test(
				test_expr,
				make_stmt_of_instr(s_if),
				make_empty_block_statement()
			)
		);
		list_of_statements = gen_nconc(list_of_statements, CONS(STATEMENT, make_stmt_of_instr(select_case), NULL));
	}

	return list_of_statements;
}

/**
 * @brief build a DATA statement, filling blanks with zeroes.
 *
 * TODO:
 * - add variables which tell when split the declaration in parts or not
 * - change this function into one returning a set of DATA statements for each sequence instead or filling with zeroes (it is what will be done in the memory anyway) ?
 */
instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym){
	pips_debug(3,"%s\n",sym->name);
	entity e1 = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, DATA_LIST_FUNCTION_NAME);
	entity_initial(e1) = MakeValueUnknown();

	entity tagged_entity = gfc2pips_symbol2entity(sym);
	newgen_list args1 = CONS(EXPRESSION,
		entity_to_expression(tagged_entity),
		NULL
	);/**/
	//add references for DATA variable(offset +-difference due to offset)
	//if(sym->component_access)

	//list of variables used int the data statement
	newgen_list init = CONS( EXPRESSION, make_call_expression( e1, args1 ), NULL );
	//list of values
	/*
	 * data (z(i), i = min_off, max_off) /1+max_off-min_off*val/
	 * gfc doesn't now the range between min_off and max_off it just state, one by one the value
	 * how to know the range ? do we have to ?
	 ** data z(offset+1) / value /
	 * ? possible ?
	 */
	newgen_list values = NULL;
	if(sym->value && sym->value->expr_type==EXPR_ARRAY){
		gfc_constructor *constr = sym->value->value.constructor;
		gfc_constructor *prec = NULL;
		int i,j;
		for (; constr; constr = constr->next){
			pips_debug(	9, "offset: %d\trepeat: %d\n", mpz_get_si(constr->n.offset), mpz_get_si(constr->repeat) );

			//add 0 to fill the gap at the beginning
			if(prec==NULL && mpz_get_si(constr->n.offset)>0){
				pips_debug(9,"we do not start the DATA statement at the beginning !\n");
				values = CONS( EXPRESSION,
					MakeBinaryCall(
						CreateIntrinsic("*"),
						gfc2pips_int2expression(mpz_get_si(constr->n.offset)),
						gfc2pips_make_zero_for_symbol(sym)
					),
					values
				);
				/*for(i=0 , j=mpz_get_si(constr->n.offset) ; i<j ; i++ ){
					values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
				}*/
			}
			//if precedent, we need to know if there has been a repetition of some kind
			if(prec){
				int offset;
				//if there is a repetition, we need to compare to the end of it
				if( mpz_get_si(prec->repeat) ){
					offset = mpz_get_si(constr->n.offset)-mpz_get_si(prec->n.offset)-mpz_get_si(prec->repeat);
				}else{
					offset = mpz_get_si(constr->n.offset)-mpz_get_si(prec->n.offset);
				}

				//add 0 to fill the gaps between the values
				if( offset >1 ){
					pips_debug(9,"We have a gap in DATA %d\n",offset);
					values = CONS( EXPRESSION,
						MakeBinaryCall(
							CreateIntrinsic("*"),
							gfc2pips_int2expression(offset-1),
							gfc2pips_make_zero_for_symbol(sym)
						),
						values
					);
					/*for( i=1 , j=offset ; i<j ; i++ ){
						values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
					}*/
				}
			}
			//if repetition on the current value, repeat, else just add
			if(mpz_get_si(constr->repeat)){
				//if repeat  =>  offset*value
				values = CONS( EXPRESSION,
					MakeBinaryCall(
						CreateIntrinsic("*"),
						gfc2pips_int2expression(mpz_get_si(constr->repeat)),
						gfc2pips_expr2expression(constr->expr)
					),
					values
				);
				/*for( i=0,j=mpz_get_si(constr->repeat) ; i<j ; i++ ){
					values = CONS( EXPRESSION, gfc2pips_expr2expression(constr->expr), values );
				}*/
			}else{
				values = CONS( EXPRESSION, gfc2pips_expr2expression(constr->expr), values );
			}
			prec = constr;
		}


		//add 0 to fill the gap at the end
		//we patch the size of a single object a little
		int size_of_unit = gfc2pips_symbol2size(sym);
		if(sym->ts.type==BT_COMPLEX) size_of_unit*=2;
		if(sym->ts.type==BT_CHARACTER) size_of_unit=1;

		int total_size;
		SizeOfArray(tagged_entity,&total_size);
		pips_debug(9,"total size: %d\n",total_size);
		int offset_end;
		if( prec ){
			if( mpz_get_si(prec->repeat) ){
				offset_end = mpz_get_si(prec->n.offset) + mpz_get_si(prec->repeat);
			}else{
				offset_end = mpz_get_si(prec->n.offset);
			}
		}

		if( prec && offset_end+1 < ((double)total_size) / (double)size_of_unit ){
			pips_debug(9,"We fill all the remaining space in the DATA %d\n",offset_end);
			values = CONS( EXPRESSION,
				MakeBinaryCall(
					CreateIntrinsic("*"),
					gfc2pips_int2expression(total_size/size_of_unit-offset_end-1),
					gfc2pips_make_zero_for_symbol(sym)
				),
				values
			);
			/*for( i=1 , j=total_size/size_of_unit-offset_end ; i<j ; i++ ){
				values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
			}*/
		}
		//fill in the remaining parts
		values = gen_nreverse(values);
	}else if (sym->value){
		values = CONS( EXPRESSION, gfc2pips_expr2expression(sym->value), NULL );
	}else{
		pips_user_error("No value, incoherence\n");
		return instruction_undefined;
	}


	entity e2 = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STATIC_INITIALIZATION_FUNCTION_NAME);
	entity_initial(e2) = MakeValueUnknown();
	values = gfc2pips_reduce_repeated_values(values);
	newgen_list args2 = gen_nconc(init, values);
	call call_ = make_call( e2, args2 );
	return make_instruction_call( call_);
}

expression gfc2pips_make_zero_for_symbol(gfc_symbol* sym){
	int size_of_unit = gfc2pips_symbol2size(sym);
	if(sym->ts.type==BT_CHARACTER) size_of_unit=1;
	if(sym->ts.type==BT_COMPLEX){
		return MakeComplexConstantExpression(
			gfc2pips_real2expression(0.),
			gfc2pips_real2expression(0.)
		);
	}else if(sym->ts.type==BT_REAL){
		return gfc2pips_real2expression(0.);
	}else{
		return gfc2pips_int2expression(0);
	}
}

/**
 * @brief look for repeated values in the list (list for DATA instructions) and transform them in a FORTRAN repeat syntax
 *
 * DATA x /1,1,1/ =>  DATA x /3*1/
 *
 * TODO: look into the consistency issue
 */
newgen_list gfc2pips_reduce_repeated_values(newgen_list l){
	//return l;
	expression curr=NULL, prec = NULL;
	newgen_list local_list = l, last_pointer_on_list=NULL;
	int nb_of_occurences=0;
	pips_debug(9, "begin reduce of values\n");
	//add recognition of /3*1, 4*1/ to make it a /7*1/
	while(local_list){
		curr = (expression)local_list->car.e;
		if(expression_is_constant_p(curr)){
			if( prec && expression_is_constant_p(prec) ){
				if( reference_variable(syntax_reference(expression_syntax(curr))) == reference_variable(syntax_reference(expression_syntax(prec))) ){
					pips_debug(10, "same as before\n");
					nb_of_occurences++;
				}else if(nb_of_occurences>1){
					//reduce
					pips_debug(9, "reduce1 %s %d\n",entity_name(reference_variable(syntax_reference(expression_syntax(prec)))),nb_of_occurences);
					last_pointer_on_list->car.e = MakeBinaryCall(
						CreateIntrinsic("*"),
						gfc2pips_int2expression(nb_of_occurences),
						prec
					);
					last_pointer_on_list->cdr = local_list;

					nb_of_occurences=1;
					last_pointer_on_list = local_list;
				}else{
					pips_debug(10, "skip to next\n");
					nb_of_occurences=1;
					last_pointer_on_list = local_list;
				}
			}else{
				pips_debug(10, "no previous\n");
				nb_of_occurences=1;
				last_pointer_on_list = local_list;
			}
			prec = curr;
		}else{//we will not be able to reduce
			pips_debug(10, "not a constant\n");
			if(nb_of_occurences>1){
				//reduce
				pips_debug(9, "reduce2 %s %d\n", entity_name(reference_variable(syntax_reference(expression_syntax(prec)))), nb_of_occurences );
				last_pointer_on_list->car.e = MakeBinaryCall(
					CreateIntrinsic("*"),
					gfc2pips_int2expression(nb_of_occurences),
					prec
				);
				last_pointer_on_list->cdr = local_list;
			}
			nb_of_occurences=0;//no dump, thus no increment
			last_pointer_on_list = NULL;//no correct reference needed
		}
		POP(local_list);
	}
	//a last sequence of data ?
	if(nb_of_occurences>1){
		//reduce
		pips_debug(9, "reduce3 %s %d\n",entity_name(reference_variable(syntax_reference(expression_syntax(prec)))),nb_of_occurences);
		last_pointer_on_list->car.e = MakeBinaryCall(
			CreateIntrinsic("*"),
			gfc2pips_int2expression(nb_of_occurences),
			prec
		);
		last_pointer_on_list->cdr = local_list;
		last_pointer_on_list = local_list;
	}
	pips_debug(9, "reduce of values done\n");
	return l;
}


entity gfc2pips_code2get_label(gfc_code *c){
	if(!c) return entity_empty_label() ;
	pips_debug(9,
		"test label: %d %d %d %d\t"
		"next %d block %d %d\n",
		(int)(c->label?c->label->value:0),
		(int)(c->label2?c->label2->value:0),
		(int)(c->label3?c->label3->value:0),
		(int)(c->here?c->here->value:0),
		(int)c->next,
		(int)c->block,
		(int)c->expr
	);
	if( c->here ) return gfc2pips_int2label(c->here->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label2(gfc_code *c){
	if(!c) return entity_empty_label() ;
	pips_debug(9,
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(int)(c->label?c->label->value:0),
		(int)(c->label2?c->label2->value:0),
		(int)(c->label3?c->label3->value:0),
		(int)(c->here?c->here->value:0),
		(int)c->next,
		(int)c->block,
		(int)c->expr
	);
	if( c->label )return gfc2pips_int2label(c->label->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label3(gfc_code *c){
	if(!c) return entity_empty_label() ;
	pips_debug(9,
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(int)(c->label?c->label->value:0),
		(int)(c->label2?c->label2->value:0),
		(int)(c->label3?c->label3->value:0),
		(int)(c->here?c->here->value:0),
		(int)c->next,
		(int)c->block,
		(int)c->expr
	);
	if( c->label )return gfc2pips_int2label(c->label2->value);
	return entity_empty_label() ;
}
entity gfc2pips_code2get_label4(gfc_code *c){
	if(!c) return entity_empty_label() ;
	pips_debug(9,
		"test label2: %d %d %d %d\t"
		"next %d block %d %d\n",
		(int)(c->label?c->label->value:0),
		(int)(c->label2?c->label2->value:0),
		(int)(c->label3?c->label3->value:0),
		(int)(c->here?c->here->value:0),
		(int)c->next,
		(int)c->block,
		(int)c->expr
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
			pips_debug(5, "op\n");
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
					pips_user_warning("intrinsic not yet recognized: %d\n",(int)expr->value.op.op);
					c="";
				break;
			}
			//c = gfc_op2string(expr->value.op.op);
			if(strlen(c)>0){
				pips_debug(6, "intrinsic recognized: %s\n",c);
				expression e1 = gfc2pips_expr2expression(expr->value.op.op1);//fprintf(stderr,"toto\n");
				if(expr->value.op.op2==NULL){
					if(!e1 || e1==expression_undefined){
						pips_user_error("e1 is null or undefined\n");
					}else if(expr->value.op.op == INTRINSIC_UMINUS ){
						return MakeFortranUnaryCall(CreateIntrinsic("--"), e1);
					}else if(expr->value.op.op == INTRINSIC_UPLUS ){
						return e1;
					}else{
						pips_user_error("No second expression member for intrinsic %s\n",c);
					}
				}
				expression e2 = gfc2pips_expr2expression(expr->value.op.op2);//fprintf(stderr,"tata\n");
				if( e1 && e2 && e1!=expression_undefined && e2!=expression_undefined ){
					return MakeBinaryCall(
						CreateIntrinsic(c),
						e1,
						e2
					);
				}else{
					pips_user_warning("e1 or e2 is null or undefined: %d\n",__LINE__);
				}
			}
		}break;
		case EXPR_VARIABLE:{
			pips_debug(5, "var\n");
			//use ri-util only functions
		    //add array recognition (multi-dimension variable)
		    //créer une liste de tous les indices et la balancer en deuxième arg  gen_nconc($1, CONS(EXPRESSION, $3, NIL))
			newgen_list ref_list = NULL;
			syntax s = syntax_undefined;

			entity ent_ref = gfc2pips_symbol2entity(expr->symtree->n.sym);
			//entity ent_ref = FindOrCreateEntity(CurrentPackage, str2upper(strdup(expr->symtree->n.sym->name)));
			//entity_type(ent_ref) = gfc2pips_symbol2type(expr->symtree->n.sym);
			if( strcmp( CurrentPackage, entity_name(ent_ref) )==0 ){
				pips_debug(9,"Variable %s is put in return storage\n",entity_name(ent_ref));
				entity_storage(ent_ref) = make_storage_return(ent_ref);
			}else{
				if(entity_storage(ent_ref)==storage_undefined){
					entity_storage(ent_ref) = MakeStorageRom();//fprintf(stderr,"expr2expression ROM %s\n",expr->symtree->n.sym->name);
				}
			}
			//entity_initial(ent_ref) = MakeValueUnknown();

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
						pips_debug(9,"ref array\n");
						if(r->u.ar.type == AR_FULL){
							//a=b  where a and b are full array, we are handling one of the expressions
							return make_expression(
								make_syntax_reference(
									make_reference(ent_ref, NULL)
								),
								normalized_undefined
							);
						}else if( gfc2pips_there_is_a_range(&r->u.ar) ){
							pips_debug(9,"We have a range\n");
							/*
							 * here we have something like x( a:b ) or y( c:d , e:f )
							 * it is not implemented in PIPS at all, to emulate the substring syntax,
							 * we create a list where each pair of expression represent end/start values
							 */
							return gfc2pips_mkRangeExpression(ent_ref, &r->u.ar);
						}
						ref_list = gfc2pips_array_ref2indices(&r->u.ar);
						break;
					/*}else if(r->type==REF_COMPONENT){
						fprintf (dumpfile, " %% %s", p->u.c.component->name);
					*/}else if(r->type==REF_SUBSTRING){
						pips_debug(9,"ref substring\n");
						expression ref = make_expression(
							make_syntax_reference(
								make_reference(ent_ref, NULL)
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
						s = make_syntax_call( make_call(substr, lexpr));
						return make_expression( s, normalized_undefined );
					}else{
						fprintf(stderr,"Unable to understand the ref %d\n",(int)r->type);
					}
					r=r->next;
				}
			}
			s = make_syntax_reference(
				make_reference(
					ent_ref,
					ref_list
				)
			);

			return make_expression( s, normalized_undefined );
		}break;
		case EXPR_CONSTANT:
			pips_debug(5, "cst %d %d\n",(int)expr, (int)expr->ts.type);
			switch(expr->ts.type){
				case BT_INTEGER: e = gfc2pips_int2expression(mpz_get_si(expr->value.integer)); break;
				case BT_LOGICAL: e = gfc2pips_logical2expression(expr->value.logical); break;
				case BT_REAL:
					//convertir le real en qqch de correct au niveau sortie
					e = gfc2pips_real2expression(mpfr_get_d(expr->value.real,GFC_RND_MODE));
				break;
				case BT_CHARACTER:{
					char *char_expr = gfc2pips_gfc_char_t2string(expr->value.character.string,expr->value.character.length);
					e = MakeCharacterConstantExpression(char_expr);
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
					pips_user_warning("type not implemented %d\n", (int)expr->ts.type);
				break;

			}
			//if(expr->ref)
			return e;
		break;
		case EXPR_FUNCTION:
			pips_debug(5, "func\n");
			//beware the automatic conversion here, some conversion functions may be automatically called here, and we do not want them in the code
			if( strncmp( expr->symtree->n.sym->name, "__convert_", strlen("__convert_") )==0 ){
				//fprintf(stderr,"gfc2pips_expr2expression: auto-convert detected %s\n",expr->symtree->n.sym->name);
				if(expr->value.function.actual->expr){
					pips_debug(6, "expression not null !\n");
					//show_expr(expr->value.function.actual->expr);
					return gfc2pips_expr2expression(expr->value.function.actual->expr);
				}else{
					pips_debug(6, "expression null !\n");
				}
			}else{
				//functions whose name begin with __ should be used by gfc only therefore we put the old name back
				if( strncmp(expr->value.function.name, "__" , strlen("__"))==0 ){
					expr->value.function.name = expr->symtree->n.sym->name;
				}
				//this is a regular call
				//on dump l'appel de fonction

				//actual est la liste
				newgen_list list_of_arguments = NULL,list_of_arguments_p = NULL;
				gfc_actual_arglist *act=expr->value.function.actual;

				if(act){
					do{
						//gfc add default parameters for some FORTRAN functions, but it is NULL in this case (could break ?)
						if(act->expr){
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
						}else{
							break;
						}

					}while(act = act->next);
				}


				entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(gfc2pips_get_safe_name(expr->value.function.name)));
				if( entity_initial(e)==entity_undefined )
					entity_initial(e) = make_value(is_value_intrinsic, e );
				//entity_initial(e) = make_value(is_value_constant,make_constant(is_constant_int, (void *) CurrentTypeSize));
				entity_type(e) = make_type_functional(make_functional(NIL, MakeOverloadedResult()));
				call call_ = make_call(e,list_of_arguments);
				//update_called_modules(e);//syntax !!
				return make_expression(
					make_syntax_call( call_),
					normalized_undefined
				);
			}
		break;
		case EXPR_ARRAY:
			pips_debug(5, "\n\narray\n\n");

			//show_constructor (p->value.constructor);
			//show_ref (p->ref);

			//MakeDataStatement($2, $4);
			//return CONS(EXPRESSION, $1, NIL);
		//break;
		default:
			pips_user_error("gfc2pips_expr2expression: dump not yet implemented, type of gfc_expr not recognized %d\n", (int)expr->expr_type);
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

/**
 * @brief create an entity based on an expression, assume it is used only for incremented variables in loops
 */
entity gfc2pips_expr2entity(gfc_expr *expr){
	message_assert("No expression to dump.",expr);

	if(expr->expr_type==EXPR_VARIABLE){
		message_assert("No symtree in the expression.",expr->symtree);
		message_assert("No symbol in the expression.",expr->symtree->n.sym);
		message_assert("No name in the expression.",expr->symtree->n.sym->name);
		entity e = FindOrCreateEntity(CurrentPackage, str2upper(gfc2pips_get_safe_name(expr->symtree->n.sym->name)));
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

newgen_list gfc2pips_arglist2arglist(gfc_actual_arglist *act){
	newgen_list list_of_arguments = NULL,list_of_arguments_p = NULL;
	while(act){
		expression ex = gfc2pips_expr2expression(act->expr);

		if(ex!=expression_undefined){

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
	return list_of_arguments;
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
newgen_list gfc2pips_exprIO3(char* s, string e, newgen_list l){
	return CONS(EXPRESSION, MakeCharacterConstantExpression(s),
		CONS(EXPRESSION, MakeNullaryCall(entity_to_expression(FindOrCreateEntity(CurrentPackage,gfc2pips_get_safe_name(e)))), l )
	);
}


/**
 * @brief create *DYNAMIC*, *STATIC*, *HEAP* and *STACK*
 */
void gfc2pips_initAreas(void){
	initialize_common_size_map();//Syntax !!
	DynamicArea = FindOrCreateEntity(CurrentPackage, DYNAMIC_AREA_LOCAL_NAME);
//	DynamicArea = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, DYNAMIC_AREA_LOCAL_NAME);
	entity_type(DynamicArea) = make_type_area( make_area(0, NIL));
    entity_storage(DynamicArea) = MakeStorageRom();
    entity_initial(DynamicArea) = MakeValueUnknown();
    AddEntityToDeclarations(DynamicArea, get_current_module_entity());
    set_common_to_size(DynamicArea, 0);//Syntax !!

    StaticArea = FindOrCreateEntity(CurrentPackage, STATIC_AREA_LOCAL_NAME);
//    StaticArea = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STATIC_AREA_LOCAL_NAME);
    entity_type(StaticArea) = make_type_area( make_area(0, NIL));
    entity_storage(StaticArea) = MakeStorageRom();
    entity_initial(StaticArea) = MakeValueUnknown();
    AddEntityToDeclarations(StaticArea, get_current_module_entity());
    set_common_to_size(StaticArea, 0);//Syntax !!

    HeapArea = FindOrCreateEntity(CurrentPackage, HEAP_AREA_LOCAL_NAME);
//    HeapArea = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, HEAP_AREA_LOCAL_NAME);
    entity_type(HeapArea) = make_type_area( make_area(0, NIL));
    entity_storage(HeapArea) = MakeStorageRom();
    entity_initial(HeapArea) = MakeValueUnknown();
    AddEntityToDeclarations(HeapArea, get_current_module_entity());
    set_common_to_size(HeapArea, 0);//Syntax !!

    StackArea = FindOrCreateEntity(CurrentPackage, STACK_AREA_LOCAL_NAME);
//    StackArea = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STACK_AREA_LOCAL_NAME);
    entity_type(StackArea) = make_type_area( make_area(0, NIL));
    entity_storage(StackArea) = MakeStorageRom();
    entity_initial(StackArea) = MakeValueUnknown();
    AddEntityToDeclarations(StackArea, get_current_module_entity());
    set_common_to_size(StackArea, 0);//Syntax !!

}

/**
 * @brief compute addresses of the stack, heap, dynamic and static areas
 */
//aller se balader dans "static segment_info * current_segment;" qui contient miriade d'informations sur la mémoire
void gfc2pips_computeAdresses(void){
	//check les déclarations, si UNBOUNDED_DIMENSION_NAME dans la liste des dimensions => direction *STACK*
	gfc2pips_computeAdressesHeap();
	gfc2pips_computeAdressesDynamic();
	gfc2pips_computeAdressesStatic();
}
/**
 * @brief compute the addresses of the entities declared in StaticArea
 */
void gfc2pips_computeAdressesStatic(void){
	gfc2pips_computeAdressesOfArea(StaticArea);
}
/**
 * @brief compute the addresses of the entities declared in DynamicArea
 */
void gfc2pips_computeAdressesDynamic(void){
	gfc2pips_computeAdressesOfArea(DynamicArea);
}
/**
 * @brief compute the addresses of the entities declared in StaticArea
 */
void gfc2pips_computeAdressesHeap(void){
	gfc2pips_computeAdressesOfArea(HeapArea);
}

/**
 * @brief compute the addresses of the entities declared in the given entity
 */
int gfc2pips_computeAdressesOfArea( entity _area ){
	//compute each and every addresses of the entities in this area. Doesn't handle equivalences.
	if(
		!_area
		|| _area==entity_undefined
		|| entity_type(_area)==type_undefined
		|| !type_area_p(entity_type(_area))
	){
		pips_user_warning("Impossible to compute the given object as an area\n");
		return 0;
	}
	int offset = 0;
	newgen_list _pcv=code_declarations(EntityCode(get_current_module_entity()));
	newgen_list pcv = gen_copy_seq(_pcv);
	pips_debug(9,"Start \t\t%s %d element(s) to check\n",entity_local_name(_area),gen_length(pcv));
	for( pcv=gen_nreverse(pcv) ; pcv != NIL; pcv = CDR(pcv) ) {
		entity e = ENTITY(CAR(pcv));//ifdebug(1)fprintf(stderr,"%s\n",entity_local_name(e));
		if(
			entity_storage(e) != storage_undefined
			&& storage_ram_p(entity_storage(e))
			&& ram_section(storage_ram(entity_storage(e)))==_area
		){
			//we need to skip the variables in commons and commons
			pips_debug(9,"Compute address of %s - offset: %d\n",entity_name(e), offset);

			entity section = ram_section(storage_ram(entity_storage(e)));
			if(
				type_tag(entity_type(e))!=is_type_variable
				|| (section!=StackArea && section!=StaticArea && section!=DynamicArea && section!=HeapArea)
			){
				pips_user_warning("We don't know how to do that - skip %s\n",entity_local_name(e));
				//size = gfc2pips_computeAdressesOfArea(e);
			}else{
				ram_offset(storage_ram(entity_storage(e))) = offset;

				area ca = type_area(entity_type(_area));
				area_layout(ca) = gen_nconc( area_layout(ca), CONS(ENTITY, e, NIL) );

				int size;
				SizeOfArray(e,&size);
				offset += size;
			}
		}
	}

	set_common_to_size( _area, offset );
	pips_debug( 9, "next offset: %d\t\tEnd %s\n", offset, entity_local_name(_area) );
	return offset;
}

void gfc2pips_computeEquiv(gfc_equiv *eq){
	//ComputeEquivalences();//syntax/equivalence.c
	//offset = calculate_offset(eq->expr);  enlever le static du fichier


/* how does PIPS know there is an equivalence to output ?
lequivchain: equivchain | lequivchain TK_COMMA equivchain;

equivchain: TK_LPAR latom TK_RPAR{ StoreEquivChain($2); };

latom: atom {
		$$ = make_chain(CONS(ATOM, MakeEquivAtom($1), (newgen_list) NULL));
	} | latom TK_COMMA atom {
		chain_atoms($1) = CONS(ATOM, MakeEquivAtom($3),chain_atoms($1));
		$$ = $1;
    };
*/

    for( ; eq ; eq=eq->next ){
    	gfc_equiv * save = eq;
    	gfc_equiv *eq_;
    	pips_debug(9,"sequence of equivalences\n");
    	entity storage_area = entity_undefined;
    	entity not_moved_entity = entity_undefined;
    	int offset=0;
    	int size=-1;
    	int not_moved_entity_size;
    	for( eq_=eq ; eq_ ; eq_=eq_->eq ){
			//check in same memory storage, not formal, not *STACK* ('cause size unknown)
			//take minimum offset
			//calculate the difference in offset to the next variable
			//set the offset to the variable with the greatest offset
			//add if necessary the difference of offset to all variable with an offset greater than the current one (and not equiv too ? or else need to proceed in order of offset ...)
			// ?? gfc2pips_expr2int(eq->expr); ??

			message_assert("expression to compute in equivalence\n",eq_->expr);
			pips_debug(9,"equivalence of %s\n",eq_->expr->symtree->name);

			//we have to absolutely know if it is an element in an array or a single variable
			entity e = gfc2pips_check_entity_exists(eq->expr->symtree->name);//this doesn't give an accurate idea for the offset, just an idea about the storage
			message_assert("entity has been founded\n",e!=entity_undefined);
			if(size==-1)not_moved_entity = e;

			message_assert("Storage is defined\n", entity_storage(e) != storage_undefined );
			message_assert("Storage is not STACK\n", entity_storage(e) != StackArea );
			message_assert("Storage is not HEAP\n", entity_storage(e) != HeapArea );
			message_assert("Storage is RAM\n", storage_ram_p(entity_storage(e)) );

			if(!storage_area)
				storage_area = ram_section(storage_ram(entity_storage(e)));
			message_assert("Entities are in the same area\n", ram_section(storage_ram(entity_storage(e)))==storage_area );

			storage_area = ram_section(storage_ram(entity_storage(e)));


			//expression ex = gfc2pips_expr2expression(eq_->expr);
			//fprintf(stderr,"toto %x\n",ex);

			//int offset_of_expression = gfc2pips_offset_of_expression(eq_->expr);
			//relative offset from the beginning of the variable (null if simple variable or first value of array)
			int offset_of_expression = calculate_offset(eq_->expr);//gcc/fortran/trans-common.c
			//int offset_of_expression = ram_offset(storage_ram(entity_storage(e)));
			offset_of_expression += ram_offset(storage_ram(entity_storage(e)));

			if(size!=-1){
				//gfc2pips_shiftAdressesOfArea( storage_area, not_moved_entity, e, eq_->expr );
			}else{
				size=0;
			}
		}
    }
}


//we need 2 offsets, one is the end of the biggest element, another is the cumulated size of each moved element
void gfc2pips_shiftAdressesOfArea( entity _area, int old_offset, int size, int max_offset, int shift ){
	newgen_list _pcv = code_declarations(EntityCode(get_current_module_entity()));
	newgen_list pcv = gen_copy_seq(_pcv);
	for( pcv=gen_nreverse(pcv) ; pcv != NIL; pcv = CDR(pcv) ) {
		entity e = ENTITY(CAR(pcv));
		if(
			entity_storage(e) != storage_undefined
			&& storage_ram_p(entity_storage(e))
			&& ram_section(storage_ram(entity_storage(e)))==_area
		){
/*
 * put those two lines in one go (to process everything in one loop only)
			when shift, if offset of variable after <c>, retrieve size of <c>
			add to every variable after <a>+sizeof(<a>) the difference of offset

			when shift, if offset of variable after <b>, retrieve size of <b>
			add to every variable after <c(2)>+sizeof(<c>)-sizeof(<c(1)>) the difference of offset

			=> when we move an array or a variable, use the full size of the array/variable
				when an element, use the full size of the array minus the offset of the element
*/
			pips_debug(9,"%s\told_offset: %d\tsize: %d\tmax_offset: %d\tshift: %d\tram_offset: %d\n", entity_name(e), old_offset, size, max_offset, shift, ram_offset(storage_ram(entity_storage(e))) );
			int personnal_shift = 0;
			//if( ram_offset(storage_ram(entity_storage(e))) > old_offset+size ){
				personnal_shift -= shift;
			//}
			if(ram_offset(storage_ram(entity_storage(e)))> old_offset){
				personnal_shift -= size;
			}
			ram_offset(storage_ram(entity_storage(e))) +=personnal_shift;
			pips_debug(9,"%s shifted of %d\n",entity_name(e),personnal_shift);
		}
	}
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
	//fprintf(stderr,"push comments %d\n",l.lb->location);

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
					b = (char*)malloc(
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
				b = (char*)malloc(sizeof(char)*(strlen(a) + 2));
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

gfc2pips_comments gfc2pips_pop_comment(void){
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
	//fprintf(stderr,"gfc2pips_replace_comments_num: replace %d by %d\n", old, new );
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
	//fprintf(stderr,"gfc2pips_comment_num_exists: %d\n", num );
	while(retour){
		if(retour->num==num)return true;
		retour = retour->prev;
	}
	return false;
}

void gfc2pips_pop_not_done_comments(void){
	while(gfc2pips_comments_stack && gfc2pips_comments_stack->done==false){
		gfc2pips_pop_comment();
	}
}
void gfc2pips_shift_comments(void){
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


gfc_code* gfc2pips_get_last_loop(void){
	if(gfc2pips_list_of_loops) return gfc2pips_list_of_loops->car.e;
	return NULL;
}
void gfc2pips_push_loop(gfc_code *c){
	gfc2pips_list_of_loops = gen_cons(c,gfc2pips_list_of_loops);
}
void gfc2pips_pop_loop(void){
	POP(gfc2pips_list_of_loops);
}













/**
 * @brief generate an union of unique elements taken from A and B
 */
newgen_list gen_union(newgen_list a, newgen_list b){
	newgen_list c = NULL;
	while(a){
		if(!gen_in_list_p(CHUNK(CAR(a)),c))
			c = gen_cons(CHUNK(CAR(a)),c);
		POP(a);
	}
	while(b){
		if(!gen_in_list_p(CHUNK(CAR(b)),c))
			c = gen_cons(CHUNK(CAR(b)),c);
		POP(b);
	}
	return c;
}

/**
 * @brief generate an intersection of unique elements taken from A and B
 */
newgen_list gen_intersection(newgen_list a, newgen_list b){
	newgen_list c = NULL;
	if(!a||!b)return NULL;
	while(a){
		if( !gen_in_list_p(CHUNK(CAR(a)),c) && gen_in_list_p(CHUNK(CAR(a)),b) )
			c = gen_cons(CHUNK(CAR(a)),c);
		POP(a);
	}
	while(b){
		if( !gen_in_list_p(CHUNK(CAR(b)),c) && gen_in_list_p(CHUNK(CAR(b)),a) )
			c = gen_cons(CHUNK(CAR(b)),c);
		POP(b);
	}
	return c;
}



