#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"


bool 
variable_entity_p(entity e)
{
    bool variable = 
	(entity_storage(e)!=storage_undefined) && 
	storage_ram_p(entity_storage(e));

    return variable;
}

bool 
symbolic_constant_entity_p(entity e)
{
    bool symbolic_constant = entity_storage(e)!= storage_undefined 
	&& storage_rom_p(entity_storage(e))
	&& entity_initial(e) != value_undefined
	&& value_symbolic_p(entity_initial(e));

    return symbolic_constant;
}


/* this function adds an entity to the list of variables of the
CurrentFunction. it does nothing if e is already in the list. */

void 
AddEntityToDeclarations(e, f)
entity e;
entity f;
{
    cons *pc, *l;

    l = code_declarations(EntityCode(f));

    for (pc = l; pc != NULL; pc = CDR(pc)) {
	if (e == ENTITY(CAR(pc)))
		return;
    }

    code_declarations(EntityCode(f)) = CONS(ENTITY, e, l);
}

/* entity make_scalar_entity(name, module_name, base)
 */
entity 
make_scalar_entity(name, module_name, base)
string name;
string module_name;
basic base;
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

    entity_type(e) = (type) MakeTypeVariable(b, NIL);
    f = local_name_to_top_level_entity(module_name);
    a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME); 

    entity_storage(e) = 
	make_storage(is_storage_ram,
		     make_ram(f, a,
			      (basic_tag(base)!=is_basic_overloaded)?
			      (add_variable_to_area(a, e)):(0),
			      NIL));

    /* FI: I would have expected is_value_unknown, especially with a RAM storage! */
    entity_initial(e) = make_value(is_value_constant,
				   MakeConstantLitteral());

    return(e);
}


/* -------------------------------------------------------------
 *
 * New Temporary Variables MANAGEMENT
 *
 */

static int 
    unique_integer_number = 0,
    unique_float_number = 0,
    unique_logical_number = 0,
    unique_complex_number = 0,
    unique_string_number = 0;

void 
reset_unique_variable_numbers()
{
    unique_integer_number=0;
    unique_float_number=0;
    unique_logical_number=0;
    unique_complex_number=0;
}

/* Default prefixes */
#define DEFAULT_INT_PREFIX 	"I_"
#define DEFAULT_FLOAT_PREFIX 	"F_"
#define DEFAULT_LOGICAL_PREFIX 	"L_"
#define DEFAULT_COMPLEX_PREFIX	"C_"
#define DEFAULT_STRING_PREFIX	"S_"

entity
make_new_scalar_variable_with_prefix(string prefix,
				     entity module,
				     basic b)
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
   
   e = make_scalar_entity(&buffer[0], module_name, b);
   AddEntityToDeclarations(e, module);
   
   return e;
}

entity
make_new_scalar_variable(entity module,
                         basic b)
{
    return make_new_scalar_variable_with_prefix("", module, b);
}


/* looks for an entity which should be a scalar of the specified
 * basic. If found, returns it, else one is created.
 */
entity 
find_or_create_scalar_entity(name, module_name, base)
string name;
string module_name;
tag base;
{
    entity e = entity_undefined;
    string nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

    if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined) 
    {
	pips_assert("find_or_create_scalar_entity",
		    (entity_scalar_p(e) && entity_basic_p(e, base)));

	return(e);
    }

    return(make_scalar_entity(name, module_name, MakeBasic(base)));
}

entity 
find_or_create_typed_entity(
   string name,
   string module_name,
   tag base)
{
    entity e = entity_undefined;
    string nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

    if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined) 
    {
	pips_assert("type is okay", entity_basic_p(e, base));

	return(e);
    }

    return(make_scalar_entity(name, module_name, MakeBasic(base)));
}

entity 
make_scalar_integer_entity(name, module_name)
char *name;
char *module_name;
{
    string full_name;
    entity e, f, a ;
    basic b ;

    debug(8,"make_scalar_integer_entity", "begin name=%s, module_name=%s\n",
	  name, module_name);

    full_name = concatenate(module_name, MODULE_SEP_STRING, name, NULL);
    hash_warn_on_redefinition();
    e = make_entity(strdup(full_name),
		    type_undefined, 
		    storage_undefined, 
		    value_undefined);

    b = make_basic(is_basic_int, (void*) 4); 

    entity_type(e) = (type) MakeTypeVariable(b, NIL);

    f = local_name_to_top_level_entity(module_name);
    a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME);
    pips_assert("make_scalar_integer_entity", !entity_undefined_p(f) && !entity_undefined_p(a));

    entity_storage(e) = make_storage(is_storage_ram,
				     (make_ram(f, a,
					       add_variable_to_area(a, e),
					       NIL)));

    entity_initial(e) = make_value(is_value_constant,
				   MakeConstantLitteral());

    debug(8,"make_scalar_integer_entity", "end\n");

    return(e);
}


bool 
entity_scalar_p(e)
entity e;
{
    type t = entity_type(e);

    pips_assert("entity_scalar_p", type_variable_p(t));

    return(ENDP(variable_dimensions(type_variable(t))));
}


/* for variables (like I), not constants (like 1)!
 * use integer_constant_p() for constants
 */
bool 
entity_integer_scalar_p(e)
entity e;
{
    return(entity_scalar_p(e) &&
	   basic_int_p(variable_basic(type_variable(entity_type(e)))));
}


/* integer_scalar_entity_p() is obsolete; use entity_integer_scalar_p() */
bool 
integer_scalar_entity_p(e)
entity e;
{
    return type_variable_p(entity_type(e)) && 
	basic_int_p(variable_basic(type_variable(entity_type(e)))) &&
	    variable_dimensions(type_variable(entity_type(e))) == NIL;
}


dimension 
entity_ith_dimension(e, i)
entity e;
int i;
{
    cons *pd;
    type t = entity_type(e);

    pips_assert("entity_ith_dimension", type_variable_p(t));

    pd = variable_dimensions(type_variable(t));

    while (pd != NIL && --i > 0)
	pd = CDR(pd);
    
    pips_assert("entity_ith_dimension", pd != NIL);

    return(DIMENSION(CAR(pd)));
}


    
/* boolean entity_unbounded_p(entity e)
 * input    : an array entity
 * output   : TRUE if the last dimension of the array is unbounded (*),
 *            FALSE otherwise.
 * modifies : nothing
 * comment  : 
 */
boolean 
entity_unbounded_p(e)
entity e;
{
    int nb_dim = NumberOfDimension(e);
    
    return(unbounded_dimension_p(entity_ith_dimension(e, nb_dim)));    
}



/* variable_entity_dimension(entity v): returns the dimension of variable v;
 * scalar have dimension 0
 */
int 
variable_entity_dimension(v)
entity v;
{
    int d = 0;

    pips_assert("variable_entity_dimension", type_variable_p(entity_type(v)));

    MAPL(cd, {
	d++;
    },
	 variable_dimensions(type_variable(entity_type(v))));

    return d;
}


void 
remove_variable_entity(v)
entity v;
{
    /* FI: this is pretty dangerous as it may leave tons of dangling pointers;
     * I use it to correct early declarations of types functions as variables;
     * I assume that no pointers to v exist in statements because we are still
     * in the declaration phasis.
     *
     * Memory leaks: I do not know if NewGen free_entity() is recursive.
     */
    storage s = entity_storage(v);
    entity f = entity_undefined;
    code c = code_undefined;

    if(storage_undefined_p(s)) {
	string fn = entity_module_name(v);
	f = local_name_to_top_level_entity(fn);
    }
    else if(storage_ram_p(s)) {
	f = ram_function(storage_ram(s));
    }
    else if(storage_rom_p(s)) {
	f = entity_undefined;
    }
    else {
	pips_error("remove_variable_entity", "unexpected storage %d\n", storage_tag(s));
    }

    if(!entity_undefined_p(f)) {
	pips_assert("remove_variable_entity", entity_module_p(f));
	c = value_code(entity_initial(f));
	gen_remove(&code_declarations(c), v);
    }
    free_entity(v);
}


/* entity make_integer_constant_entity(int c)
 * make entity for integer constant c
 */
entity 
make_integer_constant_entity(c)
int c;
{
    entity ce;
    char *num = (char*) malloc(32);
    string cn;

    sprintf(num, "%d", c);
    cn = concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,num,(char *)NULL);
    ce = gen_find_tabulated(cn,entity_domain);
    if (ce==entity_undefined) {		/* make entity for the constant c */ 
	functional cf = 
	    make_functional(NIL, 
	          make_type(is_type_variable, 
			    make_variable(make_basic(is_basic_int, (void*)sizeof(int)),
					  NIL)));
	type ct = make_type(is_type_functional, cf);
	ce = make_entity(strdup(cn), ct, MakeStorageRom(),
			 make_value(is_value_constant, 
				    make_constant(is_constant_int, (void*)c)));
    }
    
    else 
	free(num);

    return(ce);
}

/* entity make_float_constant_entity(float c)
 * make entity for float constant c
 */
entity 
make_float_constant_entity(float c)
{
  entity ce;
  char buffer[100];
  string cn;
  sprintf(buffer, "%f", c);

  cn = strdup(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, buffer, NULL));
  ce = gen_find_entity(cn);

  if (ce==entity_undefined) {		/* make entity for the constant c */ 
    functional cf = 
      make_functional(NIL, 
		      make_type(is_type_variable, 
				make_variable(make_basic(is_basic_float,(void*)sizeof(float)),
					      NIL)));
    type ct = make_type(is_type_functional, cf);
    ce = make_entity(cn, ct, MakeStorageRom(),
		     make_value(is_value_constant, 
				make_constant(is_constant_litteral, UU)));
  }
  else 
    free(cn);
  
  return ce;
}



/* 
 * This function computes the current offset of the area a passed as
 * argument. The length of the variable v is also computed and then added
 * to a's offset. The initial offset is returned to the calling function.
 * v is added to a's layout.
 */
int 
add_variable_to_area(a, v)
entity a, v;
{
    int OldOffset=-1;
    type ta = entity_type(a);
    area aa = type_area(ta);

    if(top_level_entity_p(a)) {
	/* COMMONs are supposed to havethe same layout in each routine */
	pips_error("add_variable_to_area", "COMMONs should not be modified\n");
    }
    else {
	/* the local areas are StaticArea and DynamicArea */
	int s = 0;
	OldOffset = area_size(aa);
	if(!SizeOfArray(v, &s)) {
	    pips_error("add_variable_to_area",
		       "Varying size array \"%s\"\n", entity_name(v));
	}
	area_size(aa) = OldOffset+s;
    }

    area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));

    return(OldOffset);
}


void 
add_variable_declaration_to_module(m, v)
entity m;
entity v;
{
    value val = entity_initial(m);
    code c = code_undefined;

    pips_assert("add_variable_declaration_to_module", value_code_p(val));

    c = value_code(val);
    code_declarations(c) = gen_nconc(code_declarations(c),
				     CONS(ENTITY, v, NIL));
}


bool
formal_parameter_p(entity v)
{
    storage s = entity_storage(v);
    bool formal_p = storage_formal_p(s);

    return formal_p;
}


/* True if a variable is the pseudo-variable used to store value
   returned by a function: */
bool
variable_return_p(entity v)
{
    storage s = entity_storage(v);
    bool return_p = storage_return_p(s);

    return return_p;
}


bool
variable_is_a_module_formal_parameter_p(entity a_variable,
                                        entity a_module)
{
   MAP(ENTITY, e,
       {
          storage s = entity_storage(e);
          if (e == a_variable) {
             if (storage_formal_p(s))
                /* Well, the variable is a formal parameter of the
                   module: */
                return TRUE;
             else
                /* The variable is in the declaration of the module
                   but is not a formal parameter: */
                return FALSE;
	  }
       },
          code_declarations(value_code(entity_initial(a_module))));

   /* The variable is not in the declaration of the module: */
   return FALSE;
}

/* true if v is in a common. */
bool
variable_in_common_p(
    entity v)
{
    return type_variable_p(entity_type(v)) &&
	storage_ram_p(entity_storage(v)) &&
	!SPECIAL_AREA_P(ram_section(storage_ram(entity_storage(v)))) ;
}

/* true if v appears in a SAVE statement, or in a DATA statement */
bool
variable_static_p(entity v)
{
    return(type_variable_p(entity_type(v)) &&
	   storage_ram_p(entity_storage(v)) &&
	   static_area_p(ram_section(storage_ram(entity_storage(v)))));
}
bool
variable_dynamic_p(entity v)
{
    return(type_variable_p(entity_type(v)) &&
	   storage_ram_p(entity_storage(v)) &&
	   dynamic_area_p(ram_section(storage_ram(entity_storage(v)))));
}

/* This test can only be applied to variables, not to functions, subroutines or
 * commons visible from a module.
 */
bool
variable_in_module_p(entity v,
                     entity m)
{
   bool in_module_1 = 
       strcmp(module_local_name(m), entity_module_name(v)) == 0;
   bool in_module_2 = 
    entity_is_argument_p(v, code_declarations(value_code(entity_initial(m))));

   pips_assert ("both coherency",  in_module_1==in_module_2);

   return in_module_1;
}

bool 
variable_in_list_p(e, l)
entity e;
list l;
{
    bool is_in_list = FALSE;
    for( ; (l != NIL) && (! is_in_list); l = CDR(l))
	if(same_entity_p(e, ENTITY(CAR(l))))
	    is_in_list = TRUE;
    return(is_in_list);
}


/* Discard the decls_text string of the module code to make the
   prettyprinter ignoring the textual declaration and remake all from
   the declarations without touching the corresponding property
   (PRETTYPRINT_ALL_DECLARATIONS). RK, 31/05/1994. */
void 
discard_module_declaration_text(a_module)
entity a_module;
{
    code c = entity_code(a_module);
    string s = code_decls_text(c);
    
    free(s);
    code_decls_text(c) = strdup("");
}

/* Returns a numbered entity the name of which is suffix + number,
 * the module of which is prefix. Used by some macros to return
 * dummy and primed variables for system of constraints.
 *
 * moved to ri-util from hpfc on BC's request. FC 08/09/95
 */
entity 
get_ith_dummy(prefix, suffix, i)
string prefix, suffix;
int i;
{
    char buffer[100]; 
    assert(i>=1 && i<=7);
    (void) sprintf(buffer, "%s%d", suffix, i);

    return find_or_create_scalar_entity(buffer, prefix, is_basic_int);
}


entity 
make_new_module_variable(entity module,int d)	       
{ 

    static char name[ 64 ];
    string name1;
    entity ent1=entity_undefined;
    string full_name;
    static int num = 1;
    name[0] = 'X';
    if (d != 0) {
	(void) sprintf(&name[1],"%d",d);
	num = d;
    }
    else { (void) sprintf(&name[1],"%d",num);
	   num++;}
	
    name1 = strdup(name);
    full_name=strdup(concatenate(module_local_name(module), 
				 MODULE_SEP_STRING,
				 name1,
				 NULL));
    while ((ent1 = gen_find_tabulated(full_name,entity_domain)) 
	   != entity_undefined) {
	free(name1);
	free(full_name);
	name[0] = 'X';
	(void) sprintf(&name[1],"%d",num);
	num++;
	name1 = strdup(name);
	full_name=strdup(concatenate(module_local_name(module), 
				     MODULE_SEP_STRING,
				     name1,
				     NULL));
    }
    ent1 = make_scalar_integer_entity(name1,
				      module_local_name(module));
    free(full_name);
    return ent1;
}


/* These globals variables count the number of temporary and auxiliary
 * entities. Each time such a variable is created, the corresponding
 * counter is incremented.
 *
 * FI: this must be wrong. A function to reset count_tmp and count_aux 
 * is needed if tpips or wpips are to work in a consistent way!
 */
/* gcc complains that they are not used... but they are defined! */
static int count_tmp = 0;
static int count_aux = 0;


/*============================================================================*/
/* entity make_new_entity(basic ba, int kind): Returns a new entity.
 * This entity is either a new temporary or a new auxiliary variable.
 * The parameter "kind" gives the kind of entity to produce.
 * "ba" gives the basic (ie the type) of the entity to create.
 *
 * The number of the temporaries is given by a global variable named
 * "count_tmp".
 * The number of the auxiliary variables is given by a global variable named
 * "count_aux".
 *
 * Called functions:
 *       _ current_module() : loop_normalize/utils.c
 *       _ FindOrCreateEntity() : syntax/declaration.c
 *       _ CurrentOffsetOfArea() : syntax/declaration.c
 */
entity 
make_new_entity(ba, kind)
basic ba;
int kind;
{
    extern list integer_entities, 
	real_entities, logical_entities, complex_entities,
	double_entities, char_entities;

    entity new_ent, mod_ent;
    char prefix[4], *name, *num;
    int number = 0;
    entity dynamic_area;
  
    /* The first letter of the local name depends on the basic:
     *       int --> I
     *     real  --> F (float single precision)
     *    others --> O
     */
    switch(basic_tag(ba))
    {
    case is_basic_int: { (void) sprintf(prefix, "I"); break;}
    case is_basic_float:
    {
	if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
	    (void) sprintf(prefix, "O");
	else
	    (void) sprintf(prefix, "F");
	break;
    }
    default: (void) sprintf(prefix, "O");
    }

    /* The three following letters are whether "TMP", for temporaries
     * or "AUX" for auxiliary variables.
     */
    switch(kind)
    {
    case TMP_ENT:
    {
	number = (++count_tmp);
	(void) sprintf(prefix+1, "TMP");
	break;
    }
    case AUX_ENT:
    {
	number = (++count_aux);
	(void) sprintf(prefix+1, "AUX");
	break;
    }
    default: user_error("make_new_entity", "Bad kind of entity: %d", kind);
    }

    mod_ent = get_current_module_entity();
    num = (char*) malloc(32);
    (void) sprintf(num, "%d", number);

    /* The first part of the full name is the concatenation of the define
     * constant ATOMIZER_MODULE_NAME and the local name of the module
     * entity.
     */
    /* ATOMIZER_MODULE_NAME discarded : it is a bug ! RK, 31/05/1994.
       name = strdup(concatenate(ATOMIZER_MODULE_NAME, entity_local_name(mod_ent),
       MODULE_SEP_STRING, prefix, num, (char *) NULL));
    */
    name = strdup(concatenate(entity_local_name(mod_ent),
			      MODULE_SEP_STRING, prefix, num, (char *) NULL));
    /*
      new_ent = make_entity(name,
      make_type(is_type_variable,
      make_variable(ba,
      NIL)),
      make_storage(is_storage_rom, UU),
      make_value(is_value_unknown, UU));
    */
    /* Create a true dynamic variable. RK, 31/05/1994 : */
    new_ent = make_entity(name,
			  make_type(is_type_variable,
				    make_variable(ba,
						  NIL)),
			  storage_undefined,
			  make_value(is_value_unknown, UU));
    dynamic_area = global_name_to_entity(module_local_name(mod_ent),
					 DYNAMIC_AREA_LOCAL_NAME);
    entity_storage(new_ent) = make_storage(is_storage_ram,
					   make_ram(mod_ent,
						    dynamic_area,
						    add_variable_to_area(dynamic_area, new_ent),
						    NIL));
    add_variable_declaration_to_module(mod_ent, new_ent);

    /* Is the following useless : */
  
    /* The new entity is stored in the list of entities of the same type. */
    switch(basic_tag(ba))
    {
    case is_basic_int:
    {
	integer_entities = CONS(ENTITY, new_ent, integer_entities);
	break;
    }
    case is_basic_float:
    {
	if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
	    double_entities = CONS(ENTITY, new_ent, double_entities);
	else
	    real_entities = CONS(ENTITY, new_ent, real_entities);
	break;
    }
    case is_basic_logical:
    {
	logical_entities = CONS(ENTITY, new_ent, logical_entities);
	break;
    }
    case is_basic_complex:
    {
	complex_entities = CONS(ENTITY, new_ent, complex_entities);
	break;
    }
    case is_basic_string:
    {
	char_entities = CONS(ENTITY, new_ent, char_entities);
	break;
    }
    default:break;
    }

    return new_ent;
}
