#if 0
typedef struct _err_error ERR_error;
typedef struct _err_manager ERR_mgr;
typedef unsigned long int STD_uint32;
typedef long STD_int32;
typedef STD_uint32 ERR_code ;

typedef struct _hdl_stream
{
  const char * name;
  const char * (* format) (struct _hdl_stream * ptthis,const char * fmt,va_list pa);
  const char * (* free) (const char * message);
  void (* print) (struct _hdl_stream * ptthis,const char * message);
  void * data;
} HDL_stream;
#endif
struct _err_error
{
  char function [500]; /* fonction ou s'est produite l'erreur */
  //char file [500]; /* fichier ou s'est produite l'erreur  */
  //char description [500]; /* texte personalise de l'erreur eventuellement */
  //STD_uint32 line; /* ligne dans le code ou on a generer l'erreur */
  //ERR_code code; /* le code de l'erreur generee */
};

struct _err_manager
{
  char function[500]; /* fonction ou est cree le manager */
  //char file[500]; /* fichier ou est cree le manager */
  //STD_uint32 line; /* ligne dans le code */
  //ALGMEM_allocator * mallocated; /* allocateur */
  struct _err_pile
  {
    struct _err_error * pile; /* pointeur de pile */
    //struct _err_error * pile_end; /* fin de la pile */
    //struct _err_error pile_tab[50]; /* base de la pile */

  } pile;

  /* gestion pile */
  void (* push)(struct _err_manager * _this,
		//const ERR_code code,
		//const STD_uint32 line,
		//const char * file,
		const char * function);
  //void (* pop)(struct _err_manager * _this);
  //ERR_error * (* top)(const struct _err_manager * _this);
  //STD_int32 (* occured)(const struct _err_manager * _this);
  //STD_int32 (* top_is)(const struct _err_manager * _this,const ERR_code code);
  //void (* format_message)(struct _err_manager * _this/*, const ERR_error * error*/);
  //char message[500];
  //HDL_stream * message_stream;
} ;


void ERR_mgr_ctor(//ERR_mgr * mgr,
		  //HDL_stream * message_stream,
		  //const STD_uint32 line,
		  const char * file,
		  const char * function);

void ERR_mgr_dtor(/*ERR_mgr * emgr*/);
