struct _err_manager
{
  char function/*[500]*/; /* fonction ou est cree le manager */
  void (* push)(/* const char * */ int function);
} ;

void ERR_mgr_ctor(const char * file,
		  const char * function);

