typedef struct _hdl_progress_handler
{
  int percent_start;
  int percent_stop;
  int percent_done;





  void (* progress)(struct _hdl_progress_handler * progress_mgr,
		    const int nbfait,
		    const int nbtotal);



  void (* display) (struct _hdl_progress_handler * progress_mgr);



  void * (* end_monitor) (struct _hdl_progress_handler * progress_mgr);



  const void * display_data;

} HDL_progress_handler;

/*extern void HDL_display(HDL_progress_handler *prog); */

void HDL_progress_display(HDL_progress_handler * prog);

void HDL_progress_progress(HDL_progress_handler * prog,
			   const int nbfait,
			   const int nbtotal);

void * HDL_progress_end (HDL_progress_handler * prog);

extern void HDL_progress_display(HDL_progress_handler *prog);

//void HDL_progress_display(HDL_progress_handler *prog)
//{
//}

extern void HDL_progress_progress(HDL_progress_handler *prog, const int nbfait, const int nbtotal);

//void HDL_progress_progress(HDL_progress_handler *prog, const int nbfait, const int nbtotal)
//{
//}


extern void *HDL_progress_end(HDL_progress_handler *prog);

void *HDL_progress_end(HDL_progress_handler *prog)
{
  return 0;
}

void HDL_progress_start_monitor(HDL_progress_handler * progress /*,ERR_mgr * emgr,const void * data*/)
{
  /* ERR_MGR_IF_DO( ! progress , emgr , ERR_TBD , return ); */

  if(progress)
    {
      progress->percent_start = 0;
      progress->percent_stop  = 100;
      progress->percent_done  = 0;
      progress->progress      = HDL_progress_progress;
      progress->display       = HDL_progress_display;
      progress->display_data  = 0;
      progress->end_monitor   = HDL_progress_end;
    }
}
