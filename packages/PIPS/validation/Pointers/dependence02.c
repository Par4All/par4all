int use_def_elim01() 
{   
  int i, *x, *y; 
    i = 2;
       x = &i;
       y = x;  
	 *y = 1;
	       return *y;
	  }
