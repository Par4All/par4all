BEGIN{  
  n_modules=0;
  base=75;
  n_calls=0;
  n_success=0;
  n_part_failure=0;
  n_non_lin=0;
}
NF!=0	{
	
  n_modules++;
  
  n_calls += $(base+0);
  n_success += $(base+1);
  n_part_failure += $(base+2);
  n_non_lin += $(base+3);  

  tot_tmp = $(base+0);
  if (tot_tmp !=0)
    print $1,"&",tot_tmp,"&",$(base+1),"&",int($(base+1)*100/tot_tmp),"&",$(base+2),"&",int($(base+2)*100/tot_tmp),"&",$(base+3),"&",int($(base+3)*100/tot_tmp),"\\\\";
  else
    print $1,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";


}
END {	
  print "\\hline";
  
  if (n_calls !=0)
    print WORKSPACE, "&",n_calls,"&", n_success, "&", int(n_success*100/n_calls), "&", n_part_failure, "&",int(n_part_failure*100/n_calls),"&",n_non_lin, "&", int(n_non_lin*100/n_calls),"\\\\";
  else
   print WORKSPACE,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\"; 
}
