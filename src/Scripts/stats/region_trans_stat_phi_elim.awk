BEGIN{  
  n_modules=0;
  base=89;
  n_calls=0;
  n_exact_in=0;
  n_exact_out=0;
}
NF!=0	{
	
  n_modules++;
  
  n_calls += $(base+0);
  n_exact_in += $(base+1);
  n_exact_out += $(base+2);

  tot_tmp = $(base+0);
  if (tot_tmp !=0)
    print $1,"&",tot_tmp,"&",$(base+1),"&",int($(base+1)*100/tot_tmp),"&",$(base+2),"&",int($(base+2)*100/tot_tmp),"\\\\";
  else
    print $1,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";


}
END {	
  print "\\hline";
  if (n_calls !=0)
    print WORKSPACE, "&",n_calls,"&", n_exact_in, "&", int(n_exact_in*100/n_calls), "&", n_exact_out, "&",int(n_exact_out*100/n_calls),"\\\\";
  else
    print WORKSPACE,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";
}
