BEGIN{  n_modules=0;
	for (i=0; i<5; i++)
	    n_num[i]=0;
	tot = 0;
}
NF!=0	{
	
  n_modules++;
  
  base=70;
  
  for(i=0; i<5; i++)
    {
      tot +=$(base+i);
      n_num[i] += $(base+i);
    }
  
}
END {	

  if (tot !=0)
    print "&",n_num[0],"&",int(n_num[0]*100/tot),"&",n_num[1],"&",int(n_num[1]*100/tot),"&",n_num[2],"&",int(n_num[2]*100/tot),"&",n_num[3],"&",int(n_num[3]*100/tot),"&",n_num[4],"&",int(n_num[4]*100/tot),"\\\\";
  else
    print "&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";
}
