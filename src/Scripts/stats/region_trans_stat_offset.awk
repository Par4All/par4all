BEGIN{  
  n_modules=0;
  tot_array=0;
  tot=0;
}
NF!=0	{
	
  n_modules++;

  tot_array += $3+$4;
  tot += $5;
}
END {	
  if (tot_array !=0)
    print tot, "(",tot*100/tot_array, "\\%).";
  else
    print 0, "(0\\%).";
}
