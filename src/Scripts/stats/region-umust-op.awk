BEGIN{  n_modules=0;
	n_umust=0;
	n_umust_must_must=0;
	n_umust_must_must_must=0;
	n_umust_must_may=0;
	n_umust_must_may_must=0;
	n_umust_sc_rn=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r | r | r | r |} \\hline";
		print "Module", "&", "nb", "&", "must/must",\
		      "&", "must res", "&", "must/may", "&",  "must res", "&", \
		       "sc\_rn", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_umust += $2;
	n_umust_must_must += $3;
	n_umust_must_must_must += $4;
	n_umust_must_may += $5;	
	n_umust_must_may_must += $6;
	n_umust_sc_rn += $7;
	print $1, "&", $2, "&", $3, "&", $4, "&", $5, "&", $6, "&", $7, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_umust, "&", n_umust_must_must, "&", \
	      n_umust_must_must_must, "&", n_umust_must_may, "&", \
	      n_umust_must_may_must, "&", \
	      n_umust_sc_rn, "\\\\ \\hline";
}
