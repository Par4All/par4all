BEGIN{  n_modules=0;
	n_proj=0;
	n_proj_pot_must=0;
	n_proj_must=0;
	n_proj_ofl=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r | r |} \\hline";
		print "Module", "&", "nb", "&", "pot must",\
		      "&", "must res", "&", "ofl errors", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_proj += $2;
	n_proj_pot_must += $3;
	n_proj_must += $4;
	n_proj_ofl += $5;	
	print $1, "&", $2, "&", $3, "&", $4, "&", $5, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_proj, "&", n_proj_pot_must, "&", \
	      n_proj_must, "&", n_proj_ofl, "\\\\ \\hline";
}
