BEGIN{  n_modules=0;
	n_op=0;
	n_op_pot_must=0;
	n_op_must=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r |} \\hline";
		print "Module", "&", "invocation number", "&", "pot. must results",\
		    "&", "must results", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_op += $2;
	n_op_pot_must += $3;
	n_op_must += $4;
	n_op_ofl += $5;	
	print $1, "&", $2, "&", $3, "&", $4, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_op, "&", n_op_pot_must, "&", \
	n_op_must, "\\\\ \\hline";
}
