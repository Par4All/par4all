BEGIN			{ 
				extract = 0 ; 
			}

			{
				if (extract == 1) {
					print $0 ;
				}
			}

/^\/\*/			{ 
				if (extract == 1) {
					print "bad structure at line " NR > "/dev/tty" ;
					exit 1;
				}
				else {
					extract = 1;
					print $0 ;
				}
			}

/^\/\*.*\*\//		{ 
				extract = 0 ; 
			}

/(^\*\/)|(^ \*\/)/	{
				if (extract == 0) {
					print "bad structure at line " NR > "/dev/tty" ;
					exit 1;
				}
				else {
					extract = 0;
					print "\n\n\n\n";
				}
			}
