.\" @(#)pips.l 96/06/07 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH PIPS 1 "7 Juin 1996"
.\"
.SH NAME
pips \- Interprocedural Parallelizer of Scientific Programs, Unix interface.
.SH SYNOPSIS
pips [\-f 
.IR source_file ].\|.\|. 
[\-m
.IR module ]
[\-s
.IR selected_rule ].\|.\|.
[\-p
.IR performed_rule ]
[\-b
.IR build_resource ] 
[\-0
.IR boolean_property ].\|.\|.
[\-1
.IR boolean_property ].\|.\|.
.I workspace
.\"
.SH DESCRIPTION 
PIPS (Paralleliseur Interprocedural de Programmes Scientifiques) is a research project from the computer science research center (CRI) at Ecole des Mines de Paris (France). It is a source to source compiler, currently supporting most of Fortran 77, to analyze and parallelize real scientific codes for shared memory multi-processor machines, and to distribute simple codes for processor networks.

pips is one of the major binary programs of the PIPS project. All of
its functionalities are available through more user-friendly shell
scripts: Pips, Init, Display, Select, Perform and Build.

As you specify source files with the -f option, a subdirectory of the
current directory is created:
.BR workspace.database.
The interprocedural database build by pips is stored in this subdirectory. 
It is reused when you do not specify the -f option.
.\"
.SH OPTIONS
.TP
.B \-f
introduces a list of Fortran files containing the program to analyze. Multiple -f arguments are allowed. Source code must be available for all called subroutines and functions, but a main program is not necessary.
.TP
.B \-m
defines the module of interest. Must be upper case.
.TP
.B \-s 
selection of an option. Must be upper case. Does not allow aliases. More than one option can be specified by using multiple -s arguments. For the list of options, see argument of
.I Select
.TP
.B \-p
perform a specific transformation. Must be upper case. Does not allow aliases. More than one transformation can be specified by using multiple -p arguments. For the list of transformations, see argument of
.I Perform
.TP
.B \-b
compute a specific result. Must be upper case. Does not allow aliases. Displayable results are computed and displayed by
.IR Display.
For a list of results, see
.IR Build.
.TP
.B \-0
reset the given boolean property. See the file
.IR property-rc.tex
for a list of all the properties.
.TP
.B \-1 
set the given boolean property.
.\"
.SH EXAMPLE
Interprocedural parallelization of module MATMUL in program pgm (source code in pgm-src.f):
.IP
pips -f pgm-src.f -m MATMUL -b PARALLELPRINTED_FILE pgm
.\"
.SH ENVIRONMENT
Before working with pips, you must add a directory to your path and set some environment variables. In order to do that, you should run one of the shell scripts: 
.IR Pips/pipsrc.sh 
(under any sh compatible shell) or 
.IR Pips/pipsrc.csh 
(under any csh compatible shell). 
.\"
.SH FILES
.\"
.SH "SEE ALSO"
.BR Build (1),
.BR Delete (1),
.BR Display (1),
.BR Init (1),
.BR Perform (1),
.BR Pips (1),
.BR Select (1),
.BR wpips (1).

Browse the main site
.I http://www.cri.ensmp.fr/pips
to have deeper information about
.I PIPS.

For more information see
.I Semantical Interprocedural Parallelization: 
.I An Overview of the PIPS Project
(Francois Irigoin and Pierre Jouvelot), International Conference on
Supercomputing, Cologne (Germany) 1991.

For detailed information on the usage of Pips, see
.I Manuel Utilisateur du Paralleliseur Batch
(file Documentation/dret144.f.tex).
.\"
.SH DIAGNOSTICS

