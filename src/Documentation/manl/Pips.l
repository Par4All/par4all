.\" @(#)Init.l 91/05/14 ;
.TH PIPS 1 "10 March 1997"
.\"
.SH NAME
Pips \- PIPS Shell script to interface pips
.SH SYNOPSIS
Pips [\-q] [\-w 
.I wspace
] [\-f 
.IR source_file[,source_file]... ]
[\-s
.IR selected_rule[,selected_rule]... ]
[\-m
.IR module ]
[\-p
.IR performed_rule ]
[\-b
.IR build_resource ] 

.\"
.SH AVAILABILITY

This command is available with the PIPS Software, an interprocedural 
Fortran analyzer and automatic parallelizer. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on its usage. More user-friendly PIPS interfaces 
are available. See
.B epips,
.B tpips,
and
.B wpips.
.\"
.SH DESCRIPTION 
The
.B Pips
command is not designed to be used directly but as an interface
between the batch commands,
.B Init,
.B Display,
.B Build,
.B Perform,
.B Select,
.B Delete,
and the executable
.B pips.

.B Pips
can nevertheless be used directly to write Shell scripts based on PIPS.
Like
.B Display(1),
.B Pips
uses shorthand notations for rules and resources. For instance,
a
.B CALLGRAPH_FILE
can be required as
.B CGF.

The
.B Pips
based interface uses files in the current user directory to memorize 
the current workspace and the current module.

On-line html documentation about the batch interface built on top of
.B Pips
is available at

.B <http://www.cri.ensmp.fr/pips/batch-interface.html>.

The PIPS project as a whole is presented at

.B <http://www.cri.ensmp.fr/pips>.

.\"
.SH OPTIONS
.TP
.B \-q
selects the verbose mode for debugging the Pips Shell script.
.TP
.B \-w
.I wspace
is the name of the selected workspace. 
.B Pips
creates or opens a directory called
.B wspace.database.
.TP
.B \-f
.I source_file[,source_file]...]
is the list of source files containing the code to analyze. The source 
code for all functions and subroutines called in these files is
necessary to perform interprocedural analyses.
.TP
.B \-s
.I selected_rule[,selected_rule]
see 
.B Select(1).
.TP
.B \-m 
.I module
is the name of the module which is used.
.TP
.B \-p
.I performed_rule
See
.B Perform(1)
.TP
.B \-b 
.I resource
is one of many resources, such as 
.B CALLGRAPH,
.B PRECONDITIONS,
.B ...
There is no default resource. See
.B Build(1)
and
.B Display(1).

.RS
.RE
.\"
.\" 
.SH FILES

.B .pipsprogram
contains the workspace name and the current module name.

.\"
.SH "SEE ALSO"
.BR epips (l),
.BR pips (l),
.BR tpips (l),
.BR wpips (l),
.BR Build (l),
.BR Delete (l),
.BR Display (l),
.BR Init (l),
.BR Perform (l),
.BR Select (l).
.\"
.SH BUGS
