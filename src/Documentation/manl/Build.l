.\" @(#)Init.l 91/05/14 ;
.TH BUILD 1 "10 March 1997"
.\"
.SH NAME
Build \- pips simple script command to build a resource
.SH SYNOPSIS
Build [\-w 
.I wspace
] [\-m 
.I module
] [\-v] 
.I resource
.\"
.SH AVAILABILITY

This command is available with the PIPS Software, an interprocedural Fortran analyzer and automatic parallelizer. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on the usage of Build. More user-friendly PIPS interfaces 
are available. See
.B epips,
.B tpips,
and
.B wpips.
.\"
.SH DESCRIPTION
 
.B Build
is one of the Shell scripts which provide a minimal batch-style 
user interface to PIPS.
It is better understood in the context of the basic commands 
of this interface, 
.B Init,
.B Display,
and
.B Delete.
Two other commands are necessary to acccess PIPS functionalities
from a Shell script,
.B Perform
and
.B Select. Many of these scripts are based on a generic script,
.B Pips,
which calls an executable,
.B pips.

.B Build 
can be used on any workspace created with Init. It produces
a 
.I resource, 
printable or non-printable, according to the 
.I pipsmake 
rules. 
Printable-resources can be produced and viewed with the 
.B Display
command. Non-printable resources are mostly useful for debugging purposes.

.B Build
calls 
.B Pips
with appropriate arguments, and close the workspace afterwards.

On-line html documentation is available. See <http://www.cri.ensmp.fr/pips>
for an overview of the PIPS project, and <http://www.cri.ensmp.fr/pips/batch-interface.html> for 
.B Build
and its companion commands.
.\"
.SH OPTIONS
.TP
.B \-w
.I wspace
is the name of the workspace on which applies 
.B Build.
Default is last workspace used (current workspace).
.TP
.B \-m
is the name of the module which 
.I owns
the required resource. By
default, the last module used, e.g. current module, is assumed.
.TP
.B \-v
.B Build
is silent. Use this option, verbose, to display on stderr information about
the actions performed by 
.B pips
to build interprocedurally the requested resource.
.TP
.B resource
is one of many
.B CALLGRAPH,
.B PRECONDITIONS,
.B ...
There is no default resource.

.RS
.RE
.\"
.\" .SH FILES
.\"
.SH "SEE ALSO"
.BR epips (l),
.BR pips (l),
.BR tpips (l),
.BR wpips (l),
.BR Delete (l),
.BR Display (l),
.BR Init (l),
.BR Perform (l),
.BR Pips (l),
.BR Select (l).
.\"
.SH BUGS
