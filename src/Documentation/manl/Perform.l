.\" @(#)Init.l 91/05/14 ;
.TH PERFORM 1 "8 June 1993"
.\"
.SH NAME
Perform \- pips simple command to apply a transformation
.SH SYNOPSIS
Perform [\-w 
.I wspace
] [\-m 
.I module
] [\-v] [
.I rule
]
.\"
.SH AVAILABILITY
This command is available with the Pips Software. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on the usage of Perform.
.\"
.SH DESCRIPTION
Perform can be used on any workspace. It is the most simple command to compile with pips and browse results of all text types. It calls 
.B Pips
with appropriate arguments, and close the workspace afterwards.
.\"
.SH OPTIONS
.TP
.B \-w
.I wspace
is the name of the workspace on which applies 
.B Display.
Default is last workspace used (current workspace).
.TP
.B \-m
is the name of the module on which applies 
.B Display. 
Default is last module used (current module).
.TP
.B \-v
.B Display
is silent. Use this option (verbose) to get log information.
.TP
.B rule
is one of the following key: 
.B privatizer
.B partial eval
There is no default value.

Each rule is associated with one special pips code transformation, and leads to 
creation of one new CODE resource.
.RS
.RE
.\"
.\" .SH FILES
.\"
.SH "SEE ALSO"
.BR pips (l),
.BR wpips (l),
.BR Build (l),
.BR Delete (l),
.BR Init (l),
.BR Perform (l),
.BR Pips (l),
.BR Select (l).
.\"
.SH BUGS
