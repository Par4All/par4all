
.\" @(#)Init.l 91/05/14 ;
.TH Select 1 "8 June 1993"
.\"
.SH NAME
Select \- pips simple command to select an option
.SH SYNOPSIS
Perform [\-w 
.I wspace
]
.I rule [
.I rule
]...
.\"
.SH AVAILABILITY
This command is available with the Pips Software. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on the usage of Select.
.\"
.SH DESCRIPTION
Select can be used on any workspace.
It calls 
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
.B rule
is one of the following keys:
.B tf, tfinter
.B pc, pcinter
.B rdg, rfulldg, rsdg
.B atomic_chains, region_chains
There is no default value.

Each rule is associated with one special pips resource, and leads to the deletion
of each resource created with a conflicting rule, regardless of the current module.
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
