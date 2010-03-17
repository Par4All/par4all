.\" @(#)Delete.l 91/05/14 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH Delete 1 "15 May 1991"
.\"
.SH NAME
Delete \- remove the workspace(s) of pips
.SH SYNOPSIS
.B Delete [\-a]
.IR workspace  .\|.\|. 
.\"
.SH AVAILABILITY
.LP
This command is available with the Pips Software. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on the usage of Pips.
.\"
.SH DESCRIPTION
.LP
Delete removes a list of workspaces whose names are provided after the command, no matter which the 
.IR current 
.IR workspace 
is. Filename generation rules can be used to express the workspace name, provided that special characters are quoted. For instance, you can delete every workspace begining with ws using:
.IP
.B
example% Delete 'ws*'
.\"
.SH OPTIONS
.TP
.B \-a
Delete all workspaces in the current directory, if allowed according to UNIX file protection system.
.SH ENVIRONMENT
.LP
In order to add a directory to your path and to set some environment variables, you should run one of the shell script: Pips/pipsrc.sh (under any sh compatible shell) or Pips/pipsrc.csh (under any csh compatible shell).
.\"
.LP
.SH FILES
$DOCDIR/manl/Delete.l
.\"
.SH SEE ALSO
.BR pips (1),
.BR Init (1),
.BR Build (1),
.BR Perform (1), 
.BR Pips (1),
.BR Select (1),
.BR Display (1)
.\"
.LP
.SH BUGS
Not available for the moment.
