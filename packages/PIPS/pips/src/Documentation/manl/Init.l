.\" @(#)Init.l 91/05/14 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH INIT 1 "14 Mai 1991"
.\"
.SH NAME
Init \- pips command to create a workspace or/and set current workspace 
.SH SYNOPSIS
Init [\-f 
.IR file .f].\|.\|. 
.I workspace
.\"
.SH AVAILABILITY
This command is available with the Pips Software. Refer to the 
.I Manuel Utilisateur du Paralleliseur Batch
for information on the usage of Pips.
.\"
.SH DESCRIPTION
Init creates a new workspace when a readable 
.IR file .f
is provided and sets it to current workspace. Otherwise, it sets current workspace to 
.IR workspace .
.\"
.SH OPTIONS
.TP
.B \-f
Case creating a new workspace, 
.IR file .f 
is the name of a fortran source file. This file will not be modified by any Pips software, but copied in the workspace database to work on.
.TP
.B \-d
In order to execute "Delete
.I workspace
" before creating a new one.
.\"
.SH ENVIRONMENT 
In order to add a directory to your path and to set some environment variables, you should run one of the shell script: Pips/pipsrc.sh (under any sh compatible shell) or Pips/pipsrc.csh (under any csh compatible shell).
.\"
.SH FILES
.\"
.SH "SEE ALSO"
.BR pips (1),
.BR Build (1),
.BR Delete (1),
.BR Display (1),
.BR Perform (1),
.BR Pips (1),
.BR Select (1).
.\"
.SH BUGS
