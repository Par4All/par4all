.\" @(#)tpips.l 92/02/12 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH TPIPS 1 "7 March 1997"
.\"
.SH NAME
tpips \- Interprocedural Parallelizer of Scientific Programs, 
PIPS tty interface.
.SH SYNOPSIS
tpips [-nscvh?jw] [-l 
.IR logfile ] 
[-r 
.IR rcfile ] 
[-e 
.IR tpips-cmds ] 
.\"
.SH DESCRIPTION

.B tpips 
is the 
.B line interface
and scripting language of the
PIPS project. All PIPS functionalities are available, with
on-line help and automatic completion.

.B tpips
is more efficient than the pips based scripts ( Init,
Display,...) because the database is not open, read, written and closed
for each command. 
.B tpips
is less user-friendly than wpips and epips,
although it provides the unique opportunity to apply a transformation or
display some information on several modules with one command, using
pipsmake metavariables, 
.BR %CALLERS
and  
.BR %CALLEES,  
or environment  user defined variables such
as 
.BR $MODULES
The jpips future user
interface will provide these pips/tpips functionalities too.

.B tpips
is recommended for large benchmarks and experiments, when
analysis and optimization results do not require interactive working. 
.B  tpips
is used for the validation of each new pips version.

.B tpips
can be used to automatically replay a manual wpips or 
.B  tpips
session by using 
.BR logfile_to_tpips
The
.BR logfile_to_tpips 
command derives a
.B tpips 
script
from a logfile in a workspace. A PIPS beginner
should start with a window interface, wpips or epips, and not with 
.B  tpips



Access to Unix Shell and to PIPS properties is provided by 
.B tpips
PIPS properties can be modified from 
.B tpips
and only from 
.B  tpips
during a 
.B PIPS 
session. On the other hand,
.B Wpips
reads at the
beginning of the session the default properties and the new ones
(re)defined in the 
.B properties.rc
file, but no change can be done
afterwards.  Resetting properies during a session can be used to rebuild
an application when some modules have been updated by 
.B PIPS,
e.g. after
partial evaluation.

The
.Bt tpips
commands are listed on-line by the help command.
Possible arguments are completed or suggested by automatic
completion (TAB and TAB/TAB). A command may be spread on
several lines by guarding the line feed characters by
backslashes.


.\"
.SH OPTIONS

.TP
.B \-n    
 no execution mode. Just check the script syntax.
.TP
.B \-s    
 behaves like a shell. 
.B tpips
 commands simply extend a shell.
.TP
.B \-c    
 behaves like a command, not a shell (it is the default option).
.TP
.B \-v    
 displays version and architecture information.
.TP
.B \-h or \-?    
 provides some help.
.TP
.B \-j   
 jpips special mode.
.TP
.B \-w    
 starts with a wrapper (jpips special again)...
.TP
.B \-l logfile 
 log to 
.BR logfile
.TP
.B \-r rcfile 
source the 
.BR rcfile
file. (default
.B \~.tpipsrc)
.TP
.B \-e tpips-cmds
executes the 
.B tpips-cmds
commands.

.\"
.SH EXAMPLE
Interactive analysis of module MATMUL from source code pgm-src.f, 
in workspace pgm:
.IP "$ tpips"
.IP "> create pgm pgm-src.f"
.IP "> module MATMUL"
.IP "> display PARALLELPRINTED_FILE"
.IP "> quit"
.IP $

.\"
.SH ENVIRONMENT

Before using
.B tpips, 
you need to add a PIPS root directory to your path and to set
some PIPS environment variables. In order to do that, you can execute the
shell script
.B Pips/pipsrc.sh 
for any sh compatible shell, e.g. ksh or
bash, or 
.B Pips/pipsrc.csh 
for any csh compatible shell, e.g. tcsh.

.\"
.SH FILES
.\"
.SH "SEE ALSO"
.BR epips (l),
.BR pips (l),
.BR wpips (l),
.BR Build (l),
.BR Delete (l),
.BR Display (l),
.BR Init (l),
.BR Perform (l),
.BR Pips (l),
.BR Select (l).

For more information about the initial goals of the PIPS project, see

.I Semantical Interprocedural Parallelization: 
.I An Overview of the PIPS Project
(Francois Irigoin and Pierre Jouvelot), International Conference on
Supercomputing, Cologne (Germany), 1991.

For detailed information on the usage of Pips, see:

.I Manuel Utilisateur du Paralleliseur Batch
(file Documentation/dret144.f.tex).

.I WPips and EPips User Manual
(Ronan Keryell)

.I PIPS High-Level Software Interface: Pipsmake
(Remi Triolet, Francois Irigoin)

.I PIPS: Internal Representation of Fortran Code
(Francois Irigoin, Pierre Jouvelot, Ronan Keryell, Remi Triolet)

.I Properties: Low Level Tuning of PIPS
(Lei Zhou, Francois Irigoin)

On-line up-to-date 1997 html documentation is available for the PIPS project,
<http://www.cri.ensmp.fr/~pips>, and specifically for tpips,
<http://www.cri.ensmp.fr/~pips/line-interface.html>. They contains links
to all relevant PIPS documentation, including the above mentionned technical
reports and papers.

.\"
.SH DIAGNOSTICS

