.\" @(#)epips.l 97/03/07 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH EPIPS 1 "7 March 1997"
.\"
.SH NAME
epips \- Interprocedural Parallelizer of Scientific Programs, 
GNU-Emacs interface.
.SH SYNOPSIS
epips 
.\"
.SH DESCRIPTION 

PIPS (Paralleliseur Interprocedural de Programmes Scientifiques) is a
research project from the computer science research center (CRI) at Ecole
des Mines de Paris (France). It is an interprocedural source-to-source
compiler, currently supporting most of Fortran 77, to analyze and
parallelize real scientific codes for shared memory multi-processor
machines, and to distribute automatically simple codes for processor networks.

.B epips 
is an GNU-Emacs-based window user interface of the PIPS project. All PIPS
functionalities are available through buttons and menus. Emacs display
capabilities, extended with a PIPS interface written in Emacs-LISP, 
are used to highlight relevant information and to add hypertextual
interaction.

.B epips
is a Shell script which launches 
.B emacs 
as main process and
.B wpips
as a sub-process. The log and display windows are Emacs windows
instead of Textedit windows as for 
.B wpips, 
but the control panel still is
an Xview object.

.B epips
provides advanced display capabilities based on Emacs internal mechanisms. But its other functionalities are the same as 
.B wpips. 
See
.B wpips(1).

An epips mode is also available for GNU-Emacs. Using this mode,
.B epips
can be launched from GNU-Emacs by 
.B "M-X epips".

.\"
.SH EXAMPLE

.B epips

.B "M-X epips"

.\"
.SH ENVIRONMENT

To use
.B epips, 
the PIPS root directory must be added to your command path and
some PIPS-related Shell environment variables must be set. Two
shell scripts
.B Pips/pipsrc.sh 
for any 
.B sh 
compatible shell, e.g. 
.B ksh 
or
.B bash, 
or 
.B Pips/pipsrc.csh 
for any 
.B csh 
compatible shell, e.g. 
.B tcsh, 
are provided to update your path 
and to set the necessary variables.

.\"
.SH FILES

.B epips.el

.\"
.SH "SEE ALSO"
.BR pips (l),
.BR tpips (l),
.BR wpips (l),
.BR Build (l),
.BR Delete (l),
.BR Display (l),
.BR Init (l),
.BR Perform (l),
.BR Pips (l),
.BR Select (l).

For more information see
.I Semantical Interprocedural Parallelization: 
.I An Overview of the PIPS Project
(Francois Irigoin and Pierre Jouvelot), International Conference on
Supercomputing, Cologne (Germany) 1991.

For detailed information on the usage of Pips, see:

.I Manuel Utilisateur du Paralleliseur Batch
(file Documentation/dret144.f.tex).

.I WPips and EPips User Manual 
(Ronan Keryell)

On-line
.B epips 
html documentation is available:

.B <http://www.cri.ensmp.fr/pips/wpips-epips-user-manual/wpips-epips-user-manual.html>. 

The entry point for the PIPS project is:

.B <http://www.cri.ensmp.fr/pips>.

.\"
.SH DIAGNOSTICS
