.\" @(#)wpips.l 92/02/12 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH WPIPS 1 "8 June 1993"
.\"
.SH NAME
wpips \- Interprocedural Parallelizer of Scientific Programs, 
X-Window interface.
.SH SYNOPSIS
wpips [
.IR X-Window options] [\-workspace
.IR workspace name [\-module
.IR module name]
.IR
[\-files
.IR source_files ]].\|.\|. 
.\"
.SH DESCRIPTION 

PIPS (Paralleliseur Interprocedural de Programmes Scientifiques) is a
research project from the computer science research center (CRI) at Ecole
des Mines de Paris (France). It is an interprocedural source-to-source
compiler, currently supporting most of Fortran 77, to analyze and
parallelize real scientific codes for shared memory multi-processor
machines, and to distribute automatically simple codes for processor networks.

wpips is the main user interface of the PIPS project. All PIPS
functionalities are available through buttons and menus.

First, a workspace must be selected, and a subdirectory of the current
directory is open or created to store and retrieve code and information
computed by PIPS:

.BR workspace.database.

The interprocedural database build by wpips is stored in this
subdirectory. wpips will display a selection of Fortran files which are
available in the current directory. The current directory may be change

Then, you have to select a module, unless there is only one in the Fortran
file(s) that are analyzed, in which case it is automatically selected.

To keep it simple the first time, you should then try to display the
sequential code by selecting the option

.BR Sequential View

in the

.BR Edit/View

menu. Other options let you visualize the parallelized code, the distributed code

.\"
.SH OPTIONS
.TP
.B \X-Window options

are the standard X-window options

.TP
.B \-workspace
defines the initial workspace. Must be lower case (?).

.TP
.B \-module
defines the initial module of interest. Must be upper case (?).

.TP
.B \-files

introduces a list of Fortran files containing the program to analyze. Multiple -f arguments are allowed. Source code must be available for all called subroutines and functions, but a main program is not necessary.

.\"
.SH EXAMPLE
Analysis of module MATMUL in program pgm (source code in pgm-src.f):
.IP
wpips -workspace pgm -module MATMUL -files pgm-src.f
.\"
.SH ENVIRONMENT

Before working with wpips, you must add a directory to your path and set
some environment variables. In order to do that, you should run one of the
shell scripts: Pips/pipsrc.sh (under any sh compatible shell, e.g. ksh or
bash) or Pips/pipsrc.csh (under any csh compatible shell, e.g. tcsh).

.\"
.SH FILES
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

For more information see
.I Semantical Interprocedural Parallelization: 
.I An Overview of the PIPS Project
(Francois Irigoin and Pierre Jouvelot), International Conference on
Supercomputing, Cologne (Germany) 1991.

For detailed information on the usage of Pips, see
.I Manuel Utilisateur du Paralleliseur Batch
(file Documentation/dret144.f.tex).
.I WPips and EPips User Manual (Ronan Keryell)

On-line html documentation is available.
.\"
.SH DIAGNOSTICS
