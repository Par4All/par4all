.\" @(#)tpips.l 92/02/12 ;
.\" Use "man 7 man" to understand the format of this troff file.
.TH TPIPS 1 "7 March 1997"
.\"
.SH NAME
tpips \- Interprocedural Parallelizer of Scientific Programs, 
PIPS tty interface.
.SH SYNOPSIS
tpips [-n] [-h] [-?] [-v] [-l
.IR logfilename ] [ 
.IR scriptfilename ]
.\"
.SH DESCRIPTION

.B PIPS 
(Paralleliseur Interprocedural de Programmes Scientifiques) 
is a
research project from the Computer Science Research Center (Centre de
Recherche en Informatique, CRI) at Ecole des Mines de Paris (France). It
is an interprocedural source-to-source compiler, currently supporting most
of Fortran 77, to analyze and parallelize real scientific codes for shared
memory multi-processor machines, and to distribute automatically simple
codes for distributed-memory machines and processor networks (e.g. NOW). 
It also includes an HPF compiler prototype, hpfc.

.B tpips 
is the line interface and scripting language of the PIPS
project. All PIPS functionalities are available, with on-line help and
automatic completion.  Access to Unix Shell and to PIPS properties also is
provided. This can be used to rebuild an application when some modules
have been updated by PIPS, e.g. after partial evaluation.

.B tpips 
is more efficient than the 
.B pips 
based scripts (
.B Init, 
.B Display,...)
because the database is not open, read, written and closed for each
command. 
.B tpips 
is less user-friendly than 
.B wpips 
and 
.B epips, 
although it
provides the unique opportunity to apply a transformation or display some
information on several modules with one command, using pipsmake
meta-variables, $ALL, $CALLERS and $CALLEES. PIPS 
.B properties
can also be modified from tpips, and only from tpips.
.B tpips 
can be used to replay a
wpips session by using 
.B logfile_to_tpips
to derive a tpips script from a logfile in a workspace. A PIPS beginner
should start with a window interface, 
.B wpips 
or 
.B epips,
and not with
.B tpips.

The tpips commands are listed on-line by the 
.B help
command. Possible
arguments are completed or suggested by automatic completion (TAB and
TAB/TAB). A command may be spread on several lines by guarding the line
feed characters by backslashes.

A simple tpips session is made of a few steps. First, a workspace
must be created:

.BR create
foo source1.f source2.f

and a subdirectory of the current directory, here 
.B foo.database, 
is opened
or created to store and retrieve code and information computed by PIPS about
subroutines and functions found in module1.f or module2.f.

Then, you have to select a module, let say
.B foo, 
by typing:

.BR module
foo

unless there is only one function or subroutine in the Fortran
file(s) which were passed to the 
.B create
command, in which case it is automatically selected.

To keep it simple the first time, you should then try to display the
sequential code by typing:

.BR display
PRINTED_FILE

Other resource names let you visualize the parallelized code
(PARALLELPRINTED_FILE), or the distributed code.

The commands for workspaces are: 
.B create 
a new workspace (see Shell command 
.B Init),
.B open 
an existing workspace, 
.B close 
the current workspace,
and 
.B delete 
the current workspace or a specific one (see Shell command
.B Delete). 

These commands can be used several times during a session or within a
tpips script file.

A specific current module is selected as current module with: 

.B module 
.I module_name

When several rules, i.e. functions, can be used to compute the same
resource, for instance a parallel version of a module, the rule selection
command is: 

.B activate 
.I rule

It is equivalent to the Shell
.B Select
command.

A resource, i.e. any piece of information computed by PIPS, 
can be required by command:

.B make 
.I resource

or

.B make 
.I resource(owner)

which uses interproceduraly and recursively default rules (see Shell command
.B Build). 
Or it is produced by
a specific rule:

.B apply 
.I rule

which will require all its resource arguments by recursively calling 
.B make
(see Shell command
.B Perform).

The 
.I owner 
of a resource is the module it is attached to. By default,
the owner is assumed to be the current module.



File resources can be displayed on screen:

.B display 
.I printable_resource

as with the Shell
.B Display
script.

Names for printable resources usually contain the string 
.B FILE. 
They are
stored as human readable ASCII file in the workspace, whereas
non-printable resources are ASCII-encoded internal data structures.

The tpips Unix-related commands are:
.B cd,
.B setenv,
.B echo,
and
.B shell.

The standard command to leave 
.B tpips 
is
.B quit
but a control-D or end-of-file condition are valid too.

.\"
.SH OPTIONS

.TP
.B \-n
requires a syntax check of a Script file (no PIPS action is performed)

.TP
.B \-l
specifies a non-default log file.

.TP
.B \-h
.TP
.B \-?
provides some help.

.TP
.B \-v
returns the version number (in fact, no; if you are good, 
you can guess if you are executing the 32-bit or 
the 64-bit version of PIPS...) before starting the 
tpips interpretation loop.

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
