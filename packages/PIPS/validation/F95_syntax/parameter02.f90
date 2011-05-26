! At the moment in PIPS we cannot keep trac of the parameter
! because the substution is done by the gfortran parser.
! Need to hack the gfortran parser to ba able to keep trac of
! such parameter as it is done in F77

program main
  INTEGER, PARAMETER :: SIZE = 10
  INTEGER :: ARRAY (SIZE)
end program main
