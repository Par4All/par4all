! toto in include from module foo and is renamed titi locally
! in pips we are not able to produce the correct output actually
program use
use foo, titi => toto
implicit none
integer toto
toto = 4
titi = 5
end program use

