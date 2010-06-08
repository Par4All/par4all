program test1
  CHARACTER*8 :: UHF_STR
  EQUIVALENCE (UHF, UHF_STR)
  DATA UHF_STR/"UHF     "/
  PRINT *, UHF
  PRINT *, UHF_STR
end program test1
