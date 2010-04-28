! Try to parse and print again ALLOCATABLE & ALLOCATE
      SUBROUTINE ALLOCATABLE ()

      ALLOCATABLE UBAR(:), RBAR(:)
      INTEGER ERR
      JMAX = 10
      ALLOCATE(UBAR(JMAX-1), RBAR(JMAX-1), STAT=ERR)
      UBAR=0D0
      RBAR=0D0
      print *, ubar
      print *, rbar
      RETURN
      END

program main
  call allocatable
end
