      SUBROUTINE COMPUTE_INDEXMAP_PARDO1(AINDEXMAP, D)

! Check that the declaration of D happens before the declaration of AINDEXMAP
!      INTEGER INDEXMAP(1:D(1)+1, 1:D(2), 1:D(3)), D(1:3), DIMS(1:3)
      INTEGER D(1:3), AINDEXMAP(1:D(1))

! Check that i declaration is added
      new = 1

      end
