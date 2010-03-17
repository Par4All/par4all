      SUBROUTINE COMPUTE_INDEXMAP_PARDO1(K, K_L, K_U, KK, NZ, KK2, J
     &, DIMS, JJ, NY, KJ2, I, II, NX, AINDEXMAP, D)
      INTEGER NX, II, I, KJ2, NY, JJ, J, KK2, NZ, KK, K_U, K_L, K
! Check that the declaration of D happens before the declaration of AINDEXMAP
!      INTEGER INDEXMAP(1:D(1)+1, 1:D(2), 1:D(3)), D(1:3), DIMS(1:3)
      INTEGER D(1:3), AINDEXMAP(1:D(1)+1, 1:D(2), 1:D(3)), DIMS(1:3)

! Check that i declaration is added
      new = 1

      end
