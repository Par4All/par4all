      program check_intrinsics
      real x(10)
      character*80 w

      x(3.5) = 0.

      x(int(3.25)) = 0.

      x(ifix(3.25)) = 0.

      x(idint(1.1D0)) = 0.

      x(int(real(3))) = 0.

      x(len(w)) = 0.

      end
