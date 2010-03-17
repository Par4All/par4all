      program external08

C     check the impact of signature refinment: since it comes earlier, g
C     signature should be OK. Variation of external06.f

      external f

      call f(i, j)

      call g(f)

      end
