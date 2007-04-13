      program external06

C     check the impact of signature refinment: since it comes later, g
C     signature remains unchanged, unless some sharing is introduced in
C     the symbole table for types.

      external f

      call g(f)

      call f(i, j)

      end
