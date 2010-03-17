      program external05

C     Bug: detect the illegal assignment to f

      external f

      call g(f)

      f = 1.

      call f()

      end
