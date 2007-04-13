      program external04

C     Bug: detect the illegal assignment to f

      external f

      f = 1.

      call g(f)

      call f()

      end
