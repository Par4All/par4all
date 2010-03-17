      program mtrinit
      external trinit
      common /ctrinit/ i

      print *, i

      end

      blockdata trinit
      common /ctrinit/ i
      data i /3/
      end
