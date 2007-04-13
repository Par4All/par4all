      program checkdreal

      complex z

      z = (1,0)

      print *, dreal(3), dreal(z)

      call foo

      end

      subroutine foo

      print *, dreal(3.)

      end
