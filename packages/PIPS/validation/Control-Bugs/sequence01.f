! Debug the destructuration of a Fortran sequence by the new controlizer

      program sequence01

      i = 4

      if(mod(i,2).eq.0) go to 100

      i = i + 10

 100  continue

      i = i + 20

      end
