      program substr
      character*80 f
      character*80 g

      f(i:j) = 'arthur'
      g = f(i:j)

      read(*,*) f(i:j)
      write(*,*) f(i:j)

      end
