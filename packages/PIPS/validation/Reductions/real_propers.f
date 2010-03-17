      program propers
      real p,i,j,k,l
      s = 0
      p = 1
      i = 2
      j = 3
      k = 4
      l = 5
!
      p = p * 2
      p = 2 * p
      p = (2*l)*(k*p)
!
      s = s + 1
      s = 1 + s
      s = i + s + j
      s = s + i + j
      s = i + j + s
      s = s - i
      s = (s+3) - j
      s = (l+(k+s))+j
!
      p = 3 * (i * p) / j
      p = p / l
!
      p = 2 * i / p
      s = k - s
      end
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE
