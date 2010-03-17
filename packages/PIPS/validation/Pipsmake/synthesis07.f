      program synthesis07

C     Check for non-trivial symbolic constants

      integer lload
      integer nbr
      parameter (nbr = 2)
      parameter  (lload  = 3*nbr )
      parameter (pi=3.14159)
      real lhiden
      parameter (lhiden = nbr*pi)

      call  hidenh ( lhiden, lload, 9999)

      end
