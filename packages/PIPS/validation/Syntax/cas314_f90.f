C     Exemple de la section 3.1.4 du rapport sur les dependances

C     Version F90 avec motifs. La semantique est un peu differente
C     puisqu'il y a un effet de bord dans f() et que f() est appele
C     moins souvent que dans la version f77.

C     A compiler avec f90

      program cas314_f90

      real X(0:159), Y(0:11)

      do i1 = 0, 39
            X(4*i1:4*i1+3) = f()
      enddo

      print *, X

      do i2 = 0, 11
         Y(i2) = g(X(i2+10:i2+19))
      enddo

c     Valeurs attendues: 72, avec comme increments la suite (+3, +3, +2, +2)*
C     72.0 75.0 78.0 80.0 82.0 85.0 88.0 90.0 92.0 95.0 98.0 100.0
      print *, Y

      end

      function f()
c      static c
      data c /3./
      c = c + 1.
      f = c
      end

      function g(z)
      real z(0:9)

      print *, 'z(0)=', z(0)

      g = 0.
      do i = 0, 9
         g = g + z(i)
      enddo
      end
