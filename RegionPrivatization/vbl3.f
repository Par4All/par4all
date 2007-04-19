      subroutine VBL3(Hydro, Frequence, Voies, Energie, BL, INTC,
     &     VSTAB, OUTPUT, N)

C     After time and energie fusion

C     parameter (N=1024*1024)

      real Hydro(0:N,0:511)
      real Frequence(0:N,0:255,0:511)
      real Voies(0:N,0:199,0:127)
      real Energie(0:N,0:199,0:127)
      real BL(0:N,0:127)
      real INTC(0:N,0:127)
      real VSTAB(0:N,0:127)
      real OUTPUT(0:N,0:127)

      integer t1, t2, t3, t4, t5, t6, t7, t8
      integer f4, f3
      integer v8, v7, v6, v5, v4, v3 
c     integer h2
      integer h1, h2

      DO t8 = 0, N
         DO t7 = 8*t8, 8*t8+7
            DO t5 = 8*t7, 8*t7 + 7
               DO t1 = 512*t5, 512*t5+511
                  read 5, (hydro(t1,h1), h1 = 0 , 511)
               ENDDo
               DO h2 = 0, 511
                  call FFTr(Frequence,t5, h2,
     &                 Hydro, N)
               ENDDO
               DO v3 = 0, 127
                  DO f3 = 0, 199
                     call FV(Voies, t5, f3, v3,
     &                    Frequence, N)
                     call MOD2(Energie(t5,f3,v3), Voies(t5,f3,v3))
                  ENDDO
               ENDDO
               DO v5 = 0, 127
                  call RtBL(BL, Energie,t5,v5,N)
               ENDDO
            ENDDO
            DO v6 = 0, 127
               call INTnL(INTC,t7,v6, BL, N)
            ENDDO
            DO v7 = 0, 127
               call STABAz(VSTAB, INTC,t7,v7,N)
            ENDDO
         ENDDO
         DO v8 = 0, 127
            call INTnL(OUTPUT,t8,v8, VSTAB, N)
         ENDDO
         print 6, (OUTPUT(t8,i), i = 0, 127)
      ENDDO

      end

      subroutine FFTr(Frequence, t2, h2, Hydro, N)
      real Frequence(0:N,0:255,0:511)
      real Hydro(0:N,0:511)

      integer t2, h2

      do i = 0, 255
         do j = 0, 511
            Frequence(t2, i, h2) = Hydro(512*t2+j,h2)
         enddo
      enddo

      end

      subroutine FV(Voies, t3, f3, v3, Frequence, N)
      real Frequence(0:N,0:255,0:511)
      real Voies(0:N,0:199,0:127)

      integer t3, f3, v3, h3
      
      do i = 0, 191
         h3 = 4*v3+i
         h3 = mod(h3, 512)
         Voies(t3,f3,v3) = Frequence(t3,f3+28,h3)
      enddo

      end

      subroutine MOD2(x,y)
      real x, y
      x = y*y
      end

      subroutine RtBL(BL, Energie,t5,v5,N)
      real Energie(0:N,0:199,0:127)
      real BL(0:N,0:127)

      integer t5, v5

      integer f

      BL(t5,v5) = 0.
      do f = 0, 199
         BL(t5,v5) = BL(t5, v5) + Energie(t5,f,v5)
      enddo

      end

      subroutine INTnC(x, y)
      real y(0:7)

      x = 0.
      do i = 0, 7
         x = x + y(i)
      enddo

      end

      subroutine STABAz(VSTAB, INTC, t7, v7, N)
      real INTC(0:N,0:127)
      real VSTAB(0:N,0:127)

      integer t7, v7

      integer v, v6

      VSTAB(t7,v7) = 0.
      do v = v7, v7+7
         v6 = mod(v, 128)
         VSTAB(t7,v7) = VSTAB(t7, v7) + INTC(t7,v6)
      enddo

      end

      subroutine INTnL(OUTPUT, t8, v8, VSTAB, N)
      real OUTPUT(0:N,0:127)
      real VSTAB(0:N,0:127)

      integer t8, v8

      integer t7

      OUTPUT(t8,v8) = 0.
      do t7 = 8*t8, 8*t8+7
         OUTPUT(t8,v8) = OUTPUT(t8,v8) + VSTAB(t7,v8)
      enddo

      end

