/*
 * $Id$
 *
 * $Log: StreamObserver.java,v $
 * Revision 1.1  1998/06/30 17:35:33  coelho
 * Initial revision
 *
 * Revision 1.2  1998/05/27 08:48:46  keryell
 * Small caps package name.
 *
 * Revision 1.1  1998/03/12 16:30:42  keryell
 * Initial revision
 *
 */

/*
 * Copyright (c) 1998, Subrahmanyam Allamaraju. All Rights Reserved.
 * 
 * Permission to use, copy, modify, and distribute this software for
 * NON-COMMERCIAL purposes and without fee is hereby granted provided that this
 * copyright notice appears in all copies.
 *
 * This software is intended for demonstration purposes only, and comes without
 * any explicit or implicit warranty.
 *
 * Send all queries about this software to sallamar@cvimail.cv.com
 *
 */


package JPips;


public interface StreamObserver 
{
    public abstract void streamChanged();
}
