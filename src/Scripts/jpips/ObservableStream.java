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

/*
 * $Id$
 *
 * $Log: ObservableStream.java,v $
 * Revision 1.2  1998/10/16 13:52:58  coelho
 * reindentation...
 *
 * Revision 1.1  1998/06/30 17:35:33  coelho
 * Initial revision
 *
 * Revision 1.2  1998/05/27 08:45:24  keryell
 * Small caps package name.
 *
 * Revision 1.1  1998/03/12 16:24:11  keryell
 * Initial revision
 *
 */

package JPips;

import java.io.*;
import java.util.*;

public class ObservableStream extends ByteArrayOutputStream 
{
    Vector streamObservers = new Vector();
    
    void addStreamObserver(StreamObserver o) 
    {
	streamObservers.addElement(o);
    }
    
    void removeStreamObserver(StreamObserver o) 
    {
	streamObservers.removeElement(o);
    }
        
    void notifyObservers() 
    {
	for(int i = 0; i < streamObservers.size(); i++)
	    ((StreamObserver) streamObservers.elementAt(i)).streamChanged();
    }
    
    public void write(byte[] b, int off, int len) 
    {
	super.write(b, off, len);
	notifyObservers();
    }
}
