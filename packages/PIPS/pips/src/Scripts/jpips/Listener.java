/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

package fr.ensmp.cri.jpips;

import java.lang.*;
import java.util.*;
import java.io.*;
import java.applet.*;

/** A class that redirects an input stream to the output stream.
 * 
 * @author Francois Didry
 */  
public class Listener 
  implements Runnable
{
  public final String listener = "listener   : ";
  public final String signal = "[tpips_wrapper] killing tpips...";
  
  public Resetable jpips;
  public BufferedReader in;  //input stream from tpips
  public Pawt.PFrame frame;
  
  public Listener(BufferedReader in, Resetable jpips)
  {
    this.frame = frame;
    this.jpips = jpips;
    this.in = in;
  }
  
  /** Listens and print the specified stream.
   */  
  public void run()
  {
    try
    {
      System.out.println(listener + "Tpips listener running");
      boolean tpipsRunning = true;
      while(tpipsRunning)
      {
        String s = in.readLine();
        System.out.println(listener + s);
        if(s.indexOf(signal) != -1) tpipsRunning = false;
      }
      System.out.println(listener+"tpips down... restarting tpips...");
      jpips.reset();
    }
    catch(IOException e)
    {
      System.out.println(e);
    }
  }
}
