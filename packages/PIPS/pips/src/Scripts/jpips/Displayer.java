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

import java.io.*;
import java.awt.*;
import java.util.*;

import fr.ensmp.cri.jpips.Pawt.*;

import javax.swing.*;
import javax.swing.border.*;


/** A window manager.
 It manages the displaying of windows on the screen.
 @author Francois Didry
 */
abstract class Displayer implements JPipsComponent
{
  public  Vector frameVector; //contains the displayed windows
  public  int  noWindows; //number of displayed windows
  public  PPanel panel;  //panel of the displayer
  
  /** Sets the number of displayed windows to 0.
   * Creates the displayer panel for JPips.
   */
  public Displayer()
  {
    noWindows = 0;
    frameVector = new Vector();
    buildPanel();
  }
  
  /** Builds the panel for JPips.
   */
  public void buildPanel()
  {
    panel = new PPanel(new BorderLayout());
    panel.setPreferredSize(new Dimension(300,150));      
    panel.setBorder(Pawt.createTitledBorder("Windows"));
  }
  
  /** Adds a window to the count.
   */
  public void incNoWindows()
  {
    noWindows++;
  }
  
  /** Removes a window from the count.
   */
  public void decNoWindows()
  {
    noWindows--;
  }
  
  /** @return the current number of windows
   */
  public int getNoWindows()
  {
    return noWindows;
  }
  
  /** @return the displayer panel for JPips
   */
  public Component getComponent()
  {
    return panel;
  }
  
  abstract boolean display(File file, boolean locked, boolean writable);
  
  abstract void display(String name, String string,
                        boolean locked, boolean writable);
  
  public PMenu getMenu() { return null; }
  
  public void setActivated(boolean yes) {}
  
  public void reset() {}
  
  /** A link between an extension and its representative string..
   * @author Francois Didry
   */
  static public class Extension
  {
    
    
    public String  extension, //extension 
      info;  //information about the extension
    
    
    /** Sets the extension and its info..
     */
    public Extension(String extension, String info)
    {
      this.extension = extension;
      this.info = info;
    }
    
    
    /** @return the extension.
     */
    public String getExtension()
    {
      return extension;
    }
    
    
    /** @return the info.
     */
    public String getInfo()
    {
      return info;
    }
    
    public String toString()
    { 
      return this.getExtension();
    }
  }
  
}
