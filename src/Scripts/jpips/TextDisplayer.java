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

import java.applet.*;
import java.net.*;
import java.util.*;
import java.io.*;

import fr.ensmp.cri.jpips.Pawt.*;

import java.awt.event.*;
import java.awt.*;


/** A window manager for text windows.
 * It manages the displaying of windows on the screen.
 * It regulates the amount of windows displayed at the same time.
 * @author Francois Didry
 */
public class TextDisplayer extends Displayer
{
  public Vector infos;  //contains the extensions
  public PFrame frame;  //frame of jpips
  
  /** Sets the number of displayed windows to 0.
   * Creates the vector of windows.
   */
  public TextDisplayer(PFrame frame)
  {
    super();
    this.frame = frame;
    setInfos();
  }
  
  /** Sets the extension info table.
   */
  public void setInfos()
  {
    infos = new Vector();
    infos.addElement(new Extension("pref","Basic sequential view"));
    infos.addElement(new Extension("tran","Transformers sequential view"));
    infos.addElement(new Extension("prec","Preconditions sequential view"));
    infos.addElement(new Extension("reg","Regions sequential view"));
    infos.addElement(new Extension("parf","Parallel view"));
    infos.addElement(new Extension("icfg","Basic ICFG view"));
    infos.addElement(new Extension("icfgl","Loops ICFG view"));
    infos.addElement(new Extension("icfgc","Control ICFG view"));
    infos.addElement(new Extension("cg","Call graph view"));
    infos.addElement(new Extension("adfg_file","Array data flow graphview"));  
    infos.addElement(new Extension("bdt_file","Scheduling view"));  
    infos.addElement(new Extension("plc_file","Mapping view"));  
    infos.addElement(new Extension("comp","Complexity program view"));  
    infos.addElement(new Extension("inreg"," Regions IN view"));  
    infos.addElement(new Extension("outreg","Regions OUT view"));  
    infos.addElement(new Extension("prop","Proper effects view"));  
    infos.addElement(new Extension("cumu","Cumulated effects view"));  
    infos.addElement(new Extension("dg","Dependence graph view"));  
    infos.addElement(new Extension("stco","Static control view"));  
  }
  
  /** Displays a text frame from a text file.
   @param file the file to display.
   @param locked whether the display is locked.
   @param writable whether the text can be edited.
   @return whether a new frame was created to display the file.
   */
  public boolean display(File file, boolean locked, boolean writable)
  {
    // the title includes the description, the filename and a date.
    String title = 
      getInfo(file) + ": " + file.getName() + " (" + file.lastModified() + ")";
    
    // first look for a frame with already holds this file.
    for(int i=0; i<frameVector.size(); i++)
    {
      PTextFrame f = (PTextFrame) frameVector.elementAt(i);
      if(title.equals(f.getTitle()))
      {
        f.setVisible(true);
        f.toFront();
        return false;
      }
    }
    
    PTextFrame frame = getFreeFrame();
    String text = getText(file);
    
    if(text != null)
    {
      if(frame != null)
      {
        frame.setTitle(title);
        frame.ta.setText(text);
        frame.panelButton.setText(title);
        frame.locked = locked;
        frame.writable = writable;
      }
      else
      {
        frame = new PTextFrame(title, text, locked, writable);
        register(frame);
        frame.setVisible(true);
      }
    }
    return true;
  }
  
  
  /** Displays a text frame from a string.
   */
  public void display(String title, String text,
                      boolean locked, boolean writable)
  {
    PTextFrame f = getFreeFrame();
    if(f != null)
    {
      f.setTitle(title);
      f.ta.setText(text);
      f.panelButton.setText(title);
      f.locked = locked;
      f.writable = writable;
    }
    else
    {
      f = new PTextFrame(title, text, locked, writable);
      register(f);
      f.setVisible(true);
    }
  }
  
  
  /** @return info from the specified file
   */
  public String getInfo(File file)
  {
    String name = file.getName();
    int index = name.indexOf(".");
    if(index == -1) return "Unknown file";
    String extension = name.substring(index+1);
    for(int i=0; i<infos.size(); i++)
    {
      Extension e = (Extension)infos.elementAt(i);
      if(e.getExtension().equals(extension)) return e.getInfo();
    }
    return "Unknown file";
  }
  
  
  /** Adds a windowlistener to a window.
   */
  public void register(PTextFrame f)
  {
    frameVector.addElement(f);
    WindowListener w = new WindowListener()
    {
      public void windowClosing(WindowEvent e) { removeWindow(e); }
      public void windowOpened(WindowEvent e) { addWindow(e); }
      public void windowClosed(WindowEvent e) { removeWindow(e); }
      public void windowActivated(WindowEvent e) {}
      public void windowDeactivated(WindowEvent e) {}
      public void windowIconified(WindowEvent e) {}
      public void windowDeiconified(WindowEvent e) {}
    }; 
    f.addWindowListener(w);
  }
  
  
  /** Adds a window.
   */
  public void addWindow(WindowEvent e)
  {
    PTextFrame f = (PTextFrame)e.getSource();
    incNoWindows();
    updatePanel();
  }
  
  
  /** Removes a window.
   */
  public void removeWindow(WindowEvent e)
  {
    PTextFrame f = (PTextFrame)e.getSource();
    decNoWindows();
    frameVector.removeElement(f);
    updatePanel();
  }
  
  
  /** @return a free window if possible
   */
  public PTextFrame getFreeFrame()
  {
    for(int i=0; i<frameVector.size(); i++)
    {
      PTextFrame f = (PTextFrame)frameVector.elementAt(i);
      if(!f.locked) return f;
    }
    return null;
  }
  
  
  /** Closes all the windows.
   */
  public void closeAll()
  {
    for(int i=0; i<frameVector.size(); i++)
    {
      PTextFrame f = (PTextFrame)frameVector.elementAt(i);
      f.dispose();
    }
  }
  
  
  /** @return the text of a text file
   */
  public static String getText(File f)
  {
    try
    {
      StringBuffer s = new StringBuffer();
      char[] buff = new char[5000];
      InputStreamReader reader = new FileReader(f);
      int nch;
      while ((nch = reader.read(buff, 0, buff.length)) != -1)
      {
        s.append(buff, 0, nch);
      }
      return s.toString();
    }
    catch (IOException e)
    {
      System.out.println("Could not load file: " + f);
    }
    return null;
  }
  
  
  /** Updates the panel.
   */
  public void updatePanel()
  {
    panel.removeAll();
    PPanel p = new PPanel(new GridBagLayout());
    GridBagConstraints c = new GridBagConstraints();
    c.gridx = 0;
    c.insets = new Insets(0,0,0,0);
    c.fill = GridBagConstraints.HORIZONTAL;
    c.anchor = GridBagConstraints.NORTH;
    for(int i=0; i<frameVector.size(); i++)
    {
      c.gridy = i;
      PTextFrame f = (PTextFrame)frameVector.elementAt(i);
      p.add(f.panelButton,c);
    }
    PScrollPanel scrollPanel = new PScrollPanel((Component)p);
    panel.add(scrollPanel);
    frame.repaint();
    frame.pack();
  }
  
  
}
