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

import fr.ensmp.cri.jpips.Pawt.*;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.border.*;


/** A module manager for TPips.
 * @author Francois Didry
 */  
public class ModuleManager implements JPipsComponent
{
  public TPips tpips;  //tpips instance
  public PList list;  //contains the modules
  public DefaultListModel modules;
  public PPanel panel;  //jpips module panel
  
  
  /** Sets the model of the list containing the modules names.
   Sets the tpips instance.
   */  
  public ModuleManager(TPips tpips)
  {
    this.tpips = tpips;
    buildPanel();
  }
  
  /** Creates the module panel for jpips.
   */  
  public void buildPanel()
  {
    panel = new PPanel(new BorderLayout());
    panel.setBorder(Pawt.createTitledBorder("Modules"));
    modules = new DefaultListModel();
    list = new PList(modules);
    list.setSelectionMode(2);
    PScrollPanel scrollPanel = new PScrollPanel((Component)list);
    scrollPanel.setPreferredSize(new Dimension(200,100));
    panel.add(scrollPanel,BorderLayout.WEST);
    PButton b = new PButton("Select All");
    b.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) { 
        setAllSelected(); 
      }
    });
    panel.add(b,BorderLayout.SOUTH);
  }
  
  /** @return the module panel for JPips
   */  
  public Component getComponent()
  {
    return (Component) panel;
  }
  
  /** Sets as selected all the modules of the list.
   */  
  public void setAllSelected()
  {
    DefaultListModel dlm = (DefaultListModel) list.getModel();
    int tab[] = new int[dlm.size()-1];
    // skip first one.
    for(int i=1; i<dlm.size(); i++) tab[i-1] = i;
    list.setSelectedIndices(tab);
  }
  
  
  /** @return the selected modules
   */  
  public Object[] getSelectedModules()
  {
    return list.getSelectedValues();
  }
  
  /** Sets the modules of tpips in the modules list.
   */
  public void setModules()
  {
    //DefaultListModel dlm = (DefaultListModel) list.getModel();
    String all_modules = tpips.sendCommand("info modules");
    if (all_modules != null)
    {
      StringTokenizer tok = new StringTokenizer(all_modules, " ", false);
      String module;
      while(tok.hasMoreTokens())
      {
        module = tok.nextToken();
        // quite obscure swing bug...
        // modules.addElement(module);
        modules.add(0, module); 
      }
      modules.add(0, "%MAIN"); // add so as to select the main.
    }
  }
  
  /** Clears the modules list.
   */
  public void unsetModules()
  {
    DefaultListModel dlm = (DefaultListModel) list.getModel();
    dlm.removeAllElements();
  }
  
  public PMenu getMenu()
  {
    return null;
  }
  
  public void setActivated(boolean yes) 
  {
    // nope.
  }
  
  public void reset()
  {
    DefaultListModel dlm = (DefaultListModel) list.getModel();
    dlm.removeAllElements();
  }
}
