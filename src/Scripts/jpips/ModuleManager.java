/*
  $Id$

  $Log: ModuleManager.java,v $
  Revision 1.5  1998/11/20 17:32:18  coelho
  %MAIN added to the menu.

  Revision 1.4  1998/10/17 12:19:24  coelho
  border++.

  Revision 1.3  1998/10/16 18:06:37  coelho
  obscure bug tmp fix...

  Revision 1.2  1998/10/16 17:16:40  coelho
  attempt to port to jdk1.2b4

  Revision 1.1  1998/06/30 17:35:33  coelho
  Initial revision
*/


package JPips;

import java.lang.*;
import java.util.*;
import java.io.*;

import JPips.Pawt.*;

import java.awt.*;
import java.awt.event.*;

import com.sun.java.swing.*;
import com.sun.java.swing.border.*;


/** A module manager for TPips.
  * @author Francois Didry
  */  
public class ModuleManager implements JPipsComponent
{
  public TPips	tpips;		//tpips instance
  public PList	list;		//contains the modules
  public DefaultListModel modules;
  public PPanel panel;		//jpips module panel


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
