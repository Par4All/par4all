
/** $Id$
  * $Log: ModuleManager.java,v $
  * Revision 1.1  1998/06/30 17:35:33  coelho
  * Initial revision
  *
  */


package JPips;


import java.lang.*;
import java.util.*;
import java.io.*;
import java.awt.swing.*;
import JPips.Pawt.*;
import java.awt.*;
import java.awt.swing.border.*;
import java.awt.event.*;


/** A module manager for TPips  
  * @author Francois Didry
  */  
public class ModuleManager implements JPipsComponent
{


  public	TPips		tpips;		//tpips instance
  public	PList		list;		//contains the modules
  public	PPanel		panel;		//jpips module panel


  /** Sets the model of the list containing the modules names.
    * Sets the tpips instance.
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
      panel.setBorder(new TitledBorder("Modules"));
      list = new PList(new DefaultListModel());
      list.setSelectionMode(2);
      PScrollPanel scrollPanel = new PScrollPanel((Component)list);
      scrollPanel.setPreferredSize(new Dimension(200,100));
      panel.add(scrollPanel,BorderLayout.WEST);
      PButton b = new PButton("Select All");
      ActionListener a = new ActionListener()
        {
	  public void actionPerformed(ActionEvent e) { setAllSelected(); }
	};
      b.addActionListener(a);
      panel.add(b,BorderLayout.SOUTH);
    }


  /** @return the module panel for JPips
    */  
  public Component getComponent()
    {
      return (Component)panel;
    }



  /** Sets as selected all the modules of the list.
    */  
  public void setAllSelected()
    {
      DefaultListModel dlm = (DefaultListModel) list.getModel();
      int tab[] = new int[dlm.size()];
      for(int i=0; i<dlm.size(); i++) tab[i] = i;
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
      DefaultListModel dlm = (DefaultListModel) list.getModel();
      String s = tpips.sendCommand("info modules");
      if(s != null)
        {
	  StringTokenizer tok = new StringTokenizer(s," ",false);
          String response;
          while(tok.hasMoreTokens())
            {
	      response = tok.nextToken();
	      dlm.addElement(response);
	    }
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
  
  
  public void setActivated(boolean yes) {}


  public void reset()
    {
      DefaultListModel dlm = (DefaultListModel) list.getModel();
      dlm.removeAllElements();
    }



}


