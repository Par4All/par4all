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

import java.awt.event.*;
import java.awt.*;

import javax.swing.*;

import fr.ensmp.cri.jpips.Pawt.*;

/** A parser manager that uses a Parser object.
 * It creates a menu and a panel from a part of a text file.
 *
 * @see Parser
 * @author Francois Didry
 */
public class OptionParser
{
  public TPips  tpips;  //tpips instance
  
  public String  title;  //title of the menu
  public String  title2;  //title of the frame
  
  public PFrame  frame;  //generated frame
  public PMenu  menu;  //generated menu
  
  public Vector  state,  //contains merker components
    vector;  //contains all the components
  
  
  public Parser  p;  //the object used to read the text
  public PPanel  optionPanel; //panel included into the frame
  public int  position; //current line in optionPanel
  
  public PButton  close;  //close button of the frame
  public PCheckBoxMenuItem cbdisplay; //checkbox of the menu
  
  public GridBagConstraints c; //layout of optionPanel
  
  //these tags represents fonctionalities both used in the menu and the panel 
  public final String SINGLE_BUTTON  = "SINGLE_BUTTON",
    LARGE_BUTTON  = "LARGE_BUTTON",
    SINGLE_CHECKBOX  = "SINGLE_CHECKBOX",
    SEPARATOR  = "SEPARATOR",
    BUTTON_WITH_TEXT = "BUTTON_WITH_TEXT",
    BUTTON_WITH_MENU = "BUTTON_WITH_MENU",
    LABEL_WITH_MENU  = "LABEL_WITH_MENU",
    BUTTON_WITH_CHOICE = "BUTTON_WITH_CHOICE",
    LABEL_WITH_CHOICE = "LABEL_WITH_CHOICE",
    IBUTTON_WITH_CHOICE = "IBUTTON_WITH_CHOICE",
    ILABEL_WITH_CHOICE = "ILABEL_WITH_CHOICE",
    MENU   = "MENU",
    RADIOBUTTONGROUP = "RADIOBUTTONGROUP",
    RADIOBUTTON  = "RADIOBUTTON",
    GROUP   = "GROUP",
    CLOSE   = "CLOSE";
  
  public final String image    = "DownArrow.gif",
    displayButton   = "Display Frame",
    closeButton   = "Close";
  
  
  /** Defines frame, menu, and optionPanel.
   * Adds a "display frame" checkbox in the menu.
   * Launches the parsing that fills menu and optionPanel.
   * Adds a "close" button at the end of optionPanel.
   */
  public OptionParser(Parser p, TPips tpips)
  {
    this.p = p;
    this.tpips = tpips;
    state = new Vector();
    vector = new Vector();
    
    this.title = p.nextNonEmptyLine();
    this.title2 = p.nextNonEmptyLine();
    
    frame = new PFrame(title2);
    GridBagLayout fLayout = new GridBagLayout();
    GridBagConstraints cc = new GridBagConstraints();
    cc.insets = new Insets(0,0,0,0);
    frame.getContentPane().setLayout(fLayout);
    frame.setResizable(false);
    frame.setVisible(false);
    
    menu = new PMenu(title);
    
    //display frame checkbox
    cbdisplay = new PCheckBoxMenuItem(displayButton);
    ActionListener a = new ActionListener()
    {
      public void actionPerformed(ActionEvent e)
      { frame.setVisible(cbdisplay.isSelected()); } 
    };
    cbdisplay.addActionListener(a);
    menu.add(cbdisplay);
    menu.add(new PSeparator());
    
    GridBagLayout opLayout = new GridBagLayout();
    optionPanel = new PPanel(opLayout);
    c = new GridBagConstraints();
    position = 0;
    
    parseCommand(null, null, null, null, null);
    
    //line and close button
    PLabel b = new PLabel();
    b.setPreferredSize(new Dimension(1,2));
    
    add(optionPanel,b,
        0,position++,3,1,1,1,0.0,0.0,5,GridBagConstraints.HORIZONTAL,c);
    close = new PButton(closeButton);
    a = new ActionListener()
    {
      public void actionPerformed(ActionEvent e)
      {
        frame.setVisible(false);
        cbdisplay.setSelected(false);
      } 
    };
    close.addActionListener(a);
    add(optionPanel,close,
        0,position,3,1,1,1,0.0,0.0,0,GridBagConstraints.HORIZONTAL,c);
    
    WindowListener w = new WindowAdapter()
    {
      public void windowClosing(WindowEvent e)
      { cbdisplay.setSelected(false); }
      public void windowOpened(WindowEvent e)
      { cbdisplay.setSelected(true); }
    };
    frame.addWindowListener(w);
    
    Dimension opDim = opLayout.minimumLayoutSize(optionPanel);
    optionPanel.setPreferredSize(opDim);
    
    add(frame.getContentPane(),optionPanel,
        0,0,1,1,1,1,0.0,0.0,0,GridBagConstraints.HORIZONTAL,cc);
    
    Dimension fDim = fLayout.minimumLayoutSize(frame);
    frame.setSize(fDim.width+20, fDim.height+40);
  }
  
  
  /** Recursively calls the parsing and identifies the tags.
   * Calls a single method for each tag.
   * @param m1 is the current menu in optionPanel
   * @param m2 is the current menu in menu
   * @param is the current "display" label
   * @param bg1 is the current buttongroup in optionPanel
   * @param bg2 is the current buttongroup in menu
   */
  public void parseCommand(PMenu m1, PMenu m2, PLabel l, 
                           PButtonGroup bg1, PButtonGroup bg2)
  {
    String s = p.nextNonEmptyLine().trim();
    
    while(s != null && !s.equals(CLOSE))
    {
      if(s.equals(SINGLE_BUTTON))
        addSingleButton(false);
      else if(s.equals(LARGE_BUTTON))
        addSingleButton(true);
      else if(s.equals(SINGLE_CHECKBOX))
        addSingleCheckBox();
      else if(s.equals(SEPARATOR))
        addSeparator();
      else if(s.equals(BUTTON_WITH_TEXT))
        addButtonWithText();
      else if(s.equals(BUTTON_WITH_MENU))
        addButtonWithMenu();
      else if(s.equals(LABEL_WITH_MENU))
        addLabelWithMenu();
      else if(s.equals(BUTTON_WITH_CHOICE))
        addButtonWithChoice(true);
      else if(s.equals(IBUTTON_WITH_CHOICE))
        addButtonWithChoice(false);
      else if(s.equals(LABEL_WITH_CHOICE))
        addLabelWithChoice(true);
      else if(s.equals(ILABEL_WITH_CHOICE))
        addLabelWithChoice(false);
      else if(s.equals(RADIOBUTTON))
        addRadioButtonMenuItem(m1, m2, l, bg1, bg2);
      
      // recursion...
      else if(s.equals(MENU))
        addMenu(m1, m2, l, bg1, bg2);
      else if(s.equals(RADIOBUTTONGROUP))
        addRadioButtonGroup(m1, m2, l, bg1, bg2);
      else if (s.equals(GROUP))
        addGroup(m1, m2, l, bg1, bg2);
      
      // error...
      else
        System.err.println("unexpected: " + s);
      
      s = p.nextNonEmptyLine();
    }
  }
  
  
  /** A short add method for a GridBagLayout.
   */
  public void add(Container cont, 
                  Component comp,
                  int x, int y, 
                  int w, int h,
                  int px, int py, 
                  double wex, double wey, 
                  int in, int f, 
                  GridBagConstraints c)
  {
    c.anchor = GridBagConstraints.EAST;
    c.insets = new Insets(in,in,in,in);
    c.gridx = x;
    c.gridy = y ;
    c.gridwidth = w;
    c.gridheight = h;
    c.ipadx = px;
    c.ipady = py;
    c.weightx = wex;
    c.weighty = wey;
    c.fill = f;
    cont.add(comp,c);
    vector.addElement(comp);
  }
  
  
  /** Adds a button to optionPanel.
   * Adds an executable menuItem to menu.
   * This method can be called everywhere in the tree structure.
   */
  public void addSingleButton(boolean large)
  {
    String name, command, tip;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PButton b = new PButton(name, command, tip);
    add(optionPanel,b,
        0,position++,large? 3:1,1,1,1,0.0,0.0,2,
        GridBagConstraints.HORIZONTAL,c);
    
    PMenuItem mi = new PMenuItem(name,command);
    menu.add(mi);
    
    b.addActionListener(getBListener());
    mi.addActionListener(getMIListener());
  }
  
  
  /** Adds a checkbox to optionPanel.
   * Adds a checkbox menuItem to menu.
   * This method can be called everywhere in the tree structure.
   */
  public void addSingleCheckBox()
  {
    String name, command, checking, tip;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    checking = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PCheckBox cb = new PCheckBox(name,command,checking,tip);
    add(optionPanel,cb,
        0,position++,3,1,1,1,0.0,0.0,2,GridBagConstraints.NONE,c);
    
    PCheckBoxMenuItem cbmi = new PCheckBoxMenuItem(name,command,checking,cb);
    menu.add(cbmi);
    cb.cbmi = cbmi;
    
    cb.addActionListener(getCBListener());
    cbmi.addActionListener(getCBMIListener());
  }
  
  /** Adds a line to optionPanel.
   * Adds a separator to menu.
   * This method can be called everywhere in the tree structure.
   */
  public void addSeparator()
  {
    PLabel b = new PLabel();
    b.setPreferredSize(new Dimension(1,2));
    
    add(optionPanel,b,
        0,position++,3,1,1,1,0.0,0.0,5,GridBagConstraints.HORIZONTAL,c);
    
    PSeparator s = new PSeparator();
    menu.add(s);
  }
  
  
  /** Adds a button and a textfield to optionPanel.
   * This method can only be called in the root of the tree structure.
   */
  public void addButtonWithText()
  {
    String name, command, tip;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PTextField tf = new PTextField();
    PButton b = new PButton(name, command, tip, tf, null);
    
    add(optionPanel,b,
        0,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.NONE,c);
    
    add(optionPanel,tf,
        1,position++,2,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PMenuItem mi = new PMenuItem(name,command);
    menu.add(mi);
    
    b.addActionListener(getBTListener());  
    mi.addActionListener(getMITListener());
  }
  
  /** @return the down arrow file name.
   */
  private String getDownArrowImageFileName()
  {
    String down = System.getProperty("jpips.downarrow");
    
    File f = new File(down);
    if (f.exists()) return down;
    
    // default
    f = new File(image);
    if (f.exists()) return image;
    
    return null;      
  }
  
  /** common part for the labeled-menu in label and button with menu.
   * it creates a label and a menu which updates the label on selections.
   */
  private void addWithMenu(PMenu m2)
  {
    PLabel l = new PLabel();
    add(optionPanel,l,
        1,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PMenuBar mb = new PMenuBar();
    PMenu m1 = new PMenu();
    ImageIcon icon = new ImageIcon(getDownArrowImageFileName());
    m1.setIcon(icon);
    
    parseCommand(m1, m2, l, null, null);
    
    mb.add(m1);
    add(optionPanel,mb,
        2,position++,1,1,1,1,0.0,0.0,2,GridBagConstraints.NONE,c);
  }
  
  /** Adds a button, a label, and a single menu menubar to optionPanel.
   * The selection of the menu is displayed in the label.
   * Adds an executable menuItem to menu.
   * This method can only be called in the root of the tree structure.
   */
  public void addButtonWithMenu()
  {
    String name, command, tip;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PButton b = new PButton(name,command,tip);
    add(optionPanel,b,
        0,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    b.addActionListener(getBListener());
    
    addWithMenu(null);
    
    PMenuItem mi = new PMenuItem(name,command);
    menu.add(mi);
    mi.addActionListener(getMIListener());
  }
  
  /** Adds 2 labels, and a single menu menubar to optionPanel.
   * The selection of the menu is displayed in the second label.
   * Adds a menu to menu.
   * This method can only be called the root of the tree structure.
   */
  public void addLabelWithMenu()
  {
    String name, tip;
    name = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PLabel l1 = new PLabel(name, tip);
    add(optionPanel,l1,
        0,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PMenu m2 = new PMenu(name);
    addWithMenu(m2);
    menu.add(m2);
  }
  
  
  /** Adds a button and a combobox to optionPanel.
   * Adds an executable menuItem to menu.
   * This method can only be called the root of the tree structure.
   */
  public void addButtonWithChoice(boolean direct)
  {
    String name, command, checking, tip;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    checking = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PButton b = new PButton(name, command, tip, null, null);
    add(optionPanel,b,
        0,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PComboBox cob = new PComboBox(checking, null, direct);
    
    PMenuItem mi = new PMenuItem(name,command);
    menu.add(mi);
    
    name = p.nextNonEmptyLine();
    while(!name.equals(CLOSE))
    {
      cob.addItem(name);
      cob.vCommand.addElement(p.nextNonEmptyLine());
      cob.vChecking.addElement(p.nextNonEmptyLine());
      name = p.nextNonEmptyLine();
    }
    
    add(optionPanel,cob,
        1,position++,2,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    b.addActionListener(getBCOBListener(cob));
    cob.addActionListener(getSingleCOBListener());
    mi.addActionListener(getMIListener());
  }
  
  
  /** Adds a label and a combobox to optionPanel.
   * Adds a menu to menu.
   * This method can only be called the root of the tree structure.
   */
  public void addLabelWithChoice(boolean direct)
  {
    String name, checking, mark, tip;
    
    name = p.nextNonEmptyLine();
    checking = p.nextNonEmptyLine();
    mark = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PLabel l = new PLabel(name, tip);
    add(optionPanel,l,
        0,position,1,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PComboBox cob = new PComboBox(checking, mark, direct);
    state.addElement(cob);
    
    PMenu m = new PMenu(name);
    PButtonGroup bg = new PButtonGroup();
    
    name = p.nextNonEmptyLine();
    while(!name.equals(CLOSE))
    {
      String command = p.nextNonEmptyLine();
      checking = p.nextNonEmptyLine();
      
      cob.addItem(name);
      cob.vCommand.addElement(command);   
      cob.vChecking.addElement(checking);
      
      PRadioButtonMenuItem rbmi = new PRadioButtonMenuItem
        (name, command, cob, cob.getItemAt(cob.getItemCount()-1), checking);
      rbmi.addActionListener(getComplexRBMIListener());
      bg.add(rbmi);
      m.add(rbmi);
      
      cob.vRbmi.addElement(rbmi);   
      
      name = p.nextNonEmptyLine();
    }
    
    // select the first element.
    ((PRadioButtonMenuItem) cob.vRbmi.elementAt(0)).setSelected(true);
    
    add(optionPanel,cob,
        1,position++,2,1,1,1,0.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    cob.addActionListener(getCOBListener());
    
    menu.add(m);
  }
  
  
  /** Adds a label and a combobox to optionPanel.
   * Adds a menu to menu.
   * This method can only be called in a menu of the tree structure.
   */
  public void addMenu(PMenu m1, PMenu m2, PLabel l, 
                      PButtonGroup bg1, PButtonGroup bg2)
  {
    PMenu newm1, newm2;
    String name = p.nextNonEmptyLine();
    
    newm1 = new PMenu(name);
    newm2 = new PMenu(name);
    
    parseCommand(newm1, newm2, l, bg1, bg2);
    
    m1.add(newm1);
    if(m2 != null) m2.add(newm2);
  }
  
  /** group some stuff together in the panel.
   */
  public void addGroup(PMenu m1, PMenu m2, PLabel l, 
                       PButtonGroup bg1, PButtonGroup bg2)
  {
    String name, tip;
    
    name = p.nextNonEmptyLine();
    tip = p.nextNonEmptyLine();
    
    PPanel newp = new PPanel(new GridBagLayout());
    newp.setBorder(Pawt.createTitledBorder(name));
    newp.setToolTipText(tip);
    
    add(optionPanel, newp, 
        0,position++,3,1,1,1,1.0,0.0,2,GridBagConstraints.HORIZONTAL,c);
    
    PPanel saved = optionPanel;
    optionPanel = newp;
    
    parseCommand(m1, m2, l, bg1, bg2);
    
    optionPanel = saved;
  }
  
  /** Sets the beggining of a button group.
   * This method can only be called in a menu of the tree structure.
   */
  public void addRadioButtonGroup(PMenu m1, PMenu m2, PLabel l, 
                                  PButtonGroup bg1, PButtonGroup bg2)
  {
    String checking = p.nextNonEmptyLine();
    
    PButtonGroup newbg1 = new PButtonGroup(checking);
    PButtonGroup newbg2 = new PButtonGroup(checking);
    
    parseCommand(m1,m2,l,newbg1,newbg2);
    
    // select first menu
    PRadioButtonMenuItem rbmi
      = (PRadioButtonMenuItem)(newbg1.getElements().nextElement());
    rbmi.setSelected(true);
    JPopupMenu pm = (JPopupMenu) rbmi.getParent();
    PMenu m = (PMenu) pm.getInvoker();
    rbmi.label.setText(m.getText()+" / "+rbmi.getText());
    
    // also.
    rbmi = (PRadioButtonMenuItem)(newbg2.getElements().nextElement());
    rbmi.setSelected(true);
    pm = (JPopupMenu) rbmi.getParent();
    m = (PMenu) pm.getInvoker();
    rbmi.label.setText(m.getText()+" / "+rbmi.getText());
  }
  
  
  
  /** Adds a radiobutton on its own.
   * This method can only be called in a menu of the tree structure.
   */
  public void addRadioButtonMenuItem(PMenu m1, PMenu m2, PLabel l, 
                                     PButtonGroup bg1, PButtonGroup bg2)
  {
    String name, command, checking;
    
    name = p.nextNonEmptyLine();
    command = p.nextNonEmptyLine();
    checking = p.nextNonEmptyLine();
    
    if (bg1==null) 
      throw new Error("unexpected RadioButton (no group)! " +
                      name + "/" + command + "/" + checking);
    
    if(bg2 != null && m2 != null)
    {
      PRadioButtonMenuItem rbmi1 = 
        new PRadioButtonMenuItem(name,command,l,null,checking);
      PRadioButtonMenuItem rbmi2 =
        new PRadioButtonMenuItem(name,command,l,rbmi1,checking);
      rbmi1.rbmi = rbmi2;
      rbmi1.addActionListener(getRBMILListener());
      rbmi2.addActionListener(getRBMILListener());
      bg1.add(rbmi1);
      bg2.add(rbmi2);
      m1.add(rbmi1); 
      m2.add(rbmi2);
    }
    else
    {
      PRadioButtonMenuItem rbmi1 = 
        new PRadioButtonMenuItem(name,command,l,null);
      rbmi1.addActionListener(getSingleRBMILListener());
      bg1.add(rbmi1);
      m1.add(rbmi1);
    }     
  }
  
  
  /** @return an ActionListener for a button
   */
  public ActionListener getBListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) { 
        tpips.sendCommand(((PButton)e.getSource()).command); 
      }
    };
  }
  
  /** an action on a comboboxed button.
   */
  class BCOBActionListener 
    implements ActionListener
  {
    private PComboBox cob;
    
    BCOBActionListener(PComboBox cob) { this.cob = cob; }
    
    /** a button of a combobox is pressed.
     * its action depends on whether cob is direct or not.
     */
    public void actionPerformed(ActionEvent e)
    {
      PButton b = (PButton) e.getSource();
      if (cob.direct)
      {
        tpips.sendCommand(((PButton)e.getSource()).command);
      }
      else
      {
        int selectedIndex = cob.getSelectedIndex();
        String command = (String)cob.vCommand.elementAt(selectedIndex);
        tpips.sendCommand(command);
      }
    }
  }
  
  /** @return an ActionListener for a button in a combobox.
   */
  public ActionListener getBCOBListener(PComboBox cob)
  {
    return new BCOBActionListener(cob);
  }
  
  /** @return an ActionListener for a button with a text
   */
  public ActionListener getBTListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PButton b = (PButton)e.getSource();
        tpips.sendCommand(b.command+b.tf.getText());
      }
    };
  }
  
  
  /** @return an ActionListener for a menu item
   */
  public ActionListener getMIListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) { 
        tpips.sendCommand(((PMenuItem)e.getSource()).command); 
      }
    };
  }
  
  
  /** @return an ActionListener for a menu item with text
   */
  public ActionListener getMITListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PMenuItem mi = (PMenuItem)e.getSource();
        tpips.sendCommand(mi.command+mi.tf.getText());
      }
    };
  }
  
  
  /** @return an ActionListener for a checkbox
   */
  public ActionListener getCBListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PCheckBox cb = (PCheckBox)e.getSource();
        cb.cbmi.setSelected(cb.isSelected());
        tpips.sendCommand(cb.command);
      }
    };
  }
  
  
  /** @return an ActionListener for a checkbox menu item
   */
  public ActionListener getCBMIListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PCheckBoxMenuItem cbmi = (PCheckBoxMenuItem)e.getSource();
        tpips.sendCommand(cbmi.command);
        cbmi.cb.setSelected(cbmi.isSelected());
      }
    };
  }
  
  
  /** @return an ActionListener for combobox on its own
   */
  public ActionListener getSingleCOBListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PComboBox cob = (PComboBox)e.getSource();
        if (cob.direct) 
        {
          int selectedIndex = cob.getSelectedIndex();
          String command = 
            (String)cob.vCommand.elementAt(selectedIndex);
          tpips.sendCommand(command);
        }
      }
    };
  }
  
  
  /** @return an ActionListener for a combobox linked to a radiobutton group
   */
  public ActionListener getCOBListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PComboBox cob = (PComboBox)e.getSource();
        int selectedIndex = cob.getSelectedIndex();
        PRadioButtonMenuItem rbmi = 
          (PRadioButtonMenuItem)cob.vRbmi.elementAt(selectedIndex);
        rbmi.setSelected(true);
        if (cob.direct)
        {
          String command = (String)
            cob.vCommand.elementAt(selectedIndex);
          tpips.sendCommand(command);
        }
      }
    };
  }
  
  /** @return an ActionListener for a radiobuttonmenuitem linked to a combobox
   */
  public ActionListener getComplexRBMIListener() {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e)
      {
        PRadioButtonMenuItem rbmi = (PRadioButtonMenuItem)e.getSource();
        rbmi.cob.setSelectedItem(rbmi.o);
        tpips.sendCommand(rbmi.command);
      }
    };
  }
  
  /** @return an ActionListener for a radiobuttonmenuitem on its own
   */
  public ActionListener getSingleRBMILListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PRadioButtonMenuItem rbmi = (PRadioButtonMenuItem)e.getSource();
        rbmi.label.setText(rbmi.getText());
        tpips.sendCommand(rbmi.command);
      } 
    };
  }
  
  
  /** @return an ActionListener for a radiobuttonmenuitem linked to another one
   */
  public ActionListener getRBMILListener()
  {
    return new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        PRadioButtonMenuItem rbmi = (PRadioButtonMenuItem)e.getSource();
        rbmi.rbmi.setSelected(true);
        JPopupMenu pm = (JPopupMenu) rbmi.getParent();
        PMenu m = (PMenu) pm.getInvoker();
        rbmi.label.setText(m.getText() + " / " + rbmi.getText());
        tpips.sendCommand(rbmi.command);
      } 
    };
  }
}
