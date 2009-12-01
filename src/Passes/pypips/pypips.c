/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
#include <stdlib.h>
#include <stdio.h>

#include "linear.h"
#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "ri-util.h" /* ri needed for statement_mapping in pipsdbm... */
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "properties.h"
#include "pipsmake.h"

#include "top-level.h"

void create(char* workspace_name, char ** filenames)
{
    /* init various composants */
    initialize_newgen();
    initialize_sc((char*(*)(Variable))entity_local_name);

    // SG: should check this !
    // pips_log_handler = tpips_user_log;

    gen_array_t filename_list = gen_array_make(0);
    static bool exception_callback_set = false;
    if(!exception_callback_set)
    {
        exception_callback_set = true;
        set_exception_callbacks(push_pips_context, pop_pips_context);
    }
    while(*filenames)
    {
        //printf("appending '%s'\n",*filenames);
        gen_array_append(filename_list,*filenames);
        filenames++;
    }


    if (workspace_exists_p(workspace_name))
        pips_user_error
            ("Workspace %s already exists. Delete it!\n", workspace_name);
    else if (db_get_current_workspace_name()) {
        pips_user_error("Close current workspace %s before "
                "creating another!\n", 
                db_get_current_workspace_name());
    } 
    else
    {
        if (db_create_workspace(workspace_name))
        {
            if (!create_workspace(filename_list))
            {
                db_close_workspace(FALSE);
                /* If you need to preserve the workspace
                   for debugging purposes, use property
                   ABORT_ON_USER_ERROR */
                if(!get_bool_property("ABORT_ON_USER_ERROR")) {
                    user_log("Deleting workspace...\n");
                    delete_workspace(workspace_name);
                }
                pips_user_error("Could not create workspace %s\n", 
                        workspace_name);
            }

            string main_module_name = get_first_main_module();

            if (!string_undefined_p(main_module_name)) {
                /* Ok, we got it ! Now we select it: */
                user_log("Main module PROGRAM \"%s\" selected.\n",
                        main_module_name);
                lazy_open_module(main_module_name);
            }
        }
        else {
            pips_user_error("Cannot create directory for workspace"
                    ", check rights!\n");
        }
    }
}

void quit()
{
    close_workspace(FALSE);
}

void set_property(char* propname, char* value)
{
    size_t len =strlen(propname) + strlen(value) + 2;
    char * line = calloc(len,sizeof(char));
    strcat(line,propname);
    strcat(line," ");
    strcat(line,value);
    parse_properties_string(line);
    free(line);
}

char* info(char * about)
{
    string sinfo = NULL;
    if (same_string_p(about, "workspace"))
    {
        sinfo = db_get_current_workspace_name();
        if(sinfo) sinfo=strdup(sinfo);
    }
    else if (same_string_p(about, "module"))
    {
        sinfo = db_get_current_module_name();
        if(sinfo) sinfo=strdup(sinfo);
    }
    else if (same_string_p(about, "modules") && db_get_current_workspace_name())
    {
        gen_array_t modules = db_get_module_list();
        int n = gen_array_nitems(modules), i;

        size_t sinfo_size=0;
        for(i=0; i<n; i++)
        {
            string m = gen_array_item(modules, i);
            sinfo_size+=strlen(m)+1;
        }
        sinfo=(char*)calloc(1+sinfo_size,sizeof(char));
        if(!sinfo) fprintf(stderr,"not enough meory to hold all module names\n");
        else {
            for(i=0; i<n; i++)
            {
                string m = gen_array_item(modules, i);
                strcat(sinfo,m); /* suboptimum*/
                strcat(sinfo," ");
            }
        }
        gen_array_full_free(modules);
    }
    else if (same_string_p(about, "directory"))
    {
        char pathname[MAXPATHLEN];
        sinfo=getcwd(pathname, MAXPATHLEN);
        if(sinfo)
            sinfo=strdup(sinfo);
        else
            fprintf(stderr,"failer to retreive current working directory\n");
    }

    if(!sinfo)
        sinfo=strdup("");
    return sinfo;
}

void apply(char * phasename, char * target)
{
    safe_apply(phasename,target);
}

void display(char *rname, char *mname)
{
    bool reset = db_get_current_module_name()==NULL;
    if(reset) db_set_current_module_name(mname);
    string fname = build_view_file(rname);
    if(reset) db_reset_current_module_name();

    if (!fname)
    {
        pips_user_error("Cannot build view file %s\n", rname);
        return;
    }

    if (!file_exists_p(fname))
    {
        pips_user_error("View file \"%s\" not found\n", fname);
        return;
    }

    FILE * in = safe_fopen(fname, "r");
    safe_cat(stdout, in);
    safe_fclose(in, fname);

    free(fname);
    return;
}

char* show(char * rname, char *mname)
{
    if (!db_resource_p(rname, mname)) {
        pips_user_warning("no resource %s[%s].\n", rname, mname);
        return  strdup("");
    }

    if (!displayable_file_p(rname)) {
        pips_user_warning("resource %s cannot be displayed.\n", rname);
        return strdup("");
    }

    /* now returns the name of the file.
    */
    return strdup(db_get_memory_resource(rname, mname, TRUE));
}

