#define pips_user_error(...) do { fprintf(stderr, __VA_ARGS__); return; } while(0)

static void begin_catch_stdout()
{
    stdout=freopen(".pyps.out","w+",stdout);
}
static char* end_catch_stdout()
{
    long end = ftell(stdout);
    rewind(stdout);
    long start = ftell(stdout);
    char * whole_file=calloc(1+(end-start),sizeof(char));
    fread(whole_file,end-start,sizeof(char),stdout);
    stdout=freopen("/dev/tty","w",stdout);
    return whole_file;
}

void create(char* workspace_name, char ** filenames)
{
    string main_module_name;
    tpips_init();
    gen_array_t filename_list = gen_array_make(0);
    while(*filenames)
    {
        printf("appending '%s'\n",*filenames);
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

            main_module_name = get_first_main_module();

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
    tpips_close();
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
char* info(char * topic)
{
    begin_catch_stdout();
    tp_some_info(topic);
    return end_catch_stdout();
}

static void init_res_or_rule(res_or_rule* ror,const char *res, const char *rule)
{
    ror->the_name=strdup(rule);
    ror->the_owners=gen_array_make(1);
    gen_array_append(ror->the_owners,strdup(res));
}

void apply(char * phasename, char * target)
{
    res_or_rule ror;
    init_res_or_rule(&ror,target,phasename);
    perform(safe_apply,&ror);
}

void display(char *rcname, char *target)
{
    res_or_rule ror;
    init_res_or_rule(&ror,target,rcname);
    perform(display_a_resource,&ror);
}

char* show(char * rcname, char *target)
{
    begin_catch_stdout();
    res_or_rule ror;
    init_res_or_rule(&ror,target,rcname);
    perform(just_show,&ror);
    return end_catch_stdout();
}



