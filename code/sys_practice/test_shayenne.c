int     main()
{
    int e;
    char *envp[] = { NULL };
    char *argv[] = {"/bin/ls", "-l", "-a"};
    char* example;
    pid_t pid;


    line_read =  readline ("[shayenne's prompt]");

    // Funcao que trata a entrada de comandos e retorna os argumentos para argv
    if ((pid = fork()) == 0)
        e = execve(argv[0], argv, envp);
    else
        waitpid(-1, NULL, 0);

    chdir("..");
    if ((pid = fork()) == 0)
        e = execve(argv[0], argv, envp);
    else
        waitpid(-1, NULL, 0);

    chdir("ep1");
    if ((pid = fork()) == 0) {
        e = execve(argv[0], argv, envp);
    }
    else
        waitpid(-1, NULL, 0);

    printf("\n%s\n", get_current_dir_name());

    if (e == -1)
        fprintf(stderr, "Error: %s\n", strerror(errno));

    char *buf;

    rl_bind_key('\t',rl_abort);//disable auto-complete

    // Deixar o prompt bonitinho []
    const char *prompt = (const char *) get_current_dir_name();

    while((buf = readline("[",prompt,"]"))!=NULL)
    {
        if (strcmp(buf,"quit")==0)
            break;

        printf("[%s]\n",buf);

        if (buf[0]!=0)
            add_history(buf);
    }

    free(buf);


    return 0;
}

