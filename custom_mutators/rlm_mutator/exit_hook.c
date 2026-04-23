/*
 * exit_hook.c — report child exit code to rlm.py via a tmpfile.
 *
 * AFL++'s post_run() Python hook receives zero arguments; the child's exit
 * status is captured by AFL++'s waitpid() but never forwarded to Python.
 * This library intercepts exit() and _exit() in every forked child and writes
 * the code to RLM_EXIT_FILE before handing off to the real exit path.
 * rlm.py::post_run() then reads that file — AFL++ is synchronous: the child
 * has fully exited before post_run() is called, so there is no race.
 *
 * Signal deaths (SIGSEGV etc.) leave no file; rlm.py treats a missing file
 * as an abnormal termination (reward = -1.0).
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c -ldl
 *
 * Usage (RLM_EXIT_FILE should be exported by run_afl.sh before afl-fuzz):
 *   AFL_PRELOAD=/path/to/exit_hook.so  afl-fuzz -i seeds -o out -- ./target @@
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void write_exit(int code) {
    const char *path = getenv("RLM_EXIT_FILE");
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n", code);
    fclose(f);
}

__attribute__((constructor))
static void exit_hook_init(void) {
    const char *path = getenv("RLM_EXIT_FILE");
    if (!path) {
        fprintf(stderr, "[exit_hook] WARNING: RLM_EXIT_FILE not set — exit codes will not be recorded\n");
    }
}

void exit(int code) {
    write_exit(code);
    _exit(code);
}

void _exit(int code) {
    write_exit(code);
    /* call the real _exit via dlsym to terminate — if unavailable, syscall */
    typedef void (*exit_fn)(int);
    exit_fn real_exit = (exit_fn) dlsym(RTLD_NEXT, "_exit");
    if (real_exit) real_exit(code);
    syscall(231 /* SYS_exit_group on x86_64 */, code);
    __builtin_unreachable();
}
