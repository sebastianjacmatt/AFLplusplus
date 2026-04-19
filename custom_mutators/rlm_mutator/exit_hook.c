/*
 * exit_hook.c — report child exit code to rlm.py via a tmpfile.
 *
 * AFL++'s post_run() Python hook receives zero arguments; the child's exit
 * status is captured by AFL++'s waitpid() but never forwarded to Python.
 * This library intercepts exit() in every forked child and writes the code
 * to RLM_EXIT_FILE before handing off to _exit().  rlm.py::post_run() then
 * reads that file safely — AFL++ is synchronous: the child has fully exited
 * before post_run() is called, so there is no race.
 *
 * Only exit() is intercepted.  _exit() / abort() / signals (SIGSEGV etc.)
 * leave no file; rlm.py treats a missing file as a signal-based crash and
 * falls back to the coverage reward (AFL++ handles signal crashes separately).
 *
 * Performance: ~10–20 µs per execution on tmpfs.  Negligible for targets
 * that take > 1 ms per run (JS engines, interpreters, etc.).
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
 *
 * Usage (RLM_EXIT_FILE is set automatically by rlm.py::init()):
 *   LD_PRELOAD=/path/to/exit_hook.so  afl-fuzz -i seeds -o out -- ./target @@
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void exit(int code) {

    const char *path = getenv("RLM_EXIT_FILE");
    if (path) {
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "%d\n", code);
            fclose(f);
        }
    }
    _exit(code);

}
