/* Built and run by capi/tests/run-smoke.sh. Usage: smoke MODEL.gguf */
#include <stdio.h>
#include <string.h>
#include <titler.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s)\n", msg, titler_last_error()); return 1; } } while (0)

int main(int argc, char **argv) {
    CHECK(argc == 2, "usage: smoke MODEL.gguf");
    CHECK(titler_new_from_file("no-such-file.gguf") == NULL, "missing file must fail");
    CHECK(strlen(titler_last_error()) > 0, "error message set");

    titler_t *t = titler_new_from_file(argv[1]);
    CHECK(t != NULL, "load model");

    char out[256];
    int64_t n = titler_title(t, "how do I sort a list of dicts by a key in python", 0, out, sizeof out);
    CHECK(n > 0 && n < (int64_t)sizeof out, "title length");
    CHECK(strlen(out) == (size_t)n, "title is NUL-terminated at its length");
    printf("%s\n", out);

    char tiny[8];
    n = titler_title(t, "how do I sort a list of dicts by a key in python", 1, tiny, sizeof tiny);
    CHECK(n >= (int64_t)sizeof tiny && strlen(tiny) == sizeof tiny - 1, "short buffer is cut, full length returned");

    CHECK(titler_title(t, NULL, 4, out, sizeof out) == -1, "NULL prompt fails");
    CHECK(titler_title(NULL, "x", 4, out, sizeof out) == -1, "NULL handle fails");
    CHECK(titler_title_n(t, (const uint8_t *)"", 0, 4, out, sizeof out) >= 0, "empty prompt works");

    titler_free(t);
    titler_free(NULL);
    printf("titler %s ok\n", titler_version());
    return 0;
}
