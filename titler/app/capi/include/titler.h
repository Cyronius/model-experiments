/* titler: short titles for prompts, from a 4 MB model that runs on the CPU.
 *
 * Load the model once with titler_new_*, then call titler_title from any number of threads at the same time.
 * Strings are UTF-8. Functions that can fail return NULL or -1 and set a message for titler_last_error().
 */
#ifndef TITLER_H
#define TITLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(TITLER_DLL_IMPORT)
#define TITLER_API __declspec(dllimport)
#else
#define TITLER_API
#endif

typedef struct TitlerHandle titler_t;

/* Load the model from a GGUF file. Returns NULL on error. */
TITLER_API titler_t *titler_new_from_file(const char *path);

/* Load the model from GGUF bytes already in memory. The bytes are copied; free them afterwards if you like. */
TITLER_API titler_t *titler_new_from_memory(const uint8_t *data, size_t len);

/* Free a handle. NULL is fine. Do not use the handle afterwards, and do not free it while another thread uses it. */
TITLER_API void titler_free(titler_t *t);

/* Write the title for a NUL-terminated prompt into out (at most cap bytes, including the NUL).
 * beams: 4 is the default (use 0 or less for it); 1 is fastest; the maximum is 8.
 * Returns the full length of the title in bytes, not counting the NUL, or -1 on error.
 * If the return value is cap or more, the title was cut short at a character boundary. A buffer of 256 bytes is enough
 * for any title (8 words). Invalid UTF-8 in the prompt is replaced, not rejected. */
TITLER_API int64_t titler_title(const titler_t *t, const char *prompt, int beams, char *out, size_t cap);

/* Same, for a prompt that is not NUL-terminated or contains NUL bytes. */
TITLER_API int64_t titler_title_n(const titler_t *t, const uint8_t *prompt, size_t len, int beams, char *out, size_t cap);

/* The message for the last failure on the calling thread. Valid until the next titler call on this thread. */
TITLER_API const char *titler_last_error(void);

/* Library version, for example "0.1.0". */
TITLER_API const char *titler_version(void);

#ifdef __cplusplus
}
#endif

#endif
