/**
 * @file libm_rounding.h
 * @brief Picks a TFLite Micro golden by how the running libm rounds expm1f,
 *        tanhf and erff.
 *
 * TFLite Micro and the runtime call the platform libm for these functions,
 * so a golden that passes through one of them holds that libm's rounding.
 * The goldens recorded on glibc 2.39 are the legacy set; the *_cr headers
 * hold the outputs recorded on a correctly rounded libm, for the cases where
 * the two differ. A case is checked against the set its functions round as.
 */

#ifndef TIGRIS_TEST_LIBM_ROUNDING_H
#define TIGRIS_TEST_LIBM_ROUNDING_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { LIBM_EXPM1F = 1u, LIBM_TANHF = 2u, LIBM_ERFF = 4u };

typedef enum { LIBM_UNKNOWN, LIBM_LEGACY, LIBM_CORRECT } libm_rounding_t;

/* A case's output on a correctly rounded libm, keyed by its legacy output. */
typedef struct {
    const void *legacy;
    uint32_t functions;
    const void *output;
} libm_golden_t;

/* Inputs where glibc 2.39 and correct rounding differ, and both results. */
typedef struct {
    uint32_t function;
    const char *name;
    float input[3];
    uint32_t legacy[3], correct[3];
} libm_probe_t;

static const libm_probe_t libm_probes[3] = {
    {LIBM_EXPM1F, "expm1f", {-4.453125f, -3.515625f, -2.765625f},
     {0xbf7d0506u, 0xbf7863aau, 0xbf6fe360u}, {0xbf7d0505u, 0xbf7863abu, 0xbf6fe361u}},
    {LIBM_TANHF, "tanhf", {-1.625f, -1.375f, -0.953125f},
     {0xbf6ce37eu, 0xbf613c52u, 0xbf3dbeeeu}, {0xbf6ce37du, 0xbf613c53u, 0xbf3dbeedu}},
    {LIBM_ERFF, "erff", {-1.703125f, -0.96875f, -0.171875f},
     {0xbf7be678u, 0xbf544e42u, 0xbe44a809u}, {0xbf7be677u, 0xbf544e43u, 0xbe44a808u}},
};

/* Legacy or correct when every probe of `probe` agrees on one, else unknown
 * with `*at` the first probe that does not fit. */
static inline libm_rounding_t libm_classify(const libm_probe_t *probe, size_t *at)
{
    libm_rounding_t rounding = LIBM_UNKNOWN;
    for (size_t i = 0; i < 3u; i++) {
        /* volatile keeps the compiler from evaluating the call itself. */
        volatile float x = probe->input[i];
        float y = probe->function == LIBM_EXPM1F ? expm1f(x)
                : probe->function == LIBM_TANHF ? tanhf(x) : erff(x);
        uint32_t bits;
        memcpy(&bits, &y, sizeof(bits));
        libm_rounding_t got = bits == probe->legacy[i] ? LIBM_LEGACY
                            : bits == probe->correct[i] ? LIBM_CORRECT : LIBM_UNKNOWN;
        *at = i;
        if (got == LIBM_UNKNOWN || (i > 0u && got != rounding)) return LIBM_UNKNOWN;
        rounding = got;
    }
    return rounding;
}

/* Prints each function's class, once per test executable. */
static inline void libm_report(void)
{
    static const char *const names[] = {"unknown", "legacy", "correctly rounded"};
    printf("  libm:");
    for (size_t f = 0; f < 3u; f++) {
        size_t at;
        printf("%s %s %s", f ? "," : "", libm_probes[f].name, names[libm_classify(&libm_probes[f], &at)]);
    }
    printf("\n");
}

/* The output a case is checked against: `legacy`, or its entry in `set` when
 * the functions the case calls round correctly. NULL, after a message naming
 * the function and probe input, when they fit no recorded set. */
static inline const void *libm_golden(const libm_golden_t *set, size_t count, const void *legacy)
{
    const libm_golden_t *entry = NULL;
    const libm_probe_t *first = NULL;
    libm_rounding_t want = LIBM_UNKNOWN;
    size_t first_at = 0u;
    for (size_t i = 0; i < count; i++)
        if (set[i].legacy == legacy) entry = &set[i];
    if (entry == NULL) return legacy;
    for (size_t f = 0; f < 3u; f++) {
        const libm_probe_t *probe = &libm_probes[f];
        size_t at;
        if (!(entry->functions & probe->function)) continue;
        libm_rounding_t got = libm_classify(probe, &at);
        if (got == LIBM_UNKNOWN) {
            fprintf(stderr, "    %s(%.9g) rounds as neither the legacy nor the correctly rounded golden set\n",
                    probe->name, (double)probe->input[at]);
            return NULL;
        }
        if (first != NULL && got != want) {
            fprintf(stderr, "    %s(%.9g) and %s(%.9g) round as different golden sets\n",
                    first->name, (double)first->input[first_at], probe->name, (double)probe->input[at]);
            return NULL;
        }
        first = probe;
        first_at = at;
        want = got;
    }
    return want == LIBM_CORRECT ? entry->output : legacy;
}

#endif /* TIGRIS_TEST_LIBM_ROUNDING_H */
