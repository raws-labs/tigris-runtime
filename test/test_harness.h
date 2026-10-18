/**
 * @file test_harness.h
 * @brief Assertion macros, pass/fail counters and a file loader for the
 *        native test executables. Each executable includes it once.
 */

#ifndef TIGRIS_TEST_HARNESS_H
#define TIGRIS_TEST_HARNESS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int tests_run;
static int tests_passed;
static int tests_failed;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* Integer or pointer operands only; each is evaluated once. */
#define TEST_ASSERT_EQ(a, b, msg) do { \
    long got_ = (long)(a); \
    long want_ = (long)(b); \
    tests_run++; \
    if (got_ != want_) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %ld, expected %ld)\n", \
                __func__, __LINE__, msg, got_, want_); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

#define TEST_ASSERT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (eps)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %.6f, expected %.6f)\n", \
                __func__, __LINE__, msg, (double)(a), (double)(b)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* Prints the summary line and returns the process exit code. */
static inline int test_report(void)
{
    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}

/* Reads a whole file into malloc'd storage, which tigris_plan_load accepts. */
static inline uint8_t *load_file(const char *path, uint32_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long size;
    uint8_t *buf;
    if (!f) {
        fprintf(stderr, "  Cannot open: %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    size = ftell(f);
    if (size <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    buf = (uint8_t *)malloc((size_t)size);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (uint32_t)size;
    return buf;
}

#endif /* TIGRIS_TEST_HARNESS_H */
