/**
 * @file test_iface.c
 * @brief Boundary conversion between the model's interface and the plan.
 *
 * A plan executes on one dtype, which is not always the dtype the model file
 * declares at its boundary. These tests drive tigris_input_write and
 * tigris_output_read over the schema-6 fixture, whose model input and output
 * declare float32 while the plan stores int8, and over hand-built tensors for
 * the cases the fixture cannot reach.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

#define TEST_ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) != (b)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %ld, expected %ld)\n", \
                __func__, __LINE__, msg, (long)(a), (long)(b)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

static uint8_t *load_file(const char *path, uint32_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long size;
    uint8_t *buf;
    if (!f)
        return NULL;
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

/* The schema-6 fixture: a QDQ Conv whose boundary declares float32. */
static uint8_t *fixture_buf = NULL;
static tigris_plan_t fixture_plan;

static int load_fixture(void)
{
    uint32_t len = 0;
    char path[512];
    int n = snprintf(path, sizeof(path), "%s/schema-v6-interface-dtype.tgrs",
                     TIGRIS_SCHEMA_COMPAT_DIR);
    if (n <= 0 || (size_t)n >= sizeof(path))
        return -1;
    fixture_buf = load_file(path, &len);
    if (!fixture_buf)
        return -1;
    return tigris_plan_load(fixture_buf, len, &fixture_plan) == TIGRIS_OK ? 0 : -1;
}

static void test_declared_interface_sizes_the_caller_buffer(void)
{
    uint16_t idx = fixture_plan.model_inputs[0];
    const tigris_tensor_t *t = &fixture_plan.tensors[idx];
    printf("  test_declared_interface_sizes_the_caller_buffer...\n");
    TEST_ASSERT_EQ(t->dtype, 3, "plan stores int8");
    TEST_ASSERT_EQ(t->iface_dtype, 1, "model declares float32");
    /* Four bytes per element where the plan stores one. */
    TEST_ASSERT_EQ(tigris_iface_bytes(&fixture_plan, idx),
                   t->size_bytes * 4u, "float interface is four times as wide");
}

static void test_float_roundtrips_through_the_quantized_boundary(void)
{
    uint16_t in_idx = fixture_plan.model_inputs[0];
    const tigris_tensor_t *t = &fixture_plan.tensors[in_idx];
    uint32_t elements = t->size_bytes;
    uint32_t bytes = tigris_iface_bytes(&fixture_plan, in_idx);
    const tigris_quant_param_t *qp =
        tigris_tensor_quant(&fixture_plan, t);
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors,
                                  sizeof(void *));
    int8_t *storage = (int8_t *)calloc(elements, 1);
    float *sent = (float *)malloc(bytes);
    tigris_mem_t mem;

    printf("  test_float_roundtrips_through_the_quantized_boundary...\n");
    if (!ptrs || !storage || !sent) {
        free(ptrs); free(storage); free(sent);
        return;
    }
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = fixture_plan.header->num_tensors;
    ptrs[in_idx] = storage;

    /* Values on the quantization grid survive the round trip exactly. */
    for (uint32_t i = 0; i < elements; i++)
        sent[i] = (float)((int)i - 4) * qp->scale;

    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, sent, bytes),
                   TIGRIS_OK, "float input is accepted");
    for (uint32_t i = 0; i < elements; i++)
        TEST_ASSERT_EQ(storage[i], (int8_t)((int)i - 4 + qp->zero_point),
                       "input lands on the stored grid");

    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, sent,
                                      bytes - 4u),
                   TIGRIS_ERR_BAD_SIZE, "a short buffer is refused");
    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, NULL, bytes),
                   TIGRIS_ERR_NULL, "a null buffer is refused");

    free(sent);
    free(storage);
    free(ptrs);
}

static void test_output_is_read_back_in_the_declared_dtype(void)
{
    uint16_t out_idx = fixture_plan.model_outputs[0];
    const tigris_tensor_t *t = &fixture_plan.tensors[out_idx];
    uint32_t elements = t->size_bytes;
    uint32_t bytes = tigris_iface_bytes(&fixture_plan, out_idx);
    const tigris_quant_param_t *qp = tigris_tensor_quant(&fixture_plan, t);
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors,
                                  sizeof(void *));
    int8_t *storage = (int8_t *)calloc(elements, 1);
    float *received = (float *)malloc(bytes);
    tigris_mem_t mem;

    printf("  test_output_is_read_back_in_the_declared_dtype...\n");
    if (!ptrs || !storage || !received) {
        free(ptrs); free(storage); free(received);
        return;
    }
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = fixture_plan.header->num_tensors;
    ptrs[out_idx] = storage;
    for (uint32_t i = 0; i < elements; i++)
        storage[i] = (int8_t)((int)i - 4);

    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem, out_idx, received,
                                      bytes),
                   TIGRIS_OK, "float output is produced");
    for (uint32_t i = 0; i < elements; i++) {
        float expect = ((float)storage[i] - (float)qp->zero_point) * qp->scale;
        TEST_ASSERT(received[i] == expect, "output dequantizes exactly");
    }

    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem, out_idx, received,
                                      bytes + 4u),
                   TIGRIS_ERR_BAD_SIZE, "a mis-sized buffer is refused");

    free(received);
    free(storage);
    free(ptrs);
}

static void test_non_boundary_tensors_are_refused(void)
{
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors,
                                  sizeof(void *));
    tigris_mem_t mem;
    uint16_t interior = 0xFFFFu;
    uint16_t i;

    printf("  test_non_boundary_tensors_are_refused...\n");
    if (!ptrs)
        return;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = fixture_plan.header->num_tensors;

    for (i = 0; i < fixture_plan.header->num_tensors; i++) {
        uint8_t flags = fixture_plan.tensors[i].flags;
        if ((flags & (TIGRIS_TENSOR_MODEL_INPUT |
                      TIGRIS_TENSOR_MODEL_OUTPUT)) == 0u) {
            interior = i;
            break;
        }
    }
    if (interior != 0xFFFFu) {
        float scratch = 0.0f;
        TEST_ASSERT_EQ(tigris_iface_bytes(&fixture_plan, interior), 0u,
                       "an interior tensor has no interface");
        TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, interior,
                                          &scratch, sizeof(scratch)),
                       TIGRIS_ERR_BAD_TENSOR,
                       "an interior tensor is not writable as an input");
    }
    TEST_ASSERT_EQ(tigris_iface_bytes(&fixture_plan, 0xFFFFu), 0u,
                   "an out-of-range index has no interface");
    TEST_ASSERT_EQ(tigris_iface_bytes(NULL, 0), 0u, "a null plan has none");
    free(ptrs);
}

static void test_plain_interface_copies(void)
{
    /* A plan whose boundary needs no conversion: schema-v2, all float32. */
    uint8_t *buf;
    uint32_t len = 0;
    char path[512];
    tigris_plan_t plan;
    int n = snprintf(path, sizeof(path), "%s/schema-v2-linear.tgrs",
                     TIGRIS_SCHEMA_COMPAT_DIR);

    printf("  test_plain_interface_copies...\n");
    if (n <= 0 || (size_t)n >= sizeof(path))
        return;
    buf = load_file(path, &len);
    TEST_ASSERT(buf != NULL, "float fixture is readable");
    if (!buf)
        return;
    if (tigris_plan_load(buf, len, &plan) == TIGRIS_OK) {
        uint16_t idx = plan.model_inputs[0];
        const tigris_tensor_t *t = &plan.tensors[idx];
        uint32_t bytes = tigris_iface_bytes(&plan, idx);
        void **ptrs = (void **)calloc(plan.header->num_tensors, sizeof(void *));
        uint8_t *storage = (uint8_t *)calloc(t->size_bytes, 1);
        uint8_t *sent = (uint8_t *)malloc(bytes);
        tigris_mem_t mem;
        TEST_ASSERT_EQ(t->iface_dtype, 0, "no declared interface is recorded");
        TEST_ASSERT_EQ(bytes, t->size_bytes, "the caller hands over what is stored");
        if (ptrs && storage && sent) {
            memset(&mem, 0, sizeof(mem));
            mem.tensor_ptrs = ptrs;
            mem.num_tensors = plan.header->num_tensors;
            ptrs[idx] = storage;
            memset(sent, 0x5A, bytes);
            TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, idx, sent, bytes),
                           TIGRIS_OK, "a matching interface is copied");
            TEST_ASSERT_EQ(memcmp(storage, sent, bytes), 0,
                           "the bytes arrive unchanged");
        }
        free(sent); free(storage); free(ptrs);
    }
    free(buf);
}

static void test_unconvertible_interfaces_are_refused(void)
{
    /* The plan validated on load, so an unsupported pair is produced here by
     * editing the loaded record, the same way the loader tests reach opcode
     * branches no emitted plan carries. */
    uint16_t in_idx = fixture_plan.model_inputs[0];
    tigris_tensor_t *t = (tigris_tensor_t *)&fixture_plan.tensors[in_idx];
    uint8_t saved_iface = t->iface_dtype;
    uint16_t saved_qp = t->quant_param_idx;
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors,
                                  sizeof(void *));
    int8_t *storage = (int8_t *)calloc(t->size_bytes, 1);
    float scratch[64];
    uint32_t bytes = tigris_iface_bytes(&fixture_plan, in_idx);
    tigris_mem_t mem;

    printf("  test_unconvertible_interfaces_are_refused...\n");
    if (!ptrs || !storage || bytes > sizeof(scratch)) {
        free(ptrs); free(storage);
        return;
    }
    memset(&mem, 0, sizeof(mem));
    memset(scratch, 0, sizeof(scratch));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = fixture_plan.header->num_tensors;
    ptrs[in_idx] = storage;

    /* A dtype this runtime has no conversion for. */
    t->iface_dtype = 11u; /* float64 */
    TEST_ASSERT_EQ(tigris_iface_bytes(&fixture_plan, in_idx), 0u,
                   "an unsupported declared dtype has no byte count");
    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, scratch,
                                      bytes),
                   TIGRIS_ERR_BAD_INTERFACE,
                   "an unsupported declared dtype is refused");

    /* A float interface over a tensor carrying no quantization. */
    t->iface_dtype = saved_iface;
    t->quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, scratch,
                                      bytes),
                   TIGRIS_ERR_BAD_INTERFACE,
                   "a float interface needs quantization to convert through");
    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem, in_idx, scratch,
                                      bytes),
                   TIGRIS_ERR_BAD_TENSOR,
                   "an input is not readable as an output");

    t->quant_param_idx = saved_qp;
    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, NULL,
                                      fixture_plan.model_outputs[0], scratch,
                                      bytes),
                   TIGRIS_ERR_NULL, "a null memory manager is refused");
    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem,
                                      fixture_plan.model_outputs[0], NULL,
                                      bytes),
                   TIGRIS_ERR_NULL, "a null destination is refused");
    TEST_ASSERT_EQ(tigris_input_write(NULL, &mem, in_idx, scratch, bytes),
                   TIGRIS_ERR_BAD_TENSOR, "a null plan is refused");

    /* The output tensor has no storage allocated in this fixture-only mem. */
    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem,
                                      fixture_plan.model_outputs[0], scratch,
                                      tigris_iface_bytes(
                                          &fixture_plan,
                                          fixture_plan.model_outputs[0])),
                   TIGRIS_ERR_NULL, "an unallocated tensor is refused");

    free(storage);
    free(ptrs);
}

int main(void)
{
    printf("TiGrIS Model Interface Tests\n\n");
    if (load_fixture() != 0) {
        fprintf(stderr, "Cannot load the schema-v6 fixture\n");
        return 1;
    }

    test_declared_interface_sizes_the_caller_buffer();
    test_float_roundtrips_through_the_quantized_boundary();
    test_output_is_read_back_in_the_declared_dtype();
    test_non_boundary_tensors_are_refused();
    test_plain_interface_copies();
    test_unconvertible_interfaces_are_refused();

    free(fixture_buf);
    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
