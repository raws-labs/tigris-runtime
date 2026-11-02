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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_iface.h"
#include "tigris_iface_internal.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "test_harness.h"

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
    TEST_ASSERT_EQ(t->dtype, TIGRIS_DTYPE_INT8, "plan stores int8");
    TEST_ASSERT_EQ(t->iface_dtype, TIGRIS_DTYPE_FLOAT32, "model declares float32");
    TEST_ASSERT_EQ(tigris_tensor_declared_dtype(t), TIGRIS_DTYPE_FLOAT32,
                   "the declared dtype is the interface dtype");
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

static void test_quantization_rounds_as_tflite_does(void)
{
    const tigris_quant_param_t unit = {1.0f, 0, 1u, 0u, 0u, 0u};
    const tigris_quant_param_t shifted = {0.5f, -3, 1u, 0u, 0u, 0u};
    printf("  test_quantization_rounds_as_tflite_does...\n");
    /* Just below one half rounds down; adding 0.5 first would give 1. */
    TEST_ASSERT_EQ(tigris_quantize_value(0.49999997f, &unit), 0, "below a half rounds down");
    TEST_ASSERT_EQ(tigris_quantize_value(-0.49999997f, &unit), 0, "above minus a half rounds up");
    TEST_ASSERT_EQ(tigris_quantize_value(0.5f, &unit), 1, "a half rounds away from zero");
    TEST_ASSERT_EQ(tigris_quantize_value(-2.5f, &unit), -3, "minus two and a half rounds away");
    TEST_ASSERT_EQ(tigris_quantize_value(1.25f, &shifted), 0, "the zero point follows rounding");
    TEST_ASSERT_EQ(tigris_quantize_value(3.0e9f, &unit), 127, "beyond int32 clamps high");
    TEST_ASSERT_EQ(tigris_quantize_value(-3.0e9f, &unit), -128, "beyond int32 clamps low");
}

/* Both rules on the same inputs: they differ only where value / scale is an
 * exact tie with an even integer below it in magnitude. */
static void test_quantization_rules_on_ties(void)
{
    static const struct {
        float value, scale;
        int32_t zero_point;
        int8_t away, even;
        const char *what;
    } cases[] = {
        {0.5f, 1.0f, 0, 1, 0, "one half"},
        {-0.5f, 1.0f, 0, -1, 0, "minus one half"},
        {1.5f, 1.0f, 0, 2, 2, "one and a half"},
        {-1.5f, 1.0f, 0, -2, -2, "minus one and a half"},
        {2.5f, 1.0f, 0, 3, 2, "two and a half"},
        {-2.5f, 1.0f, 0, -3, -2, "minus two and a half"},
        {0.49999997f, 1.0f, 0, 0, 0, "just below one half"},
        {0.50000006f, 1.0f, 0, 1, 1, "just above one half"},
        {-0.49999997f, 1.0f, 0, 0, 0, "just above minus one half"},
        {-0.50000006f, 1.0f, 0, -1, -1, "just below minus one half"},
        {2.4999998f, 1.0f, 0, 2, 2, "just below two and a half"},
        {2.5000002f, 1.0f, 0, 3, 3, "just above two and a half"},
        {-2.4999998f, 1.0f, 0, -2, -2, "just above minus two and a half"},
        {-2.5000002f, 1.0f, 0, -3, -3, "just below minus two and a half"},
        {0.15625f, 0.0625f, 0, 3, 2, "a tie at scale 2^-4"},
        {-0.15625f, 0.0625f, 0, -3, -2, "a negative tie at scale 2^-4"},
        {1.0f, 0.4f, 0, 3, 2, "1 / 0.4 is 2.5 in float32"},
        {1.25f, 0.5f, -3, 0, -1, "a tie offset by a negative zero point"},
        {-1.25f, 0.5f, 7, 4, 5, "a negative tie offset by a positive zero point"},
        {126.5f, 1.0f, 0, 127, 126, "a tie below the upper clamp"},
        {127.5f, 1.0f, 0, 127, 127, "a tie at the upper clamp"},
        {-127.5f, 1.0f, 0, -128, -128, "a tie at the lower clamp"},
        {-128.5f, 1.0f, 0, -128, -128, "a tie beyond the lower clamp"},
        {-126.5f, 1.0f, -1, -128, -127, "a tie the zero point moves to the clamp"},
        {300.0f, 1.0f, 0, 127, 127, "beyond the upper clamp"},
        {-300.0f, 1.0f, 0, -128, -128, "beyond the lower clamp"},
        {4194304.5f, 1.0f, -4194304, 1, 0, "a tie above 2^22"},
        {4194305.5f, 1.0f, -4194304, 2, 2, "an odd tie above 2^22"},
        {8388606.5f, 1.0f, -8388606, 1, 0, "the last ties below 2^23"},
        {16777216.0f, 1.0f, -16777216, 0, 0, "an integer at 2^24"},
        {3.0e9f, 1.0f, 0, 127, 127, "beyond int32"},
        {-3.0e9f, 1.0f, 0, -128, -128, "beyond minus int32"},
        {-0.0f, 1.0f, 3, 3, 3, "negative zero"},
    };
    printf("  test_quantization_rules_on_ties...\n");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const tigris_quant_param_t q = {cases[c].scale, cases[c].zero_point, 1u, 0u, 0u, 0u};
        TEST_ASSERT_EQ(tigris_quantize_value(cases[c].value, &q), cases[c].away, cases[c].what);
        TEST_ASSERT_EQ(tigris_quantize_value_even(cases[c].value, &q), cases[c].even, cases[c].what);
    }
    const tigris_quant_param_t unit = {1.0f, 0, 1u, 0u, 0u, 0u};
    TEST_ASSERT_EQ(tigris_quantize_value_even(INFINITY, &unit), 127, "infinity clamps high");
    TEST_ASSERT_EQ(tigris_quantize_value_even(-INFINITY, &unit), -128, "minus infinity clamps low");
    TEST_ASSERT_EQ(tigris_quantize_value_even(NAN, &unit), tigris_quantize_value(NAN, &unit),
                   "NaN takes the same level under both rules");
}

/* Every 4099th float32 pattern below 2^31 in magnitude, with the zero point
 * set to minus its floor so the level shows the rounding as 0 or 1 exactly. */
static void test_quantization_rules_match_libm(void)
{
    uint32_t mismatches_away = 0u, mismatches_even = 0u, checked = 0u;
    printf("  test_quantization_rules_match_libm...\n");
    for (uint64_t bits = 0u; bits <= UINT32_MAX; bits += 4099u) {
        const uint32_t pattern = (uint32_t)bits;
        float value;
        memcpy(&value, &pattern, sizeof(value));
        if (!(fabsf(value) < 2147483520.0f))
            continue;
        const double floor_value = floor((double)value);
        const tigris_quant_param_t q = {1.0f, (int32_t)-floor_value, 1u, 0u, 0u, 0u};
        const int away = (int)(round((double)value) - floor_value);
        const int even = (int)(nearbyint((double)value) - floor_value);
        if (tigris_quantize_value(value, &q) != away) mismatches_away++;
        if (tigris_quantize_value_even(value, &q) != even) mismatches_even++;
        checked++;
    }
    TEST_ASSERT(checked > 500000u, "the sweep covers the float32 range");
    TEST_ASSERT_EQ(mismatches_away, 0u, "half away from zero matches round()");
    TEST_ASSERT_EQ(mismatches_even, 0u, "half to even matches nearbyint() in the default mode");
}

/* The input boundary takes the plan's rule: the flag counts from schema 12. */
static void test_input_write_follows_the_plan_rule(void)
{
    static const struct {
        uint32_t version, flags;
        int even;
        const char *what;
    } plans[] = {
        {TIGRIS_SCHEMA_VERSION_V12, TIGRIS_FLAG_QUANTIZE_TIES_EVEN, 1, "schema 12 with the flag"},
        {TIGRIS_SCHEMA_VERSION_V12, TIGRIS_FLAG_QUANTIZE_TIES_EVEN | TIGRIS_FLAG_XIP, 1,
         "schema 12 with the flag and XIP"},
        {TIGRIS_SCHEMA_VERSION_V12, 0u, 0, "schema 12 without the flag"},
        {TIGRIS_SCHEMA_VERSION_V11, TIGRIS_FLAG_QUANTIZE_TIES_EVEN, 0, "schema 11 ignores the bit"},
    };
    uint16_t in_idx = fixture_plan.model_inputs[0];
    const tigris_tensor_t *t = &fixture_plan.tensors[in_idx];
    uint32_t elements = t->size_bytes;
    uint32_t bytes = tigris_iface_bytes(&fixture_plan, in_idx);
    const tigris_quant_param_t *qp = tigris_tensor_quant(&fixture_plan, t);
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors, sizeof(void *));
    int8_t *storage = (int8_t *)calloc(elements, 1);
    float *sent = (float *)malloc(bytes);
    tigris_mem_t mem;

    printf("  test_input_write_follows_the_plan_rule...\n");
    if (!ptrs || !storage || !sent) {
        free(ptrs); free(storage); free(sent);
        return;
    }
    /* Odd multiples of half a step, both signs, kept inside the clamp. */
    uint32_t inexact = 0u;
    for (uint32_t i = 0; i < elements; i++) {
        float half_step = (float)((int)(i % 16u) - 8) + 0.5f;
        sent[i] = half_step * qp->scale;
        if (sent[i] / qp->scale != half_step) inexact++;
    }
    TEST_ASSERT_EQ(inexact, 0u, "every input is an exact tie at the input scale");
    for (size_t p = 0; p < sizeof(plans) / sizeof(plans[0]); p++) {
        tigris_file_header_t header = *fixture_plan.header;
        tigris_plan_t plan = fixture_plan;
        header.version = plans[p].version;
        header.flags = plans[p].flags;
        plan.header = &header;
        TEST_ASSERT_EQ(tigris_plan_quantizes_ties_even(&plan), plans[p].even != 0, plans[p].what);
        memset(&mem, 0, sizeof(mem));
        mem.tensor_ptrs = ptrs;
        mem.num_tensors = fixture_plan.header->num_tensors;
        ptrs[in_idx] = storage;
        TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, in_idx, sent, bytes), TIGRIS_OK,
                       plans[p].what);
        uint32_t wrong = 0u;
        for (uint32_t i = 0; i < elements; i++) {
            /* k + 0.5 goes to the even one of k and k + 1, or away from zero. */
            int k = (int)(i % 16u) - 8;
            int level = plans[p].even ? (k % 2 == 0 ? k : k + 1) : (k >= 0 ? k + 1 : k);
            level += qp->zero_point;
            if (level < -128) level = -128;
            if (level > 127) level = 127;
            if (storage[i] != (int8_t)level) wrong++;
        }
        TEST_ASSERT_EQ(wrong, 0u, plans[p].what);
    }
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
    /* A plan whose boundary needs no conversion: schema-v4, all float32. */
    uint8_t *buf;
    uint32_t len = 0;
    char path[512];
    tigris_plan_t plan;
    int n = snprintf(path, sizeof(path), "%s/schema-v4-transpose.tgrs",
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

static void test_uint8_boundary_moves_by_128(void)
{
    /* A uint8 boundary: the compiler moved the zero point by 128 with the
     * values, so the conversion is exactly that shift, both ways. */
    uint16_t in_idx = fixture_plan.model_inputs[0];
    uint16_t out_idx = fixture_plan.model_outputs[0];
    tigris_tensor_t *ti = (tigris_tensor_t *)&fixture_plan.tensors[in_idx];
    tigris_tensor_t *to = (tigris_tensor_t *)&fixture_plan.tensors[out_idx];
    uint8_t saved_in = ti->iface_dtype;
    uint8_t saved_out = to->iface_dtype;
    void **ptrs = (void **)calloc(fixture_plan.header->num_tensors,
                                  sizeof(void *));
    int8_t *in_storage = (int8_t *)calloc(ti->size_bytes, 1);
    int8_t *out_storage = (int8_t *)calloc(to->size_bytes, 1);
    uint8_t *bytes_in = (uint8_t *)calloc(ti->size_bytes, 1);
    uint8_t *bytes_out = (uint8_t *)calloc(to->size_bytes, 1);
    tigris_mem_t mem;

    printf("  test_uint8_boundary_moves_by_128...\n");
    if (!ptrs || !in_storage || !out_storage || !bytes_in || !bytes_out)
        goto done;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = fixture_plan.header->num_tensors;
    ptrs[in_idx] = in_storage;
    ptrs[out_idx] = out_storage;

    ti->iface_dtype = 2u;
    to->iface_dtype = 2u;
    TEST_ASSERT_EQ(tigris_iface_bytes(&fixture_plan, in_idx), ti->size_bytes,
                   "a uint8 interface is one byte per element");

    for (uint32_t i = 0; i < ti->size_bytes; i++)
        bytes_in[i] = (uint8_t)(i * 37u);
    TEST_ASSERT_EQ(tigris_input_write(&fixture_plan, &mem, in_idx, bytes_in,
                                      ti->size_bytes),
                   TIGRIS_OK, "a uint8 input is written");
    int shifted = 1;
    for (uint32_t i = 0; i < ti->size_bytes; i++)
        shifted &= (int32_t)in_storage[i] == (int32_t)bytes_in[i] - 128;
    TEST_ASSERT(shifted, "every input byte lands 128 lower");

    for (uint32_t i = 0; i < to->size_bytes; i++)
        out_storage[i] = (int8_t)((int32_t)(i * 29u % 256u) - 128);
    TEST_ASSERT_EQ(tigris_output_read(&fixture_plan, &mem, out_idx, bytes_out,
                                      to->size_bytes),
                   TIGRIS_OK, "a uint8 output is read");
    shifted = 1;
    for (uint32_t i = 0; i < to->size_bytes; i++)
        shifted &= (int32_t)bytes_out[i] == (int32_t)out_storage[i] + 128;
    TEST_ASSERT(shifted, "every output byte reads 128 higher");

done:
    ti->iface_dtype = saved_in;
    to->iface_dtype = saved_out;
    free(bytes_out);
    free(bytes_in);
    free(out_storage);
    free(in_storage);
    free(ptrs);
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

static void test_bool_boundary(void)
{
    uint8_t storage[3] = {0};
    uint8_t values[3] = {0, 1, 1};
    uint8_t received[3] = {0};
    void *ptrs[] = {storage};
    tigris_file_header_t header = {0};
    tigris_tensor_t tensor = {0};
    tigris_plan_t plan = {0};
    tigris_mem_t mem = {0};
    header.num_tensors = 1;
    tensor.dtype = 9;
    tensor.flags = TIGRIS_TENSOR_MODEL_INPUT | TIGRIS_TENSOR_MODEL_OUTPUT;
    tensor.quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensor.size_bytes = sizeof(storage);
    plan.header = &header;
    plan.tensors = &tensor;
    mem.num_tensors = 1;
    mem.tensor_ptrs = ptrs;
    TEST_ASSERT_EQ(tigris_iface_bytes(&plan, 0), sizeof(storage), "bool element size is one byte");
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, values, sizeof(values)), TIGRIS_OK, "bool input copies");
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 0, received, sizeof(received)), TIGRIS_OK, "bool output copies");
    TEST_ASSERT(memcmp(values, received, sizeof(values)) == 0, "bool bytes unchanged");
    values[1] = 2;
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, values, sizeof(values)), TIGRIS_ERR_BAD_INTERFACE, "noncanonical bool refused");
}

static void test_index_output(void)
{
    int32_t indices[] = {0, 17, INT32_MAX};
    int64_t wide[3] = {0};
    int32_t narrow[3] = {0};
    void *ptrs[] = {indices};
    tigris_file_header_t header = {0};
    tigris_tensor_t tensor = {0};
    tigris_plan_t plan = {0};
    tigris_mem_t mem = {0};
    header.num_tensors = 1;
    tensor.dtype = 6;
    tensor.iface_dtype = 7;
    tensor.flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    tensor.quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensor.size_bytes = sizeof(indices);
    plan.header = &header;
    plan.tensors = &tensor;
    mem.num_tensors = 1;
    mem.tensor_ptrs = ptrs;
    TEST_ASSERT_EQ(tigris_iface_bytes(&plan, 0), sizeof(wide), "index interface doubles storage bytes");
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 0, wide, sizeof(wide)), TIGRIS_OK, "indices widen exactly");
    for (int i = 0; i < 3; i++) TEST_ASSERT_EQ(wide[i], indices[i], "widened index");
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 0, narrow, sizeof(narrow)), TIGRIS_ERR_BAD_SIZE, "stored size is not interface size");
    tensor.iface_dtype = 0;
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 0, narrow, sizeof(narrow)), TIGRIS_OK, "int32 interface copies");
    TEST_ASSERT(memcmp(indices, narrow, sizeof(indices)) == 0, "int32 bytes unchanged");
    tensor.iface_dtype = 7;
    tensor.flags |= TIGRIS_TENSOR_MODEL_INPUT;
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, wide, sizeof(wide)), TIGRIS_OK, "index input narrows exactly");
    wide[0] = INT32_MIN;
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, wide, sizeof(wide)), TIGRIS_OK, "int32 minimum is representable");
    TEST_ASSERT_EQ(indices[0], INT32_MIN, "negative starts stay signed");
    wide[0] = (int64_t)INT32_MAX + 1;
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, wide, sizeof(wide)), TIGRIS_ERR_BAD_INTERFACE, "positive narrowing overflow refused");
    wide[0] = (int64_t)INT32_MIN - 1;
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, wide, sizeof(wide)), TIGRIS_ERR_BAD_INTERFACE, "negative narrowing overflow refused");
    tensor.size_bytes = UINT32_MAX - 3u;
    TEST_ASSERT_EQ(tigris_iface_bytes(&plan, 0), 0, "widening overflow refused");
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 0, wide, 0), TIGRIS_ERR_BAD_SIZE, "overflow cannot wrap buffer length");
}

int main(void)
{
    test_index_output();
    test_bool_boundary();
    printf("TiGrIS Model Interface Tests\n\n");
    if (load_fixture() != 0) {
        fprintf(stderr, "Cannot load the schema-v6 fixture\n");
        return 1;
    }

    test_declared_interface_sizes_the_caller_buffer();
    test_float_roundtrips_through_the_quantized_boundary();
    test_quantization_rounds_as_tflite_does();
    test_quantization_rules_on_ties();
    test_quantization_rules_match_libm();
    test_input_write_follows_the_plan_rule();
    test_output_is_read_back_in_the_declared_dtype();
    test_non_boundary_tensors_are_refused();
    test_plain_interface_copies();
    test_uint8_boundary_moves_by_128();
    test_unconvertible_interfaces_are_refused();

    free(fixture_buf);
    return test_report();
}
