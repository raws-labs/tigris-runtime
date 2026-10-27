/**
 * @file test_accel_routing.c
 * @brief Host tests for ESP-NN/CMSIS-NN pre-routing without vendor libraries.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_accel_policy.h"
#include "tigris_kernels_s8.h"
#include "tigris_mem.h"
#include "test_harness.h"

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    int32_t shapes[8];
    uint16_t indices[2];
    tigris_weight_entry_t weight_entry;
    int8_t weights[3];
    tigris_quant_param_t quant_param;
    int32_t quant_data[2];
    tigris_plan_t plan;
} route_fixture_t;

static const int8_t test_input[5] = {1, 2, 3, 4, 5};
static const int8_t expected_dilated[5] = {11, 16, 22, 10, 13};

static void build_fixture(route_fixture_t *fx, tigris_op_type_t op_type,
                          int horizontal)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->indices[0] = 0;
    fx->indices[1] = 1;

    for (uint16_t i = 0; i < 2; i++) {
        fx->tensors[i].size_bytes = sizeof(test_input);
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].dtype = 3; /* ONNX INT8 */
        fx->tensors[i].quant_param_idx = i == 0
            ? TIGRIS_NO_QUANT_PARAM : 0;
        fx->shapes[i * 4] = 1;
        fx->shapes[i * 4 + 1] = horizontal ? 1 : 5;
        fx->shapes[i * 4 + 2] = horizontal ? 5 : 1;
        fx->shapes[i * 4 + 3] = 1;
    }

    fx->op.op_type = (uint8_t)op_type;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;
    fx->op.weight_idx = 0;
    fx->op.bias_idx = TIGRIS_NO_WEIGHT;
    fx->op.act_min = -128;
    fx->op.act_max = 127;
    fx->op.spatial.kernel_h = horizontal ? 1 : 3;
    fx->op.spatial.kernel_w = horizontal ? 3 : 1;
    fx->op.spatial.stride_h = 1;
    fx->op.spatial.stride_w = 1;
    fx->op.spatial.pad_top = horizontal ? 0 : 2;
    fx->op.spatial.pad_bottom = horizontal ? 0 : 2;
    fx->op.spatial.pad_left = horizontal ? 2 : 0;
    fx->op.spatial.pad_right = horizontal ? 2 : 0;
    fx->op.spatial.dilation_h = horizontal ? 1 : 2;
    fx->op.spatial.dilation_w = horizontal ? 2 : 1;
    fx->op.spatial.group = 1;

    fx->weight_entry.offset = 0;
    fx->weight_entry.size_bytes = sizeof(fx->weights);
    fx->weights[0] = 1;
    fx->weights[1] = 2;
    fx->weights[2] = 3;

    /* Q0.31 multiplier 0.5 with shift +1 is an identity requantization. */
    fx->quant_param.scale = 1.0f;
    fx->quant_param.num_channels = 1;
    fx->quant_param.multiplier_off = 0;
    fx->quant_param.shift_off = 1;
    fx->quant_data[0] = 1073741824;
    fx->quant_data[1] = 1;

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.weight_entries = &fx->weight_entry;
    fx->plan.weight_blob = (const uint8_t *)fx->weights;
    fx->plan.quant_params = &fx->quant_param;
    fx->plan.quant_data = fx->quant_data;
    fx->plan.num_quant_params = 1;
}

static void init_mem(tigris_mem_t *mem, void **ptrs,
                     int8_t *output, const route_fixture_t *fx,
                     int tile_active)
{
    memset(mem, 0, sizeof(*mem));
    ptrs[0] = (void *)test_input;
    ptrs[1] = output;
    mem->tensor_ptrs = ptrs;
    mem->num_tensors = 2;
    if (tile_active) {
        mem->tile.active = 1;
        mem->tile.in_h = fx->shapes[1];
        mem->tile.out_h = fx->shapes[5];
        mem->tile.in_w = fx->shapes[2];
        mem->tile.out_w = fx->shapes[6];
        mem->tile.pad_top = fx->op.spatial.pad_top;
        mem->tile.pad_bottom = fx->op.spatial.pad_bottom;
    }
}

static int run_reference(const route_fixture_t *fx, int8_t *output,
                         int tile_active)
{
    tigris_mem_t mem;
    void *ptrs[2];
    init_mem(&mem, ptrs, output, fx, tile_active);
    return tigris_dispatch_kernel_s8(
        &fx->plan, &fx->op, 0, &mem, NULL);
}

static int run_pre_route(const route_fixture_t *fx,
                         tigris_accel_backend_t backend,
                         int8_t *output, int tile_active, int *handled)
{
    tigris_mem_t mem;
    void *ptrs[2];
    init_mem(&mem, ptrs, output, fx, tile_active);
    return tigris_accel_try_s8_ref(
        backend, &fx->plan, &fx->op, 0, &mem, NULL, handled);
}

/* Route a tile carrying line-buffer roll offsets. out_h bounds the rows the
 * s8_ref kernel writes when the guard fires so execution stays in the caller's
 * buffer. */
static int run_pre_route_rolled(const route_fixture_t *fx,
                                tigris_accel_backend_t backend,
                                int8_t *output, int32_t out_row_start,
                                int32_t in_row_start, int32_t out_h,
                                int *handled)
{
    tigris_mem_t mem;
    void *ptrs[2];
    init_mem(&mem, ptrs, output, fx, 1);
    mem.tile.out_row_start = out_row_start;
    mem.tile.in_row_start = in_row_start;
    if (out_h > 0)
        mem.tile.out_h = out_h;
    return tigris_accel_try_s8_ref(
        backend, &fx->plan, &fx->op, 0, &mem, NULL, handled);
}

/* Route a row-banded op. Only row_tiled is set, so the test isolates that
 * branch of the guard. */
static int run_pre_route_row_tiled(const route_fixture_t *fx,
                                   tigris_accel_backend_t backend,
                                   int8_t *output, int *handled)
{
    tigris_mem_t mem;
    void *ptrs[2];
    init_mem(&mem, ptrs, output, fx, 1);
    mem.tile.row_tiled = 1;
    mem.tile.out_h = 1;
    mem.tile.in_h = 1;
    return tigris_accel_try_s8_ref(
        backend, &fx->plan, &fx->op, 0, &mem, NULL, handled);
}

/* Route a 2D (width) tiled op. Only width_tiled is set here; row offsets stay
 * zero so the test isolates the width_tiled branch of the guard. */
static int run_pre_route_width_tiled(const route_fixture_t *fx,
                                     tigris_accel_backend_t backend,
                                     int8_t *output, int width_tiled,
                                     int *handled)
{
    tigris_mem_t mem;
    void *ptrs[2];
    init_mem(&mem, ptrs, output, fx, 1);
    mem.tile.width_tiled = (uint8_t)width_tiled;
    return tigris_accel_try_s8_ref(
        backend, &fx->plan, &fx->op, 0, &mem, NULL, handled);
}

static void assert_array_eq(const int8_t *actual, const int8_t *expected,
                            const char *message)
{
    tests_run++;
    if (memcmp(actual, expected, sizeof(test_input)) != 0) {
        fprintf(stderr, "  FAIL: %s: %s (got [%d,%d,%d,%d,%d])\n",
                __func__, message,
                actual[0], actual[1], actual[2], actual[3], actual[4]);
        tests_failed++;
    } else {
        tests_passed++;
    }
}

static void test_esp_dilated_ops_route_reference(void)
{
    static const struct {
        const char *name;
        tigris_op_type_t op_type;
        int horizontal;
    } cases[] = {
        { "Conv", TIGRIS_OP_CONV, 0 },
        { "Depthwise", TIGRIS_OP_DEPTHWISE, 1 },
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        printf("  test_esp_dilated_ops_route_reference %s...\n", cases[c].name);
        route_fixture_t fx;
        int8_t direct[5] = {0};
        int8_t routed[5] = {0};
        int handled = 0;
        build_fixture(&fx, cases[c].op_type, cases[c].horizontal);

        TEST_ASSERT_EQ(tigris_accel_pre_route(
                           TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                       TIGRIS_ACCEL_ROUTE_S8_REF,
                       "ESP-NN dilated op pre-routes to s8_ref");
        TEST_ASSERT_EQ(run_reference(&fx, direct, 0), 0,
                       "direct reference succeeds");
        TEST_ASSERT_EQ(run_pre_route(&fx, TIGRIS_ACCEL_ESP_NN,
                                     routed, 0, &handled), 0,
                       "ESP-NN fallback succeeds");
        TEST_ASSERT_EQ(handled, 1, "ESP-NN adapter is bypassed");
        assert_array_eq(routed, direct, "ESP-NN fallback matches s8_ref");
        assert_array_eq(routed, expected_dilated,
                        "ESP-NN fallback matches dilated result");
    }
}

static void test_cmsis_dilation_and_tile_routes(void)
{
    printf("  test_cmsis_dilation_and_tile_routes...\n");
    route_fixture_t fx;
    int8_t untouched[5];
    int8_t direct[5] = {0};
    int8_t routed[5] = {0};
    int handled = 1;
    build_fixture(&fx, TIGRIS_OP_CONV, 0);

    memset(untouched, 0x5a, sizeof(untouched));
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "CMSIS-NN non-tiled dilation stays native");
    TEST_ASSERT_EQ(run_pre_route(&fx, TIGRIS_ACCEL_CMSIS_NN,
                                 untouched, 0, &handled), 0,
                   "CMSIS-NN native route returns to adapter");
    TEST_ASSERT_EQ(handled, 0, "CMSIS-NN non-tiled adapter is selected");
    {
        const int8_t sentinel[5] = {0x5a, 0x5a, 0x5a, 0x5a, 0x5a};
        assert_array_eq(untouched, sentinel,
                        "pre-route does not execute the native adapter");
    }

    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN tiled dilated Conv routes to s8_ref");
    TEST_ASSERT_EQ(run_reference(&fx, direct, 1), 0,
                   "tiled Conv reference succeeds");
    TEST_ASSERT_EQ(run_pre_route(&fx, TIGRIS_ACCEL_CMSIS_NN,
                                 routed, 1, &handled), 0,
                   "CMSIS-NN tiled fallback succeeds");
    TEST_ASSERT_EQ(handled, 1, "CMSIS-NN tiled adapter is bypassed");
    assert_array_eq(routed, direct,
                    "CMSIS-NN tiled fallback matches s8_ref");

    build_fixture(&fx, TIGRIS_OP_DEPTHWISE, 1);
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "CMSIS-NN non-tiled dilated Depthwise stays native");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN tiled Depthwise routes to s8_ref");
}

static void test_unit_dilation_keeps_esp_adapter(void)
{
    printf("  test_unit_dilation_keeps_esp_adapter...\n");
    route_fixture_t fx;
    build_fixture(&fx, TIGRIS_OP_CONV, 0);

    fx.op.spatial.dilation_h = 0;
    fx.op.spatial.dilation_w = 0;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "zero/default dilation is effective unit dilation");

    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "explicit unit dilation keeps ESP-NN adapter");
}

static void test_depthwise_multiplier_routes_adapter(void)
{
    printf("  test_depthwise_multiplier_routes_adapter...\n");
    route_fixture_t fx;
    build_fixture(&fx, TIGRIS_OP_DEPTHWISE, 1);
    fx.op.spatial.dilation_w = 1;
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER, "ESP-NN keeps a unit multiplier");
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER, "CMSIS-NN keeps a unit multiplier");
    fx.shapes[7] = 2;
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER, "ESP-NN handles a channel multiplier");
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER, "CMSIS-NN handles a channel multiplier");
    fx.shapes[0] = 2;
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF, "ESP-NN multiplier batch stays on reference");
}

static void test_esp_asymmetric_pad_workspace_policy(void)
{
    printf("  test_esp_asymmetric_pad_workspace_policy...\n");

    uint32_t required = 0;
    /* (H + top + bottom) * (W + left + right) * C
     * = (3 + 1 + 2) * (4 + 3 + 4) * 2 = 132 bytes. */
    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       3, 4, 2, 1, 2, 3, 4, 132, &required),
                   1, "exact asymmetric pad workspace fits");
    TEST_ASSERT_EQ(required, 132,
                   "asymmetric pad workspace reports exact byte count");

    required = 0;
    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       3, 4, 2, 1, 2, 3, 4, 131, &required),
                   0, "undersized asymmetric pad workspace falls back");
    TEST_ASSERT_EQ(required, 132,
                   "undersized workspace still reports required bytes");

    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       3, 4, 2, 1, 2, 3, 4, 132, NULL),
                   1, "required-byte output is optional");
    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       3, 4, 0, 1, 2, 3, 4, UINT32_MAX, &required),
                   0, "invalid channel count fails closed");
    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       INT32_MAX, 1, 1, 1, 0, 0, 0,
                       UINT32_MAX, &required),
                   0, "padded dimension overflow fails closed");
    TEST_ASSERT_EQ(tigris_accel_esp_pad_workspace_fits(
                       65536, 65536, 1, 0, 0, 0, 0,
                       UINT32_MAX, &required),
                   0, "padded byte-count overflow fails closed");
}

/* Column-reduced window op with the vendor padding convention: output row r
 * reads input rows r*stride - pad_top + k, k < kernel; rows outside [0, in_h)
 * are padding. Row sums are position-weighted so a shifted window cannot pass
 * by accident. */
static void window_rows(const int32_t *in, int32_t in_h, int32_t pad_top,
                        int32_t stride, int32_t kernel, int32_t out_h,
                        int32_t *out)
{
    for (int32_t r = 0; r < out_h; r++) {
        int32_t acc = 0;
        for (int32_t k = 0; k < kernel; k++) {
            int32_t row = r * stride - pad_top + k;
            if (row >= 0 && row < in_h)
                acc += in[row] * (k + 1);
        }
        out[r] = acc;
    }
}

static void test_row_band_matches_unsplit_op(void)
{
    printf("  test_row_band_matches_unsplit_op...\n");

    enum { MAX_IN = 23, MAX_OUT = 24 };
    int32_t in[MAX_IN];
    for (int32_t i = 0; i < MAX_IN; i++)
        in[i] = 3 * i + 1;

    int32_t whole[MAX_OUT];
    int32_t band_out[MAX_OUT];
    int mismatches = 0;
    int bands_run = 0;

    for (int32_t in_h = 1; in_h <= MAX_IN; in_h += 2) {
        for (uint16_t kernel = 1; kernel <= 5; kernel++) {
            for (uint16_t stride = 1; stride <= 3; stride++) {
                for (uint16_t pad_top = 0; pad_top < kernel; pad_top++) {
                    for (int32_t pad_bottom = 0; pad_bottom < kernel;
                         pad_bottom++) {
                        int32_t span = in_h + pad_top + pad_bottom - kernel;
                        if (span < 0)
                            continue;
                        int32_t out_h = span / stride + 1;
                        if (out_h > MAX_OUT)
                            continue;
                        window_rows(in, in_h, pad_top, stride, kernel,
                                    out_h, whole);

                        for (int32_t band = 1; band <= out_h; band++) {
                            for (int32_t o0 = 0; o0 < out_h; o0 += band) {
                                int32_t rows = out_h - o0 < band
                                             ? out_h - o0 : band;
                                int32_t start = -1, count = -1;
                                uint16_t bpad = 0xFFFFu;
                                if (!tigris_accel_row_band(
                                        in_h, pad_top, stride, kernel, o0,
                                        rows, &start, &count, &bpad)) {
                                    mismatches++;
                                    continue;
                                }
                                window_rows(in + start, count, bpad, stride,
                                            kernel, rows, band_out);
                                bands_run++;
                                for (int32_t r = 0; r < rows; r++) {
                                    if (band_out[r] != whole[o0 + r])
                                        mismatches++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    TEST_ASSERT(bands_run > 1000, "sweep ran a meaningful number of bands");
    TEST_ASSERT_EQ(mismatches, 0,
                   "every band reproduces its rows of the unsplit op");

    int32_t start = 0, count = 0;
    uint16_t bpad = 0;
    /* in_h 10, pad_top 2, stride 2, kernel 5: rows 3..4 read input 4..10. */
    TEST_ASSERT_EQ(tigris_accel_row_band(10, 2, 2, 5, 3, 2,
                                         &start, &count, &bpad), 1,
                   "interior band resolves");
    TEST_ASSERT_EQ(start, 4, "interior band input start");
    TEST_ASSERT_EQ(count, 6, "interior band input rows clip at in_h");
    TEST_ASSERT_EQ(bpad, 0, "interior band has no leading padding");

    TEST_ASSERT_EQ(tigris_accel_row_band(10, 2, 2, 5, 0, 1,
                                         &start, &count, &bpad), 1,
                   "first band resolves");
    TEST_ASSERT_EQ(bpad, 2, "first band keeps the leading padding");

    /* Windows entirely below the input: nothing to read. */
    TEST_ASSERT_EQ(tigris_accel_row_band(4, 0, 1, 3, 6, 1,
                                         &start, &count, &bpad), 0,
                   "band past the input is refused");
    TEST_ASSERT_EQ(tigris_accel_row_band(4, 0, 0, 3, 0, 1,
                                         &start, &count, &bpad), 0,
                   "zero stride is refused");
    TEST_ASSERT_EQ(tigris_accel_row_band(4, 0, 1, 3, 0, 0,
                                         &start, &count, &bpad), 0,
                   "empty band is refused");
    TEST_ASSERT_EQ(tigris_accel_row_band(4, 0, 1, 3, 0, 1,
                                         NULL, &count, &bpad), 0,
                   "missing output is refused");
}

static int view_is(const tigris_accel_tile_view_t *v, int32_t in_h, int32_t in_w,
                   int32_t out_h, int32_t out_w, uint16_t top, uint16_t bottom,
                   uint16_t left, uint16_t right, int32_t in_offset, int32_t out_offset)
{
    return v->in_h == in_h && v->in_w == in_w && v->out_h == out_h && v->out_w == out_w &&
           v->pad_top == top && v->pad_bottom == bottom && v->pad_left == left &&
           v->pad_right == right && v->in_offset == in_offset && v->out_offset == out_offset;
}

static void test_tile_view(void)
{
    printf("  test_tile_view...\n");
    static const int32_t x_shape[4] = {1, 9, 8, 3};
    static const int32_t y_shape[4] = {1, 5, 7, 4};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV;
    op.spatial.stride_h = 2;
    op.spatial.pad_top = 1;
    op.spatial.pad_bottom = 2;
    op.spatial.pad_left = 3;
    op.spatial.pad_right = 4;

    /* Tile fields that differ from the op in every value. */
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tile.in_h = 6;
    mem.tile.out_h = 3;
    mem.tile.in_w = 5;
    mem.tile.out_w = 2;
    mem.tile.pad_top = 5;
    mem.tile.pad_bottom = 6;
    mem.tile.pad_left = 7;
    mem.tile.pad_right = 8;
    mem.tile.width_tiled = 1;
    mem.tile.out_row_start = 2;
    mem.tile.in_row_start = 1;

    tigris_accel_tile_view_t v;
    tigris_accel_tile_view(&op, &mem, x_shape, y_shape, &v);
    TEST_ASSERT(view_is(&v, 9, 8, 5, 7, 1, 2, 3, 4, 0, 0),
                "inactive tile: whole tensors and the op's pads");

    mem.tile.active = 1;
    mem.tile.width_tiled = 0;
    mem.tile.out_row_start = 0;
    mem.tile.in_row_start = 0;
    tigris_accel_tile_view(&op, &mem, x_shape, y_shape, &v);
    TEST_ASSERT(view_is(&v, 6, 8, 3, 7, 5, 6, 3, 4, 0, 0),
                "height tile: tile rows and row pads, the op's widths and column pads");

    mem.tile.width_tiled = 1;
    tigris_accel_tile_view(&op, &mem, x_shape, y_shape, &v);
    TEST_ASSERT(view_is(&v, 6, 5, 3, 2, 5, 6, 7, 8, 0, 0),
                "2D tile: tile rows, columns and all four pads");

    /* Roll: output from row 2, input from row 2 * 2 - 1 = 3 of the buffer. */
    mem.tile.width_tiled = 0;
    mem.tile.out_row_start = 2;
    mem.tile.in_row_start = 1;
    tigris_accel_tile_view(&op, &mem, x_shape, y_shape, &v);
    TEST_ASSERT(view_is(&v, 3, 8, 3, 7, 5, 6, 3, 4, 3 * 8 * 3, 2 * 7 * 4),
                "rolled height tile: offsets in full rows, input rows reduced");

    mem.tile.width_tiled = 1;
    tigris_accel_tile_view(&op, &mem, x_shape, y_shape, &v);
    TEST_ASSERT(view_is(&v, 3, 5, 3, 2, 5, 6, 7, 8, 3 * 5 * 3, 2 * 2 * 4),
                "rolled 2D tile: offsets in tile rows");
}

/* Quant data spanning two pages, so a param on page 1 reads past the first. */
static int32_t channel_quant_data[TIGRIS_QUANT_PAGE_ELEMS + 8];

static void test_channel_quant(void)
{
    printf("  test_channel_quant...\n");
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.quant_data = channel_quant_data;
    int32_t *page1 = channel_quant_data + TIGRIS_QUANT_PAGE_ELEMS;
    /* Page 0 holds other values at the same offsets. */
    for (int i = 0; i < 8; i++) {
        channel_quant_data[i] = -100 - i;
        page1[i] = 100 + i;
    }
    tigris_quant_param_t per_channel = {0.5f, 0, 3, 0, 4, 1};
    tigris_quant_param_t per_tensor = {0.5f, 0, 1, 6, 7, 1};

    TEST_ASSERT_EQ(tigris_quant_page_start(&per_channel),
                   TIGRIS_QUANT_PAGE_ELEMS, "page 1 starts one page in");
    TEST_ASSERT_EQ(tigris_quant_multipliers(&plan, &per_channel), page1,
                   "multipliers on the param's page");
    TEST_ASSERT_EQ(tigris_quant_shifts(&plan, &per_channel), page1 + 4,
                   "shifts on the param's page");

    int32_t buffer[8];
    const int32_t *mult = NULL;
    const int32_t *shift = NULL;
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_channel, 3, NULL, 0,
                                              &mult, &shift), 1,
                   "per-channel param of the op's width");
    TEST_ASSERT_EQ(mult, page1, "per-channel multipliers used in place");
    TEST_ASSERT_EQ(shift, page1 + 4, "per-channel shifts used in place");

    for (int i = 0; i < 8; i++)
        buffer[i] = -1;
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_tensor, 3, buffer, 4,
                                              &mult, &shift), 1,
                   "per-tensor param broadcast");
    TEST_ASSERT_EQ(mult, buffer, "multipliers start the buffer");
    TEST_ASSERT_EQ(shift, buffer + 4, "shifts follow the buffer's capacity");
    for (int c = 0; c < 3; c++) {
        TEST_ASSERT_EQ(buffer[c], 106, "per-tensor multiplier per channel");
        TEST_ASSERT_EQ(buffer[4 + c], 107, "per-tensor shift per channel");
    }
    TEST_ASSERT_EQ(buffer[3], -1, "multiplier past the channels untouched");
    TEST_ASSERT_EQ(buffer[7], -1, "shift past the channels untouched");

    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, NULL, 2, buffer, 4,
                                              &mult, &shift), 1,
                   "no param broadcast");
    TEST_ASSERT_EQ(buffer[0], 1, "no param multiplies by 1");
    TEST_ASSERT_EQ(buffer[1], 1, "no param multiplies by 1 per channel");
    TEST_ASSERT_EQ(buffer[4], 0, "no param shifts by 0");
    TEST_ASSERT_EQ(buffer[5], 0, "no param shifts by 0 per channel");

    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_channel, 0, buffer, 4,
                                              &mult, &shift), 0,
                   "zero channels refused");
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_channel, 4, buffer, 4,
                                              &mult, &shift), 0,
                   "per-channel param of another width refused");
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_tensor, 3, NULL, 4,
                                              &mult, &shift), 0,
                   "broadcast without a buffer refused");
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_tensor, 5, buffer, 4,
                                              &mult, &shift), 0,
                   "broadcast past the capacity refused");
    TEST_ASSERT_EQ(tigris_accel_channel_quant(&plan, &per_channel, 3, NULL, 0,
                                              NULL, &shift), 0,
                   "missing output refused");
}

/* A line-buffered chain sets mem->tile.out_row_start / in_row_start on rolled
 * interior tiles. 1.5b: the ESP-NN/CMSIS-NN Conv/Depthwise adapters now honor
 * those offsets natively, so a rolled conv/depthwise stays on the vendor path;
 * every other op still falls back to s8_ref. pre_route cannot see the offsets
 * (it only takes tile_active), so the roll routing lives in
 * tigris_accel_try_s8_ref. */
static void test_rolled_tile_routing(void)
{
    printf("  test_rolled_tile_routing...\n");
    route_fixture_t fx;
    build_fixture(&fx, TIGRIS_OP_CONV, 0);

    /* Non-dilated 1x1 height-preserving conv. */
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    fx.op.spatial.kernel_h = 1;
    fx.op.spatial.kernel_w = 1;
    fx.op.spatial.pad_top = 0;
    fx.op.spatial.pad_bottom = 0;
    fx.op.spatial.pad_left = 0;
    fx.op.spatial.pad_right = 0;

    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "tiled non-dilated conv is an ESP adapter op by pre_route");

    int8_t out[8];
    int handled;

    /* Non-rolled tiled: guard must not over-fire; ESP keeps its adapter. */
    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_rolled(&fx, TIGRIS_ACCEL_ESP_NN, out, 0, 0, 3, &handled);
    TEST_ASSERT_EQ(handled, 0,
                   "non-rolled tiled conv stays on the ESP adapter");

    /* 1.5b: a rolled conv now runs on the vendor adapter (it folds the roll
     * offsets into the input/output pointers), on both backends. */
    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_rolled(&fx, TIGRIS_ACCEL_ESP_NN, out, 2, 0, 3, &handled);
    TEST_ASSERT_EQ(handled, 0,
                   "rolled conv (out_row_start) stays on the ESP adapter");

    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_rolled(&fx, TIGRIS_ACCEL_ESP_NN, out, 0, 2, 3, &handled);
    TEST_ASSERT_EQ(handled, 0,
                   "rolled conv (in_row_start) stays on the ESP adapter");

    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_rolled(&fx, TIGRIS_ACCEL_CMSIS_NN, out, 2, 0, 3, &handled);
    TEST_ASSERT_EQ(handled, 0,
                   "rolled conv stays on the CMSIS adapter");

    /* A rolled NON-conv/depthwise op still routes to s8_ref: only conv/depthwise
     * fold the roll offsets natively. Max-pool honors the roll in s8_ref, so the
     * fallback execution stays in bounds. */
    build_fixture(&fx, TIGRIS_OP_MAX_POOL, 0);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    fx.op.spatial.kernel_h = 1;
    fx.op.spatial.kernel_w = 1;
    fx.op.spatial.pad_top = 0;
    fx.op.spatial.pad_bottom = 0;
    fx.op.spatial.pad_left = 0;
    fx.op.spatial.pad_right = 0;
    memset(out, 0, sizeof(out));
    handled = 0;
    run_pre_route_rolled(&fx, TIGRIS_ACCEL_ESP_NN, out, 2, 0, 3, &handled);
    TEST_ASSERT_EQ(handled, 1,
                   "rolled max-pool still routes ESP to s8_ref");
}

/* A 2D-tiled op sets mem->tile.width_tiled on tiles that partition width as
 * well as height. 1.5c: the ESP-NN/CMSIS-NN Conv/Depthwise adapters now honor
 * the packed-width tile (in_w/out_w/pad_left, right pad implicit), so a
 * width-tiled conv/depthwise stays on the vendor path; every other op still
 * routes to s8_ref. */
static void test_width_tiled_routing(void)
{
    printf("  test_width_tiled_routing...\n");
    route_fixture_t fx;
    build_fixture(&fx, TIGRIS_OP_CONV, 0);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;

    int handled;
    int8_t out[8];

    /* width_tiled conv/depthwise now runs on the adapter on both backends. */
    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_width_tiled(&fx, TIGRIS_ACCEL_ESP_NN, out, 1, &handled);
    TEST_ASSERT_EQ(handled, 0, "2D-tiled conv stays on the ESP adapter");

    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_width_tiled(&fx, TIGRIS_ACCEL_CMSIS_NN, out, 1, &handled);
    TEST_ASSERT_EQ(handled, 0, "2D-tiled conv stays on the CMSIS adapter");

    build_fixture(&fx, TIGRIS_OP_DEPTHWISE, 1);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route_width_tiled(&fx, TIGRIS_ACCEL_CMSIS_NN, out, 1, &handled);
    TEST_ASSERT_EQ(handled, 0, "2D-tiled depthwise stays on the CMSIS adapter");

    /* A width-tiled NON-conv/depthwise op still routes to s8_ref. */
    build_fixture(&fx, TIGRIS_OP_MAX_POOL, 0);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    fx.op.spatial.kernel_h = 1;
    fx.op.spatial.kernel_w = 1;
    fx.op.spatial.pad_top = 0;
    fx.op.spatial.pad_bottom = 0;
    fx.op.spatial.pad_left = 0;
    fx.op.spatial.pad_right = 0;
    memset(out, 0, sizeof(out));
    handled = 0;
    run_pre_route_width_tiled(&fx, TIGRIS_ACCEL_ESP_NN, out, 1, &handled);
    TEST_ASSERT_EQ(handled, 1, "2D-tiled max-pool still routes ESP to s8_ref");
}

/* A row band says how many rows the buffer holds in mem->tile, not in the
 * tensor shape. The vendor FullyConnected kernels read the row count from the
 * tensor, so they would read and write the whole matrix through a band-sized
 * allocation. Every backend must route a row band to the reference kernel,
 * which honors it. */
static void test_row_banded_ops_route_to_reference(void)
{
    printf("  test_row_banded_ops_route_to_reference...\n");

    route_fixture_t fx;
    int8_t out[8];
    int handled;

    build_fixture(&fx, TIGRIS_OP_FULLY_CONN, 0);
    memset(out, 0, sizeof(out));
    handled = 0;
    run_pre_route_row_tiled(&fx, TIGRIS_ACCEL_CMSIS_NN, out, &handled);
    TEST_ASSERT_EQ(handled, 1,
                   "a row-banded fully-connected routes CMSIS to s8_ref");

    memset(out, 0, sizeof(out));
    handled = 0;
    run_pre_route_row_tiled(&fx, TIGRIS_ACCEL_ESP_NN, out, &handled);
    TEST_ASSERT_EQ(handled, 1,
                   "a row-banded fully-connected routes ESP to s8_ref");

    /* Conv is not a row-banding operator, but the guard is on the tile mode
     * rather than the operator, so it routes too rather than relying on the
     * compiler never to emit it. */
    build_fixture(&fx, TIGRIS_OP_CONV, 0);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    memset(out, 0, sizeof(out));
    handled = 0;
    run_pre_route_row_tiled(&fx, TIGRIS_ACCEL_CMSIS_NN, out, &handled);
    TEST_ASSERT_EQ(handled, 1, "a row band routes whatever the operator");
}

/* 1.5a: a plain-height tile (tile.active, no roll offset, not width_tiled) with
 * unit dilation now runs on the CMSIS-NN Conv/Depthwise adapter, matching what
 * the ESP-NN adapter already does. Dilated, rolled, and 2D-width tiles still
 * route to s8_ref. */
static void test_cmsis_plain_height_tile_routes_adapter(void)
{
    printf("  test_cmsis_plain_height_tile_routes_adapter...\n");
    route_fixture_t fx;
    int handled;
    int8_t out[8];

    /* unit-dilation plain-height Conv tile -> CMSIS adapter */
    build_fixture(&fx, TIGRIS_OP_CONV, 0);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "CMSIS-NN plain-height tiled Conv routes to the adapter");
    memset(out, 0, sizeof(out));
    handled = 1;
    run_pre_route(&fx, TIGRIS_ACCEL_CMSIS_NN, out, 1, &handled);
    TEST_ASSERT_EQ(handled, 0,
                   "plain-height tiled Conv selects the CMSIS adapter");

    /* unit-dilation plain-height Depthwise tile -> CMSIS adapter */
    build_fixture(&fx, TIGRIS_OP_DEPTHWISE, 1);
    fx.op.spatial.dilation_h = 1;
    fx.op.spatial.dilation_w = 1;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "CMSIS-NN plain-height tiled Depthwise routes to the adapter");

    /* a dilated plain-height tile still routes to reference: native
     * dilated-tile parity is out of scope for this step. */
    build_fixture(&fx, TIGRIS_OP_CONV, 0);  /* fixture dilation_h = 2 */
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 1),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN dilated tiled Conv still routes to s8_ref");
}

static void test_binary_routes(void)
{
    tigris_file_header_t header = { .num_tensors = 3, .num_ops = 1 };
    tigris_tensor_t tensors[3] = {0};
    tigris_quant_param_t quant[3] = {0};
    uint16_t indices[3] = {0, 1, 2};
    int32_t shapes[12] = {1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2};
    tigris_op_t op = { .op_type = TIGRIS_OP_MUL, .num_inputs = 2, .num_outputs = 1,
                       .outputs_off = 2, .weight_idx = TIGRIS_NO_WEIGHT,
                       .bias_idx = TIGRIS_NO_WEIGHT };
    tigris_plan_t plan = { .header = &header, .tensors = tensors, .ops = &op,
                          .index_pool = indices, .shape_pool = shapes,
                          .quant_params = quant, .num_quant_params = 3 };
    for (uint16_t i = 0; i < 3; i++) {
        tensors[i].dtype = 3; tensors[i].ndim = 4; tensors[i].shape_off = i * 4u;
        tensors[i].size_bytes = 8; tensors[i].quant_param_idx = i;
        quant[i].scale = 0.125f; quant[i].zero_point = -17; quant[i].num_channels = 1;
    }
    tigris_accel_binary_t params;
    TEST_ASSERT(tigris_accel_binary_params(&plan, &op, NULL, &params), "same-shape Mul is native");
    TEST_ASSERT_EQ(params.count, 8, "untiled count");
    tensors[1].flags = TIGRIS_TENSOR_CONSTANT;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "constant stays on reference");
    tensors[1].flags = 0;
    shapes[5] = 1; shapes[6] = 4;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "equal byte counts do not imply equal shapes");
    shapes[5] = 2; shapes[6] = 2;
    tensors[1].size_bytes = 2;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "channel broadcast stays on reference");
    tensors[1].size_bytes = 8;
    quant[1].num_channels = 2;
    TEST_ASSERT(tigris_accel_binary_params(&plan, &op, NULL, &params),
                "an input record carrying its producer's channels stays native");
    quant[1].num_channels = 1;
    quant[2].num_channels = 2;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "per-channel output stays on reference");
    quant[2].num_channels = 1;
    quant[2].scale = 0x1p-21f;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "unsafe left shift stays on reference");
    quant[2].scale = 0x1p27f;
    TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "out-of-range right shift stays on reference");
    quant[2].scale = 0.125f;
    const uint8_t kinds[] = {TIGRIS_OP_MUL, TIGRIS_OP_MAXIMUM, TIGRIS_OP_MINIMUM};
    for (unsigned k = 0; k < sizeof(kinds); k++) {
        op.op_type = kinds[k];
        TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, &plan, &op, 0, 1),
                       TIGRIS_ACCEL_ROUTE_ADAPTER, "CMSIS binary has an adapter");
        TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, &plan, &op, 0, 1),
                       k == 0 ? TIGRIS_ACCEL_ROUTE_ADAPTER : TIGRIS_ACCEL_ROUTE_S8_REF,
                       "ESP only has a Mul adapter");
        if (k != 0) {
            quant[1].scale = 0.25f;
            TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "comparison requires equal scales");
            quant[1].scale = 0.125f;
            quant[2].zero_point++;
            TEST_ASSERT(!tigris_accel_binary_params(&plan, &op, NULL, &params), "comparison requires equal zero points");
            quant[2].zero_point--;
        }
        int8_t a[8] = {-128, -3, 17, 127, 1, 3, -8, 31};
        int8_t b[8] = {127, 5, -128, 17, 3, 1, 31, -8};
        int8_t output[8] = {0};
        void *ptrs[3] = {a, b, output};
        tigris_mem_t mem = { .tensor_ptrs = ptrs, .num_tensors = 3 };
        mem.tile.active = 1; mem.tile.in_h = 1; mem.tile.out_h = 1;
        mem.tile.in_w = 2; mem.tile.out_w = 2;
        TEST_ASSERT(tigris_accel_binary_params(&plan, &op, &mem, &params), "plain height band is native");
        TEST_ASSERT_EQ(params.count, 4, "height band count");
        for (int mode = 0; mode < 3; mode++) {
            mem.tile.width_tiled = (uint8_t)(mode == 0);
            mem.tile.row_tiled = (uint8_t)(mode == 1);
            mem.tile.in_row_start = mode == 2 ? 1 : 0;
            mem.tile.out_row_start = mode == 2 ? 1 : 0;
            for (int backend = TIGRIS_ACCEL_ESP_NN; backend <= TIGRIS_ACCEL_CMSIS_NN; backend++) {
                int handled = 0;
                TEST_ASSERT_EQ(tigris_accel_try_s8_ref((tigris_accel_backend_t)backend,
                    &plan, &op, 0, &mem, NULL, &handled), 0, "binary fallback executes");
                TEST_ASSERT_EQ(handled, 1, "width, row and rolled binary tiles stay on reference");
            }
        }
    }
}

static void test_float_section_routes_reference(void)
{
    route_fixture_t fx;
    build_fixture(&fx, TIGRIS_OP_RELU, 1);
    fx.op.weight_idx = TIGRIS_NO_WEIGHT;
    for (uint8_t i = 0; i < 2; i++) {
        fx.tensors[i].dtype = 1;
        fx.tensors[i].size_bytes = 5u * sizeof(float);
        fx.tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }
    float input[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, output[5];
    const float expected[5] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    void *ptrs[2] = {input, output};
    tigris_mem_t mem = { .tensor_ptrs = ptrs, .num_tensors = 2 };
    for (int backend = TIGRIS_ACCEL_ESP_NN; backend <= TIGRIS_ACCEL_CMSIS_NN; backend++) {
        int handled = 0;
        TEST_ASSERT_EQ(tigris_accel_pre_route((tigris_accel_backend_t)backend,
                       &fx.plan, &fx.op, 0, 0), TIGRIS_ACCEL_ROUTE_S8_REF,
                       "float section avoids the int8 adapter");
        TEST_ASSERT_EQ(tigris_accel_try_s8_ref((tigris_accel_backend_t)backend,
                       &fx.plan, &fx.op, 0, &mem, NULL, &handled), 0, "float section executes");
        TEST_ASSERT_EQ(handled, 1, "float section handled by reference");
        TEST_ASSERT(memcmp(output, expected, sizeof(expected)) == 0, "float section result");
    }
}

static void test_cmsis_global_pool_rounding_routes(void)
{
    printf("  test_cmsis_global_pool_rounding_routes...\n");
    route_fixture_t fx;
    const uint8_t rounding = TIGRIS_POOL_ROUNDING_AVERAGE;
    tigris_op_attribute_t attr = { .op_index = 0, .type = TIGRIS_OP_ATTR_POOL_ROUNDING,
                                   .data_len = 1, .data_offset = 0 };
    build_fixture(&fx, TIGRIS_OP_GLOBAL_AVG, 0);
    memset(&fx.op.spatial, 0, sizeof(fx.op.spatial));
    fx.op.weight_idx = TIGRIS_NO_WEIGHT;
    fx.tensors[0].quant_param_idx = 0;

    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF, "CMSIS-NN global mean routes to s8_ref");
    fx.plan.op_attributes = &attr;
    fx.plan.op_attribute_data = &rounding;
    fx.plan.num_op_attributes = 1;
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER, "CMSIS-NN rounded global pool stays native");
    TEST_ASSERT_EQ(tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF, "ESP-NN global pool routes to s8_ref");
}

int main(void)
{
    printf("TiGrIS Accelerator Routing Tests\n\n");
    test_esp_dilated_ops_route_reference();
    test_cmsis_dilation_and_tile_routes();
    test_cmsis_plain_height_tile_routes_adapter();
    test_unit_dilation_keeps_esp_adapter();
    test_depthwise_multiplier_routes_adapter();
    test_float_section_routes_reference();
    test_cmsis_global_pool_rounding_routes();
    test_binary_routes();
    test_rolled_tile_routing();
    test_width_tiled_routing();
    test_row_banded_ops_route_to_reference();
    test_esp_asymmetric_pad_workspace_policy();
    test_row_band_matches_unsplit_op();
    test_tile_view();
    test_channel_quant();

    return test_report();
}
