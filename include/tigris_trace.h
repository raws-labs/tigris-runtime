#ifndef TIGRIS_TRACE_H
#define TIGRIS_TRACE_H

#include <stdint.h>

typedef enum {
    TIGRIS_TRACE_STAGE_BEGIN = 1, /* stage, path */
    TIGRIS_TRACE_STAGE_END   = 2, /* stage, fast_used/slow_used after cleanup */
    TIGRIS_TRACE_TILE_BEGIN  = 3, /* stage, tile index in `op`, input rows/cols incl. halo */
    TIGRIS_TRACE_LOAD        = 4, /* slow -> fast: tensor, bytes, src_offset, offset */
    TIGRIS_TRACE_SPILL       = 5, /* fast -> slow: tensor, bytes, src_offset, offset, rows/cols */
    TIGRIS_TRACE_ALLOC       = 6, /* tensor, pool, offset, bytes */
    TIGRIS_TRACE_RESET       = 7, /* pool reset to fast_reserved */
    TIGRIS_TRACE_MOVE        = 8, /* compaction or roll: tensor, pool, src_offset, offset, bytes */
    TIGRIS_TRACE_WEIGHTS     = 9, /* weight block into fast: bytes, offset, src bytes in src_offset */
    TIGRIS_TRACE_OP          = 10,/* kernel call: op index, stage */
    TIGRIS_TRACE_COPY        = 11 /* state or control-flow copy: tensor, pool, bytes */
} tigris_trace_kind_t;

typedef enum {
    TIGRIS_TRACE_PATH_NORMAL = 0, TIGRIS_TRACE_PATH_TILED = 1, TIGRIS_TRACE_PATH_TILED_2D = 2,
    TIGRIS_TRACE_PATH_BY_INPUT = 3, TIGRIS_TRACE_PATH_TRANSPOSE = 4, TIGRIS_TRACE_PATH_ROWS = 5,
    TIGRIS_TRACE_PATH_RESHAPE = 6, TIGRIS_TRACE_PATH_CHAIN = 7, TIGRIS_TRACE_PATH_CONTROL = 8
} tigris_trace_path_t;

#define TIGRIS_TRACE_NONE_U16 0xFFFFu
#define TIGRIS_TRACE_POOL_FAST 0u
#define TIGRIS_TRACE_POOL_SLOW 1u
#define TIGRIS_TRACE_POOL_NONE 0xFFu

typedef struct {
    uint8_t  kind;        /* tigris_trace_kind_t */
    uint8_t  pool;        /* destination pool, or TIGRIS_TRACE_POOL_NONE */
    uint8_t  path;        /* tigris_trace_path_t, STAGE_BEGIN only */
    uint8_t  _pad;
    uint16_t stage;       /* stage index, or TIGRIS_TRACE_NONE_U16 */
    uint16_t tensor;      /* tensor index, or TIGRIS_TRACE_NONE_U16 */
    uint16_t op;          /* op index (OP), tile index (TILE_BEGIN), else NONE */
    uint16_t _pad2;
    uint32_t offset;      /* destination byte offset within its pool */
    uint32_t src_offset;  /* source byte offset within the other pool */
    uint32_t bytes;       /* bytes moved or allocated */
    int32_t  row0;
    int32_t  row1;  /* row range [row0, row1) for tile events, -1 when whole */
    int32_t  col0;
    int32_t  col1;  /* column range for 2D tiles, -1 when not 2D */
    uint32_t fast_used;   /* after the event */
    uint32_t slow_used;   /* after the event */
} tigris_trace_event_t;

typedef void (*tigris_trace_fn)(const tigris_trace_event_t *event, void *ctx);

#endif
