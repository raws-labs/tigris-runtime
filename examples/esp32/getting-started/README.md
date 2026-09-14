# TiGrIS getting started: a U-Net that does not fit

This example runs a real **256x256 int8 U-Net** (encoder-decoder segmentation,
17 ops) on an ESP32-S3. The point is what it demonstrates:

- The network's **largest single activation is ~1.19 MiB** and its **naive peak
  is ~2.38 MiB**, so an arena-based runtime such as TFLite Micro cannot allocate
  it against internal SRAM (it OOMs at `AllocateTensors`).
- TiGrIS **2D-tiles** the stages and **spills the long-lived skip tensors to
  PSRAM**, so the working set fits a **232 KiB** fast arena (about 10% of the naive
  peak). The encoder convolutions run on the **ESP-NN accelerated** kernels.

The compiled plan (`unet.tgrs`) and a golden output (`unet_ref.bin`) are embedded
in the app, so there is nothing to flash separately.

## Requirements

- An **ESP32-S3 with PSRAM** (developed on an ESP32-S3-DevKitC-1 N16R8: 16 MB
  flash, 8 MB octal PSRAM). See `sdkconfig.defaults.esp32s3` / `partitions.csv`
  to adapt to a QUAD-PSRAM or 8 MB-flash board.
- ESP-IDF **5.0 or newer**.

## Create and run

```bash
idf.py create-project-from-example "raws-labs/tigris-runtime:getting-started"
cd getting-started
idf.py set-target esp32s3
idf.py flash monitor
```

## Expected output

```
=== TiGrIS getting-started: U-Net that does not fit ===

Model:            unet_matched
Ops / stages:     17 / 17
Plan (flash):     146.4 KiB
Fast arena:       232 KiB
Largest tensor:   1.19 MiB  <- an arena runtime must hold this whole
...
Fast (SRAM): 232 KiB   Slow (PSRAM): 5248 KiB   Workspace: 467 B
...
Stages tiled:       4 of 17   (total tiles 0, chains 10)
Peak fast arena:    232 KiB of a 2.38 MiB naive peak
Output vs golden:   524288 int8 values, mean|diff|=0.517, max|diff|=3, 0.045% over 2 LSB

SELF_CHECK: PASS
TIGRIS_DONE
```

`SELF_CHECK` compares the device output against the embedded ORT float oracle
(`unet_ref.bin`). Int8 kernels track the oracle to within a few LSB from requant
rounding, so the check is statistical. PASS means non-degenerate output whose
**mean** absolute difference stays well under 1 LSB with no far-drifting element.
For the rigorous host-side parity check, see the benchmark suite.

## The model interface

The U-Net declares **float32** inputs and outputs and quantizes inside itself,
so the plan stores int8 while the model's interface is float. The example hands
its input over in the declared dtype and lets the runtime apply the scale and
zero point:

```c
uint32_t want = tigris_iface_bytes(&plan, tensor_idx);
float *feed = heap_caps_malloc(want, MALLOC_CAP_SPIRAM);
/* fill feed with float32 samples */
tigris_input_write(&plan, &mem, tensor_idx, feed, want);
```

`tigris_output_read()` is the matching call for results. This example reads the
output tensor raw instead, because the self-check compares int8 against an int8
golden reference; an application wanting float32 results calls
`tigris_output_read()`.

That conversion buffer is four times the stored int8 tensor (768 KiB here) and
comes from the same PSRAM as the slow arena, so the example reserves it before
sizing the arena.
