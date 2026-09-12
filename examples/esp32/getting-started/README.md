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
Stages tiled:     7 of 17   (total tiles 22, chains ...)
Output vs golden: 524288 int8 values, mean|diff|=0.52, max|diff|=3, 0.045% over 2 LSB

SELF_CHECK: PASS
TIGRIS_DONE
```

`SELF_CHECK` compares the device output against the embedded ORT float oracle
(`unet_ref.bin`). Int8 kernels track the oracle to within a few LSB from requant
rounding, so the check is statistical. PASS means non-degenerate output whose
**mean** absolute difference stays well under 1 LSB with no far-drifting element.
For the rigorous host-side parity check, see the benchmark suite.
