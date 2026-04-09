# TiGrIS ESP32 Example

ESP-IDF example app for running [TiGrIS](https://github.com/raws-labs/tigris) inference plans on ESP32 hardware.

**[Documentation](https://tigris-ml.dev/docs)**

## Prerequisites

- ESP-IDF v5.x

## Build

```
cd examples/esp32
idf.py set-target esp32s3
idf.py build
idf.py flash
```

## Flash a Plan

```
scripts/flash_plan.sh model.tgrs
```

## License

Apache 2.0
