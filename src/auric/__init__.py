"""auric: tiled detection of truck classes in large overhead images.

Modules (importing the package does no work; each function takes its paths and parameters explicitly)
    config        YAML experiment/env configs (inheritance, overrides, path checks)
    structures    Annotation / Detections containers and the detections file format
    boxes         box geometry, size buckets, NMS
    data          raw dataset access, attribute tables, dev split, training tile cache
    model         YOLO training and full-image (tiled) inference
    evaluation    full-image detection metrics and error diagnostics
    runs          run directories: resolved config, stage record, artifact paths
    visualize     drawing boxes and crops (arrays only)
    configs/      experiment and environment YAML files, the dev split file
"""
__version__ = "0.2.0"
