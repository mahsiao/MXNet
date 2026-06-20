# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import dualyolo
from ultralytics.nn.tasks import (
    DetectionModel,
    OBBModel,
)
from ultralytics.utils import ROOT, YAML

class DualYOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def names(self):
        return self._names

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "obb": {
                "model": OBBModel,
                "trainer": dualyolo.obb.OBBMultimodalTrainer,
                "validator": dualyolo.obb.OBBValidator,
                "predictor": dualyolo.obb.OBBPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": dualyolo.detect.MultimodalDetectionTrainer,
                "validator": dualyolo.detect.DetectionValidator,
                "predictor": dualyolo.detect.DetectionPredictor,
            },

        }

    @names.setter
    def names(self, value):
        self._names = value


