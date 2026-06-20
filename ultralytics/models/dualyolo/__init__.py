# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.dualyolo import detect, obb

from .model import DualYOLO

__all__ = "DualYOLO", "detect", "obb"
