from experiments.semantic import Semantic
from dataloader.Transformation import InstanceLabelTransformation

class Instance(Semantic):
  # Igual que Semantic, pero con una clase por diente (mapa de dataset.json)
  label_transform = InstanceLabelTransformation
