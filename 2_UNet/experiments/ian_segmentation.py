from experiments.pulp_segmentation import PulpSegmentation

class IANSegmentation(PulpSegmentation):
  # Igual que PulpSegmentation, pero con el nervio (IAN) como etiqueta
  label_file = 'gt_ian'
