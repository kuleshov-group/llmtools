from llmtools.engine.quant.config import QuantConfig

class QuantizationAlgorithm():
	"""Quantization algorthim abstract class"""
	def __init__(self, config: QuantConfig):
		self.config = config

	def quantize(self, model, dataloader):
		raise NotImplementedError