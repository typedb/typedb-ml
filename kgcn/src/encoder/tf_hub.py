import tensorflow_hub as hub


class TensorFlowHubEncoder:

    def __init__(self, module_url):
        self._embed = hub.Module(module_url)

    def __call__(self, features):
        return self._embed(features)
