import yaml

class Config:
    def __init__(self, path_to_config):
        """
        Loads parameters from config.yaml into global object.

        Args:
          path_to_config (str): file path for config.yaml

        Attributes:
          dictionary (dict): a dictionary containing all parameter values from config.yaml
        """

        self.path_to_config = path_to_config
        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)
