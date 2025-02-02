import json
import os

class LoadConfig:
    @staticmethod
    def load_config(config_file: str):
        """
        JSON 파일에서 설정을 로드.
        Args:
            config_file (str): 설정 파일 경로
        Returns:
            dict: 설정 값
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        with open(config_file, "r") as file:
            config = json.load(file)
            return config["embedding"] 
