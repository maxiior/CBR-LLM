
from data import DataCreator
from configs import Configs

experiments_configs = Configs.get_experiments_configs()

dc = DataCreator(experiment_configs=experiments_configs[0])
dataset = dc.create_structured_datasets()


dataset['train'].to_csv("tes.csv")