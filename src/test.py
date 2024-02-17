
from data import DataCreator
from configs import Configs

dc = DataCreator(Configs.get())

print(dc.create_structured_datasets().keys())