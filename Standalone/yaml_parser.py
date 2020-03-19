import yaml

stream = open("layouts/config.yaml","r+")
data = yaml.load(stream)
print(data)