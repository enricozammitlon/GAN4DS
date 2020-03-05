import yaml

stream = open("layouts/discriminator_layout.yaml","r+")
data = yaml.load(stream)
print(data)