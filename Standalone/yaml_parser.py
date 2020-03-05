import yaml

stream = open("discriminator_layout.yaml","r+")
data = yaml.load(stream)
print(data)