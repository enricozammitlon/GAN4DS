from pykwalify.core import Core
import os

subdirs=[x[0] for x in os.walk('../layouts/') if 'subconfig.yaml' in x[2]]

c = Core(source_file="../layouts/config.yaml", schema_files=["config_schema.yaml"])
c.validate(raise_exception=True)
for i in subdirs:
  c = Core(source_file=i+"/subconfig.yaml", schema_files=["config_schema.yaml"])
  c.validate(raise_exception=True)