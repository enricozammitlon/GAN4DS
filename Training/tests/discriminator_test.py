from pykwalify.core import Core
import os
subdirs=[x[0] for x in os.walk('../layouts/') if 'discriminator_layout.yaml' in x[2]]
for i in subdirs:
  c = Core(source_file=i+"/discriminator_layout.yaml", schema_files=["layout_schema.yaml"])
  c.validate(raise_exception=True)