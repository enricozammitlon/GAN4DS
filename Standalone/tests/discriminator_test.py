from pykwalify.core import Core
c = Core(source_file="../layouts/discriminator_layout.yaml", schema_files=["discriminator_schema.yaml"])
c.validate(raise_exception=True)