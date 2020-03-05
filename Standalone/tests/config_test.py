from pykwalify.core import Core
c = Core(source_file="../layouts/config.yaml", schema_files=["config_schema.yaml"])
c.validate(raise_exception=True)