from . import world_cityflow
try:
    from . import world_sumo
except Exception:
    # Allow CityFlow-only setups without SUMO_HOME or SUMO installed.
    pass
try:
    from . import world_openengine
except Exception:
    # OpenEngine is optional and may not be available in all environments.
    pass
