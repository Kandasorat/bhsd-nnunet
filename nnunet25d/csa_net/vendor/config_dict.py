class ConfigDict(dict):
    """Small attribute-access dictionary used by the vendored CSA-Net config.

    The upstream project uses ``ml_collections.ConfigDict`` only for this
    behavior. Keeping the tiny compatible surface here avoids adding a new
    runtime dependency to the established Gadi environment.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value
