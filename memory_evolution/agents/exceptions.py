
##### Base Agents Exceptions #####


class EnvironmentNotSetError(Exception):
    """Raised when an environment is needed
    but an environment has not been set yet."""

    default_msg = (
        "An environment is needed but an environment"
        " has not been set yet."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if not self.args:
            self.args = (self.default_msg,)


##### Neat Agents Exceptions #####


class NotEvolvedError(Exception):
    """Raised when asking the genome of an agent which has never been
    evolved before to perform an action."""

    default_msg = (
        "Agent has never been evolved before, "
        "evolve the agent before asking for "
        "any property of it."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if not self.args:
            self.args = (self.default_msg,)

