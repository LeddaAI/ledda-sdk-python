"""Exceptions raised by the Ledda SDK in strict mode."""


class LeddaError(Exception):
    """Base exception for all Ledda SDK errors."""


class PromptNotFoundError(LeddaError):
    """Prompt name does not exist."""

    def __init__(self, name: str, message: str = ""):
        self.prompt_name = name
        super().__init__(message or f"Prompt '{name}' not found")


class PromptNotDeployedError(LeddaError):
    """Prompt exists but is not deployed to the requested environment."""

    def __init__(self, name: str, label: str, message: str = ""):
        self.prompt_name = name
        self.label = label
        super().__init__(
            message or f"Prompt '{name}' exists but has no '{label}' environment. Deploy it first."
        )


class LeddaConnectionError(LeddaError):
    """Could not reach the Ledda API."""

    def __init__(self, message: str = "Could not connect to Ledda API"):
        super().__init__(message)
