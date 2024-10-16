class LocalConfigurationError(Exception):
    """
    General error thrown when a local configuration issue should prevent
    the validator from starting.
    """


class GenesisError(Exception):
    """
    General Error thrown when an error occurs as a result of an incomplete
    or erroneous genesis action.
    """


class InvalidGenesisStateError(GenesisError):
    """
    Error thrown when there is an invalid initial state during the genesis
    block generation process.
    """


class InvalidGenesisConsensusError(GenesisError):
    """
    Error thrown when the consensus algorithm refuses or fails to initialize
    or finalize the genesis block.
    """


class NotAvailableException(Exception):
    """
    Indicates a required service is not available and the action should be
    tried again later.
    """


class UnknownConsensusModuleError(Exception):
    """Error thrown when there is an invalid consensus module configuration.
    """


class PeeringException(Exception):
    """
    Indicates that a request to peer with this validator should not be allowed.
    """


class PossibleForkDetectedError(Exception):
    """Exception thrown when a possible fork has occurred while iterating
    through the block store.
    """


class NoProcessorVacancyError(Exception):
    """Error thrown when no processor has occupancy to handle a transaction
    """


class WaitCancelledException(Exception):
    """Exception thrown when a wait function has detected a cancellation event
    """
