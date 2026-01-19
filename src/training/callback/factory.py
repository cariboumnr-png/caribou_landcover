'''Compose a list of callback classes.'''

# standard imports
import dataclasses
# local imports
import training.callback
import training.common

@dataclasses.dataclass
class CallbackSet:
    '''Wrapper for concrete callback classes.'''
    progress: training.common.ProgressCallbackLike
    train: training.common.TrainCallbackLike
    validate: training.common.ValCallbackLike
    logging: training.common.LoggingCallbackLike

    def __iter__(self):
        return iter((getattr(self, f.name) for f in dataclasses.fields(self)))

def build_callbacks() -> CallbackSet:
    '''Public API'''

    return CallbackSet(
        progress=training.callback.ProgressCallback(),
        train=training.callback.TrainCallback(),
        validate=training.callback.ValCallback(),
        logging=training.callback.LoggingCallback()
    )
