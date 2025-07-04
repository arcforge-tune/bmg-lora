import os
import warnings
import logging

class WarningFilter:
    """Utility to suppress common warnings and noisy logs."""

    @staticmethod
    def suppress():
        # Suppress pkg_resources deprecation warning
        warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")
        # Suppress transformers pytree deprecation warning
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic", message="`torch.utils._pytree._register_pytree_node` is deprecated.*")
        # Suppress transformers deepspeed deprecation warning
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed", message="transformers.deepspeed module is deprecated.*")
        # Suppress bitsandbytes GPU support warning
        warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support.*")
        # General warning filter (last, to catch all others)
        warnings.filterwarnings("ignore")
        # Logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
        # Environment variables
        os.environ["PYTORCH_DISABLE_OPERATOR_OVERRIDE_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTHONWARNINGS"] = "ignore"