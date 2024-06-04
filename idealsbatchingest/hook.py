# hook.py
import speedwagon
from typing import Dict, Type

from idealsbatchingest.workflows import BatchIngesterWorkflow


@speedwagon.hookimpl
def registered_workflows() -> Dict[str, Type[speedwagon.workflow]]:
    """Register workflows with the plugin.

    Returns:
        Returns a dictionary with the name of the workflow for the key and the
        class of the workflow for the value.
    """
    return {"IDEALS Batch Ingest Builder": BatchIngesterWorkflow}