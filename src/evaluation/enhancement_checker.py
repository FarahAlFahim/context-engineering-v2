"""Check which instances have been further enhanced."""

import json
import logging
from typing import List

logger = logging.getLogger("context_engineering.evaluation.enhancement_checker")


def check_further_enhancement(input_file: str) -> List[str]:
    """Return instance_ids where further_enhanced is True."""
    with open(input_file, 'r') as f:
        data = json.load(f)

    result = []
    if isinstance(data, list):
        for instance in data:
            if instance.get('further_enhanced') is True:
                result.append(instance.get('instance_id', 'unknown'))
    elif isinstance(data, dict):
        for instance_id, instance_data in data.items():
            if instance_data.get('further_enhanced') is True:
                result.append(instance_id)

    logger.info(f"Instances with further_enhanced=true: {len(result)}")
    return result
