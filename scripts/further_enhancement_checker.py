import json

# Input file path
INPUT_FILE = "swe_results/our_plus_mini_sweagent_enhanced/scikit-learn__scikit-learn.json"

def check_further_enhancement(input_file):
    """
    Check for instances with further_enhanced set to true
    
    Args:
        input_file: Path to the JSON file to analyze
    
    Returns:
        List of instance_ids where further_enhanced is true
    """
    further_enhancement_list = []
    
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Check if data is a list or dictionary
    if isinstance(data, list):
        # Traverse all instances in the list
        for instance in data:
            # Check if further_enhanced is true
            if instance.get('further_enhanced') == True:
                # Get the instance_id (adjust key name if different)
                instance_id = instance.get('instance_id', 'unknown')
                further_enhancement_list.append(instance_id)
    else:
        # Traverse all instance_ids in dictionary
        for instance_id, instance_data in data.items():
            # Check if further_enhanced is true
            if instance_data.get('further_enhanced') == True:
                further_enhancement_list.append(instance_id)
    
    return further_enhancement_list

if __name__ == "__main__":
    result_list = check_further_enhancement(INPUT_FILE)
    print(f"Instance IDs with further_enhanced=true: {len(result_list)}")
    print(result_list)
