import json
import os

# Define the repositories list
bug_report_folder = 'agent_outputs'
repositories = [
    {"bug_reports": f"swe_results/{bug_report_folder}/astropy__astropy.json", "ground_truth": "data/ground_truth/astropy__astropy.json"},
    {"bug_reports": f"swe_results/{bug_report_folder}/django__django.json", "ground_truth": "data/ground_truth/django__django.json"},
    # {"bug_reports": f"{bug_report_folder}/ActiveMQ.json", "ground_truth": "ground_truth/methods/ActiveMQ.json"},
    # {"bug_reports": f"{bug_report_folder}/Hadoop.json", "ground_truth": "ground_truth/methods/Hadoop.json"},
    # {"bug_reports": f"{bug_report_folder}/HDFS.json", "ground_truth": "ground_truth/methods/HDFS.json"},
    # {"bug_reports": f"{bug_report_folder}/Hive.json", "ground_truth": "ground_truth/methods/Hive.json"},
    # {"bug_reports": f"{bug_report_folder}/MAPREDUCE.json", "ground_truth": "ground_truth/methods/MAPREDUCE.json"},
    # {"bug_reports": f"{bug_report_folder}/Storm.json", "ground_truth": "ground_truth/methods/Storm.json"},
    # {"bug_reports": f"{bug_report_folder}/YARN.json", "ground_truth": "ground_truth/methods/YARN.json"}
]
# Output results
output_file = f"swe_results/direct_method_match/{bug_report_folder}.json"


# List of filenames to exclude (no ground truth available)
excluded_filenames = set([])


# Function to check if a predicted method is in the ground truth
def is_method_in_ground_truth(predicted_method, ground_truth_methods):
    for gt_method in ground_truth_methods:
        # -------- check method only [last part only] --------
        if '.' in predicted_method:
            predicted_method = predicted_method.split(".")[-1]
        # -------- check method only [last part only] --------
        if gt_method.endswith(predicted_method):  # Matching based on suffix
            return True
    return False

# Function to process each repository
def process_repository(bug_report_path, ground_truth_path):
    if not os.path.exists(bug_report_path) or not os.path.exists(ground_truth_path):
        print(f"Skipping {bug_report_path} or {ground_truth_path} - File not found.")
        return None, 0, 0, []

    # Load bug reports
    with open(bug_report_path, 'r') as f:
        bug_reports = json.load(f)

    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    results = []
    total_bug_reports = 0
    total_matched = 0
    problem_location_missing_list = []  # Track missing 'problem_location' field

    # Process each bug report
    for report in bug_reports:
        instance_id = report["instance_id"]
        if instance_id in excluded_filenames:
            continue  # Skip files without ground truth

        # Check if 'problem_location' field exists
        if "bug_report" not in report or "problem_location" not in report["bug_report"]:
            problem_location_missing_list.append(instance_id)
            continue  # Skip counting this report

        predicted_methods = report["bug_report"]["problem_location"].get("methods", [])
        ground_truth_methods = ground_truth.get(instance_id, [])

        total_bug_reports += 1  # Count processed reports

        # Find matches
        matched_methods = [method for method in predicted_methods if is_method_in_ground_truth(method, ground_truth_methods)]

        if matched_methods:
            total_matched += 1  # Count reports where at least one method matched

        # Compute match percentage
        match_percentage = len(matched_methods) / len(predicted_methods) if predicted_methods else 0

        results.append({
            "instance_id": instance_id,
            "total_predicted": len(predicted_methods),
            "total_ground_truth": len(ground_truth_methods),
            "matched_methods": matched_methods,
            "match_percentage": match_percentage
        })

    return results, total_bug_reports, total_matched, problem_location_missing_list

# Loop through repositories and process each
all_results = {}
overall_bug_reports = 0
overall_matched = 0
overall_problem_location_missing = []

for repo in repositories:
    project_name = os.path.basename(repo["bug_reports"]).replace(".json", "")  # Extract project name
    print(f"Processing {project_name}...")

    results, total_bug_reports, total_matched, problem_location_missing_list = process_repository(repo["bug_reports"], repo["ground_truth"])

    if results:
        all_results[project_name] = results
        overall_bug_reports += total_bug_reports
        overall_matched += total_matched
        overall_problem_location_missing.extend(problem_location_missing_list)

# Save results
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=4)

# Print summary
print(f"\nResults saved to {output_file}")
print(f"Total Bug Reports Processed: {overall_bug_reports}")
print(f"Total Reports with Matched Methods: {overall_matched}")
print(f"Total Reports Missing 'problem_location': {len(overall_problem_location_missing)}")
if overall_problem_location_missing:
    # print("Filenames Missing 'problem_location':", overall_problem_location_missing[:10], "...")  # Show first 10 instance_ids
    print("Filenames Missing 'problem_location':", overall_problem_location_missing)  # Show first 10 instance_ids
