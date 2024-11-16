import json
from typing import List
from gwpy.segments import Segment, SegmentList

def load_segments_from_json(file_path: str) -> SegmentList:
    """
    Load segments from a JSON file and return a SegmentList.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        SegmentList: A SegmentList containing the segments from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    segments = data.get("segments", [])
    # Convert to SegmentList
    return SegmentList([Segment(start, end) for start, end in segments])

def check_detection_in_segments(detection: float, segment_list: SegmentList) -> bool:
    """
    Check if a detection is within any of the segments in a SegmentList.

    Parameters:
        detection (float): The detection time as a float.
        segment_list (SegmentList): A SegmentList object.

    Returns:
        bool: True if the detection is within any segment, False otherwise.
    """
    return detection in segment_list

def find_missing_detection(detection: float, json_files: List[str]) -> List[str]:
    """
    Check if a single detection is missing in any of the JSON files.

    Parameters:
        detection (float): The detection time as a float.
        json_files (List[str]): A list of JSON file paths.

    Returns:
        List[str]: A list of JSON files where the detection is missing.
    """
    missing_files = []

    for json_file in json_files:
        segment_list = load_segments_from_json(json_file)
        if not check_detection_in_segments(detection, segment_list):
            missing_files.append(json_file)

    return missing_files

def main(detection: float, json_files: List[str]):
    """
    Main function to check if a single detection is missing in any of the JSON files.

    Parameters:
        detection (float): The detection time as a float.
        json_files (List[str]): List of JSON file paths containing segments.

    Returns:
        List[str]: List of JSON files where the detection is missing.
    """
    # Find missing JSON files for the given detection
    missing_files = find_missing_detection(detection, json_files)

    if missing_files:
        print(f"Detection {detection} is missing in the following JSON files:")
        for json_file in missing_files:
            print(f"- {json_file}")
    else:
        print(f"Detection {detection} is present in all JSON files.")

    return missing_files


if __name__ == "__main__":
    # Example usage
    detections = ['1243305662.9310', '1241104246.7490', '1249635282.3590',
                  '1242442957.4230', '1241624696.5500', '1251009253.7240',
                  '1240050919.5040', '1250981809.4370', '1238351461.2030',
                  '1246417246.8230', '1246487209.3080', '1249035984.2120',
                  '1242827473.3700', '1248280604.5540', '1240878400.3070',
                  '1251709954.8220', '1249529264.6980', '1239155734.1820',
                  '1253638396.3240', '1251452408.2800', '1260825547.0250',
                  '1265853772.5540', '1260164276.4340', '1262676274.8130',
                  '1260258000.1600', '1259852757.6790', '1267617687.9370',
                  '1261041997.0680', '1267610457.9930', '1264316116.3950',
                  '1260288309.7230', '1263906435.1910', '1263013367.0550',
                  '1267610487.9370', '1262203619.4010', '1262002542.6820']
    detections = [float(d) for d in detections]

    injections = [
        'NO_BURST_HW_INJ',
        'NO_CBC_HW_INJ',
        'NO_DETCHAR_HW_INJ',
        'NO_STOCH_HW_INJ'
    ]

    removed_O3a = []
    removed_O3b = []

    for detection in detections[:]:
        for period in ['O3a', 'O3b']:
            # Skip detection based on period boundary
            if period == 'O3a' and detection >= 1253977218:
                continue
            if period == 'O3b' and detection < 1253977218:
                continue

            # List of JSON files to check against
            json_files = [f'data/H1_{injection}_{period}.json' for injection in injections]

            # Check if the detection is missing in any JSON files
            missing_files = main(detection, json_files)

            # If the detection is missing in any files, remove it from the list and track it
            if missing_files:
                print(f"Removing detection {detection} as it is missing in: {', '.join(missing_files)}")
                detections.remove(detection)
                if period == 'O3a':
                    removed_O3a.append(detection)
                elif period == 'O3b':
                    removed_O3b.append(detection)

    # Print the final lists and counts of removed detections
    print("\nFinal list of detections after removal:", detections)
    print("Total detections:", len(detections))

    # Print removed injections statistics
    print("\nRemoved O3a detections (< 1253977218):")
    print(removed_O3a)
    print(f"Total removed O3a detections: {len(removed_O3a)}")

    print("\nRemoved O3b detections (>= 1253977218):")
    print(removed_O3b)
    print(f"Total removed O3b detections: {len(removed_O3b)}")




# import json
# from typing import List
# from gwpy.segments import Segment, SegmentList

# def load_segments_from_json(file_path: str) -> SegmentList:
#     """
#     Load segments from a JSON file and return a SegmentList.

#     Parameters:
#         file_path (str): Path to the JSON file.

#     Returns:
#         SegmentList: A SegmentList containing the segments from the JSON file.
#     """
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     segments = data.get("segments", [])
#     return SegmentList([Segment(start, end) for start, end in segments])

# def find_intersection_between_analysis_and_injections(analysis_files: List[str], injection_files: List[str]) -> SegmentList:
#     """
#     Find the intersection between analysis segments and the complement of injection segments.

#     Parameters:
#         analysis_files (List[str]): A list of JSON file paths for analysis segments.
#         injection_files (List[str]): A list of JSON file paths for injection segments.
#         full_range (Segment): The full time range to consider.

#     Returns:
#         SegmentList: The intersection of analysis segments and the complement of injection segments.
#     """
#     # Load and merge all analysis segments
#     analysis_segments = SegmentList()
#     for file_path in analysis_files:
#         analysis_segments.extend(load_segments_from_json(file_path))
#     analysis_segments = analysis_segments.coalesce()

#     # Load and merge all injection segments, then find their complement
#     injection_segments = SegmentList()
#     for file_path in injection_files:
#         injection_segments.extend(~load_segments_from_json(file_path))
#     injection_segments = injection_segments.coalesce()

#     # Find the intersection between analysis segments and the complement of injection segments
#     intersection = analysis_segments & injection_segments

#     return intersection

# def main(analysis_files: List[str], injection_files: List[str]):
#     """
#     Main function to find the intersection between analysis segments and the complement of injection segments.

#     Parameters:
#         analysis_files (List[str]): List of JSON file paths for analysis segments.
#         injection_files (List[str]): List of JSON file paths for injection segments.
#         full_range (Segment): The full time range to consider.

#     Returns:
#         SegmentList: The intersection of analysis segments and the complement of injection segments.
#     """
#     # Find the intersection
#     intersection = find_intersection_between_analysis_and_injections(analysis_files, injection_files)

#     # Print the intersection segments
#     if intersection:
#         print("Intersection of analysis segments and injection segments:")
#         for segment in intersection:
#             print(f"[{segment[0]}, {segment[1]}]")
#     else:
#         print("No intersection found.")

#     return intersection


# if __name__ == "__main__":
#     # Example usage
#     injections = [
#         'NO_BURST_HW_INJ',
#         'NO_CBC_HW_INJ',
#         'NO_DETCHAR_HW_INJ',
#         'NO_STOCH_HW_INJ'
#     ]

#     for period in ['O3a', 'O3b']:
#         print(f'Checking {period}:')

#         # List of analysis files (e.g., Hanford and Livingston segments)
#         analysis_files = [
#             f'data/{period}_Hanford_segments.json',
#             f'data/{period}_Livingston_segments.json'
#         ]

#         # List of injection files
#         injection_files = [f'data/H1_{injection}_{period}.json' for injection in injections]

#         # Find and print the intersection
#         intersection = main(analysis_files, injection_files)
