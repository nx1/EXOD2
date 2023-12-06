from download_observations import download_observations
from preprocessing import preprocess_observations
from source_detection import detect_sources_in_all
from postprocessing import postprocess_all

def main():
    target_directory = 'observations'
    observation_list_file = 'observations.txt'
    output_directory = 'output'

    downloaded_files = download_observations(target_directory, observation_list_file)
    preprocessed_files = preprocess_observations(downloaded_files, output_directory)
    detected_sources = detect_sources_in_all(preprocessed_files, output_directory)
    postprocessed_results = postprocess_all(detected_sources, output_directory)

if __name__ == "__main__":
    main()
