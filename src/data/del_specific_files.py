import os
import glob

def delete_files_in_folder(folder_path, file_extensions):
    """
    Delete files with specified extensions in the given folder and its subfolders.

    Parameters:
    folder_path (str): The path to the folder where files should be deleted.
    file_extensions (list of str): List of file patterns to delete (e.g., ['*.png', '*.eps', '*.csv', '*.txt']).
    """
    for file_extension in file_extensions:
        # Recursively search for files matching the extension in the folder and all subfolders
        files = glob.glob(os.path.join(folder_path, '**', file_extension), recursive=True)
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

def main():
    reports_folder = "reports"
    file_extensions = ["*.csv", "*.txt", "*.png", "*.eps"]
    
    if not os.path.exists(reports_folder):
        print(f"Folder does not exist: {reports_folder}")
        return

    delete_files_in_folder(reports_folder, file_extensions)

if __name__ == "__main__":
    main()