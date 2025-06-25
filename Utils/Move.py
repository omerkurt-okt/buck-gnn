import os
import shutil

def move_files_from_subfolders(main_folder):
    # List all subfolders in the main folder
    for root, dirs, files in os.walk(main_folder):
        # Skip if we're in the main folder
        if root == main_folder:
            continue
            
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            
            # Determine the destination path
            destination_path = os.path.join(main_folder, file)

            # Resolve name conflicts
            if os.path.exists(destination_path):
                base, extension = os.path.splitext(file)
                counter = 1
                # Create a new filename with a counter
                while os.path.exists(destination_path):
                    new_file_name = f"{base}_{counter}{extension}"
                    destination_path = os.path.join(main_folder, new_file_name)
                    counter += 1
            
            # Move the file
            shutil.move(file_path, destination_path)

# Call the function with your main folder path
main_folder_path = r'D:\Projects_Omer\DLNet\GNN_Project\0_Data\TEST\wo_stiffener\Shapes_100'
move_files_from_subfolders(main_folder_path)