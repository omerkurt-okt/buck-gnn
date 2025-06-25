import os
import multiprocessing as mp
from pathlib import Path
import subprocess
import math
from typing import List

def normalize_path(path: str) -> str:
    """Convert Windows path to TCL-friendly format"""
    return path.replace('\\', '/')

def create_tcl_script(directory: str, hm_files: List[str]) -> str:
    """Create TCL script content for processing specific HyperMesh files"""
    # Convert directory path to TCL format
    tcl_directory = normalize_path(directory)
    
    # Convert the list of files to TCL list format with normalized paths
    tcl_files_list = ' '.join(f'"{f}"' for f in hm_files)
    
    tcl_content = f'''
set directory "{tcl_directory}"
set hm_files {{{tcl_files_list}}}

if {{![file exists $directory]}} {{
    puts "Error: Directory '$directory' does not exist"
    return
}}

cd $directory

foreach hm_file $hm_files {{
    puts "Processing file: $hm_file"
    
    *clearmarkall
    *clearmark nodes 1
    *clearmark elems 2
    
    catch {{
        hm_answernext yes
        *closemodel
    }}
    
    *readfile "$hm_file" 0
    
    catch {{
        *createmark elements 2 "displayed"
        *assignsystem_option elements 2 0 0 1 0 2 0
        *vectorsoff 
    }}
    
    catch {{
        *createentity mats cardimage=MAT1 includeid=0 name="Aluminum"
        *setvalue mats id=1 STATUS=1 1=76000
        *setvalue mats id=1 STATUS=1 3=0.3
        
        *createentity props cardimage=PSHELL includeid=0 name="Shell_Default_Prop"
        *clearmark properties 1
        *setvalue props id=1 STATUS=1 95=1.5
        *setvalue props id=1 materialid={{mats 1}}
        
        *setvalue comps id=1 propertyid={{props 1}}
        *createmark nodes 1 "all"
        *nodemarkaddtempmark 1
        *nodecleartempmark 
        *retainmarkselections 0
    }}
    
    set bdf_name [file rootname $hm_file]
    set template_dir [hm_info -appinfo SPECIFIEDPATH TEMPLATES_DIR]
    set template [file join $template_dir "feoutput" "nastran" "general"]
    
    *createstringarray 5 "HM_NODEELEMS_SET_COMPRESS_SKIP " "EXPORT_DMIG_LONGFORMAT " \
    "INCLUDE_RELATIVE_PATH " "HMCOMMENTS_SKIP" "IDRULES_SKIP"
    
    hm_answernext yes
    *feoutputwithdata "$template" "$directory/$bdf_name.bdf" 0 0 1 1 5
    
    puts "Completed processing: $hm_file"
    puts "Exported as: $bdf_name.bdf"
}}

puts "All files processed"
*quit 1
'''
    return tcl_content

def split_list(lst: list, n: int) -> List[list]:
    """Split a list into n roughly equal parts"""
    k, m = divmod(len(lst), n)
    
    return [lst[i * int(k) + min(i, int(m)):(i + 1) * int(k) + min(i + 1, int(m))] for i in range(int(n))]

def process_hm_batch(batch_files: List[str], hypermesh_path: str, work_dir: str):
    """Process a specific batch of HyperMesh files"""
    # Create TCL script for this batch
    tcl_content = create_tcl_script(work_dir, batch_files)
    tcl_file = os.path.join(work_dir, f"process_batch_{os.getpid()}.tcl")
    
    # Ensure the TCL file path is also properly formatted
    tcl_file = normalize_path(tcl_file)
    
    with open(tcl_file, 'w') as f:
        f.write(tcl_content)
    
    print(f"Process {os.getpid()} starting on {len(batch_files)} files")
    print(f"TCL script created at: {tcl_file}")
    
    # Run HyperMesh with the TCL script
    try:
        subprocess.run([hypermesh_path, "-clientconfig", "hwfepre.dat", "-nocommand", "-uNastran","-nouserprofiledialog","-tcl", tcl_file], check=True)
        print(f"Process {os.getpid()} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error in process {os.getpid()}: {e}")
    finally:
        # Clean up TCL script
        try:
            os.remove(tcl_file)
        except Exception as e:
            print(f"Warning: Could not remove TCL script {tcl_file}: {e}")

def main():
    # Configuration
    HYPERMESH_PATH = r"C:\Program Files\Altair\2022.3\hwdesktop\hw\bin\win64\hw.exe"
    WORK_DIR = r"D:\Projects_Omer\GNN_Project\0_Data\TEST\SHAPES\Shapes_OnlyWO\wo_cutout"  # Working directory
    NUM_PROCESSES = max(1, mp.cpu_count()/4)
    
    if not os.path.exists(WORK_DIR):
        print(f"Error: Working directory does not exist: {WORK_DIR}")
        return
        
    if not os.path.exists(HYPERMESH_PATH):
        print(f"Error: HyperMesh executable not found at: {HYPERMESH_PATH}")
        return
    
    # Get all HM files
    hm_files = [f.name for f in Path(WORK_DIR).glob("*.hm")]
    if not hm_files:
        print("No .hm files found in the directory")
        return
    
    # Split files into batches
    file_batches = split_list(hm_files, NUM_PROCESSES)
    
    print(f"Found {len(hm_files)} .hm files")
    print(f"Splitting work across {NUM_PROCESSES} processes")
    print(f"Working directory: {WORK_DIR}")
    
    for i, batch in enumerate(file_batches):
        print(f"Batch {i+1} will process {len(batch)} files")
    
    # Create process pool and process files in parallel
    with mp.Pool(int(NUM_PROCESSES)) as pool:
        pool.starmap(process_hm_batch, 
                    [(batch, HYPERMESH_PATH, WORK_DIR) for batch in file_batches])
    
    print("All processing completed")

if __name__ == "__main__":
    main()