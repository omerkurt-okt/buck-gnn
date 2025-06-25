import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from tqdm import tqdm
import re

class NastranRunner:
    def __init__(self, nastran_path: str, num_threads: int = 4):
        self.nastran_path = nastran_path
        self.num_threads = num_threads
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nastran_runs.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def find_hidden_nodes(self, lines):
        """Find nodes that are not referenced in any element"""
        nodes = set()
        used_nodes = set()
        
        # First pass: collect all node IDs
        for line in lines:
            if line.strip().startswith('GRID'):
                node_id = int(line[8:16].strip())
                nodes.add(node_id)
            
            # Check for element definitions (CQUAD4, CTRIA3, CBAR, etc.)
            elif any(line.strip().startswith(elem) for elem in ['CQUAD4', 'CTRIA3', 'CBAR']):
                try:
                    if line.strip().startswith('CQUAD4'):
                        # Get the 4 node fields for CQUAD4 (fixed positions)
                        node_ids = [
                            int(line[24:32].strip()),  # Node 1
                            int(line[32:40].strip()),  # Node 2
                            int(line[40:48].strip()),  # Node 3
                            int(line[48:56].strip())   # Node 4
                        ]
                    elif line.strip().startswith('CTRIA3'):
                        # Get the 3 node fields for CTRIA3
                        node_ids = [
                            int(line[24:32].strip()),  # Node 1
                            int(line[32:40].strip()),  # Node 2
                            int(line[40:48].strip())   # Node 3
                        ]
                    elif line.strip().startswith('CBAR'):
                        # Get the 2 node fields for CBAR
                        node_ids = [
                            int(line[24:32].strip()),  # Node 1
                            int(line[32:40].strip())   # Node 2
                        ]
                    
                    # Add all valid node IDs to used_nodes
                    for node_id in node_ids:
                        if node_id > 0:  # Ensure it's a valid node ID
                            used_nodes.add(node_id)
                            
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error processing element line: {line.strip()}, Error: {str(e)}")
                    continue
        # Return nodes that exist but are not used in any element
        return nodes - used_nodes

    def modify_bdf_file(self, bdf_path: str) -> bool:
        """Modify BDF file as needed"""
        try:
            # Read the file
            with open(bdf_path, 'r') as file:
                lines = file.readlines()

            modified = False
            
            # Find hidden nodes
            hidden_nodes = self.find_hidden_nodes(lines)
            if hidden_nodes:
                self.logger.info(f"Found hidden nodes in {bdf_path}: {hidden_nodes}")
                
                # Create new lines excluding hidden nodes
                new_lines = []
                for line in lines:
                    if line.strip().startswith('GRID'):
                        node_id = int(line[8:16].strip())
                        if node_id not in hidden_nodes:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                lines = new_lines
                modified = True

            # Look for and modify the EIGRL line
            for i, line in enumerate(lines):
                if line.strip().startswith('EIGRL          1') and '0.0' not in line:
                    lines[i] = 'EIGRL          1     0.0               1\n'
                    modified = True
                    break
                    
            for i, line in enumerate(lines):
                if line.strip().startswith('MAT1           4'):
                    lines[i] = 'MAT1           4  76000.              .3\n'
                    modified = True
                    break

            # If modification was made, write back to file
            if modified:
                with open(bdf_path, 'w') as file:
                    file.writelines(lines)
                self.logger.info(f"Modified file {bdf_path}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error modifying file {bdf_path}: {str(e)}")
            return False

    def run_nastran(self, bdf_path: str) -> bool:
        """Run Nastran analysis for a single BDF file"""
        try:
            bdf_path = os.path.abspath(bdf_path)
            output_dir = os.path.abspath(os.path.dirname(bdf_path))
            
            # Check and modify BDF file if needed
            self.modify_bdf_file(bdf_path)
            
            cmd = [
                self.nastran_path,
                bdf_path,
                "scr=yes",
                "mem=2048",
            ]
            
            process = subprocess.run(
                cmd,
                cwd=output_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Clean up extra files
            base_path = os.path.splitext(bdf_path)[0]
            for ext in ['.log', '.f04', '.f06']:
                file_to_remove = base_path + ext
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
            
            if process.returncode != 0:
                self.logger.error(f"Nastran analysis failed for {bdf_path}: {process.stderr}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {bdf_path}: {str(e)}")
            return False

    def process_directory(self, directory: str):
        """Process all BDF files in directory and subdirectories"""
        # Find all BDF files
        bdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.bdf'):
                    bdf_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(bdf_files)} BDF files to process")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Use tqdm for progress bar
            list(tqdm(
                executor.map(self.run_nastran, bdf_files),
                total=len(bdf_files),
                desc="Processing BDF files"
            ))

def main():
    nastran_path = r"C:\Program Files\MSC.Software\MSC_Nastran\2020sp1\bin\nastran.exe"
    directory = r"D:\Projects_Omer\GNN_Project\0_Data\TEST\w_stiffener\Shapes_100"  # Directory containing BDF files
    
    runner = NastranRunner(nastran_path=nastran_path, num_threads=4)
    runner.process_directory(directory)

if __name__ == "__main__":
    main()