set directory D:/Projects_Omer/DLNet/GNN_Project/0_Data/Dataset_Splitted/Test

    # Check if directory exists
    if {![file exists $directory]} {
        puts "Error: Directory '$directory' does not exist"
        return
    }
    
    # Change to specified directory
    cd $directory
    
    # Get all .hm files in the directory
    set hm_files [glob -nocomplain "*.hm"]
    
    if {[llength $hm_files] == 0} {
        puts "No .hm files found in directory: $directory"
        return
    }
    
    # Process each .hm file
    foreach hm_file $hm_files {
        puts "Processing file: $hm_file"
        
        # Close any open model
        *clearmarkall
		*clearmark nodes 1
		*clearmark elems 2
		
        catch {
			hm_answernext yes
			*closemodel
		}
        
        # Open the current .hm file
        *readfile "$hm_file" 0
        
        # Set material orientation
        catch {
			# *view "top"
			# *vectorsoff 
			*createmark elements 2 "displayed"
			*assignsystem_option elements 2 0 0 1 0 2 0
			*vectorsoff 
        }
        catch {
			*createentity mats cardimage=MAT1 includeid=0 name="Aluminum"
			*setvalue mats id=1 STATUS=1 1=76000
			*setvalue mats id=1 STATUS=1 3=0.3
			
			*createentity props cardimage=PSHELL includeid=0 name="Shell_Default_Prop"
			*clearmark properties 1
			*setvalue props id=1 STATUS=1 95=1.5
			*setvalue props id=1 materialid={mats 1}
			
			*setvalue comps id=1 propertyid={props 1}
			*createmark nodes 1 "all"
			*nodemarkaddtempmark 1
			*nodecleartempmark 
			*retainmarkselections 0
		}
        # Get filename without extension for BDF output
        set bdf_name [file rootname $hm_file]
        
        # Export as BDF
        set template_dir [hm_info -appinfo SPECIFIEDPATH TEMPLATES_DIR]
        set template [file join $template_dir "feoutput" "nastran" "general"]
        
        *createstringarray 5 "HM_NODEELEMS_SET_COMPRESS_SKIP " "EXPORT_DMIG_LONGFORMAT " \
        "INCLUDE_RELATIVE_PATH " "HMCOMMENTS_SKIP" "IDRULES_SKIP"
        
        # Export BDF
		hm_answernext yes
        *feoutputwithdata "$template" "${directory}/${bdf_name}.bdf" 0 0 1 1 5
        
        puts "Completed processing: $hm_file"
        puts "Exported as: ${bdf_name}.bdf"
    }
    
    puts "All files processed"

