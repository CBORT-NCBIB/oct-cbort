
# README - oct-cbort

A python package for reconstructing raw OCT/OFDI data from the Center for Biomedical OCT Research @ The Wellman Center for Photomedicine.  
[https://octresearch.org/](https://octresearch.org/)

The library was developped by the labs of Professor Ben Vakoc and Professor Brett Bouma. 

### Background:
Unlike traditional imaging modalities, OCT contrasts requires "reconstruction" of the acquired interferometric fringes. With this in mind, the `oct-cbort` Python package serves the following purposes:

1. When coupled with acquisition frameworks, the package allows for automated processing of raw OCT data into analyzable endpoint images with no/minimal user input. This is desirable for _users_ of OCT technology, whose primary goal is to view OCT images.

2. The package has been designed in a modular format to allow for "scripting", namely for OCT contrast _developpers_. The ability to write Python scripts in Jupyter Notebooks alongside core reconstruction algorithms and validated priors allows for the rapid debugging and creation of novel algorithms.

3. Finally, the package is a general tool for dissemination, education and rapid community uptake of novel contrast algorithms developped at CBORT.


### Steps to begin processing on a new machine:

0. Install Anaconda (In Windows, choose to install with conda exec in PATH). Install GitHub Desktop.

1. Clone or download oct-cbort code from GitHub

2. Open terminal (MacOS/Linux) or command prompt (Windows) and type:    
      1. `conda create --name oct38 python=3.8`
      2. `conda activate oct38`

3. Install requirements:  
 
    1. For CPU processing :  
         `pip install -r requirements.txt`  
    2. For GPU processing (only perform if you have an NVIDIA GPU:          
         1.  Install NVIDIA Cuda toolkit (V11.0) (for nvcc)     
         2. `conda install -c anaconda cudatoolkit=11` (for cuda and cudnn)     
         3. `pip install -r requirements-gpu.txt` (for rest of dep. including cupy)     


4. Within the terminal navigate to `oct-cbort`  directory (ie. cd Documents\Github\oct-cbort) and run a first test by processing example data by typing:    
     `python examples/3.py`  
    
5. Run processing on a directory with desired settings:  
    1. For all frames in file:  
         `python -m oct E:\OFDIData\user.Damon\[p.SPARC_Phantom][s.BirefPhant_0125_7][01-25-2019_15-06-24] tomo+struct+angio+ps mgh all`   
    2. For a range of frames:  
         `python -m oct C:\Users\ofdi_process_six\Desktop\[p.SPARC][s.CalibTest_holder_ChkNerve_NoChange_2][11-24-2018_11-41-52] tomo+angio+struct                       mgh+tif 1-10`   
    3. For a very simply gui:  
         `python -m oct`   

6. For Jupyter Notebook tutorials, it is best to perform the following as well to assure your the IPkernel can see your new environment (these are not requird for post-processing) : 
   To install these type in terminal:   
      1. `conda install -c anaconda ipykernel`
      2. `python -m ipykernel install --user --name=oct38`


### Usage tutorials:

Jupyter notebook tutorials are found in /tutorials. These notebooks show how to use the library for more technical scripting purposes. Best practice is to run all tutorials right after installing oct-cbort repository to get familiar with the extended funcitonality.

### Contributing:

To contribute, please create your own branch, develop on it, and once your addition is tested,  create a "pull request" for administrators to look over before adding to the main branch.

### Issues:

If there are issues or bugs, please submit as much information as possible to the issue channel of the repository for future debugging and tracking.  
 

### Documentation

While the repository is not currently public, and therefore there is no public document hosting, we do provide some _unfinished_ documentation!

To view the documentation, simply navigate to the oct-cbort directory and type the following in the cmd\terminal:

` python docs\docs.py`

This will launch the built html files in your web browser.