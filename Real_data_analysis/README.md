# Data analysis for TCGA LUAD data

# Main Steps
## 1. Data download  
* The gene expression data and clinical data  

    The gene expression data and clinical data can be downloaded from cBioportal [https://www.cbioportal.org/study/summary?id=luad_tcga_gdc](https://www.cbioportal.org/study/summary?id=luad_tcga_gdc). We save the following two files related to gene expression and clinical data in the folder `raw_data/`:
    * `raw_data/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt` contains the gene expression data
    * `raw_data/data_clinical_patient.txt` contains the clinical features (AGE, SEX, TOBACCO_SMOKING_HISTORY_INDICATOR, SMOKING_PACK_YEARS, AJCC_PATHOLOGIC_TUMOR_STAGE), and the survival outcome (OS_MONTHS and OS_STATUS).
    
    To download the gene pathway data, we use the R package `KEGGREST` to download the cancer related pathway "hsa05223".

* Pathological imaging data  

    The raw pathological images can be downloaded from can be downloaded from GDC [https://portal.gdc.cancer.gov/projects/TCGA-LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD). Here, the images are in SVS format. 

## 2. Data preprocessing  
* Pathological imaging data 
    * Ferature extraction using the code in `image_pipeline.py`

        For each svs file, we will first extract sub-image (based on the crop over raw image) level features, and then aggregate using median over all sub-images. Specifically, we can use the following command to run the code:
        ```bash  
        python image_pipeline.py --svs_file <path_to_svs_file> --output_folder <folder_to_save_csv_file>
        ```
        (For a patient with multiple images, we will randomly select one of these images for analysis.)
    * The processed pathological imaging features are saved in `raw_data/imaging_features.csv`
    
* Gene expression data and clinical data  
    * Gene expression data  
    We first filter the genes according to the KEGG pathway. For the selected genes, we then apply standardization, remove outliers, and rescale the values to the range [0, 1].

    * Clinical data  
        * Age: standardized
        * Gender: 1 for male, 0 for female
        * Smoking1: TOBACCO_SMOKING_HISTORY_INDICATOR, 1 for smoking, 0 for non-smoking
        * Smoking2: SMOKING_PACK_YEARS, continuous variable, standardized and rescale the values to the range [0, 1].
        * Stage: AJCC_PATHOLOGIC_TUMOR_STAGE, we code 3 for "Stage III" and above, 2 for all "Stage II"/"Stage IB", and 1 for all "Stage I". We rescale the values to the range [0, 1].

* Data Merge  
    We merge the clinical data, gene expression data, and pathological image data based on the patient ID. 

* The code for the data preprocessing is in `prepare.Rmd`, which use the raw data from folder `./raw_data`, and output the processed data in folder `/processed_data`. Here, we have the following files:
    * `X_clip.csv`: imaging features
    * `Z_clip.csv`: genomic features
    * `E_clip.csv`: clinical features
    * `survival_outcome.csv`: survival outcome (time and event)

## 3. Model fitting and evaluation  
The related code can be found in `real_data_analysis.py`. The code contains the following steps:
* Data read and preprocessing
* The 100 random experiments, including data splitting, model fitting, and model evaluation. We also output the estimated linear coefficients and the variable selection reuslts for the nonlinear parts.  
* The related code can be found in `real_data_analysis.py`, which use the processed data from folder `./processed_data`, and output the results in folder `/real_data_results`. Here, we have the following files:
    * `0_result.csv`: the C-index result
    * `0_coefs_linear.npy`: the estimated linear coefficients for different methods
    * `0_coefs_nonlinear.npy`: the variable selection results for the nonlinear part for different methods