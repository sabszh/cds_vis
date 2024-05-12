# Assignment 4 - Detecting faces in historical newspapers
The aim of this assignment is to explore the occurrence of *human facial images* in historical newspapers by employing FaceNet with Torch. This inquiry seeks to address the following research question: how has the frequency of human facial depictions evolved in printed media over the past approximately two centuries? Are there discernible disparities, and if so, what implications might they carry?

## Data Source
The assignment works with a corpus of historic Swiss newspapers: the *Journal de Genève* (JDG, 1826-1994); the *Gazette de Lausanne* (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). You can read more about this corpus in the associated [research article](https://zenodo.org/records/3706863)

## Usage
To use this script, follow these steps:

1. Clone or download the repository and make sure you have the file structure as pointed out, and the needed files stored in `in`

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh
    ```

## Script Overview

## Output Summary

## Discussion of Limitations and Possible Steps to Improvement


## File Structure
The project directory should be structured as follows:

```
.
A4/
│
├── in/
│   └── newspapers/
│        ├── GDL/
│        │   ├── <filename>.jpg
│        │   └── ...
│        ├── IMP/
│        │   ├── <filename>.jpg
│        │   └── ...
│        ├── JDG/
│        │   ├── <filename>.jpg
│        │   └── ...
│        └── README-images.txt
├── out/
│   └── newspaper_sample_face_counts.csv
├── src/
│   └── newspaper_face_detection.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```