# Data Cleaning and Hybrid Transformer-GRU Model

## Overview

This repository contains two main components:

1. **Cleaning Module**: A set of scripts and tools for preparing and cleaning AIS (Automatic Identification System) data.
2. **AI_V2 Module**: A hybrid Transformer-GRU model designed for predicting coordinates based on the cleaned AIS data.

---

## Folder Structure

### 1. Cleaning Module (`Cleaning/`)

The `Cleaning` folder contains scripts for preprocessing and cleaning AIS data. These scripts handle tasks such as:

-   Filtering cargo data (`cargoFilter.py`)
-   Removing outliers (`removeOutliers.py`)
-   Reducing trajectory data (`trajectoryReducer.py`)
-   Detecting and clustering ship movements (`clusterDetection.py`, `clusterRunner.py`)
-   Handling missing time intervals (`missingTime.py`)
-   Extracting geographical features (`geo.py`)

#### Key Files:

-   `cleaning.py`: Main script for cleaning AIS data.
-   `ShipNotMovingFiltre.py`: Filters out ships that are not moving.
-   `polyIntersect.py`: Handles polygon intersection for geographical data.
-   `main.py`: Entry point for running the cleaning pipeline.

### 2. AI_V2 Module (`AI_V2/`)

The `AI_V2` folder contains the implementation of a hybrid Transformer-GRU model for predicting coordinates. This module leverages the cleaned AIS data to train and evaluate the model.

#### Key Features:

-   **Transformer-GRU Architecture**: Combines the strengths of Transformers for capturing long-term dependencies and GRUs for sequential data processing.
-   **Model Training and Evaluation**: Scripts for training and evaluating the model (`Transformer.py`).
-   **Preprocessing**: Scripts for preparing data for the model (`preproccess.py`, `preproccess20.py`, `preproccess50.py`).

#### Key Files:

-   `Transformer.py`: Main script for training the model.
-   `merge.py`: Combines multiple datasets for training.

---

## Models (`AI_V2/models/`)

The `models` folder contains pre-trained models and checkpoints for the hybrid Transformer-GRU model. These can be used for inference or further fine-tuning.

---

## Data (`Data/`)

The `Data` folder includes geographical data files such as:

-   `land_poly.geojson`: Land polygon data.
-   `river_poly.geojson`: River polygon data.

---

## Scripts (`scripts/`)

The `scripts` folder contains shell scripts for automating various tasks, including:

-   Cleaning AIS data (`cleaning.sh`)
-   Running clustering algorithms (`clusterRunner.sh`)
-   Extracting features (`extractor.sh`)
-   Reducing trajectories (`trajectory_reducer.sh`)

---

## Getting Started

### Prerequisites

-   Python 3.10 or later
-   Required Python libraries (install using `requirements.txt` if available)
-   Shell environment for running `.sh` scripts

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd data-cleaning
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Cleaning AIS Data

1. Navigate to the `Cleaning` folder:
    ```bash
    cd Cleaning
    ```
2. Run the main cleaning script:
    ```bash
    python main.py
    ```
    or use the bash script:
    ```bash
    sbatch main.sh
    ```

#### Training the Hybrid Transformer-GRU Model

1. Navigate to the `AI_V2` folder:
    ```bash
    cd AI_V2
    ```
2. Preprocess the data:
    ```bash
    python preproccess.py
    ```
3. Train and eval the model:
    ```bash
    python Transformer.py
    ```

---

## Running on Supercomputers with Singularity

This project was designed to run on supercomputers using Singularity containers and the SLURM workload manager. All scripts, including the data cleaning pipeline and AI model training, are executed via `sbatch` commands for job scheduling.

### Singularity Containers

Singularity containers were used to ensure a consistent runtime environment across different systems. The AI model and data cleaning scripts were executed within these containers.

#### Setting Up Singularity

1. Build or pull the required Singularity image:

    ```bash
    singularity build <image-name>.sif <definition-file>
    ```

    or

    ```bash
    singularity pull <image-name>.sif <image-url>
    ```

2. Load the Singularity module (if required by the supercomputer):
    ```bash
    module load singularity
    ```

#### Running Scripts with Singularity

To run a script inside a Singularity container, use the following pattern:

```bash
singularity exec <image-name>.sif sbatch <script-name>.sh
```

### Example Workflow

#### Cleaning AIS Data

1. Submit the cleaning job:
    ```bash
    singularity exec <image-name>.sif sbatch Cleaning/main.sh
    ```

#### Training the Hybrid Transformer-GRU Model

1. Submit the training job:
    ```bash
    singularity exec <image-name>.sif sbatch AI_V2/Transformer.sh
    ```

This approach ensures reproducibility and efficient resource utilization on supercomputers.

### Building the Singularity Container

To build the required Singularity container for this project, simply run the `builder.sh` script. This script uses SLURM to allocate resources and builds the container using Singularity's `--fakeroot` option.

#### Steps to Build:

1. Navigate to the `AI_V2` folder:
    ```bash
    cd AI_V2
    ```
2. Submit the build job:
    ```bash
    sbatch builder.sh
    ```

This will create the Singularity container `ai_container_l4_v2.sif` based on the definition file `ai_container_l4.def`. The build process is configured to use 20 CPUs, 100GB of memory, and a maximum runtime of 12 hours.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

-   The AIS data used in this project.
-   Libraries and frameworks that made this project possible.
