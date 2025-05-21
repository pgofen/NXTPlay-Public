# P-NXTPlay

**P-NXTPlay** is a machine learning incubator project developed through P-ai in collaboration with the Pomona-Pitzer Football program. The goal was to automate the detection of offensive formations from practice footage, using computer vision and spatial data to streamline coaching workflows. This repository contains sample notebooks and materials from the project.

## Technical Approach

Our pipeline included several key stages:

1. **Field Segmentation**  
   `FieldSegmentation.ipynb` processes raw footage to delineate the playing field, laying the groundwork for accurate player detection.

2. **Player Detection and Labeling**  
   `DetectPlayersAndLabelOffenseAndDefense.ipynb` uses computer vision to locate players and distinguish offensive from defensive units.

3. **Formation Detection**  
   `FormationDetection.ipynb` builds a neural network model that ingests player coordinates and predicts offensive formations.

## Results

We developed a functional MVP capable of predicting offensive formations with over 50% accuracy, representing a threefold improvement over initial testing benchmarks. This was achieved despite limited training data and computational resources. 

## Repository Contents

- `FieldSegmentation.ipynb`: Field segmentation from practice footage  
- `DetectPlayersAndLabelOffenseAndDefense.ipynb`: Player detection and team labeling  
- `FormationDetection.ipynb`: Neural network for formation prediction  
- `initialData.csv`: Raw coordinate data  
- `simplifiedDataWithCoordinates.csv`: Labeled player positions and formations  
- `largeModel.pt`: Trained PyTorch model

  
