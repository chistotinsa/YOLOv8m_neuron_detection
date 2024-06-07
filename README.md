# YOLOv8m neuron detection

This model allows count neurons stained using the Nissl method in the CA1 field of the hippocampus.
It takes 300μm area on a hippocampus randomly and counts alive cells on it.

The model takes photos of a rat`s hippocampus specimen as input. The microscopic magnification is 20x, 
base photo resolution - 2048х1504 pixels.

![plot](./readme_pics/hippocampus_raw.jpg)
![plot](./readme_pics/detected_cells.jpg)


## Usage
#### 1. Clone the repo and install all dependencies
```python
git clone https://github.com/chistotinsa/YOLOv8m_neuron_detection
pip install -r requirements.txt
```

#### 2. Run the script
To run the main script — 
1. Put your photos 2048х1504 pxls into a single folder (if you have another photo resolution - change the PIXEL_STEP value in scripts/neuron_count.py according to the docstrings)
2. Run neuron_count.py from cmd/bash with flags --photo_path (folder with photos to process) and --results (path to the excel file with results) like this:  
`python neuron_count.py --photo_path 'F:\Desktop\projects\neuron_detection\photos' --results 'F:\Desktop\projects\neuron_detection\results\results.xlsx'`
4. The output includes Excel table with number of alive neurons on a random
   300 μm hippocampus area per each input photo and photos with detected neurons outlined in red boxes ('runs' folder)

## How we made this model
The idea arose due to the fact that many biologists, for the purpose of brain research, are forced to 
spend a lot of time manually counting brain cells in microscopic images. This job takes an incredible amount of hours. 
That's why we came up with this project.

1. The base model is YOLOv8m trained on the blood cells
   by Keremberke taken [here](https://github.com/keremberke/awesome-yolov8-models)
2. We made our own hippocampus neurons dataset, labeled it
3. and finetuned Keremberke`s YOLOv8m on it.
4. Then we wrote script that counts alive neurons on a random 300 μm hippocampus area.
