# Quantum Chemistry Lab – Artificial Intelligence Model

Project carried out as part of the rotational lab "Quantum Chemistry" by Jan Wolski, March 2025.  
Full project preparation report in `raport.pdf`  
Polska wersja README.md <u>[here](README.md)</u>

## Project Goal
The goal of this project is to develop an artificial intelligence model that predicts the electron affinity (EA) of a given chemical compound based on its chemical descriptors.

## Data Source
The data was collected using a web scraper written in Go, utilizing the `gorod` library. The data source is the [NIST Chemistry WebBook](https://webbook.nist.gov). The data is saved in CSV format and includes the compound name, electron affinity (EA), and molecular formula.

## Chemical Descriptors
The project uses 32 chemical descriptors, including:
- Total number of atoms in the molecule
- Number of atoms of specific elements (e.g., H, C, N)
- Molecular mass
- Ionic radii, Pauling electronegativities, polarizabilities, ionization energies
- Hydrogen Deficiency Index (HDI)

A full list of descriptors can be found in the `descryptors.py` file.

## Model
An advanced model utilizing convolutional layers (CNN), self-attention mechanisms, and classic dense layers:
- MAE = 0.1415  
- RMSE = 0.2205  
- R² = 0.935  

## Running the Model
1. Download or clone the repository.
2. Download or generate the data using the scraper.
3. Run the `descryptors.py` script to generate chemical descriptors.

### Training the Model
To train the model on the dataset, use:
```bash
python main.py --model <model_type> --action train
```

Example:
```
python main.py --model class --action train
```

### Using a Pretrained Model
To use a pretrained model to predict the electron affinity of a compound, use:
```bash
python main.py --model <model_type> --action use --target <chemical_formula>
```

Example:
```
python main.py --model reg --action use --target "C7H13-"
```

### Available `--model` Types
* Regression on all values – `reg`  
* Regression on positive values only – `preg`  
* Classifier for EA > 3.6 eV – `class`

### Available `--action` Types
* Use a pretrained model from the `src` folder – `use`  
* Train a new model – `train`

## Project Structure
- `descryptors.py` – chemical descriptor generation  
- `models.py` – neural network implementation  
- `main.py` – main script for training and inference  
- `src` – contains model files: tokenizer, scaler, and trained models  

## Libraries Used
- [PyTorch](https://pytorch.org)  
- [scikit-learn](https://scikit-learn.org)  
- [pandas](https://pandas.pydata.org)  
- [mendeleev](https://github.com/lmmentel/mendeleev)

## Potential Development Directions
- Increasing dataset size  
- Integrating an autoencoder to improve prediction quality  
- Expanding the set of chemical descriptors  

## References
- [NIST Chemistry WebBook](https://webbook.nist.gov)  
- [PyTorch](https://pytorch.org)  
- [scikit-learn](https://scikit-learn.org)
