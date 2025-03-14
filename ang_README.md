# Quantum Chemistry Laboratory – Artificial Intelligence Model

Project prepared as part of the rotational laboratory course "Quantum Chemistry" by Jan Wolski, March 2025.
Full project report available in `raport.pdf` (unfortunately in Polish)

## Project Goal
The goal of this project is to create an artificial intelligence model that predicts the electron affinity (EA) of chemical compounds based on chemical descriptors.

## Data Source
The data was obtained using a web scraper written in Go, utilizing the gorod library. The source of the data is the [NIST Chemistry WebBook](https://webbook.nist.gov). Data is saved in CSV format containing the compound name, electron affinity (EA) value, and molecular formula.

## Chemical Descriptors
The project utilizes 32 chemical descriptors, including:
- Number of atoms in a molecule
- Number of atoms of individual elements (e.g., H, C, N)
- Molecular mass
- Ionic radii, Pauling electronegativities, polarizabilities, ionization energies
- Hydrogen Deficiency Index (HDI)

The complete list of descriptors is available in the file `descryptors.py`.

## Model
An advanced model utilizing convolutional layers (CNN), a self-attention mechanism, and classical Dense layers:
- MAE = 0.1415
- RMSE = 0.2205
- R² = 0.935

## Running the Model
1. Download or clone the repository.
2. Download or generate the data using the scraper.
3. Run the script `descryptors.py` to generate chemical descriptors.
4. Train the model using `models.py`.

### Using the Trained Model
To use the trained model for predicting electron affinity, execute:
```bash
python use_model.py [chemical_formula]
```

Example:
```
python use_model.py "C7H13-"
```

## Project Structure
- `descryptors.py` – calculation of chemical descriptors
- `models.py` – implementation of the neural network
- `main.py` – script to use the model
- `src` – contains the model files: tokenizer, scaler, and trained model

## Libraries Used
- [PyTorch](https://pytorch.org)
- [scikit-learn](https://scikit-learn.org)
- [pandas](https://pandas.pydata.org)
- [mendeleev](https://github.com/lmmentel/mendeleev)

## Potential Development Directions
- Increasing dataset size
- Integration of an autoencoder to improve prediction accuracy
- Expansion of the descriptor set

## References
- [NIST Chemistry WebBook](https://webbook.nist.gov)
- [PyTorch](https://pytorch.org)
- [scikit-learn](https://scikit-learn.org)


