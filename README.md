# Through the looking glass: Deep interpretable dynamic directed connectivity in resting fMRI
# DICE model




Usman Mahmood, Zengin Fu, Satrajit S. Ghosh, Vince D. Calhoun, Sergey M. Plis

Accepted at NeuroImage 
Paper : https://www.sciencedirect.com/science/article/pii/S1053811922008588?via%3Dihub



#### Dependencies:
* PyTorch
* Scikit-Learn

```bash
conda install pytorch torchvision -c pytorch
conda install sklearn
```

### Installation 

```bash
# PyTorch
conda create --name dice python=3.9
conda activate dice
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -e .
pip install -r requirements.txt
```

```
conda activate dice
PYTHONPATH=./ python scripts/run_experiments.py --ds fbirn --prefix test
```