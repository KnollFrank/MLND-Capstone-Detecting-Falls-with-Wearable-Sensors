# Machine Learning Engineer Nanodegree
## Setup

In order to execute the Jupyter Notebook DetectingFalls.ipynb, execute the following shell commands:

```shell
conda create -n capstone python=3.6
source activate capstone
pip install -r requirements/requirements.txt
python -m ipykernel install --user --name capstone
cd data; 7z x FallDataSet.7z.001; cd ..
jupyter notebook DetectingFalls.ipynb
```
