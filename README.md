**Dependencies**
  
```
conda create -n $env python=3.10
conda activate $env
pip install -r requirements.txt 
```

**How to use**

Train model: 
```
python main.py
```
Inference: 
```
python main.py --action inference --checkpoint_path $path_to_model
```
