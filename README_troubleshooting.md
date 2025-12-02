___

## Troubleshooting

### OpenMP Error on Windows

If you encounter an OpenMP error like:
```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

This is already handled in the scripts by setting `KMP_DUPLICATE_LIB_OK=TRUE`. If you still see this error, you can manually set it:

**Windows (Command Prompt):**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
python train_vae_image_retrieval.py
```

**Windows (PowerShell):**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_vae_image_retrieval.py
```

**Linux/Mac:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python train_vae_image_retrieval.py
```

### CUDA/GPU Issues

The scripts automatically detect and use GPU if available. To force CPU usage, modify the device selection in the scripts or set:
```python
device = torch.device("cpu")
```

### Missing Dependencies

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For `pythonocc-core` (required for STEP file processing), use conda:
```bash
conda install -c conda-forge pythonocc-core
```

