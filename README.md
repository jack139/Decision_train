# Train decision model use unsloth

```bash
sudo pip3.12 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
sudo pip3.12 install https://download.pytorch.org/whl/cu118/xformers-0.0.30-cp312-cp312-manylinux_2_28_x86_64.whl

cd ~/Codes/LLMs/source/unsloth
sudo pip3.12 install --upgrade --no-cache-dir .
```


```bash
UNSLOTH_ENABLE_LOGGING=1 python3.12 test.py
```
