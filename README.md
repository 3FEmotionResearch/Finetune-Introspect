#Finetune emotion introspect

This project uses Meta-Llama-3-8B-Instruct to perform inference on emotional datasets. And use based on the model's output, it finetune the model to have the capability of introspect when analyizng emotion in transcript with confidence level. 

## Setup
Create a A100 machine on paperspace, with ML-in-the-box for A100 model. 

git config --global credential.helper store
huggingface-cli login (with "read" permission key)
Request model access in hugginface and get approved in email.

1. Install dependencies:
   ```bash
   # Create a clean sandbox
   python3 -m venv venv

   #Activate it
   source venv/bin/activate


   pip install -r requirements.txt
   ```
2. Place your CSV dataset in the `data/` directory.

3. Finetune:
   ```bash
   python run_finetuning.py (this will finetune the model to be introspective)
   ```

4. Pure infer step (no need to run):
   ```bash
   python infer_emotions.py (this will let the model infer 10 times for each question in data folder)
   ```

5. Create fine tune dataset (no need to run, data already exist in repo):
   ```bash
   python create_introspection_dataset.py (this will create formatted dataset based on above infer output to finetune the model)
   ```


## Notes
- The script will be updated to process your dataset once provided. 