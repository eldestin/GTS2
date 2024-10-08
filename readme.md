# GTS2

*GTS2: Integrating Graph Temporal-Structural Dependencies and Textual Semantics for Outdated Fact Detection*

## Architecture
Our model's architecture is shown as below:
![image](https://github.com/user-attachments/assets/d6c86ea4-3375-4bad-8944-cad4b552076b)


## Experiment

To reproduce the experiments, please configure an environment:

- Install all the packages using  `pip install -r requirements.txt`
- You may also download Bert initial weight on [bert-base-uncased · Hugging Face](https://huggingface.co/bert-base-uncased) in advance.

Then, start training:

- First specify the dataset: go to `train.py` and change parameter `wiki = TRUE / FALSE` to specify dataset as `wiki / Yago`.
- Run `python train.py`

Start Testing:

- Test function will be automatically used after training procedure.
- Go to `train.py` and comment `run_with_amp` function if you only want to test.
- Then run `python train.py`
