
# CA-AT
The codes for Conflict-Aware Adversarial Training
=======
## Env Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
sh cifar10_at.sh # Vanllia AT
sh cifar10_ca_at.sh # CA-AT
```

## Evaluation

```bash
sh eval_pgd.sh # Test Adv Accs on PGD attack with Different Steps
sh eval_untarget.sh # Test Adv Accs on 
```
